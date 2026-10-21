# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the PrismML PTQ1_0 type registration and Hadamard transforms."""

import math

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, LlamaFileType

from vllm_gguf_plugin.ggml_ext import (
    GGML_TYPE_PQ2_0,
    GGML_TYPE_PTQ1_0,
    PTQ1_0_BLOCK_BYTES,
    QK_PTQ1_0,
    register_prism_types,
)
from vllm_gguf_plugin.gguf_utils import is_remote_gguf, is_valid_gguf_quant_type
from vllm_gguf_plugin.quantization.hadamard import (
    GGUFHadamardLayout,
    build_hadamard_layouts,
    fwht,
    hadamard_matrix,
    read_prism_hadamard_spec,
)

# Two 28-byte blocks packed by the reference C codec of PrismML-Eng/llama.cpp
# (ggml-quants.c, quantize_row_ptq1_0_ref) from the trits below with d = 1.0.
_GOLDEN_BLOCKS = bytes.fromhex(
    "2ba3f2ffd5d8b65e977451087fdd4daa04e05602fe9fbef498bb003c"
    "eb0f9a3a73ecf5122420a7a0fc81596c7fe6c1e39c2e78dbd4ea003c"
)
_GOLDEN_TRIT_CODES = (
    "0122222111001201122211002120112212121110002012221012120220121002"
    "1112111222110212021021220100220202002121110022020201100012201102"
    "2010122000112111202212201122210001100011002121022100022221121111"
    "1221121210212200122210121201211110221102101212100202112022121012"
)
_GOLDEN_TRITS = np.array([int(c) - 1 for c in _GOLDEN_TRIT_CODES], dtype=np.float32)


def test_types_are_registered():
    register_prism_types()  # idempotent
    assert GGMLQuantizationType.PTQ1_0 == GGML_TYPE_PTQ1_0
    assert GGMLQuantizationType(GGML_TYPE_PTQ1_0).name == "PTQ1_0"
    assert GGMLQuantizationType.PQ2_0 == GGML_TYPE_PQ2_0
    assert LlamaFileType.MOSTLY_PTQ1_0 == 129
    assert gguf.GGML_QUANT_SIZES[GGMLQuantizationType.PTQ1_0] == (
        QK_PTQ1_0,
        PTQ1_0_BLOCK_BYTES,
    )
    assert gguf.GGML_QUANT_SIZES[GGMLQuantizationType.PQ2_0] == (128, 34)
    assert "PTQ1_0" in GGMLQuantizationType.__members__
    assert is_valid_gguf_quant_type("PTQ1_0")
    assert is_remote_gguf("prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0")


def test_ptq1_0_numpy_codec_matches_reference():
    blocks = np.frombuffer(_GOLDEN_BLOCKS, dtype=np.uint8).reshape(1, -1)
    assert blocks.shape[1] == 2 * PTQ1_0_BLOCK_BYTES
    dequantized = gguf.dequantize(blocks, GGMLQuantizationType.PTQ1_0)
    np.testing.assert_array_equal(dequantized.reshape(-1), _GOLDEN_TRITS)
    requantized = gguf.quantize(
        _GOLDEN_TRITS.reshape(1, -1), GGMLQuantizationType.PTQ1_0
    )
    assert requantized.tobytes() == _GOLDEN_BLOCKS


@pytest.mark.parametrize(
    "qtype", [GGMLQuantizationType.PTQ1_0, GGMLQuantizationType.PQ2_0]
)
def test_prism_codec_round_trip(qtype):
    rng = np.random.default_rng(0)
    scales = (
        rng.uniform(0.05, 2.0, size=(16, 8, 1)).astype(np.float16).astype(np.float32)
    )
    trits = rng.integers(-1, 2, size=(16, 8, 128)).astype(np.float32)
    trits[..., 0] = 1.0  # keep every block's max at the scale
    weights = (trits * scales).reshape(16, 1024)
    packed = gguf.quantize(weights, qtype)
    np.testing.assert_array_equal(gguf.dequantize(packed, qtype), weights)


@pytest.mark.parametrize("block_size", [16, 512, 1024])
def test_hadamard_matrix_is_normalized_sylvester(block_size):
    matrix = hadamard_matrix(block_size)
    idx = np.arange(block_size)
    parity = np.vectorize(lambda v: bin(v).count("1") & 1)(idx[:, None] & idx[None, :])
    expected = np.where(parity, -1.0, 1.0) / math.sqrt(block_size)
    np.testing.assert_allclose(matrix.numpy(), expected)
    torch.testing.assert_close(
        matrix @ matrix, torch.eye(block_size), atol=1e-5, rtol=0
    )


def test_fwht_blockwise_and_dtype():
    x = torch.randn(5, 3 * 512, dtype=torch.bfloat16)
    y = fwht(x, 512)
    assert y.dtype == x.dtype and y.shape == x.shape
    expected = (x.float().reshape(-1, 512) @ hadamard_matrix(512)).reshape(x.shape)
    torch.testing.assert_close(y.float(), expected.to(torch.bfloat16).float())
    torch.testing.assert_close(
        fwht(fwht(x.float(), 512), 512), x.float(), atol=1e-4, rtol=0
    )


def test_hadamard_layout_forward_inverse_and_sharding():
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 1.0], size=3072).astype(np.float32)
    layout = GGUFHadamardLayout(1024, 3072, signs, forward=True, inverse=True)
    x = torch.randn(4, 3072)
    torch.testing.assert_close(
        layout.output_to_vllm(layout.input_to_gguf(x)), x, atol=1e-4, rtol=0
    )
    forward_only = GGUFHadamardLayout(1024, 3072, signs, forward=True, inverse=False)
    assert torch.equal(forward_only.output_to_vllm(x), x)
    weight = torch.arange(2 * 24 * 28, dtype=torch.uint8).reshape(2, -1)
    shard = layout.shard_weight(
        weight, dim=1, logical_size=3072, block_size=128, tp_rank=1, tp_size=3
    )
    assert torch.equal(shard, weight[:, 8 * 28 : 16 * 28])
    with pytest.raises(ValueError, match="whole Hadamard blocks"):
        layout.shard_weight(
            weight, dim=1, logical_size=3072, block_size=128, tp_rank=0, tp_size=2
        )


def _write_folded_gguf(
    path, *, version, tied, block=512, hidden=1024, with_output=True
):
    rng = np.random.default_rng(2)
    writer = gguf.GGUFWriter(str(path), "qwen35")
    writer.add_block_count(1)
    writer.add_uint32("prism.hadamard.version", version)
    writer.add_bool("prism.hadamard.tied_output", tied)
    writer.add_uint32("prism.hadamard.block_size", block)
    writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
    writer.add_string("prism.hadamard.axis", "input-last-dimension")
    writer.add_string("prism.hadamard.sign_mode", "explicit")
    names = ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]
    if with_output:
        names.append("output.weight")
    writer.add_array("prism.hadamard.weight_names", names)
    writer.add_array("prism.hadamard.inverse_weight_names", ["token_embd.weight"])
    writer.add_array("prism.hadamard.sign_widths", [hidden, 2 * hidden])
    sign_values = rng.choice([-1, 1], size=3 * hidden)
    writer.add_array("prism.hadamard.sign_values", [int(v) for v in sign_values])

    def add_quantized(name, shape):
        qtype = GGMLQuantizationType.PTQ1_0
        data = rng.integers(-1, 2, size=shape).astype(np.float32)
        writer.add_tensor(
            name,
            gguf.quantize(data, qtype),
            raw_dtype=qtype,
            raw_shape=gguf.quant_shape_to_byte_shape(shape, qtype),
        )

    add_quantized("token_embd.weight", (32, hidden))
    if with_output:
        add_quantized("output.weight", (32, hidden))
    add_quantized("blk.0.ffn_gate.weight", (2 * hidden, hidden))
    add_quantized("blk.0.ffn_up.weight", (2 * hidden, hidden))
    add_quantized("blk.0.ffn_down.weight", (hidden, 2 * hidden))
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sign_values


_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
    "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
    "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
}


def test_hadamard_spec_and_layouts_version_1(tmp_path):
    path = tmp_path / "v1.gguf"
    sign_values = _write_folded_gguf(path, version=1, tied=False)
    spec = read_prism_hadamard_spec([path])
    assert spec is not None and spec.version == 1 and not spec.tied_output
    assert spec.block_size == 512 and set(spec.signs) == {1024, 2048}
    np.testing.assert_array_equal(spec.signs[1024], sign_values[:1024])
    assert spec.tensor_widths["blk.0.ffn_down.weight"] == 2048

    layouts = build_hadamard_layouts(spec, _NAME_MAP)
    assert set(layouts) == {
        "model.embed_tokens",
        "lm_head",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    }
    # fused gate_up_proj shards must resolve to one shared layout object
    assert (
        layouts["model.layers.0.mlp.gate_proj"] is layouts["model.layers.0.mlp.up_proj"]
    )
    assert (
        layouts["model.embed_tokens"].inverse
        and not layouts["model.embed_tokens"].forward
    )
    assert layouts["lm_head"].forward and not layouts["lm_head"].inverse
    assert layouts["model.layers.0.mlp.down_proj"].width == 2048

    reader = gguf.GGUFReader(str(path))
    tensor = next(t for t in reader.tensors if t.name == "blk.0.ffn_down.weight")
    assert tensor.tensor_type == GGMLQuantizationType.PTQ1_0
    assert gguf.dequantize(tensor.data, tensor.tensor_type).shape == (1024, 2048)


def test_hadamard_spec_version_2_tied_output(tmp_path):
    path = tmp_path / "v2.gguf"
    _write_folded_gguf(path, version=2, tied=True, with_output=False)
    spec = read_prism_hadamard_spec([path])
    assert spec is not None and spec.tied_output
    layouts = build_hadamard_layouts(spec, _NAME_MAP)
    embed = layouts["model.embed_tokens"]
    assert embed.forward and embed.inverse and "lm_head" not in layouts


def test_hadamard_spec_rejects_inconsistent_metadata(tmp_path):
    path = tmp_path / "bad.gguf"
    _write_folded_gguf(path, version=2, tied=False, with_output=False)
    with pytest.raises(ValueError, match="version 2 requires tied_output"):
        read_prism_hadamard_spec([path])


def test_hadamard_layouts_reject_unmapped_folded_weights(tmp_path):
    path = tmp_path / "v1.gguf"
    _write_folded_gguf(path, version=1, tied=False)
    spec = read_prism_hadamard_spec([path])
    partial_map = {k: v for k, v in _NAME_MAP.items() if "ffn_down" not in k}
    with pytest.raises(ValueError, match="without a vLLM parameter mapping"):
        build_hadamard_layouts(spec, partial_map)


def test_plain_gguf_has_no_hadamard_spec(tmp_path):
    path = tmp_path / "plain.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tensor("x", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert read_prism_hadamard_spec([path]) is None
