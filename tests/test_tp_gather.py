# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather-mode TP for Hadamard-folded row-parallel linears (quantization/tp.py).

The ranks of one TP group are simulated in a single process: every simulated
rank loads its shard through the real loader (``load_row_parallel_weight``),
runs the real ``apply`` with the all-gather replaced by the concatenation of
every rank's input, and the ranks' partial outputs are summed the way the
layer's all-reduce would. Gather mode must reproduce TP 1 bit for bit; the
block-aligned row-parallel path (TP 2 on a 6-block input) up to bf16 rounding
of the partial sums.
"""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

from vllm_gguf_plugin.ggml_ext import register_prism_types
from vllm_gguf_plugin.quantization import params as params_mod
from vllm_gguf_plugin.quantization import tp as tp_mod
from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.hadamard import GGUFHadamardLayout
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

BLOCK = 1024
QTYPE = GGMLQuantizationType.PTQ1_0


class _FakeRowLayer(torch.nn.Module):
    """The attributes RowParallelLinear gives its quant method."""

    def __init__(self, input_size, output_size, tp_rank, tp_size, has_bias=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.reduce_results = True
        self.has_bias = has_bias
        self.prefix = "model.layers.0.mlp.down_proj"

    def weight_loader_v2(self, param, loaded_weight):
        param.load_row_parallel_weight(loaded_weight=loaded_weight)


def _random_ptq1_0(rows, cols, seed):
    rng = np.random.default_rng(seed)
    nblk = cols // 128
    scales = rng.uniform(0.01, 0.05, size=(rows, nblk, 1))
    scales = scales.astype(np.float16).astype(np.float32)
    trits = rng.integers(-1, 2, size=(rows, nblk, 128)).astype(np.float32)
    trits[..., 0] = 1.0
    weights = (trits * scales).reshape(rows, cols)
    return torch.from_numpy(gguf.quantize(weights, QTYPE))


def _layout(width, seed):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=width).astype(np.float32)
    return GGUFHadamardLayout(BLOCK, width, signs, forward=True, inverse=False)


def _patch_rank(monkeypatch, rank, size):
    from vllm.model_executor import parameter as vllm_parameter

    for module in (params_mod, vllm_parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: size
        )


def _load(method, layer, packed, k, n):
    method.create_weights(
        layer, k // layer.tp_size, [n], k, n, torch.bfloat16, weight_loader=None
    )
    layer.weight_type.weight_loader(layer.weight_type, torch.tensor(int(QTYPE)))
    layer.weight.weight_loader(layer.weight, packed)
    method.process_weights_after_loading(layer)


def _reference(monkeypatch, packed, x, k, n, seed):
    _patch_rank(monkeypatch, 0, 1)
    layer = _FakeRowLayer(k, n, 0, 1)
    method = GGUFLinearMethod(None, layout=_layout(k, seed))
    _load(method, layer, packed, k, n)
    return method.apply(layer, x)


def test_needs_gather_mode():
    folded = _layout(6 * BLOCK, 0)
    assert not tp_mod.needs_gather_mode(folded, 6 * BLOCK, 1, "auto")
    assert not tp_mod.needs_gather_mode(folded, 6 * BLOCK, 2, "auto")
    assert tp_mod.needs_gather_mode(folded, 6 * BLOCK, 2, "gather")
    assert tp_mod.needs_gather_mode(folded, 6 * BLOCK, 4, "auto")
    assert tp_mod.needs_gather_mode(_layout(17 * BLOCK, 0), 17 * BLOCK, 2, "auto")
    assert not tp_mod.needs_gather_mode(None, 17 * BLOCK, 4, "auto")
    inverse_only = GGUFHadamardLayout(BLOCK, 6 * BLOCK, None, forward=False, inverse=True)
    assert not tp_mod.needs_gather_mode(inverse_only, 6 * BLOCK, 4, "auto")


def test_get_quant_method_picks_gather_for_row_parallel(monkeypatch):
    from vllm.model_executor.layers.linear import RowParallelLinear

    monkeypatch.setenv(tp_mod.TP_ROW_ENV, "auto")
    monkeypatch.setenv(tp_mod.TP_OUT_ENV, "ar")
    config = GGUFConfig()
    prefix = "model.layers.0.mlp.down_proj"
    config.register_linear_layouts({prefix: _layout(17 * BLOCK, 0)})
    layer = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.input_size, layer.output_size = 17 * BLOCK, 5120
    for tp_size, expected in ((1, GGUFLinearMethod), (2, tp_mod.GGUFTPGatherLinearMethod)):
        layer.tp_rank, layer.tp_size = 0, tp_size
        method = config.get_quant_method(layer, prefix)
        assert type(method) is expected
    layer.tp_size = 4
    assert config.get_quant_method(layer, prefix).tp_size == 4


def test_gather_loader_narrows_rows_cpu(monkeypatch):
    register_prism_types()
    k, n, tp = 5 * BLOCK, 64, 4
    packed = _random_ptq1_0(n, k, 1)
    for rank in range(tp):
        _patch_rank(monkeypatch, rank, tp)
        layer = _FakeRowLayer(k, n, rank, tp)
        method = tp_mod.GGUFTPGatherLinearMethod(
            None, _layout(k, 1), tp_rank=rank, tp_size=tp
        )
        method.create_weights(layer, k // tp, [n], k, n, torch.bfloat16, weight_loader=None)
        assert layer.weight.tensor_shape == (n // tp, k)
        layer.weight.weight_loader(layer.weight, packed)
        rows = n // tp
        assert torch.equal(layer.weight.data, packed[rank * rows : (rank + 1) * rows])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA mat-vec")
@pytest.mark.parametrize("out_mode", ["ar", "ag"])
@pytest.mark.parametrize(
    "blocks, n, tp, tokens",
    [(5, 512, 2, 1), (5, 512, 4, 1), (6, 512, 4, 3), (17, 256, 2, 2), (17, 256, 4, 8)],
)
def test_gather_mode_matches_tp1_bit_for_bit(monkeypatch, out_mode, blocks, n, tp, tokens):
    register_prism_types()
    k = blocks * BLOCK
    seed = 7 * blocks + tp
    with torch.device("cuda"):
        packed = _random_ptq1_0(n, k, seed)
        x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        ref = _reference(monkeypatch, packed, x, k, n, seed)
        monkeypatch.setenv(tp_mod.TP_OUT_ENV, out_mode)
        total = torch.zeros(tokens, n, dtype=torch.float32, device="cuda")
        rows = n // tp
        for rank in range(tp):
            _patch_rank(monkeypatch, rank, tp)
            layer = _FakeRowLayer(k, n, rank, tp)
            method = tp_mod.gather_method_for(None, layer, _layout(k, seed))
            assert isinstance(method, tp_mod.GGUFTPGatherLinearMethod)
            _load(method, layer, packed, k, n)
            assert layer.reduce_results == (out_mode == "ar")
            x_local = x[:, rank * (k // tp) : (rank + 1) * (k // tp)].contiguous()
            calls = []

            def fake_all_gather(t, dim=-1):
                calls.append(t.shape)
                # first call: the input; second (ag mode): this rank's rows
                return x if len(calls) == 1 else t

            monkeypatch.setattr(tp_mod, "tensor_model_parallel_all_gather", fake_all_gather)
            out = method.apply(layer, x_local)
            if out_mode == "ag":
                assert len(calls) == 2 and out.shape == (tokens, rows)
                assert torch.equal(out, ref[:, rank * rows : (rank + 1) * rows])
                continue
            assert len(calls) == 1 and out.shape == (tokens, n)
            mine = slice(rank * rows, (rank + 1) * rows)
            assert torch.equal(out[:, mine], ref[:, mine])
            outside = torch.ones(n, dtype=torch.bool, device="cuda")
            outside[mine] = False
            assert torch.count_nonzero(out[:, outside]) == 0
            total += out.float()
        if out_mode == "ar":
            assert torch.equal(total.to(torch.bfloat16), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA mat-vec")
def test_block_aligned_row_parallel_matches_tp1(monkeypatch):
    """TP 2 on a 6-block input stays row-parallel: partial sums, bf16 rounding."""
    register_prism_types()
    k, n, tp, tokens, seed = 6 * BLOCK, 512, 2, 2, 11
    with torch.device("cuda"):
        packed = _random_ptq1_0(n, k, seed)
        x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        ref = _reference(monkeypatch, packed, x, k, n, seed)
        total = torch.zeros(tokens, n, dtype=torch.float32, device="cuda")
        for rank in range(tp):
            _patch_rank(monkeypatch, rank, tp)
            layer = _FakeRowLayer(k, n, rank, tp)
            layout = _layout(k, seed)
            assert tp_mod.gather_method_for(None, layer, layout) is None
            method = GGUFLinearMethod(None, layout=layout)
            _load(method, layer, packed, k, n)
            layout._tp_rank = rank
            local = x[:, rank * (k // tp) : (rank + 1) * (k // tp)].contiguous()
            total += method.apply(layer, local).float()
        scale = ref.float().abs().max().item()
        torch.testing.assert_close(total, ref.float(), rtol=2e-2, atol=2e-2 * scale)
