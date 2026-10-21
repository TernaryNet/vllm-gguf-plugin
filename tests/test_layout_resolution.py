# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout bookkeeping in GGUFConfig: name mapping and fused-layer resolution."""

import pytest
from vllm.model_executor.models.utils import WeightsMapper

from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.hadamard import GGUFHadamardLayout


def _layout(width: int = 1024) -> GGUFHadamardLayout:
    return GGUFHadamardLayout(512, width, None, forward=True, inverse=False)


def test_apply_vllm_mapper_renames_layout_keys_like_weights():
    config = GGUFConfig()
    head, proj = _layout(), _layout()
    config.register_linear_layouts(
        {"lm_head": head, "model.language_model.layers.0.mlp.gate_proj": proj}
    )
    config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
    )
    assert config.linear_layouts == {
        "language_model.lm_head": head,
        "language_model.model.layers.0.mlp.gate_proj": proj,
    }


def test_resolve_linear_layout_for_fused_modules():
    config = GGUFConfig()
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    shared = _layout()
    config.register_linear_layouts(
        {
            "model.layers.0.self_attn.q_proj": shared,
            "model.layers.0.self_attn.k_proj": shared,
            "model.layers.0.self_attn.v_proj": shared,
            "model.layers.0.mlp.gate_proj": shared,
            "model.layers.0.mlp.up_proj": _layout(),
            "model.layers.0.mlp.down_proj": _layout(2048),
        }
    )
    assert config.resolve_linear_layout("model.layers.0.self_attn.qkv_proj") is shared
    assert config.resolve_linear_layout("model.layers.0.mlp.down_proj").width == 2048
    assert config.resolve_linear_layout("model.layers.1.self_attn.qkv_proj") is None
    assert config.resolve_linear_layout("model.layers.0.self_attn.o_proj") is None
    with pytest.raises(ValueError, match="must share one layout"):
        config.resolve_linear_layout("model.layers.0.mlp.gate_up_proj")
