# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from .utils import is_layer_skipped_gguf

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from .layout import GGUFLinearLayout


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []
        self.linear_layouts: dict[str, GGUFLinearLayout] = {}

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        del config
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from .fused_moe import GGUFMoEMethod
        from .linear import GGUFLinearMethod
        from .vocal_embeds import GGUFEmbeddingMethod

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(
                self,
                layout=self.resolve_linear_layout(prefix),
            )
        if isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            return GGUFEmbeddingMethod(self, layout=self.resolve_linear_layout(prefix))
        if isinstance(layer, RoutedExperts):
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def register_linear_layouts(
        self,
        layouts: Mapping[str, "GGUFLinearLayout"],
        prefix: str = "",
    ) -> None:
        """Register GGUF linear layouts before model initialization."""
        self.linear_layouts.update(
            (maybe_prefix(prefix, module_name), layout)
            for module_name, layout in layouts.items()
        )

    def resolve_linear_layout(self, prefix: str) -> "GGUFLinearLayout | None":
        """Return the layout registered for *prefix* or for all of its shards.

        Layouts are registered under the checkpoint's module names. vLLM fuses
        some of them (``q_proj``/``k_proj``/``v_proj`` -> ``qkv_proj``), in which
        case every shard must carry the same layout because the fused layer
        transforms its input only once.
        """
        layout = self.linear_layouts.get(prefix)
        if layout is not None:
            return layout
        proj_name = prefix.rsplit(".", 1)[-1]
        shard_names = self.packed_modules_mapping.get(proj_name)
        if not shard_names:
            return None
        base = prefix[: len(prefix) - len(proj_name)]
        shard_layouts = [
            self.linear_layouts.get(base + shard_name) for shard_name in shard_names
        ]
        if all(shard_layout is None for shard_layout in shard_layouts):
            return None
        first = shard_layouts[0]
        if any(shard_layout is not first for shard_layout in shard_layouts):
            raise ValueError(
                f"Shards of fused GGUF layer {prefix} need different input "
                f"layouts ({dict(zip(shard_names, shard_layouts))}); all shards "
                "of a fused layer must share one layout."
            )
        return first

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )
        if self.linear_layouts:
            layouts = self.linear_layouts
            mapped_names = hf_to_vllm_mapper.apply_list(list(layouts))
            self.linear_layouts = dict(zip(mapped_names, layouts.values(), strict=True))
