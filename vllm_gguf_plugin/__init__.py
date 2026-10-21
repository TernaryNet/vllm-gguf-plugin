# SPDX-License-Identifier: Apache-2.0

# Registers the PrismML tensor types (PTQ1_0, PQ2_0) with the ``gguf`` package;
# everything below keys on ``gguf.GGMLQuantizationType`` so it must run first.
from . import ggml_ext  # noqa: F401  # isort: skip
from .config_parser import GGUFConfigParser
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
