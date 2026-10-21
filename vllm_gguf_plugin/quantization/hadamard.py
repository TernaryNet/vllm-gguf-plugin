# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation transforms for Hadamard-folded GGUF weights (``prism.hadamard.*``).

PrismML Bonsai checkpoints store every low-bit matrix in a rotated basis: each
weight ``W`` was replaced offline by ``W' = W @ diag(s) @ H`` where ``H`` is a
block-diagonal normalized Sylvester (Walsh-)Hadamard matrix with block
``block_size`` over the input dimension and ``s`` is a fixed ±1 sign vector of
the input width. Because ``H`` is symmetric and its own inverse, the matmul
``W x`` becomes ``W' (H (s ⊙ x))``, so the runtime must apply ``s`` and the
blockwise transform to the activation right before every folded matmul.

Two tensor kinds are affected:

* folded linear weights (``prism.hadamard.weight_names``): transform the
  input activation;
* latent token-embedding tables (``prism.hadamard.inverse_weight_names``):
  rows are stored rotated, so the lookup result is restored with
  ``e = s ⊙ (H z)``.

With ``prism.hadamard.tied_output`` (version 2) the embedding table also acts
as the output head and needs both. The contract is validated exactly like the
PrismML llama.cpp fork does, and anything it does not recognise is refused
rather than loaded with silently wrong math.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache

import gguf
import numpy as np
import torch
from gguf.constants import GGMLQuantizationType

HADAMARD_TRANSFORM = "normalized-sylvester-walsh-hadamard"
HADAMARD_AXIS = "input-last-dimension"
_FLOAT_TYPES = frozenset(
    {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}
)
#: The rotation is computed in this dtype. llama.cpp keeps its activations in
#: fp32, so the transform runs in fp32 and is cast back to the model dtype.
HADAMARD_COMPUTE_DTYPE = torch.float32


@dataclass(frozen=True, eq=False)
class PrismHadamardSpec:
    """Validated ``prism.hadamard.*`` metadata of one GGUF model."""

    version: int
    block_size: int
    tied_output: bool
    sign_mode: str
    weight_names: tuple[str, ...]
    inverse_weight_names: tuple[str, ...]
    gdn_v_grouped: bool
    #: ±1 sign vector per input width (``sign_mode == "explicit"``).
    signs: dict[int, np.ndarray] = field(default_factory=dict)
    #: Input width (GGML ``ne[0]``) of every tensor in the files.
    tensor_widths: dict[str, int] = field(default_factory=dict)
    tensor_types: dict[str, GGMLQuantizationType] = field(default_factory=dict)

    @property
    def folded_names(self) -> frozenset[str]:
        return frozenset(self.weight_names) | frozenset(self.inverse_weight_names)

    def signs_for(self, width: int) -> np.ndarray | None:
        if self.sign_mode == "identity":
            return None
        signs = self.signs.get(width)
        if signs is None:
            raise ValueError(
                f"prism.hadamard has no sign vector for input width {width}"
            )
        return signs


def _field_value(reader: gguf.GGUFReader, key: str, default=None):
    field_ = reader.fields.get(key)
    if field_ is None:
        return default
    value = field_.contents()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _validate_spec(
    spec: PrismHadamardSpec, has_output_weight: bool
) -> PrismHadamardSpec:
    if spec.version not in (1, 2):
        raise ValueError(f"unsupported prism.hadamard.version: {spec.version}")
    if (spec.version == 2) != spec.tied_output:
        raise ValueError(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it"
        )
    if spec.tied_output and has_output_weight:
        raise ValueError(
            "prism.hadamard.tied_output requires output.weight to be absent"
        )
    if spec.block_size <= 0 or spec.block_size & (spec.block_size - 1):
        raise ValueError(f"invalid prism.hadamard.block_size: {spec.block_size}")
    if spec.sign_mode not in ("identity", "explicit"):
        raise ValueError(f"unsupported prism.hadamard.sign_mode: {spec.sign_mode}")
    if not spec.weight_names:
        raise ValueError("prism.hadamard.weight_names is empty")
    if len(set(spec.weight_names)) != len(spec.weight_names):
        raise ValueError("duplicate entries in prism.hadamard.weight_names")
    for name in spec.inverse_weight_names:
        # llama.cpp applies the inverse only to the token-embedding lookup.
        if name != "token_embd.weight":
            raise ValueError(
                f"prism.hadamard: weight '{name}' is not a verified "
                "inverse-after-lookup table"
            )
        if name in spec.weight_names:
            raise ValueError(f"duplicate prism.hadamard inverse weight: {name}")
    if spec.tied_output and "token_embd.weight" not in spec.inverse_weight_names:
        raise ValueError("prism.hadamard.tied_output requires a latent token embedding")
    if (
        not spec.tied_output
        and "token_embd.weight" in spec.inverse_weight_names
        and not has_output_weight
    ):
        raise ValueError(
            "a tied Hadamard output requires version 2 and tied_output=true"
        )

    for name in spec.folded_names:
        width = spec.tensor_widths.get(name)
        if width is None:
            raise ValueError(f"prism.hadamard weight not found: {name}")
        if width % spec.block_size:
            raise ValueError(
                f"prism.hadamard block size {spec.block_size} does not divide "
                f"input dimension {width} for {name}"
            )
        if spec.sign_mode == "explicit":
            spec.signs_for(width)
    return spec


def _read_spec_from_reader(reader: gguf.GGUFReader) -> PrismHadamardSpec | None:
    version = _field_value(reader, "prism.hadamard.version")
    if version is None:
        return None

    sign_mode = str(_field_value(reader, "prism.hadamard.sign_mode", ""))
    transform = str(_field_value(reader, "prism.hadamard.transform", ""))
    axis = str(_field_value(reader, "prism.hadamard.axis", ""))
    if transform != HADAMARD_TRANSFORM:
        raise ValueError(f"unsupported prism.hadamard.transform: {transform}")
    if axis != HADAMARD_AXIS:
        raise ValueError(f"unsupported prism.hadamard.axis: {axis}")
    block_size = int(_field_value(reader, "prism.hadamard.block_size", 0))

    signs: dict[int, np.ndarray] = {}
    if sign_mode == "explicit":
        widths = [
            int(w) for w in _field_value(reader, "prism.hadamard.sign_widths", [])
        ]
        values = np.asarray(
            _field_value(reader, "prism.hadamard.sign_values", []), dtype=np.int64
        )
        if not widths:
            raise ValueError(
                "prism.hadamard.sign_mode is explicit but sign_widths is empty"
            )
        offset = 0
        for width in widths:
            if (
                width <= 0
                or (block_size and width % block_size)
                or offset + width > len(values)
            ):
                raise ValueError(f"invalid prism.hadamard sign width: {width}")
            vec = values[offset : offset + width]
            if not np.isin(vec, (-1, 1)).all():
                raise ValueError("prism.hadamard sign values must be +/-1")
            signs[width] = vec.astype(np.float32)
            offset += width
        if offset != len(values):
            raise ValueError("prism.hadamard.sign_values length mismatch")

    tensor_widths = {t.name: int(t.shape[0]) for t in reader.tensors}
    tensor_types = {t.name: t.tensor_type for t in reader.tensors}
    spec = PrismHadamardSpec(
        version=int(version),
        block_size=block_size,
        tied_output=bool(_field_value(reader, "prism.hadamard.tied_output", False)),
        sign_mode=sign_mode,
        weight_names=tuple(
            str(n) for n in _field_value(reader, "prism.hadamard.weight_names", [])
        ),
        inverse_weight_names=tuple(
            str(n)
            for n in _field_value(reader, "prism.hadamard.inverse_weight_names", [])
        ),
        gdn_v_grouped=bool(_field_value(reader, "prism.hadamard.gdn_v_grouped", False)),
        signs=signs,
        tensor_widths=tensor_widths,
        tensor_types=tensor_types,
    )
    return _validate_spec(spec, has_output_weight="output.weight" in tensor_widths)


@lru_cache(maxsize=8)
def _read_prism_hadamard_spec_cached(
    gguf_files: tuple[str, ...],
) -> PrismHadamardSpec | None:
    specs = [_read_spec_from_reader(gguf.GGUFReader(path)) for path in gguf_files]
    specs = [spec for spec in specs if spec is not None]
    if not specs:
        return None
    if len(specs) > 1:
        raise ValueError("prism.hadamard metadata must live in a single GGUF file")
    return specs[0]


def read_prism_hadamard_spec(
    gguf_files: Iterable[str],
) -> PrismHadamardSpec | None:
    """Return the ``prism.hadamard`` spec of *gguf_files*, or ``None``."""
    return _read_prism_hadamard_spec_cached(tuple(str(path) for path in gguf_files))


# --- the transform ----------------------------------------------------------

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def hadamard_matrix(
    block_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix.

    ``H[i, j] = (-1)^popcount(i & j) / sqrt(n)``; symmetric and its own inverse.
    """
    device = torch.device(device)
    key = (block_size, device, dtype)
    matrix = _HADAMARD_CACHE.get(key)
    if matrix is None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(
                f"Hadamard block size must be a power of two, got {block_size}"
            )
        idx = torch.arange(block_size, dtype=torch.int64)
        bits = idx[:, None] & idx[None, :]
        parity = torch.zeros_like(bits)
        for shift in range(block_size.bit_length()):
            parity ^= (bits >> shift) & 1
        matrix = torch.where(parity.bool(), -1.0, 1.0) / math.sqrt(block_size)
        matrix = matrix.to(device=device, dtype=dtype)
        _HADAMARD_CACHE[key] = matrix
    return matrix


def fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Blockwise normalized Walsh-Hadamard transform along the last dimension."""
    width = x.shape[-1]
    if width % block_size:
        raise ValueError(
            f"Hadamard block size {block_size} does not divide activation width {width}"
        )
    matrix = hadamard_matrix(block_size, x.device, HADAMARD_COMPUTE_DTYPE)
    blocks = x.reshape(-1, block_size).to(HADAMARD_COMPUTE_DTYPE)
    return (blocks @ matrix).reshape(x.shape).to(x.dtype)


class GGUFHadamardLayout:
    """Input/output transform of one Hadamard-folded GGUF weight.

    Implements the :class:`GGUFLinearLayout` protocol. The weight itself is
    already in vLLM order, so ``weight_to_vllm`` is the identity; sharding a
    row-parallel weight is only allowed at Hadamard block boundaries because
    the transform mixes the elements of a block.

    ``prepare`` moves the rotation matrix and sign vector to the weight's
    device once the weights are loaded. The forward transform must not create
    or copy tensors itself: it is traced by torch.compile and captured into
    CUDA graphs, where a host-to-device copy is an error.
    """

    def __init__(
        self,
        block_size: int,
        width: int,
        signs: np.ndarray | None,
        *,
        forward: bool,
        inverse: bool,
    ) -> None:
        if width % block_size:
            raise ValueError(
                f"Hadamard block size {block_size} does not divide width {width}"
            )
        if signs is not None and signs.shape != (width,):
            raise ValueError(
                f"sign vector of shape {signs.shape} does not match width {width}"
            )
        self.block_size = block_size
        self.width = width
        self.forward = forward
        self.inverse = inverse
        self._signs_cpu = (
            None
            if signs is None
            else torch.from_numpy(np.ascontiguousarray(signs)).to(
                HADAMARD_COMPUTE_DTYPE
            )
        )
        self._device: torch.device | None = None
        self._matrix: torch.Tensor | None = None
        self._signs: torch.Tensor | None = None
        self._tp_rank = 0

    def __repr__(self) -> str:
        return (
            f"GGUFHadamardLayout(block_size={self.block_size}, width={self.width}, "
            f"signs={'explicit' if self._signs_cpu is not None else 'identity'}, "
            f"forward={self.forward}, inverse={self.inverse})"
        )

    def prepare(self, device: torch.device | str) -> None:
        """Materialize the transform's constants on *device*."""
        device = torch.device(device)
        if self._device == device:
            return
        self._matrix = hadamard_matrix(self.block_size, device, HADAMARD_COMPUTE_DTYPE)
        self._signs = (
            None if self._signs_cpu is None else self._signs_cpu.to(device=device)
        )
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            self._tp_rank = get_tensor_model_parallel_rank()
        except (ImportError, AssertionError):
            # not running inside an initialized vLLM worker (unit tests)
            self._tp_rank = 0
        self._device = device

    def _local_signs(self, x: torch.Tensor) -> torch.Tensor | None:
        if self._signs is None:
            return None
        local_width = x.shape[-1]
        if local_width == self.width:
            return self._signs
        tp_size, remainder = divmod(self.width, local_width)
        if remainder or local_width % self.block_size:
            raise ValueError(
                f"Activation width {local_width} is not a Hadamard-block-aligned "
                f"shard of the folded width {self.width} (block {self.block_size})"
            )
        if self._tp_rank >= tp_size:
            raise ValueError(
                f"Tensor parallel rank {self._tp_rank} is out of range for the "
                f"{tp_size}-way shard of a Hadamard-folded input"
            )
        start = self._tp_rank * local_width
        return self._signs[start : start + local_width]

    def _apply(self, x: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        if self._device != x.device:
            self.prepare(x.device)
        assert self._matrix is not None
        signs = self._local_signs(x)
        out = x.to(HADAMARD_COMPUTE_DTYPE)
        if not inverse and signs is not None:
            out = out * signs
        out = (out.reshape(-1, self.block_size) @ self._matrix).reshape(x.shape)
        if inverse and signs is not None:
            out = out * signs
        return out.to(x.dtype)

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor:
        """``H (s ⊙ x)``: rotate a linear input into the folded basis."""
        if not self.forward:
            return x
        return self._apply(x, inverse=False)

    def output_to_vllm(self, x: torch.Tensor) -> torch.Tensor:
        """``s ⊙ (H z)``: restore a latent embedding lookup to the primal basis."""
        if not self.inverse:
            return x
        return self._apply(x, inverse=True)

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor:
        del dim, head_dim
        return weight

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Slice a row-parallel weight along its input dim at block boundaries."""
        if tp_size == 1:
            return weight
        local_size, remainder = divmod(logical_size, tp_size)
        if remainder or local_size % self.block_size or local_size % block_size:
            raise ValueError(
                f"Cannot shard a Hadamard-folded input of {logical_size} elements "
                f"(block {self.block_size}) across TP size {tp_size}: every rank "
                "must own whole Hadamard blocks"
            )
        shard_bytes, remainder = divmod(weight.shape[dim], tp_size)
        if remainder:
            raise ValueError(
                f"Packed weight dimension {weight.shape[dim]} is not divisible "
                f"across TP size {tp_size}"
            )
        return weight.narrow(dim, tp_rank * shard_bytes, shard_bytes).contiguous()


def build_hadamard_layouts(
    spec: PrismHadamardSpec,
    name_map: Mapping[str, str],
) -> dict[str, GGUFHadamardLayout]:
    """Map vLLM module names to the transforms their folded weights need.

    *name_map* translates GGUF tensor names to vLLM parameter names; fused
    modules (``qkv_proj`` etc.) are resolved later by the quantization config
    from their shard names.
    """
    folded_float = sorted(
        name for name in spec.folded_names if spec.tensor_types[name] in _FLOAT_TYPES
    )
    if folded_float:
        raise NotImplementedError(
            "Hadamard-folded weights stored unquantized are not supported "
            f"(use a PTQ1_0 file): {folded_float}"
        )

    unmapped = sorted(name for name in spec.folded_names if name not in name_map)
    if unmapped:
        raise ValueError(
            "Hadamard-folded GGUF tensors without a vLLM parameter mapping; "
            f"refusing to load them without their activation transform: {unmapped}"
        )

    shared: dict[tuple[int, bool, bool], GGUFHadamardLayout] = {}
    layouts: dict[str, GGUFHadamardLayout] = {}
    inverse_names = frozenset(spec.inverse_weight_names)
    for name in spec.folded_names:
        forward = (
            name in spec.weight_names or spec.tied_output and name in inverse_names
        )
        inverse = name in inverse_names
        width = spec.tensor_widths[name]
        key = (width, forward, inverse)
        layout = shared.get(key)
        if layout is None:
            layout = GGUFHadamardLayout(
                spec.block_size,
                width,
                spec.signs_for(width),
                forward=forward,
                inverse=inverse,
            )
            shared[key] = layout
        mapped = name_map[name]
        if not mapped.endswith(".weight"):
            raise ValueError(
                f"Unexpected parameter name for folded weight {name}: {mapped}"
            )
        layouts[mapped.removesuffix(".weight")] = layout
    return layouts
