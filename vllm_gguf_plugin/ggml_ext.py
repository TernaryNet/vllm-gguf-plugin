# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGML tensor types that the upstream ``gguf`` package does not know.

PrismML ships Bonsai GGUF checkpoints in two private ternary packings
(see https://github.com/PrismML-Eng/llama.cpp):

* ``PTQ1_0`` (type 143): 128 ternary weights per block, packed base-3 at five
  trits per byte like upstream ``TQ1_0`` but with one fp16 scale per 128
  weights instead of per 256. 28 bytes per block, 1.75 bits per weight.
* ``PQ2_0`` (type 142): the same weights with each trit in a 2-bit slot and one
  fp16 scale per 128 weights. 34 bytes per block.

``gguf.GGMLQuantizationType`` is an ``IntEnum`` that the reader, the quant
size table, and this plugin all key on, so the members are added to the enum
at import time rather than mirrored in a parallel table. Only ``PTQ1_0`` has
GPU kernels in this plugin; ``PQ2_0`` is registered so that such files are
read and rejected with a clear error instead of failing inside the reader.
"""

from __future__ import annotations

import enum

import numpy as np
from gguf import quants as gguf_quants
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, LlamaFileType

GGML_TYPE_PQ2_0 = 142
GGML_TYPE_PTQ1_0 = 143
LLAMA_FTYPE_MOSTLY_PQ2_0 = 128
LLAMA_FTYPE_MOSTLY_PTQ1_0 = 129

QK_PTQ1_0 = 128
#: qs (24 bytes, 5 trits each) + qh (2 bytes, 4 trits each) + fp16 scale
PTQ1_0_BLOCK_BYTES = 24 + 2 + 2
QK_PQ2_0 = 128
PQ2_0_BLOCK_BYTES = 2 + QK_PQ2_0 // 4

_PRISM_TENSOR_TYPES = (
    ("PQ2_0", GGML_TYPE_PQ2_0, (QK_PQ2_0, PQ2_0_BLOCK_BYTES)),
    ("PTQ1_0", GGML_TYPE_PTQ1_0, (QK_PTQ1_0, PTQ1_0_BLOCK_BYTES)),
)
_PRISM_FILE_TYPES = (
    ("MOSTLY_PQ2_0", LLAMA_FTYPE_MOSTLY_PQ2_0),
    ("MOSTLY_PTQ1_0", LLAMA_FTYPE_MOSTLY_PTQ1_0),
)


def _extend_int_enum(cls: type[enum.IntEnum], name: str, value: int) -> enum.IntEnum:
    """Add ``name = value`` to an existing ``IntEnum`` class."""
    existing = cls._value2member_map_.get(value)
    if existing is not None:
        if existing.name != name:
            raise RuntimeError(
                f"{cls.__name__}({value}) is already {existing.name}, "
                f"cannot register it as {name}"
            )
        return existing
    if name in cls._member_map_:
        raise RuntimeError(
            f"{cls.__name__}.{name} already exists with value "
            f"{cls._member_map_[name].value}, expected {value}"
        )

    member = int.__new__(cls, value)
    member._name_ = name
    member._value_ = value
    member._sort_order_ = len(cls._member_names_)
    # ``EnumType.__setattr__`` refuses to rebind names that are already in
    # ``_member_map_``, so the class attribute has to be set first.
    setattr(cls, name, member)
    cls._member_map_[name] = member
    cls._value2member_map_[value] = member
    cls._member_names_.append(name)
    return member


def register_prism_types() -> None:
    """Register the Prism tensor and file types with the ``gguf`` package.

    Safe to call repeatedly.
    """
    for name, value, sizes in _PRISM_TENSOR_TYPES:
        member = _extend_int_enum(GGMLQuantizationType, name, value)
        GGML_QUANT_SIZES[member] = sizes
    for name, value in _PRISM_FILE_TYPES:
        _extend_int_enum(LlamaFileType, name, value)
    _register_numpy_codecs()


# --- numpy reference codecs -------------------------------------------------
#
# PTQ1_0 block layout (28 bytes):
#   qs[0:16]   five trits per byte; byte m holds elements m + 16*n, n = 0..4
#              (elements 0..79)
#   qs[16:24]  five trits per byte; byte 16+m holds elements 80 + m + 8*n
#              (elements 80..119)
#   qh[0:2]    four trits per byte; byte h holds elements 120 + h + 2*n,
#              n = 0..3, stored in the four most significant base-3 digits
#   d          fp16 scale
# The first element of a byte is the most significant digit. Bytes are stored
# as ceil(q * 256 / 243) so that a digit is recovered with ((q * 3^n & 0xFF) * 3) >> 8.

_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)


def _ptq1_0_dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    qs, rest = np.hsplit(blocks, [24])
    qh, d = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)

    qs0, qs1 = qs[..., :16], qs[..., 16:]
    # (n_blocks, 5, 16): digit n of byte m -> element 16*n + m
    qs0 = qs0.reshape((n_blocks, 1, 16)) * _POW3.reshape((1, 5, 1))
    qs1 = qs1.reshape((n_blocks, 1, 8)) * _POW3.reshape((1, 5, 1))
    qh = qh.reshape((n_blocks, 1, 2)) * _POW3[:4].reshape((1, 4, 1))
    codes = np.concatenate(
        [
            qs0.reshape((n_blocks, -1)),
            qs1.reshape((n_blocks, -1)),
            qh.reshape((n_blocks, -1)),
        ],
        axis=-1,
    )
    trits = ((codes.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)
    return d * trits.astype(np.float32)


def _ptq1_0_quantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1 / d)
    trits = np.rint(blocks * inv_d).astype(np.int8)
    codes = (trits + np.int8(1)).astype(np.uint8)

    c0, c1, ch = codes[..., :80], codes[..., 80:120], codes[..., 120:]
    weights = np.array([81, 27, 9, 3, 1], dtype=np.uint8)
    qs0 = (c0.reshape((n_blocks, 5, 16)) * weights.reshape((1, 5, 1))).sum(axis=1)
    qs1 = (c1.reshape((n_blocks, 5, 8)) * weights.reshape((1, 5, 1))).sum(axis=1)
    qh = (ch.reshape((n_blocks, 4, 2)) * weights[:4].reshape((1, 4, 1))).sum(axis=1)
    qs = np.concatenate([qs0, qs1, qh], axis=-1).astype(np.uint16)
    qs = ((qs * 256 + (243 - 1)) // 243).astype(np.uint8)
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([qs, d], axis=-1)


def _pq2_0_dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4))
    codes = (qs.reshape((n_blocks, -1, 1)) >> shifts) & np.uint8(3)
    trits = codes.reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)
    return d * trits.astype(np.float32)


def _pq2_0_quantize_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1 / d)
    codes = (np.rint(blocks * inv_d).astype(np.int8) + np.int8(1)).astype(np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4))
    qs = (codes.reshape((n_blocks, -1, 4)) << shifts).sum(axis=-1).astype(np.uint8)
    d = d.astype(np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1)


def _register_numpy_codecs() -> None:
    quant_base = getattr(gguf_quants, "__Quant")  # noqa: B009
    for name, value, _ in _PRISM_TENSOR_TYPES:
        qtype = GGMLQuantizationType(value)
        if qtype in gguf_quants._type_traits:
            continue
        if name == "PTQ1_0":
            dequantize, quantize = _ptq1_0_dequantize_blocks, _ptq1_0_quantize_blocks
        else:
            dequantize, quantize = _pq2_0_dequantize_blocks, _pq2_0_quantize_blocks
        # ``__init_subclass__`` registers the class in ``gguf.quants._type_traits``.
        type(
            name,
            (quant_base,),
            {
                "dequantize_blocks": classmethod(
                    lambda cls, blocks, _f=dequantize: _f(blocks)
                ),
                "quantize_blocks": classmethod(
                    lambda cls, blocks, _f=quantize: _f(blocks)
                ),
            },
            qtype=qtype,
        )


register_prism_types()
