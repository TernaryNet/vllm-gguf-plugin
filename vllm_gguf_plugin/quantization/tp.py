# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor parallelism for Hadamard-folded row-parallel GGUF linears.

A folded weight ``W' = W diag(s) H`` (see ``hadamard.py``) needs its input
rotated in whole ``block_size`` blocks. Row parallelism splits that input
across the ranks, so it only works when every rank's share is whole blocks;
Bonsai-27B's ``down_proj`` input has 17 blocks of 1024 (prime: no TP above 1)
and its 6144-wide output projections 6 (TP 2 or 3).

For those layers this module computes the layer column-parallel over its
outputs instead ("gather mode"): all-gather the input (rank ``r`` holds input
columns ``[r * K / TP, (r + 1) * K / TP)``, the order the column-parallel
producer or the per-head core leaves them in), rotate the full width, multiply
by this rank's ``N / TP`` output rows over the full K, and hand the layer a
zero-padded partial sum (own rows in place, zeros elsewhere), so the
``RowParallelLinear`` all-reduce assembles the output. Every output row is
computed by the same kernel over the same K as without TP and the padding adds
exact zeros, so the result is bit-identical to TP 1. The row-parallel contract
holds even for a layer built with ``reduce_results=False``.

Switches (read once per process, at model construction):

* ``VLLM_GGUF_TP_ROW``: ``auto`` (default) uses gather mode only where the
  per-rank input is not whole Hadamard blocks; ``gather`` uses it for every
  folded row-parallel layer (bit-exact A/Bs against TP 1).
* ``VLLM_GGUF_TP_OUT``: ``ar`` (default) returns the zero-padded partial sum to
  the layer's all-reduce; ``ag`` (EXPERIMENTAL) all-gathers the ``N / TP``
  output rows instead and clears the layer's ``reduce_results`` (layers with a
  bias, or built with ``reduce_results=False``, always use ``ar``). On 4x L4
  over PCIe the all-gather is the cheaper collective (Bonsai-27B TP 4 spec-off
  step 17.7 -> 16.6 ms, replies sane), but with MTP speculative decoding and
  four concurrent streams ``ag`` produced a garbage reply and a non-finite
  logprob (HTTP 400) on 2026-10-25 while ``ar`` did not: do not enable it until
  that is understood.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.utils import set_weight_attrs

from .hadamard import GGUFHadamardLayout
from .linear import GGUFLinearMethod

logger = init_logger(__name__)

TP_ROW_ENV = "VLLM_GGUF_TP_ROW"
TP_OUT_ENV = "VLLM_GGUF_TP_OUT"
_LOGGED: set[tuple[int, int, int, str]] = set()


def tp_row_mode() -> str:
    mode = os.environ.get(TP_ROW_ENV, "auto").strip().lower()
    if mode not in ("auto", "gather"):
        raise ValueError(f"{TP_ROW_ENV} must be auto or gather, got {mode!r}")
    return mode


def tp_out_mode() -> str:
    mode = os.environ.get(TP_OUT_ENV, "ar").strip().lower()
    if mode not in ("ar", "ag"):
        raise ValueError(f"{TP_OUT_ENV} must be ar or ag, got {mode!r}")
    return mode


def needs_gather_mode(
    layout: object, input_size: int, tp_size: int, row_mode: str | None = None
) -> bool:
    """Whether a row-parallel layer with *layout* runs in gather mode.

    Only forward Hadamard layouts constrain the row split; every other layout
    keeps vLLM's row parallelism.
    """
    if tp_size <= 1 or not isinstance(layout, GGUFHadamardLayout):
        return False
    if not layout.forward:
        return False
    if (row_mode or tp_row_mode()) == "gather":
        return True
    local, remainder = divmod(input_size, tp_size)
    return bool(remainder or local % layout.block_size)


def gather_method_for(quant_config, layer: torch.nn.Module, layout) -> GGUFLinearMethod | None:
    """The gather-mode method for a ``RowParallelLinear`` that needs it, else None.

    Called from ``GGUFConfig.get_quant_method`` while ``LinearBase.__init__``
    runs: ``RowParallelLinear`` has already set ``tp_rank`` / ``tp_size`` and
    ``LinearBase`` the logical ``input_size`` / ``output_size``.
    """
    tp_size = int(getattr(layer, "tp_size", 1) or 1)
    input_size = int(getattr(layer, "input_size", 0))
    if not needs_gather_mode(layout, input_size, tp_size):
        return None
    output_size = int(getattr(layer, "output_size", 0))
    if output_size % tp_size:
        raise ValueError(
            f"Cannot run the Hadamard-folded row-parallel layer "
            f"{getattr(layer, 'prefix', '?')} ({output_size} x {input_size}) "
            f"column-parallel at TP {tp_size}: {output_size} output rows do not "
            "divide evenly"
        )
    out_mode = tp_out_mode()
    key = (input_size, output_size, tp_size, out_mode)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "GGUF TP: Hadamard-folded row-parallel %d x %d at TP %d runs in gather "
            "mode (all-gather input, %d output rows a rank over the full K, output "
            "via %s)",
            output_size,
            input_size,
            tp_size,
            output_size // tp_size,
            "padded all-reduce" if out_mode == "ar" else "all-gather",
        )
    return GGUFTPGatherLinearMethod(
        quant_config,
        layout=layout,
        tp_rank=int(getattr(layer, "tp_rank", 0)),
        tp_size=tp_size,
        out_mode=out_mode,
    )


@register_weight_loader_v2_supported_method
class GGUFTPGatherLinearMethod(GGUFLinearMethod):
    """A row-parallel GGUF linear computed column-parallel over its outputs."""

    def __init__(
        self,
        quant_config,
        layout: GGUFHadamardLayout,
        *,
        tp_rank: int,
        tp_size: int,
        out_mode: str = "ar",
    ) -> None:
        super().__init__(quant_config, layout=layout)
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank {tp_rank} out of range for TP {tp_size}")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.out_mode = out_mode
        self.local_rows = 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size_per_partition
        if len(output_partition_sizes) != 1:
            raise ValueError(
                "gather mode expects a row-parallel layer (one output partition), "
                f"got {output_partition_sizes}"
            )
        rows, remainder = divmod(output_partition_sizes[0], self.tp_size)
        if remainder:
            raise ValueError(
                f"{output_partition_sizes[0]} output rows do not divide across "
                f"TP {self.tp_size}"
            )
        self.local_rows = rows
        # This rank's rows over the full input: (N / TP, K).
        super().create_weights(
            layer,
            input_size,
            [rows],
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        set_weight_attrs(layer.weight, {"gguf_tp_gather": True})
        if self.out_mode == "ag" and (
            not getattr(layer, "reduce_results", True)
            or getattr(layer, "has_bias", False)
        ):
            self.out_mode = "ar"
        if self.out_mode == "ag":
            # The output is complete on every rank after the all-gather.
            layer.reduce_results = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_full = tensor_model_parallel_all_gather(x, dim=-1)
        out = super().apply(layer, x_full)
        if self.out_mode == "ag":
            return tensor_model_parallel_all_gather(out, dim=-1)
        before = self.tp_rank * self.local_rows
        after = (self.tp_size - 1 - self.tp_rank) * self.local_rows
        out = F.pad(out, (before, after))
        if bias is not None:
            # RowParallelLinear passes the bias on rank 0 only.
            out = out + bias
        return out
