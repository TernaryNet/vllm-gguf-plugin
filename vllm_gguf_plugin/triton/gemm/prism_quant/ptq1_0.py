import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from ..utils import GGML_TYPE_PTQ1_0, _validate_args, load_f16_from_u8

# Activations are quantized to Q8 per 32-element sub-block exactly like the
# MMVQ kernels do (roundf(x / d) with d = amax / 127 stored as fp16), once per
# matmul in a separate pass, so every N tile of the GEMM reads int8 tiles and
# all dispatch tiers see the same activation rounding.
PTQ1_0_GEMM_BLOCK_M = 128
# Swept on an L4 for the Bonsai-27B shapes: a 128-wide N tile wins at every
# batch, 4 warps up to a 64-row M tile and 8 warps at 128, and software
# pipelining only pays off for the 128-row tile.
PTQ1_0_GEMM_BLOCK_N = 128


@triton.jit
def quantize_q8_sub_blocks_kernel(
    x_ptr,
    q_ptr,
    d_ptr,
    isum_ptr,
    m,
    num_sub_blocks,
    stride_xm,
    stride_xk,
    BLOCK_M: tl.constexpr,
):
    """One program quantizes BLOCK_M rows of one 32-element sub-block."""
    pid_m = tl.program_id(axis=0)
    sub = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = sub * 32 + tl.arange(0, 32)
    m_mask = offs_m < m
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    d = libdevice.div_rn(amax, 127.0)
    scaled = tl.where(amax[:, None] == 0.0, 0.0, libdevice.div_rn(x, d[:, None]))
    q = libdevice.round(scaled).to(tl.int32)
    tl.store(
        q_ptr + offs_m[:, None] * (num_sub_blocks * 32) + offs_k[None, :],
        q.to(tl.int8),
        mask=m_mask[:, None],
    )
    tl.store(d_ptr + offs_m * num_sub_blocks + sub, d.to(tl.float16), mask=m_mask)
    tl.store(isum_ptr + offs_m * num_sub_blocks + sub, tl.sum(q, axis=1), mask=m_mask)


@triton.jit
def _trit_digit(packed, n: tl.constexpr):
    """Base-3 digit ``n`` (0 = most significant) of each packed byte."""
    pow3: tl.constexpr = 3**n
    return (((packed * pow3) & 0xFF) * 3) >> 8


@triton.jit
def _trit_digits2(packed, n_first: tl.constexpr):
    """Digits n_first and n_first + 1 of [BLOCK_N, W] bytes as [BLOCK_N, 2 * W].

    Digit index major, matching GGML's element order e = W * n + m.
    """
    pair = tl.join(_trit_digit(packed, n_first), _trit_digit(packed, n_first + 1))
    # join interleaves along a trailing axis: [BLOCK_N, W, 2] -> permute to
    # [BLOCK_N, 2, W] so the digit index is the major one
    pair = tl.permute(pair, (0, 2, 1))
    return tl.reshape(pair, (pair.shape[0], pair.shape[1] * pair.shape[2]))


@triton.jit
def _scaled_dot(dot, isum, d_x, d_w, acc):
    return acc + (dot - isum[:, None]).to(tl.float32) * (d_x[:, None] * d_w[None, :])


@triton.jit
def ptq1_0_gemm_kernel(
    q_ptr,
    d_ptr,
    isum_ptr,
    w_u8_ptr,
    y_ptr,
    m,
    n,
    num_k_blocks,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T for PTQ1_0 weights and pre-quantized Q8 activations.

    Each 128-weight block is decoded from three contiguous byte loads
    (qs[0:16], qs[16:24], qh[0:2]); see block_ptq1_0 in ggml-common.h for the
    element order, which fixes which digits feed each 32-element sub-block.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < m
    n_mask = offs_n < n
    offs_16 = tl.arange(0, 16)
    offs_8 = tl.arange(0, 8)
    offs_2 = tl.arange(0, 2)
    offs_32 = tl.arange(0, 32)
    # int8 tl.dot needs K >= 32, so sub-blocks 2 and 3 join two 16-element
    # digit groups. tl.join interleaves (a0, b0, a1, b1, ...), so the
    # activation columns are gathered in that same order.
    half = offs_32 >> 1
    odd = (offs_32 & 1) == 1
    # sub-block 2: a = elements 64..79 (digit 4 of qs[0:16]),
    #              b = elements 80..95 (digits 0, 1 of qs[16:24])
    cols2 = tl.where(odd, 80 + half, 64 + half)
    # sub-block 3: a = elements 96..111 (digits 2, 3 of qs[16:24]),
    #              b = join(112..119 (digit 4 of qs[16:24]), 120..127 (qh))
    cols3 = tl.where(
        odd, tl.where((half & 1) == 1, 120 + (half >> 1), 112 + (half >> 1)), 96 + half
    )

    num_sub_blocks = num_k_blocks * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn
    q_row_ptrs = q_ptr + offs_m * (num_sub_blocks * 32)
    d_row_ptrs = d_ptr + offs_m * num_sub_blocks
    isum_row_ptrs = isum_ptr + offs_m * num_sub_blocks

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 28
        d_w = load_f16_from_u8(block_ptrs + 26, n_mask).to(tl.float32)
        qs0 = tl.load(
            block_ptrs[:, None] + offs_16[None, :], mask=n_mask[:, None], other=0
        )
        qs1 = tl.load(
            block_ptrs[:, None] + 16 + offs_8[None, :], mask=n_mask[:, None], other=0
        )
        qh = tl.load(
            block_ptrs[:, None] + 24 + offs_2[None, :], mask=n_mask[:, None], other=0
        )
        qs0 = qs0.to(tl.int32)
        qs1 = qs1.to(tl.int32)
        qh = qh.to(tl.int32)
        q_block = q_row_ptrs[:, None] + kb * 128
        s0 = kb * 4

        # sub-block 0: elements 0..31 = digits 0, 1 of qs[0:16]
        q_tile = tl.load(q_block + offs_32[None, :], mask=m_mask[:, None], other=0)
        w_tile = _trit_digits2(qs0, 0).to(tl.int8)
        dot = tl.dot(q_tile, tl.trans(w_tile), out_dtype=tl.int32)
        d_x = tl.load(d_row_ptrs + s0, mask=m_mask, other=0.0).to(tl.float32)
        isum = tl.load(isum_row_ptrs + s0, mask=m_mask, other=0)
        acc = _scaled_dot(dot, isum, d_x, d_w, acc)

        # sub-block 1: elements 32..63 = digits 2, 3 of qs[0:16]
        q_tile = tl.load(q_block + 32 + offs_32[None, :], mask=m_mask[:, None], other=0)
        w_tile = _trit_digits2(qs0, 2).to(tl.int8)
        dot = tl.dot(q_tile, tl.trans(w_tile), out_dtype=tl.int32)
        d_x = tl.load(d_row_ptrs + s0 + 1, mask=m_mask, other=0.0).to(tl.float32)
        isum = tl.load(isum_row_ptrs + s0 + 1, mask=m_mask, other=0)
        acc = _scaled_dot(dot, isum, d_x, d_w, acc)

        # sub-block 2
        q_tile = tl.load(q_block + cols2[None, :], mask=m_mask[:, None], other=0)
        w_tile = tl.reshape(
            tl.join(_trit_digit(qs0, 4), _trit_digits2(qs1, 0)), (BLOCK_N, 32)
        ).to(tl.int8)
        dot = tl.dot(q_tile, tl.trans(w_tile), out_dtype=tl.int32)
        d_x = tl.load(d_row_ptrs + s0 + 2, mask=m_mask, other=0.0).to(tl.float32)
        isum = tl.load(isum_row_ptrs + s0 + 2, mask=m_mask, other=0)
        acc = _scaled_dot(dot, isum, d_x, d_w, acc)

        # sub-block 3
        q_tile = tl.load(q_block + cols3[None, :], mask=m_mask[:, None], other=0)
        qh_digits = tl.reshape(
            tl.permute(tl.join(_trit_digits2(qh, 0), _trit_digits2(qh, 2)), (0, 2, 1)),
            (BLOCK_N, 8),
        )  # elements 120..127 in order
        tail = tl.reshape(tl.join(_trit_digit(qs1, 4), qh_digits), (BLOCK_N, 16))
        w_tile = tl.reshape(tl.join(_trit_digits2(qs1, 2), tail), (BLOCK_N, 32)).to(
            tl.int8
        )
        dot = tl.dot(q_tile, tl.trans(w_tile), out_dtype=tl.int32)
        d_x = tl.load(d_row_ptrs + s0 + 3, mask=m_mask, other=0.0).to(tl.float32)
        isum = tl.load(isum_row_ptrs + s0 + 3, mask=m_mask, other=0)
        acc = _scaled_dot(dot, isum, d_x, d_w, acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def quantize_q8_sub_blocks(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x`` (2-D, contiguous) per 32 elements along the last dim.

    Returns the int8 codes, the fp16 scales and the exact int32 code sums,
    each laid out as ``(rows, K // 32)`` (codes as ``(rows, K)``).
    """
    rows, k = x.shape
    num_sub_blocks = k // 32
    q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    d = torch.empty((rows, num_sub_blocks), dtype=torch.float16, device=x.device)
    isum = torch.empty((rows, num_sub_blocks), dtype=torch.int32, device=x.device)
    block_m = 32
    grid = (triton.cdiv(rows, block_m), num_sub_blocks)
    quantize_q8_sub_blocks_kernel[grid](
        x, q, d, isum, rows, num_sub_blocks, x.stride(0), x.stride(1), BLOCK_M=block_m
    )
    return q, d, isum


def ggml_gemm_ptq1_0_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    W, x_2d, x_shape, num_k_blocks = _validate_args(W, X, row, GGML_TYPE_PTQ1_0)
    q, d, isum = quantize_q8_sub_blocks(x_2d)
    y_2d = torch.empty((x_2d.shape[0], row), device=X.device, dtype=X.dtype)
    # Rows beyond M are masked but still cost decode and rescale work, so keep
    # the M tile close to the batch (tl.dot needs at least 16).
    block_m = 16
    while block_m < min(x_2d.shape[0], PTQ1_0_GEMM_BLOCK_M):
        block_m *= 2
    grid = (triton.cdiv(x_2d.shape[0], block_m), triton.cdiv(row, PTQ1_0_GEMM_BLOCK_N))
    ptq1_0_gemm_kernel[grid](
        q,
        d,
        isum,
        W,
        y_2d,
        x_2d.shape[0],
        row,
        num_k_blocks,
        W.stride(0),
        y_2d.stride(0),
        y_2d.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=PTQ1_0_GEMM_BLOCK_N,
        num_warps=8 if block_m >= 128 else 4,
        num_stages=2 if block_m >= 128 else 1,
    )
    if len(x_shape) == 2:
        return y_2d
    return y_2d.view(*x_shape[:-1], row)
