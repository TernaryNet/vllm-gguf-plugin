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
PTQ1_0_GEMM_BLOCK_N = 64


@triton.jit
def _ptq1_0_byte_map(e):
    """Byte index and base-3 power for block element ``e`` (see ggml-common.h)."""
    in_qs0 = e < 80
    in_qs1 = (e >= 80) & (e < 120)
    t1 = e - 80
    t2 = e - 120
    byte_idx = tl.where(in_qs0, e & 15, tl.where(in_qs1, 16 + (t1 & 7), 24 + (t2 & 1)))
    digit = tl.where(in_qs0, e >> 4, tl.where(in_qs1, t1 >> 3, t2 >> 1))
    pow3 = tl.where(
        digit == 0,
        1,
        tl.where(digit == 1, 3, tl.where(digit == 2, 9, tl.where(digit == 3, 27, 81))),
    )
    return byte_idx, pow3


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
def _ptq1_0_sub_block(
    q_row_ptrs,
    d_row_ptrs,
    isum_row_ptrs,
    m_mask,
    block_ptrs,
    n_mask,
    sub_index,
    byte_idx,
    pow3,
    d_w,
    acc,
):
    """Accumulate one 32-element sub-block: Q8(x) . trits, rescaled."""
    offs_s = tl.arange(0, 32)
    q = tl.load(
        q_row_ptrs[:, None] + sub_index * 32 + offs_s[None, :],
        mask=m_mask[:, None],
        other=0,
    )
    d_x = tl.load(d_row_ptrs + sub_index, mask=m_mask, other=0.0).to(tl.float32)
    isum = tl.load(isum_row_ptrs + sub_index, mask=m_mask, other=0)

    raw = tl.load(
        block_ptrs[:, None] + byte_idx[None, :], mask=n_mask[:, None], other=0
    )
    raw = raw.to(tl.int32)
    # digits {0, 1, 2}; the -1 bias is applied through the activation sums
    digits = ((((raw * pow3[None, :]) & 0xFF) * 3) >> 8).to(tl.int8)
    dot = tl.dot(q, tl.trans(digits), out_dtype=tl.int32)
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_s = tl.arange(0, 32)
    m_mask = offs_m < m
    n_mask = offs_n < n
    byte0, pow0 = _ptq1_0_byte_map(offs_s)
    byte1, pow1 = _ptq1_0_byte_map(32 + offs_s)
    byte2, pow2 = _ptq1_0_byte_map(64 + offs_s)
    byte3, pow3 = _ptq1_0_byte_map(96 + offs_s)

    num_sub_blocks = num_k_blocks * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn
    q_row_ptrs = q_ptr + offs_m * (num_sub_blocks * 32)
    d_row_ptrs = d_ptr + offs_m * num_sub_blocks
    isum_row_ptrs = isum_ptr + offs_m * num_sub_blocks

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 28
        d_w = load_f16_from_u8(block_ptrs + 26, n_mask).to(tl.float32)
        s0 = kb * 4
        acc = _ptq1_0_sub_block(
            q_row_ptrs,
            d_row_ptrs,
            isum_row_ptrs,
            m_mask,
            block_ptrs,
            n_mask,
            s0,
            byte0,
            pow0,
            d_w,
            acc,
        )
        acc = _ptq1_0_sub_block(
            q_row_ptrs,
            d_row_ptrs,
            isum_row_ptrs,
            m_mask,
            block_ptrs,
            n_mask,
            s0 + 1,
            byte1,
            pow1,
            d_w,
            acc,
        )
        acc = _ptq1_0_sub_block(
            q_row_ptrs,
            d_row_ptrs,
            isum_row_ptrs,
            m_mask,
            block_ptrs,
            n_mask,
            s0 + 2,
            byte2,
            pow2,
            d_w,
            acc,
        )
        acc = _ptq1_0_sub_block(
            q_row_ptrs,
            d_row_ptrs,
            isum_row_ptrs,
            m_mask,
            block_ptrs,
            n_mask,
            s0 + 3,
            byte3,
            pow3,
            d_w,
            acc,
        )

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
        num_warps=max(2, block_m // 16),
        num_stages=2,
    )
    if len(x_shape) == 2:
        return y_2d
    return y_2d.view(*x_shape[:-1], row)
