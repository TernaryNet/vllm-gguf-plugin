import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from ..utils import GGML_TYPE_PTQ1_0, load_f16_from_u8, run_triton_kernel

# Activations are quantized to Q8 per 32-element sub-block inside the kernel,
# exactly like the MMVQ kernels do, so the products run on int8 tensor cores
# and every batch size sees the same activation rounding.
PTQ1_0_GEMM_BLOCK_M = 64
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
def _ptq1_0_sub_block(
    x_row_ptrs,
    stride_xk,
    m_mask,
    block_ptrs,
    n_mask,
    k_start,
    byte_idx,
    pow3,
    d_w,
    acc,
):
    """Accumulate one 32-element sub-block: Q8(x) . trits, rescaled."""
    offs_s = tl.arange(0, 32)
    x_tile = tl.load(
        x_row_ptrs[:, None] + (k_start + offs_s)[None, :] * stride_xk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x_tile), axis=1)
    d_x = libdevice.div_rn(amax, 127.0)
    # IEEE division keeps ties identical to the CUDA quantizer's roundf(x / d)
    scaled = tl.where(amax[:, None] == 0.0, 0.0, libdevice.div_rn(x_tile, d_x[:, None]))
    q = libdevice.round(scaled).to(tl.int32)
    isum = tl.sum(q, axis=1)
    # block_q8_1 stores the activation scale as fp16; use the same value here
    # so every dispatch tier produces identical results
    d_x = d_x.to(tl.float16).to(tl.float32)

    raw = tl.load(
        block_ptrs[:, None] + byte_idx[None, :], mask=n_mask[:, None], other=0
    )
    raw = raw.to(tl.int32)
    # digits {0, 1, 2}; the -1 bias is applied through the activation sums
    digits = ((((raw * pow3[None, :]) & 0xFF) * 3) >> 8).to(tl.int8)
    dot = tl.dot(q.to(tl.int8), tl.trans(digits), out_dtype=tl.int32)
    return acc + (dot - isum[:, None]).to(tl.float32) * d_x[:, None] * d_w[None, :]


@triton.jit
def ptq1_0_gemm_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    m,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn
    x_row_ptrs = x_ptr + offs_m * stride_xm

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 28
        d_w = load_f16_from_u8(block_ptrs + 26, n_mask).to(tl.float32)
        k0 = kb * 128
        acc = _ptq1_0_sub_block(
            x_row_ptrs, stride_xk, m_mask, block_ptrs, n_mask, k0, byte0, pow0, d_w, acc
        )
        acc = _ptq1_0_sub_block(
            x_row_ptrs,
            stride_xk,
            m_mask,
            block_ptrs,
            n_mask,
            k0 + 32,
            byte1,
            pow1,
            d_w,
            acc,
        )
        acc = _ptq1_0_sub_block(
            x_row_ptrs,
            stride_xk,
            m_mask,
            block_ptrs,
            n_mask,
            k0 + 64,
            byte2,
            pow2,
            d_w,
            acc,
        )
        acc = _ptq1_0_sub_block(
            x_row_ptrs,
            stride_xk,
            m_mask,
            block_ptrs,
            n_mask,
            k0 + 96,
            byte3,
            pow3,
            d_w,
            acc,
        )

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def ggml_gemm_ptq1_0_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    # Rows beyond M are masked but still cost decode and rescale work, so keep
    # the M tile close to the batch (tl.dot needs at least 16).
    num_rows = X.numel() // X.shape[-1]
    block_m = 16
    while block_m < min(num_rows, PTQ1_0_GEMM_BLOCK_M):
        block_m *= 2
    return run_triton_kernel(
        ptq1_0_gemm_kernel,
        W,
        X,
        row,
        GGML_TYPE_PTQ1_0,
        block_m=block_m,
        block_n=PTQ1_0_GEMM_BLOCK_N,
    )
