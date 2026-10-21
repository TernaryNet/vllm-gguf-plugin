import torch
import triton
import triton.language as tl

from ..utils import (
    GGML_TYPE_PTQ1_0,
    load_f16_from_u8,
    load_ptq1_0_trits,
    load_x_chunk,
    run_triton_kernel,
)


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
    offs_q = tl.arange(0, 32)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 28

        for chunk in range(4):
            x_tile = load_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_m,
                m,
                kb * 128 + 32 * chunk,
                CHUNK=32,
            )
            x_dtype = x_tile.dtype
            d = load_f16_from_u8(block_ptrs + 26, n_mask).to(x_dtype)
            trits = load_ptq1_0_trits(
                block_ptrs[:, None], (32 * chunk + offs_q)[None, :], n_mask[:, None]
            )
            q_tile = trits.to(x_dtype) * d[:, None]
            acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < m) & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_gemm_ptq1_0_triton(W: torch.Tensor, X: torch.Tensor, row: int) -> torch.Tensor:
    return run_triton_kernel(ptq1_0_gemm_kernel, W, X, row, GGML_TYPE_PTQ1_0)
