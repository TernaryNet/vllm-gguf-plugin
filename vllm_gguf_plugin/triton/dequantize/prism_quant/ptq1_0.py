import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_PTQ1_0, load_f16_from_u8, load_ptq1_0_trits
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def ptq1_0_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 128) * 28
    trit = load_ptq1_0_trits(block_ptrs, offs % 128, mask)
    d = load_f16_from_u8(block_ptrs + 26, mask).to(tl.float32)
    tl.store(y_ptr + offs, trit.to(tl.float32) * d, mask=mask)


def ggml_dequantize_ptq1_0_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(
        ptq1_0_dequantize_kernel, W, m, n, dtype, GGML_TYPE_PTQ1_0
    )
