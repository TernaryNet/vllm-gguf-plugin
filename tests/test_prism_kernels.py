# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU kernel tests for the PrismML PTQ1_0 ternary type."""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

from .utils import seed_everything

QUANT_TYPE = GGMLQuantizationType.PTQ1_0
DTYPES = [torch.half, torch.bfloat16, torch.float32]
HIDDEN_SIZES = [256, 1024, 2048]
NUM_ROWS = 320
NUM_TOKENS = [1, 2, 7, 83]


def _ternary_weight(num_rows: int, hidden_size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(num_rows, hidden_size // 128, 128))
    trits[..., 0] = 1  # every block reaches its scale
    scales = rng.uniform(0.25, 1.5, size=(num_rows, hidden_size // 128, 1))
    scales = scales.astype(np.float16).astype(np.float32)
    return (trits.astype(np.float32) * scales).reshape(num_rows, hidden_size)


def _packed(weight: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    packed = gguf.quantize(weight, QUANT_TYPE)
    dense = gguf.dequantize(packed, QUANT_TYPE)
    np.testing.assert_array_equal(dense, weight)
    return torch.tensor(packed, device="cuda"), torch.tensor(dense, device="cuda")


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ptq1_0_dequantize(hidden_size: int, dtype: torch.dtype):
    packed, dense = _packed(_ternary_weight(NUM_ROWS, hidden_size))
    ref = dense.to(dtype)
    if ops._CUDA_ENABLED:
        out = ops.ggml_dequantize(packed, QUANT_TYPE, NUM_ROWS, hidden_size, dtype)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    out = ggml_dequantize_triton(packed, QUANT_TYPE, NUM_ROWS, hidden_size, dtype)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ptq1_0_embedding(hidden_size: int, dtype: torch.dtype):
    packed, dense = _packed(_ternary_weight(NUM_ROWS, hidden_size))
    ids = torch.tensor([[0, NUM_ROWS - 1], [NUM_ROWS // 2, 3]], device="cuda")
    out = apply_gguf_embedding(ids, packed, QUANT_TYPE, hidden_size, dtype=dtype)
    torch.testing.assert_close(
        out, torch.embedding(dense.to(dtype), ids), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ptq1_0_mmvq(num_tokens: int, hidden_size: int, dtype: torch.dtype):
    if not ops._CUDA_ENABLED:
        pytest.skip("CUDA extension not available")
    seed_everything(0)
    packed, dense = _packed(_ternary_weight(NUM_ROWS, hidden_size))
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    ref = x @ dense.to(dtype).T
    out = ops.ggml_mul_mat_vec_a8(packed, x, QUANT_TYPE, NUM_ROWS).to(dtype)
    # activations are quantized to Q8_1 per 32 elements inside the kernel
    torch.testing.assert_close(out, ref, atol=0.5, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ptq1_0_gemm_triton(num_tokens: int, hidden_size: int, dtype: torch.dtype):
    seed_everything(0)
    packed, dense = _packed(_ternary_weight(NUM_ROWS, hidden_size))
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    ref = x @ dense.to(dtype).T
    out = ggml_mul_mat_a8_triton(packed, x, QUANT_TYPE, NUM_ROWS)
    # the GEMM quantizes activations to Q8 per 32 elements like MMVQ does
    torch.testing.assert_close(out, ref, atol=0.5, rtol=2e-2)
    if ops._CUDA_ENABLED and num_tokens <= 64:
        mmvq = ops.ggml_mul_mat_vec_a8(packed, x, QUANT_TYPE, NUM_ROWS).to(dtype)
        atol, rtol = {
            torch.float32: (1e-3, 1e-4),
            torch.half: (2e-2, 2e-3),
            torch.bfloat16: (0.1, 1.6e-2),
        }[dtype]
        torch.testing.assert_close(out, mmvq, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_tokens", [1, 6, 7, 16, 17, 64, 128, 129, 300])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_ptq1_0_fused_mul_mat_dispatch(num_tokens: int, dtype: torch.dtype):
    """Every PTQ1_0 dispatch tier (MMVQ, Triton GEMM, dequant + GEMM) agrees."""
    seed_everything(0)
    packed, dense = _packed(_ternary_weight(NUM_ROWS, 1024))
    x = torch.rand((num_tokens, 1024), dtype=dtype, device="cuda")
    ref = x @ dense.to(dtype).T
    out = _fused_mul_mat_gguf(x, packed, int(QUANT_TYPE))
    torch.testing.assert_close(out, ref, atol=0.5, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 8, 9, 16])
@pytest.mark.parametrize("hidden_size", [384, 1024, 2048, 5120])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_ptq1_0_mmv_multi_column(num_tokens: int, hidden_size: int, dtype: torch.dtype):
    """The planar multi-column mat-vec matches the reference and the MMVQ kernel."""
    if not ops.ptq1_0_mmv_available():
        pytest.skip("CUDA extension not available")
    seed_everything(0)
    packed, dense = _packed(_ternary_weight(NUM_ROWS, hidden_size))
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    ref = x @ dense.to(dtype).T
    out = ops.ggml_mul_mat_vec_ptq1_0(packed, x, NUM_ROWS).to(dtype)
    torch.testing.assert_close(out, ref, atol=0.5, rtol=2e-2)
    # Same Q8 activation quantization as the generic kernel; only the fp32
    # summation order differs, plus one output-dtype rounding on each side.
    mmvq = ops.ggml_mul_mat_vec_a8(packed, x, QUANT_TYPE, NUM_ROWS).to(dtype)
    atol, rtol = {
        torch.float32: (1e-3, 1e-4),
        torch.half: (2e-2, 2e-3),
        torch.bfloat16: (0.1, 1.6e-2),
    }[dtype]
    torch.testing.assert_close(out, mmvq, atol=atol, rtol=rtol)
