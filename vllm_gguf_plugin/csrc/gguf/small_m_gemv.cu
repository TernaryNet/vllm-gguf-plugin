// Unquantized (bf16 / fp16 / fp32) Y = X @ W^T for 2-8 rows of X and a short W.
//
// Bonsai's GDN in_proj_ba is a BF16 [96, 5120] weight. At one row cuBLAS runs it
// as a gemv (~5 us), but at 2-8 rows (speculative verify, several streams) its
// heuristic picks a 16x16-tile tensor-core GEMM that walks K serially in a
// handful of CTAs (~36 us on an L40S, 48 calls a step). Here one CTA per output
// row splits K over 128 threads (16-byte loads of W and of every X row), keeps
// one fp32 accumulator per X row, and reduces across the CTA: W is read once
// and the launch fills the GPU with N CTAs.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

#define SMALL_M_THREADS 128
#define SMALL_M_MAX_ROWS 8

static inline cudaStream_t small_m_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return static_cast<cudaStream_t>(raw_stream);
}

// 8 consecutive elements (16 bytes for 16-bit types) as floats
template <typename scalar_t>
struct small_m_vec {
  static constexpr int ELEMS = 16 / sizeof(scalar_t);
  static __device__ __forceinline__ void load(const scalar_t * p, float (&out)[ELEMS]) {
    const uint4 raw = *(const uint4 *) p;
    const scalar_t * e = (const scalar_t *) &raw;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
      out[i] = static_cast<float>(e[i]);
    }
  }
};

template <typename scalar_t, int M>
static __global__ void __launch_bounds__(SMALL_M_THREADS)
small_m_gemv_kernel(const scalar_t * __restrict__ w, const scalar_t * __restrict__ x, scalar_t * __restrict__ y,
                    const int n_rows, const int k) {
  constexpr int E = small_m_vec<scalar_t>::ELEMS;
  const int n = blockIdx.x;
  const scalar_t * wr = w + (int64_t) n * k;
  float acc[M];
#pragma unroll
  for (int j = 0; j < M; ++j) {
    acc[j] = 0.0f;
  }
  for (int k0 = threadIdx.x * E; k0 < k; k0 += SMALL_M_THREADS * E) {
    float wv[E];
    small_m_vec<scalar_t>::load(wr + k0, wv);
#pragma unroll
    for (int j = 0; j < M; ++j) {
      float xv[E];
      small_m_vec<scalar_t>::load(x + (int64_t) j * k + k0, xv);
#pragma unroll
      for (int i = 0; i < E; ++i) {
        acc[j] = fmaf(wv[i], xv[i], acc[j]);
      }
    }
  }
  __shared__ float part[SMALL_M_THREADS / 32][M];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
#pragma unroll
  for (int j = 0; j < M; ++j) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      acc[j] += VLLM_SHFL_XOR_SYNC_WIDTH(acc[j], mask, 32);
    }
    if (lane == 0) {
      part[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x < M) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < SMALL_M_THREADS / 32; ++i) {
      s += part[i][threadIdx.x];
    }
    y[(int64_t) threadIdx.x * n_rows + n] = s;
  }
}

template <typename scalar_t>
static void run_small_m_gemv(const int m, const void * w, const void * x, void * y, const int n_rows, const int k,
                             cudaStream_t stream) {
  const dim3 grid(n_rows, 1, 1);
  const dim3 block(SMALL_M_THREADS, 1, 1);
  const scalar_t * wp = (const scalar_t *) w;
  const scalar_t * xp = (const scalar_t *) x;
  scalar_t * yp = (scalar_t *) y;
  switch (m) {
    case 2: small_m_gemv_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    case 3: small_m_gemv_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    case 4: small_m_gemv_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    case 5: small_m_gemv_kernel<scalar_t, 5><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    case 6: small_m_gemv_kernel<scalar_t, 6><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    case 7: small_m_gemv_kernel<scalar_t, 7><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
    default: small_m_gemv_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(wp, xp, yp, n_rows, k); break;
  }
}

// Y [m, n] = X [m, k] @ W [n, k]^T, 2 <= m <= 8, W and X contiguous, same dtype,
// k a multiple of 16 bytes' worth of elements, both 16-byte aligned.
Tensor small_m_gemv(Tensor W, Tensor X) {
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && W.is_contiguous() && X.is_contiguous(),
                  "small_m_gemv: W and X must be contiguous 2-D tensors");
  STD_TORCH_CHECK(W.scalar_type() == X.scalar_type(), "small_m_gemv: W and X dtypes differ");
  const int64_t m = X.sizes()[0], k = X.sizes()[1], n = W.sizes()[0];
  STD_TORCH_CHECK(W.sizes()[1] == k, "small_m_gemv: K mismatch");
  STD_TORCH_CHECK(m >= 2 && m <= SMALL_M_MAX_ROWS, "small_m_gemv takes 2..8 rows of X, got ", m);
  const int64_t elem = X.scalar_type() == ScalarType::Float ? 4 : 2;
  STD_TORCH_CHECK((k * elem) % 16 == 0, "small_m_gemv: rows must be whole 16-byte chunks");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0,
                  "small_m_gemv: W and X must be 16-byte aligned");
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = small_m_stream(device_idx);
  Tensor Y = torch::stable::new_empty(X, {m, n}, X.scalar_type());
  const int im = (int) m, in = (int) n, ik = (int) k;
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "small_m_gemv", [&] {
    run_small_m_gemv<scalar_t>(im, W.data_ptr(), X.data_ptr(), Y.data_ptr(), in, ik, stream);
  });
  return Y;
}
