// PTQ1_0 planar weight layout: mat-vec and dequantize.
//
// At load time every PTQ1_0 weight row whose block count nb is a multiple of 4
// is rewritten in place (quantization/linear.py, repack_ptq1_0_planar), keeping
// 28 bytes per 128-weight block and the row width:
//   bytes [0, 16 nb)        qs[0:16] of block kb at 16 kb
//   bytes [16 nb, 24 nb)    qs[16:24] of block kb at 16 nb + 8 kb
//   bytes [24 nb, 28 nb)    qh[0], qh[1], d (fp16) of block kb at 24 nb + 4 kb
// Rows then start 16-byte aligned (28 nb is a multiple of 16), and one block
// is one 16-byte, one 8-byte and one 4-byte load, coalesced across the lanes
// that walk consecutive blocks -- instead of seven 4-byte loads at a 28-byte
// lane stride in GGML order. Rows are self-contained, so gathering rows
// (embedding lookup) and row-slicing keep working.
//
// The mat-vec gives each output row LPR lanes (8, 16 or 32) so rows of 40 / 48
// / 136 blocks keep every lane busy. It reproduces the GGML-layout kernel
// (ptq1_0_pt.cuh) bit for bit: each lane keeps one accumulator per "virtual
// lane" of the 32-lane reference (v = kb % 32), fed in the same block order,
// and the final reduction replays the reference's xor butterfly -- the masks
// at or above LPR in registers, the rest as shuffles inside the LPR-lane
// group. fp32 addition is commutative, so every partial sum is the same value.

#include <cstdint>
#include <optional>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "ptq1_0_pt.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

#define PTQ1_0_PLANAR_TYPE 1143          // internal type id, see quantization/utils.py
#define PTQ1_0_PLANAR_WARPS 4            // warps per CTA

static inline cudaStream_t ptq1_0_planar_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return static_cast<cudaStream_t>(raw_stream);
}

// One base-3 digit step on four bytes held as [b0, 0, b1, 0] / [b2, 0, b3, 0]:
// the digits of b0..b3 as four bytes, remainders advanced -- the arithmetic of
// ptq1_0_digits4 (vecdotq.cuh), with the byte gather as one PRMT instead of
// the shift/mask chain nvcc emits for it (same digits, so same results).
static __device__ __forceinline__ int ptq1_0_digits4_prmt(uint32_t & v_lo, uint32_t & v_hi) {
  const uint32_t w_lo = v_lo * 3u;
  const uint32_t w_hi = v_hi * 3u;
  v_lo = w_lo & 0x00FF00FFu;
  v_hi = w_hi & 0x00FF00FFu;
  return (int) __byte_perm(w_lo, w_hi, 0x7531);  // bytes 1, 3 of w_lo, then of w_hi
}

// The GGML-order decode of ptq1_0_pt.cuh, fed from registers: word g of
// qs[0:16] is a.{x,y,z,w}, word g of qs[16:24] is b.{x,y}, c holds qh[0],
// qh[1] in its low bytes and d in its high half. dig[w] holds the digits
// {0,1,2} of elements 4w .. 4w+3.
static __device__ __forceinline__ void ptq1_0_decode_planar(const uint4 a, const uint2 b, const uint32_t c,
                                                            int (&dig)[32]) {
  const uint32_t w16[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t v_lo = __byte_perm(w16[g], 0, 0x4140);  // [b0, 0, b1, 0]
    uint32_t v_hi = __byte_perm(w16[g], 0, 0x4342);  // [b2, 0, b3, 0]
#pragma unroll
    for (int n = 0; n < 5; ++n) {
      dig[4 * n + g] = ptq1_0_digits4_prmt(v_lo, v_hi);  // element 16n + 4g
    }
  }
  const uint32_t w8[2] = {b.x, b.y};
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    uint32_t v_lo = __byte_perm(w8[g], 0, 0x4140);
    uint32_t v_hi = __byte_perm(w8[g], 0, 0x4342);
#pragma unroll
    for (int n = 0; n < 5; ++n) {
      dig[20 + 2 * n + g] = ptq1_0_digits4_prmt(v_lo, v_hi);  // element 80 + 8n + 4g
    }
  }
  uint32_t v = __byte_perm(c, 0, 0x4140);  // [qh0, 0, qh1, 0]
#pragma unroll
  for (int n = 0; n < 4; n += 2) {
    const uint32_t w0 = v * 3u;
    v = w0 & 0x00FF00FFu;
    const uint32_t w1 = v * 3u;
    v = w1 & 0x00FF00FFu;
    dig[30 + n / 2] = (int) __byte_perm(w0, w1, 0x7531);  // element 120 + 2n
  }
}

// PF: load the next group of J blocks before decoding the current one (two
// groups of weight registers live), so a lane with more than J blocks never
// waits for DRAM between groups.
template <typename scalar_t, int NCOLS, int LPR, bool PF>
static __global__ void __launch_bounds__(WARP_SIZE_GGUF * PTQ1_0_PLANAR_WARPS)
mul_mat_vec_ptq1_0_planar_kernel(const uint8_t * __restrict__ vx, const void * __restrict__ vy,
                                 scalar_t * __restrict__ dst, const int nb, const int nrows,
                                 const int nvecs, const int nblk) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610
  static_assert(LPR == 8 || LPR == 16 || LPR == 32, "lanes per row");
  constexpr int R = WARP_SIZE_GGUF / LPR;  // rows per warp
  constexpr int J = WARP_SIZE_GGUF / LPR;  // reference lanes per lane
  const int lane = threadIdx.x % WARP_SIZE_GGUF;
  const int sub  = lane % LPR;
  const int row  = (blockIdx.x * PTQ1_0_PLANAR_WARPS + threadIdx.x / WARP_SIZE_GGUF) * R + lane / LPR;
  // rows past the end walk no blocks but still take part in the shuffles
  const int nbr = row < nrows ? nb : 0;
  const uint8_t * rowp = vx + (int64_t) (row < nrows ? row : 0) * nb * 28;
  const uint4 * pa = (const uint4 *) rowp;
  const uint2 * pb = (const uint2 *) (rowp + 16 * nb);
  const uint32_t * pc = (const uint32_t *) (rowp + 24 * nb);
  const int64_t col_stride_int4 = (int64_t) PTQ1_0_PT_PLANES * nblk;  // in 16-byte units

  float acc[J][NCOLS];
#pragma unroll
  for (int jj = 0; jj < J; ++jj) {
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
      acc[jj][j] = 0.0f;
    }
  }

  uint4 a[J];
  uint2 b[J];
  uint32_t c[J];
  // all J blocks' weight loads first, then the decodes
  auto load_group = [&](const int kb0, uint4 (&ga)[J], uint2 (&gb)[J], uint32_t (&gc)[J]) {
#pragma unroll
    for (int jj = 0; jj < J; ++jj) {
      const int kb = kb0 + jj * LPR;
      if (kb < nbr) {
        ga[jj] = __ldg(pa + kb);
        gb[jj] = __ldg(pb + kb);
        gc[jj] = __ldg(pc + kb);
      } else {
        ga[jj] = make_uint4(0, 0, 0, 0);
        gb[jj] = make_uint2(0, 0);
        gc[jj] = 0;
      }
    }
  };
  if (PF) {
    load_group(sub, a, b, c);
  }
  for (int kb0 = sub; kb0 < nbr; kb0 += WARP_SIZE_GGUF) {
    uint4 na[J];
    uint2 nb_[J];
    uint32_t nc[J];
    if (PF) {
      load_group(kb0 + WARP_SIZE_GGUF, na, nb_, nc);
    } else {
      load_group(kb0, a, b, c);
    }
#pragma unroll
    for (int jj = 0; jj < J; ++jj) {
      const int kb = kb0 + jj * LPR;
      if (kb >= nbr) {
        break;  // later jj are past the end too
      }
      int dig[32];
      ptq1_0_decode_planar(a[jj], b[jj], c[jj], dig);
      const float d = __half2float(__ushort_as_half((unsigned short) (c[jj] >> 16)));

#pragma unroll
      for (int j = 0; j < NCOLS; ++j) {
        if (j >= nvecs) {
          break;  // uniform across the block
        }
        const int4 * col = (const int4 *) vy + j * col_stride_int4;
        const int4 ds = col[8 * nblk + kb];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int4 p0 = col[(2 * k) * nblk + kb];
          const int4 p1 = col[(2 * k + 1) * nblk + kb];
          int sumi = 0;
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            sumi = __dp4a(dig[8 * k + w], ptq1_0_int4_word(p0, w), sumi);
            sumi = __dp4a(dig[8 * k + 4 + w], ptq1_0_int4_word(p1, w), sumi);
          }
          const int dsk = ptq1_0_int4_word(ds, k);
          const float d8  = __half2float(__ushort_as_half((unsigned short) (dsk & 0xFFFF)));
          const int isum  = (int) (short) (dsk >> 16);
          sum += d8 * (float) (sumi - isum);
        }
        acc[jj][j] += d * sum;
      }
    }
    if (PF) {
#pragma unroll
      for (int jj = 0; jj < J; ++jj) {
        a[jj] = na[jj];
        b[jj] = nb_[jj];
        c[jj] = nc[jj];
      }
    }
  }

  // the reference's butterfly over v = sub + LPR * jj, highest mask first
#pragma unroll
  for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
    for (int m = J / 2; m > 0; m >>= 1) {
#pragma unroll
      for (int jj = 0; jj < J; ++jj) {
        if ((jj & m) == 0) {
          const float s = acc[jj][j] + acc[jj ^ m][j];
          acc[jj][j] = s;
          acc[jj ^ m][j] = s;
        }
      }
    }
#pragma unroll
    for (int mask = LPR / 2; mask > 0; mask >>= 1) {
      acc[0][j] += VLLM_SHFL_XOR_SYNC_WIDTH(acc[0][j], mask, LPR);
    }
  }
  if (sub == 0 && row < nrows) {
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
      if (j < nvecs) {
        dst[(int64_t) j * nrows + row] = acc[0][j];
      }
    }
  }
#else
  (void) vx; (void) vy; (void) dst; (void) nb; (void) nrows; (void) nvecs; (void) nblk;
#endif
}

template <typename scalar_t, int NCOLS, int LPR, bool PF = false>
static void launch_ptq1_0_planar(const void * vx, const void * vy, scalar_t * dst, const int nb,
                                 const int nrows, const int nvecs, const int nblk, cudaStream_t stream) {
  constexpr int rows_per_cta = PTQ1_0_PLANAR_WARPS * (WARP_SIZE_GGUF / LPR);
  const dim3 grid((nrows + rows_per_cta - 1) / rows_per_cta, 1, 1);
  const dim3 block(WARP_SIZE_GGUF * PTQ1_0_PLANAR_WARPS, 1, 1);
  mul_mat_vec_ptq1_0_planar_kernel<scalar_t, NCOLS, LPR, PF>
      <<<grid, block, 0, stream>>>((const uint8_t *) vx, vy, dst, nb, nrows, nvecs, nblk);
}

// variant code: lanes per row 8 / 16 / 32, plus 1000 for the prefetching loop
template <typename scalar_t, int NCOLS>
static void launch_ptq1_0_planar_lpr(const int variant, const void * vx, const void * vy, scalar_t * dst,
                                     const int nb, const int nrows, const int nvecs, const int nblk,
                                     cudaStream_t stream) {
  switch (variant) {
    case 8:    launch_ptq1_0_planar<scalar_t, NCOLS, 8>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
    case 16:   launch_ptq1_0_planar<scalar_t, NCOLS, 16>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
    case 1008: launch_ptq1_0_planar<scalar_t, NCOLS, 8, true>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
    case 1016: launch_ptq1_0_planar<scalar_t, NCOLS, 16, true>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
    case 1032: launch_ptq1_0_planar<scalar_t, NCOLS, 32, true>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
    default:   launch_ptq1_0_planar<scalar_t, NCOLS, 32>(vx, vy, dst, nb, nrows, nvecs, nblk, stream); break;
  }
}

// The planar mat-vec for a quantized activation buffer of ncols_q columns
// (one template argument, so it can be called inside the dispatch macro).
template <typename scalar_t>
static void run_ptq1_0_planar(const int64_t ncols_q, const int variant, const void * w, const void * q,
                              scalar_t * y, const int nb, const int nrows, const int nvecs, const int nblk,
                              cudaStream_t stream) {
  switch (ncols_q) {
    case 1: launch_ptq1_0_planar_lpr<scalar_t, 1>(variant, w, q, y, nb, nrows, nvecs, nblk, stream); break;
    case 2: launch_ptq1_0_planar_lpr<scalar_t, 2>(variant, w, q, y, nb, nrows, nvecs, nblk, stream); break;
    case 4: launch_ptq1_0_planar_lpr<scalar_t, 4>(variant, w, q, y, nb, nrows, nvecs, nblk, stream); break;
    case 8: launch_ptq1_0_planar_lpr<scalar_t, 8>(variant, w, q, y, nb, nrows, nvecs, nblk, stream); break;
    default: launch_ptq1_0_planar<scalar_t, 16, 32>(w, q, y, nb, nrows, nvecs, nblk, stream); break;
  }
}

// Lanes per row when the caller passes 0. Placeholder until the micro-bench
// on the 27B shapes picks per-shape values (see tests/run_kopt_checks.py).
static int ptq1_0_planar_auto_lpr(const int nb, const int ncols_q) {
  if (ncols_q >= 16) {
    return 32;  // keeps the 16-column accumulators in registers
  }
  if (nb % 8 == 0 && nb <= 64) {
    return 8;
  }
  return 16;
}

// Y[vecs, row] = X[vecs, K] @ W^T for a planar PTQ1_0 W, vecs <= 16;
// lanes_per_row 0 = auto, else 8 / 16 / 32 (+1000 = prefetching loop); 16
// columns always run 32 lanes per row.
Tensor ggml_mul_mat_vec_ptq1_0_planar(Tensor W, Tensor X, int64_t row, int64_t lanes_per_row) {
  const int64_t col = X.sizes()[1];
  const int64_t vecs = X.sizes()[0];
  STD_TORCH_CHECK(X.dim() == 2 && X.is_contiguous(), "X must be a contiguous 2-D tensor");
  STD_TORCH_CHECK(vecs >= 1 && vecs <= PTQ1_0_PT_MAX_COLS, "planar PTQ1_0 mat-vec takes 1..16 rows of X, got ", vecs);
  STD_TORCH_CHECK(col % (4 * QK_PTQ1_0) == 0, "planar PTQ1_0 needs K % 512 == 0, got ", col);
  const int64_t nb = col / QK_PTQ1_0;
  STD_TORCH_CHECK(W.dim() == 2 && W.sizes()[0] == row && W.sizes()[1] == nb * 28 && W.is_contiguous(),
                  "planar PTQ1_0 weight must be a contiguous [row, K/128*28] byte tensor");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, "planar PTQ1_0 weight must be 16-byte aligned");
  const int64_t lpr_base = lanes_per_row % 1000;
  STD_TORCH_CHECK(lanes_per_row == 0 ||
                      (lanes_per_row < 2000 && (lpr_base == 8 || lpr_base == 16 || lpr_base == 32)),
                  "lanes_per_row must be 0, 8, 16 or 32 (+1000: prefetching loop)");
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = ptq1_0_planar_stream(device_idx);
  const int nblk = ptq1_0_pt_nblk(col);
  const int64_t ncols_q = ptq1_0_pt_ncols(vecs);
  const int lpr = lanes_per_row ? (int) lanes_per_row : ptq1_0_planar_auto_lpr((int) nb, (int) ncols_q);
  Tensor quant_X = torch::stable::new_empty(W, {ncols_q, (int64_t) PTQ1_0_PT_PLANES * nblk * 4}, ScalarType::Int);
  Tensor Y = torch::stable::new_empty(W, {vecs, row}, X.scalar_type());
  const int inb = (int) nb, irow = (int) row, ivecs = (int) vecs;
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_ptq1_0_planar", [&] {
    quantize_q8_1_pt_cuda<scalar_t>((scalar_t *) X.data_ptr(), (void *) quant_X.data_ptr(), (int) col, ivecs, nblk,
                                    stream);
    run_ptq1_0_planar<scalar_t>(ncols_q, lpr, W.data_ptr(), quant_X.data_ptr(), (scalar_t *) Y.data_ptr(), inb, irow,
                                ivecs, nblk, stream);
  });
  return Y;
}

// --- dequantize --------------------------------------------------------------

static __device__ __forceinline__ int ptq1_0_planar_trit(const uint8_t * __restrict__ rowp, const int nb,
                                                         const int kb, const int e) {
  uint8_t b;
  int n;
  if (e < 80) {
    b = rowp[16 * kb + (e & 15)];
    n = e >> 4;
  } else if (e < 120) {
    const int t = e - 80;
    b = rowp[16 * nb + 8 * kb + (t & 7)];
    n = t >> 3;
  } else {
    const int t = e - 120;
    b = rowp[24 * nb + 4 * kb + (t & 1)];
    n = t >> 1;
  }
  uint32_t v = b;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i < n) {
      v = (v * 3) & 0xFF;
    }
  }
  return (int) ((v * 3) >> 8) - 1;
}

// Same arithmetic as dequantize_block<QK_PTQ1_0, QR_PTQ1_0, dequantize_ptq1_0>:
// two consecutive elements per thread, trit * d in half, then the dst type.
template <typename dst_t>
static __global__ void dequantize_ptq1_0_planar_kernel(const uint8_t * __restrict__ vx, dst_t * __restrict__ y,
                                                       const int nb, const int64_t k) {
  const int64_t i = 2 * ((int64_t) blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= k) {
    return;
  }
  const int64_t row_elems = (int64_t) nb * QK_PTQ1_0;
  const int64_t r = i / row_elems;
  const int e = (int) (i - r * row_elems);
  const int kb = e / QK_PTQ1_0;
  const int iqs = e % QK_PTQ1_0;
  const uint8_t * rowp = vx + r * nb * 28;
  const dfloat d = *(const half *) (rowp + 24 * nb + 4 * kb + 2);
  dfloat2 v;
  v.x = __int2half_rn(ptq1_0_planar_trit(rowp, nb, kb, iqs + 0));
  v.y = __int2half_rn(ptq1_0_planar_trit(rowp, nb, kb, iqs + 1));
  v = __hmul2(v, {d, d});
  y[i + 0] = convert_from_half<dst_t>(v.x);
  y[i + 1] = convert_from_half<dst_t>(v.y);
}

template <typename dst_t>
static void launch_dequantize_ptq1_0_planar(const void * w, dst_t * y, const int nb, const int64_t k,
                                            cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t blocks = (k / 2 + threads - 1) / threads;
  dequantize_ptq1_0_planar_kernel<dst_t><<<blocks, threads, 0, stream>>>((const uint8_t *) w, y, nb, k);
}

// Dequantize a planar PTQ1_0 weight [rows, nb * 28] into the [m, n] output
// (m * n == rows * nb * 128; the caller's m/n split does not matter, the
// output is row-major element order either way).
Tensor ggml_dequantize_ptq1_0_planar(Tensor W, int64_t m, int64_t n, std::optional<ScalarType> dtype) {
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous(), "planar PTQ1_0 weight must be a contiguous 2-D byte tensor");
  STD_TORCH_CHECK(W.sizes()[1] % 28 == 0, "planar PTQ1_0 row of ", W.sizes()[1], " bytes is not whole blocks");
  const int64_t nb = W.sizes()[1] / 28;
  STD_TORCH_CHECK(m * n == W.sizes()[0] * nb * QK_PTQ1_0, "dequantize output ", m, "x", n,
                  " does not match the planar weight's ", W.sizes()[0], " rows of ", nb * QK_PTQ1_0);
  const int32_t device_idx = W.get_device_index();
  const DeviceGuard device_guard(device_idx);
  const auto dtype_ = dtype.value_or(ScalarType::Half);
  Tensor DW = torch::stable::new_empty(W, {m, n}, dtype_);
  cudaStream_t stream = ptq1_0_planar_stream(device_idx);
  const int64_t k = m * n;
  if (k == 0) {
    return DW;
  }
  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize_ptq1_0_planar", [&] {
    launch_dequantize_ptq1_0_planar<scalar_t>(W.data_ptr(), (scalar_t *) DW.data_ptr(), (int) nb, k, stream);
  });
  return DW;
}

// --- fused sign + FWHT-1024 + Q8 quantize, then the mat-vec ---------------------
//
// For a Hadamard-folded linear the activation must become H (s * x) (blockwise
// normalized Sylvester transform of 1024, signs s) before the Q8 quantize of
// the mat-vec. Instead of a sign multiply, a rotation kernel and the quantize
// kernel, one CTA per (row, 1024-block) multiplies by the signs, runs the
// butterfly in fp32 (bits 0-2 in registers, 3-7 across lanes, 8-9 through
// shared memory), scales by 1/32, rounds to the activation dtype -- the dtype
// the unfused path stores the rotation in -- and quantizes exactly like
// quantize_q8_1_pt_kernel into the planar Q8 layout. Up to fp32 summation
// order (exact in most elements) this equals the dense x @ H path.

#define PTQ1_0_ROT_BLOCK 1024
#define PTQ1_0_ROT_THREADS 128   // 8 elements per thread

template <typename scalar_t>
static __global__ void __launch_bounds__(PTQ1_0_ROT_THREADS)
rot_quantize_q8_1_pt_kernel(const scalar_t * __restrict__ x, const float * __restrict__ signs,
                            int8_t * __restrict__ y, const int ncols_x, const int nblk) {
  __shared__ float s[PTQ1_0_ROT_BLOCK];
  const int t = threadIdx.x;
  const int g = blockIdx.x;      // 1024-block of the row
  const int row = blockIdx.y;
  const int lane = t % WARP_SIZE_GGUF;
  const int64_t base = (int64_t) row * ncols_x + (int64_t) g * PTQ1_0_ROT_BLOCK + 8 * t;
  const int sbase = g * PTQ1_0_ROT_BLOCK + 8 * t;

  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = static_cast<float>(x[base + i]) * signs[sbase + i];
  }
  // element bits 0-2: in registers
#pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((i & h) == 0) {
        const float a = v[i], b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
  // element bits 3-7: lane bits 0-4
#pragma unroll
  for (int h = 1; h < WARP_SIZE_GGUF; h <<= 1) {
    const bool upper = (lane & h) != 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = VLLM_SHFL_XOR_SYNC_WIDTH(v[i], h, WARP_SIZE_GGUF);
      v[i] = upper ? p - v[i] : v[i] + p;
    }
  }
  // element bits 8-9: warp bits, through shared memory
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s[8 * t + i] = v[i];
  }
  __syncthreads();
#pragma unroll
  for (int half_ = 0; half_ < 2; ++half_) {
    const int b0 = t + 128 * half_;  // low 8 bits of the element index
    float w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      w[q] = s[b0 + 256 * q];
    }
    const float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
    w[0] = a0 + a2;
    w[2] = a0 - a2;
    w[1] = a1 + a3;
    w[3] = a1 - a3;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // 1/sqrt(1024), exact; then the activation dtype's rounding
      s[b0 + 256 * q] = static_cast<float>(static_cast<scalar_t>(w[q] * (1.0f / 32.0f)));
    }
  }
  __syncthreads();

  // quantize: thread t owns elements 8t..8t+7 of this 1024-block; four
  // consecutive threads make one 32-element sub-block
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = s[8 * t + i];
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int mask = 2; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 4));
  }
  const float d = amax / 127.0f;
  int q[8];
  int isum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = amax == 0.0f ? 0 : (int) roundf(v[i] / d);
    isum += q[i];
  }
#pragma unroll
  for (int mask = 2; mask > 0; mask >>= 1) {
    isum += VLLM_SHFL_XOR_SYNC_WIDTH(isum, mask, 4);
  }
  const int kb = g * (PTQ1_0_ROT_BLOCK / QK_PTQ1_0) + t / 16;   // Q8 block of the row
  const int plane = (t % 16) / 2;                                 // (e % 128) / 16
  int8_t * col_base = y + (int64_t) row * PTQ1_0_PT_PLANES * nblk * 16;
  int2 packed;
  packed.x = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
  packed.y = (q[4] & 0xFF) | ((q[5] & 0xFF) << 8) | ((q[6] & 0xFF) << 16) | ((q[7] & 0xFF) << 24);
  *(int2 *) (col_base + ((int64_t) plane * nblk + kb) * 16 + 8 * (t % 2)) = packed;
  if (t % 4 == 0) {
    const int sub = (t % 16) / 4;
    half * ds = (half *) (col_base + ((int64_t) 8 * nblk + kb) * 16 + sub * 4);
    ds[0] = __float2half(d);
    ((int16_t *) ds)[1] = (int16_t) isum;
  }
}

// Y = (H (s * X)) @ W^T for X [vecs <= 16, K], K % 1024 == 0, signs fp32 [K];
// W in GGML order (planar = false) or the planar layout (planar = true).
template <typename scalar_t>
static void run_ptq1_0_rot(const bool planar, const int64_t ncols_q, const int variant, const void * w,
                           const void * x, const float * signs, void * q, scalar_t * y, const int col,
                           const int nb, const int nrows, const int nvecs, const int nblk, cudaStream_t stream) {
  const dim3 qgrid(col / PTQ1_0_ROT_BLOCK, nvecs, 1);
  rot_quantize_q8_1_pt_kernel<scalar_t><<<qgrid, PTQ1_0_ROT_THREADS, 0, stream>>>(
      (const scalar_t *) x, signs, (int8_t *) q, col, nblk);
  if (planar) {
    run_ptq1_0_planar<scalar_t>(ncols_q, variant, w, q, y, nb, nrows, nvecs, nblk, stream);
  } else {
    mul_mat_vec_ptq1_0_pt_cuda<scalar_t>(w, q, y, col, nrows, nvecs, stream);
  }
}

Tensor ggml_mul_mat_vec_ptq1_0_rot(Tensor W, Tensor X, Tensor signs, int64_t row, bool planar) {
  const int64_t col = X.sizes()[1];
  const int64_t vecs = X.sizes()[0];
  STD_TORCH_CHECK(X.dim() == 2 && X.is_contiguous(), "X must be a contiguous 2-D tensor");
  STD_TORCH_CHECK(vecs >= 1 && vecs <= PTQ1_0_PT_MAX_COLS, "fused PTQ1_0 mat-vec takes 1..16 rows of X, got ", vecs);
  STD_TORCH_CHECK(col % PTQ1_0_ROT_BLOCK == 0, "fused rotation needs K % 1024 == 0, got ", col);
  STD_TORCH_CHECK(signs.dim() == 1 && signs.sizes()[0] == col && signs.is_contiguous() &&
                      signs.scalar_type() == ScalarType::Float,
                  "signs must be a contiguous fp32 vector of K elements");
  const int64_t nb = col / QK_PTQ1_0;
  STD_TORCH_CHECK(W.dim() == 2 && W.sizes()[0] == row && W.sizes()[1] == nb * 28 && W.is_contiguous(),
                  "PTQ1_0 weight must be a contiguous [row, K/128*28] byte tensor");
  if (planar) {
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, "planar PTQ1_0 weight must be 16-byte aligned");
  }
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = ptq1_0_planar_stream(device_idx);
  const int nblk = ptq1_0_pt_nblk(col);
  const int64_t ncols_q = ptq1_0_pt_ncols(vecs);
  const int lpr = ncols_q >= 16 ? 32 : ptq1_0_planar_auto_lpr((int) nb, (int) ncols_q);
  Tensor quant_X = torch::stable::new_empty(W, {ncols_q, (int64_t) PTQ1_0_PT_PLANES * nblk * 4}, ScalarType::Int);
  Tensor Y = torch::stable::new_empty(W, {vecs, row}, X.scalar_type());
  const int inb = (int) nb, irow = (int) row, ivecs = (int) vecs, icol = (int) col;
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_ptq1_0_rot", [&] {
    run_ptq1_0_rot<scalar_t>(planar, ncols_q, lpr, W.data_ptr(), X.data_ptr(), (const float *) signs.data_ptr(),
                             quant_X.data_ptr(), (scalar_t *) Y.data_ptr(), icol, inb, irow, ivecs, nblk, stream);
  });
  return Y;
}
