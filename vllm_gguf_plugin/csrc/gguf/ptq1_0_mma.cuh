// PTQ1_0 x Q8 GEMM on int8 tensor cores (mma.sync m16n8k32.s8), for prefill-sized batches.
//
//   Y[M, N] = Q8(X)[M, K] . W[N, K]^T
//
// W stays in the GGML PTQ1_0 block order (block_ptq1_0 in ggml-common.h, 28 bytes per 128
// weights, fp16 scale last). X is quantized by quantize_q8_32_kernel to int8 q[M, K] plus one
// fp16 scale per 32 elements, d[M, K / 32] -- the same values as the Triton tier's
// quantize_q8_sub_blocks (div_rn, roundf, fp16 rtne), i.e. the Q8-per-32 scheme of every PTQ1_0
// tier below 257 rows. The GEMM then reproduces the Triton tier's arithmetic exactly:
//   acc = fma(float(dot_s), dx_s * dw, acc)    per 32-element sub-block s, in K order,
// where dot_s is the exact int32 dot of the sub-block's int8 activations and ternary digits
// (weights are decoded to signed int8, so no isum correction is needed), dx_s and dw the fp16
// scales widened to fp32. Output is rounded once to the activation dtype (rtne), so Y is
// bit-identical to ggml_gemm_ptq1_0_triton for the same X and W.
//
// Kernel shape: a CTA computes a BM x BN tile; the K loop steps one PTQ1_0 block (128 = four
// Q8 sub-blocks = four mma k32 steps) at a time.
//   - A (activations): q rows via cp.async into A_STAGES smem stages of [BM][128] bytes, 16-byte
//     chunks XOR-swizzled by (row & 7) so ldmatrix.x4 is conflict-free.
//   - B (weights): the CTA's BN blocks of the next K step are loaded to registers one step
//     ahead, decoded once per CTA to signed int8 in natural element order and stored to a
//     double-buffered, identically swizzled [BN][128] tile; ldmatrix (non-transposed) of W[n][k]
//     rows gives the col-major B fragments.
//   - Epilogue per sub-block: the mma's C operand is the constant 0x4B400000 (bits of 1.5 * 2^23),
//     so D holds the bits of 1.5 * 2^23 + dot and one FADD yields the exact float of the dot
//     (|dot| <= 32 * 128 < 2^22), avoiding I2F (16 per clock per SM on sm_8x).
// Requirements (checked by the host wrapper): K % 512 == 0 (whole groups of four blocks; rows are
// then 16-byte aligned, as for the planar repack), N % BN == 0, X contiguous.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define PTQ1_0_MMA_BK 128
#define PTQ1_0_MMA_BLOCK_BYTES 28
// unroll of the four sub-block steps of a K step (full unroll hoists all four steps' fragments)
#ifndef PTQ1_0_MMA_S_UNROLL
#define PTQ1_0_MMA_S_UNROLL 1
#endif

namespace ptq1_0_mma {

// ---------------------------------------------------------------------------------------------
// Base-3 digit extraction, four bytes at a time (host + device, so it can be unit-tested on CPU).
// A byte q holds five digits, most significant first; digit n is ((q * 3^n mod 256) * 3) >> 8.
// Two bytes ride in the 16-bit lanes of a word: t = v * 3 + 0x7F00 per lane leaves (3v mod 256)
// in the low byte and digit + 0x7F (<= 0x81, no carry into the next lane) in the high byte;
// gathering the high bytes and XOR 0x80 gives the weight digit - 1 in {-1, 0, 1} as int8.
// ---------------------------------------------------------------------------------------------

static __host__ __device__ __forceinline__ uint32_t prmt(uint32_t x, uint32_t y, uint32_t s) {
#ifdef __CUDA_ARCH__
    return __byte_perm(x, y, s);
#else
    const uint64_t v = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (s >> (4 * i)) & 0x7;
        r |= (uint32_t) ((v >> (8 * sel)) & 0xFF) << (8 * i);
    }
    return r;
#endif
}

// bytes 0, 1 of w into the 16-bit lanes of lo, bytes 2, 3 into hi
static __host__ __device__ __forceinline__ void spread(const uint32_t w, uint32_t & lo, uint32_t & hi) {
    lo = prmt(w, 0u, 0x4140u);
    hi = prmt(w, 0u, 0x4342u);
}

// advance every lane to digit n: v = q * 3^n mod 256 (lane values stay < 2^16 for n <= 4)
static __host__ __device__ __forceinline__ void skip(uint32_t & lo, uint32_t & hi, const uint32_t pow3) {
    lo = (lo * pow3) & 0x00FF00FFu;
    hi = (hi * pow3) & 0x00FF00FFu;
}

// current digit of the four bytes as signed int8 weights, then advance to the next digit
static __host__ __device__ __forceinline__ uint32_t step(uint32_t & lo, uint32_t & hi) {
    const uint32_t tl = lo * 3u + 0x7F007F00u;
    const uint32_t th = hi * 3u + 0x7F007F00u;
    lo = tl & 0x00FF00FFu;
    hi = th & 0x00FF00FFu;
    return prmt(tl, th, 0x7531u) ^ 0x80808080u;
}

// qh[0..1] (low half of the block's last word) -> elements 120..123 and 124..127; element
// 120 + h + 2n is digit n of qh[h], so the digits of the two bytes interleave.
static __host__ __device__ __forceinline__ void qh_trits(const uint32_t qhd, uint32_t & a, uint32_t & b) {
    uint32_t x = prmt(qhd, 0u, 0x4140u);
    const uint32_t t0 = x * 3u + 0x7F007F00u;
    x = t0 & 0x00FF00FFu;
    const uint32_t t1 = x * 3u + 0x7F007F00u;
    x = t1 & 0x00FF00FFu;
    const uint32_t t2 = x * 3u + 0x7F007F00u;
    x = t2 & 0x00FF00FFu;
    const uint32_t t3 = x * 3u + 0x7F007F00u;
    a = prmt(t0, t1, 0x7531u) ^ 0x80808080u;
    b = prmt(t2, t3, 0x7531u) ^ 0x80808080u;
}

// Words of the 7-word block each decode unit reads: unit c covers the 16-byte chunks 2c, 2c+1
// (elements 32c .. 32c + 31, one Q8 sub-block):
//   c = 0: digits 0, 1 of qs[0:16]          (words 0..3)
//   c = 1: digits 2, 3 of qs[0:16]          (words 0..3)
//   c = 2: digit 4 of qs[0:16] | digits 0, 1 of qs[16:24]     (words 0..5)
//   c = 3: digits 2, 3 of qs[16:24] | digit 4 of qs[16:24], qh (words 4..6)
static __host__ __device__ __forceinline__ int unit_first_word(const int c) { return c == 3 ? 4 : 0; }
static __host__ __device__ __forceinline__ int unit_num_words(const int c) { return c == 2 ? 6 : (c == 3 ? 3 : 4); }

// in: the unit's words (unit_first_word .. + unit_num_words); out: 8 words = 32 int8 weights in
// natural element order.
template <int C>
static __host__ __device__ __forceinline__ void decode_unit(const uint32_t * in, uint32_t * out) {
    uint32_t lo, hi;
    if (C == 0 || C == 1) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            spread(in[w], lo, hi);
            if (C == 1) {
                skip(lo, hi, 9u);
            }
            out[w]     = step(lo, hi);
            out[4 + w] = step(lo, hi);
        }
    } else if (C == 2) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            spread(in[w], lo, hi);
            skip(lo, hi, 81u);
            out[w] = step(lo, hi);
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            spread(in[4 + u], lo, hi);
            out[4 + u] = step(lo, hi);
            out[6 + u] = step(lo, hi);
        }
    } else {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            spread(in[u], lo, hi);
            skip(lo, hi, 9u);
            out[u]     = step(lo, hi);
            out[2 + u] = step(lo, hi);
            out[4 + u] = step(lo, hi);
        }
        qh_trits(in[2], out[6], out[7]);
    }
}

static __host__ __device__ __forceinline__ void decode_unit_rt(const int c, const uint32_t * in, uint32_t * out) {
    switch (c) {
        case 0: decode_unit<0>(in, out); break;
        case 1: decode_unit<1>(in, out); break;
        case 2: decode_unit<2>(in, out); break;
        default: decode_unit<3>(in, out); break;
    }
}

#ifdef __CUDACC__

// ---------------------------------------------------------------------------------------------
// Activation quantize: one thread per 8 elements, four threads per 32-element sub-block.
// ---------------------------------------------------------------------------------------------

template <typename scalar_t>
static __global__ void quantize_q8_32_kernel(const scalar_t * __restrict__ x, int8_t * __restrict__ q,
                                             half * __restrict__ d, const int M, const int K) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int k8 = K / 8;
    // groups of four threads share a sub-block; K / 8 is a multiple of 4, so groups never straddle
    // rows, and the whole group exits together
    if (idx >= (int64_t) M * k8) {
        return;
    }
    const int64_t row = idx / k8;
    const int c8 = (int) (idx - row * k8);
    const scalar_t * xp = x + row * K + (int64_t) c8 * 8;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = (float) xp[i];
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 4));
    const float dd = __fdiv_rn(amax, 127.0f);
    uint32_t packed[2] = {0u, 0u};
    if (amax != 0.0f) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int qi = (int) roundf(__fdiv_rn(v[i], dd));
            packed[i / 4] |= ((uint32_t) (qi & 0xFF)) << (8 * (i % 4));
        }
    }
    *reinterpret_cast<uint2 *>(q + row * K + (int64_t) c8 * 8) = make_uint2(packed[0], packed[1]);
    if ((c8 & 3) == 0) {
        d[row * (K / 32) + c8 / 4] = __float2half_rn(dd);
    }
}

// ---------------------------------------------------------------------------------------------
// The GEMM.
// ---------------------------------------------------------------------------------------------

static __device__ __forceinline__ uint32_t smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void cp_async16(const uint32_t dst, const void * src, const int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

static __device__ __forceinline__ void ldmatrix_x4(uint32_t * r, const uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

static __device__ __forceinline__ void mma_s8(int * d, const uint32_t * a, const uint32_t * b, const int c) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c), "r"(c), "r"(c), "r"(c));
}

template <typename out_t>
static __device__ __forceinline__ void store_pair(out_t * p, const float a, const float b) {
    struct alignas(2 * sizeof(out_t)) Pair {
        out_t x, y;
    };
    Pair v;
    v.x = (out_t) a;
    v.y = (out_t) b;
    *reinterpret_cast<Pair *>(p) = v;
}

// 16-byte chunk c of smem row r, XOR-swizzled
static __device__ __forceinline__ int swz(const int r, const int c) {
    return r * 128 + ((c ^ (r & 7)) << 4);
}

template <int BM_, int BN_, int WARPS_M_, int WARPS_N_, int A_STAGES_>
struct Config {
    static constexpr int BM = BM_, BN = BN_, WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, A_STAGES = A_STAGES_;
    static constexpr int NWARPS = WARPS_M * WARPS_N;
    static constexpr int NT = NWARPS * 32;
    static constexpr int WM = BM / WARPS_M;
    static constexpr int WN = BN / WARPS_N;
    static constexpr int MT = WM / 16;   // m16 tiles per warp
    static constexpr int NTL = WN / 8;   // n8 tiles per warp
    // decode: warp w decodes unit c = w % 4 of rows (w / 4) * 32 + lane + j * UG * 32
    static constexpr int UG = NWARPS / 4;
    static constexpr int UPT = BN / (UG * 32);
    static constexpr int A_CPT = BM * 8 / NT;   // 16-byte A chunks per thread per K step
    static constexpr int SMEM = A_STAGES * BM * 128 + 2 * BN * 128 + 2 * BM * 4 * 4 + 2 * BN * 4;
    static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile must be whole m16 tiles and n8 pairs");
    static_assert(NWARPS % 4 == 0 && BN % (UG * 32) == 0, "decode units must tile BN");
    static_assert((BM * 8) % NT == 0, "A chunks must tile the threads");
    static_assert(NT >= BM, "one thread per row loads the activation scales");
    static_assert(A_STAGES >= 2, "at least double-buffered");
};

template <typename out_t, typename Cfg>
static __global__ void __launch_bounds__(Cfg::NT, 1)
    ptq1_0_mma_gemm_kernel(const int8_t * __restrict__ q, const half * __restrict__ dq,
                           const uint8_t * __restrict__ w, out_t * __restrict__ y, const int M, const int N,
                           const int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = Cfg::BM, BN = Cfg::BN, WM = Cfg::WM, WN = Cfg::WN, MT = Cfg::MT, NTL = Cfg::NTL;
    constexpr int S = Cfg::A_STAGES, NT = Cfg::NT, UG = Cfg::UG, UPT = Cfg::UPT;
    constexpr int MAGIC = 0x4B400000;   // bits of 1.5 * 2^23
    constexpr int kSUnroll = PTQ1_0_MMA_S_UNROLL;

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t * As = smem;                                  // S x [BM][128]
    uint8_t * Ws = As + S * BM * 128;                     // 2 x [BN][128]
    float * dxs = reinterpret_cast<float *>(Ws + 2 * BN * 128);   // 2 x [BM][4]
    float * dws = dxs + 2 * BM * 4;                       // 2 x [BN]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / Cfg::WARPS_N;
    const int wn = warp % Cfg::WARPS_N;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int nkb = K / PTQ1_0_MMA_BK;
    const int64_t wrow = (int64_t) nkb * PTQ1_0_MMA_BLOCK_BYTES;
    const int kq = K / 32;

    // weight decode assignment (warp-uniform unit)
    const int uc = warp & 3;
    const int ufirst = unit_first_word(uc);
    const int unum = unit_num_words(uc);
    const int urow0 = (warp >> 2) * 32 + lane;
    uint32_t wreg[UPT][6];
    uint2 dxreg = make_uint2(0u, 0u);

    auto load_w = [&](const int kb) {
#pragma unroll
        for (int j = 0; j < UPT; ++j) {
            const int row = urow0 + j * UG * 32;
            const uint32_t * b32 = reinterpret_cast<const uint32_t *>(
                w + (int64_t) (n0 + row) * wrow + (int64_t) kb * PTQ1_0_MMA_BLOCK_BYTES);
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                if (i < unum) {
                    wreg[j][i] = __ldg(b32 + ufirst + i);
                }
            }
        }
    };
    auto store_w = [&](const int buf) {
        uint8_t * wb = Ws + buf * BN * 128;
#pragma unroll
        for (int j = 0; j < UPT; ++j) {
            const int row = urow0 + j * UG * 32;
            uint32_t out[8];
            decode_unit_rt(uc, wreg[j], out);
            *reinterpret_cast<uint4 *>(wb + swz(row, 2 * uc)) = make_uint4(out[0], out[1], out[2], out[3]);
            *reinterpret_cast<uint4 *>(wb + swz(row, 2 * uc + 1)) = make_uint4(out[4], out[5], out[6], out[7]);
            if (uc == 3) {
                dws[buf * BN + row] = __half2float(__ushort_as_half((unsigned short) (wreg[j][2] >> 16)));
            }
        }
    };
    auto load_dx = [&](const int kb) {
        if (tid < BM) {
            const int r = m0 + tid;
            dxreg = r < M ? *reinterpret_cast<const uint2 *>(dq + (int64_t) r * kq + kb * 4) : make_uint2(0u, 0u);
        }
    };
    auto store_dx = [&](const int buf) {
        if (tid < BM) {
            const float2 f01 = __half22float2(*reinterpret_cast<const half2 *>(&dxreg.x));
            const float2 f23 = __half22float2(*reinterpret_cast<const half2 *>(&dxreg.y));
            *reinterpret_cast<float4 *>(dxs + buf * BM * 4 + tid * 4) = make_float4(f01.x, f01.y, f23.x, f23.y);
        }
    };
    auto issue_a = [&](const int kb, const int stage) {
        const uint32_t ab = smem_u32(As + stage * BM * 128);
#pragma unroll
        for (int i = 0; i < Cfg::A_CPT; ++i) {
            const int ch = tid + i * NT;
            const int row = ch >> 3;
            const int c16 = ch & 7;
            const int gr = m0 + row;
            const int8_t * src = q + (int64_t) (gr < M ? gr : M - 1) * K + (int64_t) kb * PTQ1_0_MMA_BK + c16 * 16;
            cp_async16(ab + swz(row, c16), src, gr < M ? 16 : 0);
        }
    };

    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NTL; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[i][j][e] = 0.0f;
            }
        }
    }

    // prologue: A stages 0 .. S-2, weights and scales of K step 0
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < nkb) {
            issue_a(st, st);
        }
        cp_async_commit();
    }
    load_w(0);
    load_dx(0);
    store_w(0);
    store_dx(0);
    cp_async_wait<S - 2>();
    __syncthreads();

    for (int kb = 0; kb < nkb; ++kb) {
        const int nxt = kb + S - 1;
        if (nxt < nkb) {
            issue_a(nxt, nxt % S);
        }
        cp_async_commit();
        const bool more = kb + 1 < nkb;
        if (more) {
            load_w(kb + 1);
            load_dx(kb + 1);
        }

        const int buf = kb & 1;
        const uint32_t ab = smem_u32(As + (kb % S) * BM * 128);
        const uint32_t wb = smem_u32(Ws + buf * BN * 128);
        const float * dxb = dxs + buf * BM * 4;
        const float * dwb = dws + buf * BN;
        float dwr[NTL][2];
#pragma unroll
        for (int j = 0; j < NTL; ++j) {
            const int col = wn * WN + j * 8 + 2 * t;
            dwr[j][0] = dwb[col];
            dwr[j][1] = dwb[col + 1];
        }
#pragma unroll kSUnroll
        for (int s = 0; s < 4; ++s) {
            uint32_t af[MT][4];
            uint32_t bf[NTL][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * WM + i * 16 + (lane & 15);
                ldmatrix_x4(af[i], ab + swz(r, 2 * s + (lane >> 4)));
            }
#pragma unroll
            for (int jp = 0; jp < NTL / 2; ++jp) {
                const int qd = lane >> 3;
                const int r = wn * WN + jp * 16 + (qd >> 1) * 8 + (lane & 7);
                uint32_t tmp[4];
                ldmatrix_x4(tmp, wb + swz(r, 2 * s + (qd & 1)));
                bf[2 * jp][0] = tmp[0];
                bf[2 * jp][1] = tmp[1];
                bf[2 * jp + 1][0] = tmp[2];
                bf[2 * jp + 1][1] = tmp[3];
            }
            float dxr[MT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * WM + i * 16 + g;
                dxr[i][0] = dxb[r * 4 + s];
                dxr[i][1] = dxb[(r + 8) * 4 + s];
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTL; ++j) {
                    int d[4];
                    mma_s8(d, af[i], bf[j], MAGIC);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float f = __int_as_float(d[e]) - 12582912.0f;
                        const float p = __fmul_rn(dxr[i][e >> 1], dwr[j][e & 1]);
                        acc[i][j][e] = __fmaf_rn(f, p, acc[i][j][e]);
                    }
                }
            }
        }

        if (more) {
            store_w(buf ^ 1);
            store_dx(buf ^ 1);
        }
        cp_async_wait<S - 2>();
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r = m0 + wm * WM + i * 16 + g;
#pragma unroll
        for (int j = 0; j < NTL; ++j) {
            const int c = n0 + wn * WN + j * 8 + 2 * t;
            if (r < M) {
                store_pair(y + (int64_t) r * N + c, acc[i][j][0], acc[i][j][1]);
            }
            if (r + 8 < M) {
                store_pair(y + (int64_t) (r + 8) * N + c, acc[i][j][2], acc[i][j][3]);
            }
        }
    }
#else
    (void) q; (void) dq; (void) w; (void) y; (void) M; (void) N; (void) K;
#endif
}

// Variants for the bench / dispatch (BM, BN, warps M x N, A stages):
//   0: 128 x 128, 2 x 4 warps (64 x 32 warp tile), 3 stages   (default)
//   1: 128 x 128, 4 x 4 warps (32 x 32), 3 stages
//   2: 128 x 128, 2 x 4 warps, 2 stages
//   3: 64 x 128, 2 x 4 warps (32 x 32), 3 stages
//   4: 128 x 256, 2 x 4 warps (64 x 64), 2 stages
using Cfg0 = Config<128, 128, 2, 4, 3>;
using Cfg1 = Config<128, 128, 4, 4, 3>;
using Cfg2 = Config<128, 128, 2, 4, 2>;
using Cfg3 = Config<64, 128, 2, 4, 3>;
using Cfg4 = Config<128, 256, 2, 4, 2>;
#define PTQ1_0_MMA_NUM_VARIANTS 5

static inline int variant_bn(const int variant) { return variant == 4 ? 256 : 128; }

template <typename out_t, typename Cfg>
static void launch_gemm(const int8_t * q, const half * dq, const uint8_t * w, out_t * y, const int M, const int N,
                        const int K, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(ptq1_0_mma_gemm_kernel<out_t, Cfg>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Cfg::SMEM);
        attr_set = true;
    }
    const dim3 grid((M + Cfg::BM - 1) / Cfg::BM, N / Cfg::BN, 1);
    ptq1_0_mma_gemm_kernel<out_t, Cfg><<<grid, Cfg::NT, Cfg::SMEM, stream>>>(q, dq, w, y, M, N, K);
}

template <typename out_t>
static void ptq1_0_mma_gemm_cuda(const int8_t * q, const half * dq, const void * w, out_t * y, const int M,
                                 const int N, const int K, const int variant, cudaStream_t stream) {
    const uint8_t * wb = static_cast<const uint8_t *>(w);
    switch (variant) {
        case 1: launch_gemm<out_t, Cfg1>(q, dq, wb, y, M, N, K, stream); break;
        case 2: launch_gemm<out_t, Cfg2>(q, dq, wb, y, M, N, K, stream); break;
        case 3: launch_gemm<out_t, Cfg3>(q, dq, wb, y, M, N, K, stream); break;
        case 4: launch_gemm<out_t, Cfg4>(q, dq, wb, y, M, N, K, stream); break;
        default: launch_gemm<out_t, Cfg0>(q, dq, wb, y, M, N, K, stream); break;
    }
}

template <typename scalar_t>
static void quantize_q8_32_cuda(const scalar_t * x, int8_t * q, half * d, const int M, const int K,
                                cudaStream_t stream) {
    const int64_t threads = (int64_t) M * (K / 8);
    const int block = 256;
    const int64_t grid = (threads + block - 1) / block;
    quantize_q8_32_kernel<scalar_t><<<(unsigned) grid, block, 0, stream>>>(x, q, d, M, K);
}

#endif  // __CUDACC__

}  // namespace ptq1_0_mma
