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
//   - Epilogue per sub-block, two FFMAs per output: the mma's C operand is the constant 0x4B400000
//     (bits of 1.5 * 2^23), so D holds the bits of F = 1.5 * 2^23 + dot (|dot| <= 32 * 128 < 2^22);
//     fma(F, dx, -1.5 * 2^23 * dx) = dot * dx exactly (13 + 11 significant bits, the bias term exact
//     too), and fma(dot * dx, dw, acc) equals the Triton tier's fma(float(dot), dx * dw, acc) because
//     dx * dw (11 + 11 bits) is exact as well -- no I2F (16 per clock per SM on sm_8x), no FMUL.
//     Fragments of sub-block s + 1 are loaded (ldmatrix) while sub-block s computes.
// Requirements (checked by the host wrapper): K % 512 == 0 (whole groups of four blocks; rows are
// then 16-byte aligned, as for the planar repack), N % BN == 0, X contiguous.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define PTQ1_0_MMA_BK 128
#define PTQ1_0_MMA_BLOCK_BYTES 28

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

// G = 32: the Triton tier's Q8 (bit-identical). G = 128: one scale per PTQ1_0 block (SB128 GEMM).
template <typename scalar_t, int G>
static __global__ void quantize_q8_kernel(const scalar_t * __restrict__ x, int8_t * __restrict__ q,
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
#pragma unroll
    for (int mask = 1; mask < G / 8; mask <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask, G / 8));
    }
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
    if ((c8 & (G / 8 - 1)) == 0) {
        d[row * (K / G) + c8 / (G / 8)] = __float2half_rn(dd);
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

static __device__ __forceinline__ void mma_s8_acc(int * d, const uint32_t * a, const uint32_t * b) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// volatile: keeps the mma in program order with the (volatile) ldmatrix of the next sub-block, so
// the fragment prefetch distance stays one sub-block instead of the compiler hoisting every load
static __device__ __forceinline__ void mma_s8(int * d, const uint32_t * a, const uint32_t * b, const int c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
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
    // Weight decode. HALVES: warp w decodes half (w & 1) of rows (w >> 1) * 32 + lane + j * HG * 32
    // (half 0 = elements 0..63 from qs[0:16]; half 1 = 64..127, which needs the whole block).
    // Otherwise units of 32 elements: warp w decodes unit w % 4 of rows (w / 4) * 32 + lane + j * UG * 32.
    static constexpr bool HALVES = (NWARPS % 2 == 0) && (BN % ((NWARPS / 2) * 32) == 0) && (NWARPS <= 8);
    static constexpr int GROUPS = HALVES ? NWARPS / 2 : NWARPS / 4;
    static constexpr int RPT = BN / (GROUPS * 32);   // rows per thread
    static constexpr int A_CPT = BM * 8 / NT;   // 16-byte A chunks per thread per K step
    // A stages, decoded W (2 x [BN][128]), raw W groups of four blocks (2 x [BN][112]), scales
    static constexpr int SMEM = A_STAGES * BM * 128 + 2 * BN * 128 + 2 * BN * 112 + 2 * BM * 4 * 4 + 2 * BN * 4;
    static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile must be whole m16 tiles and n8 pairs");
    static_assert(HALVES || (NWARPS % 4 == 0 && BN % ((NWARPS / 4) * 32) == 0), "decode must tile BN");
    static_assert((BM * 8) % NT == 0, "A chunks must tile the threads");
    static_assert(NT >= BM, "one thread per row loads the activation scales");
    static_assert(A_STAGES >= 2, "at least double-buffered");
    static_assert(SMEM <= 99 * 1024, "sm_8x allows at most 99 KB of shared memory per block");
};

// half 0 of a block (elements 0..63, words 0..3 = qs[0:16]) -> 16 words (chunks 0..3)
static __device__ __forceinline__ void decode_half0(const uint32_t * in, uint32_t * out) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t lo, hi;
        spread(in[w], lo, hi);
        out[w] = step(lo, hi);
        out[4 + w] = step(lo, hi);
        out[8 + w] = step(lo, hi);
        out[12 + w] = step(lo, hi);
    }
}

// half 1 (elements 64..127, words 0..6) -> 16 words (chunks 4..7)
static __device__ __forceinline__ void decode_half1(const uint32_t * in, uint32_t * out) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t lo, hi;
        spread(in[w], lo, hi);
        skip(lo, hi, 81u);
        out[w] = step(lo, hi);
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        uint32_t lo, hi;
        spread(in[4 + u], lo, hi);
        out[4 + u] = step(lo, hi);    // elements 80.. / 84..
        out[6 + u] = step(lo, hi);    // 88.. / 92..
        out[8 + u] = step(lo, hi);    // 96.. / 100..
        out[10 + u] = step(lo, hi);   // 104.. / 108..
        out[12 + u] = step(lo, hi);   // 112.. / 116..
    }
    qh_trits(in[6], out[14], out[15]);
}

template <typename out_t, typename Cfg>
static __global__ void __launch_bounds__(Cfg::NT, 1)
    ptq1_0_mma_gemm_kernel(const int8_t * __restrict__ q, const half * __restrict__ dq,
                           const uint8_t * __restrict__ w, out_t * __restrict__ y, const int M, const int N,
                           const int K, const int ablate) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = Cfg::BM, BN = Cfg::BN, WM = Cfg::WM, WN = Cfg::WN, MT = Cfg::MT, NTL = Cfg::NTL;
    constexpr int S = Cfg::A_STAGES, NT = Cfg::NT, GROUPS = Cfg::GROUPS, RPT = Cfg::RPT;
    constexpr bool HALVES = Cfg::HALVES;
    constexpr int MAGIC = 0x4B400000;          // bits of 1.5 * 2^23
    constexpr float MAGIC_F = 12582912.0f;     // 1.5 * 2^23

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t * As = smem;                                           // S x [BM][128]
    uint8_t * Ws = As + S * BM * 128;                              // 2 x [BN][128] decoded int8
    uint8_t * Rs = Ws + 2 * BN * 128;                              // 2 x [BN][112] raw block groups
    float * dxs = reinterpret_cast<float *>(Rs + 2 * BN * 112);    // 2 x [BM][4]
    float * dws = dxs + 2 * BM * 4;                                // 2 x [BN]

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

    // weight decode assignment (warp-uniform): a half block, or a 32-element unit
    const int part = HALVES ? (warp & 1) : (warp & 3);
    const int wfirst = HALVES ? 0 : unit_first_word(part);
    const int wnum = HALVES ? (part == 0 ? 4 : 7) : unit_num_words(part);
    const int urow0 = (HALVES ? (warp >> 1) : (warp >> 2)) * 32 + lane;
    uint2 dxreg = make_uint2(0u, 0u);

    // raw weights: group G = blocks 4G .. 4G+3 of the CTA's BN rows, 112 contiguous bytes a row
    // (16-byte aligned since K % 512 == 0), staged with coalesced 16-byte cp.async
    auto issue_raw = [&](const int grp, const int rbuf) {
        if (ablate & 2) return;
        const uint32_t rb = smem_u32(Rs + rbuf * BN * 112);
        for (int ch = tid; ch < BN * 7; ch += NT) {
            const int row = ch / 7;
            const int c = ch - row * 7;
            cp_async16(rb + row * 112 + c * 16, w + (int64_t) (n0 + row) * wrow + (int64_t) grp * 112 + c * 16, 16);
        }
    };
    // decode block kb of the CTA's rows from the raw stage into decoded buffer buf
    auto decode_store = [&](const int kb, const int buf) {
        if (ablate & 1) return;
        const uint8_t * rb = Rs + ((kb >> 2) & 1) * BN * 112 + (kb & 3) * PTQ1_0_MMA_BLOCK_BYTES;
        uint8_t * wb = Ws + buf * BN * 128;
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            const int row = urow0 + j * GROUPS * 32;
            const uint32_t * b32 = reinterpret_cast<const uint32_t *>(rb + row * 112);
            uint32_t win[7];
#pragma unroll
            for (int i = 0; i < 7; ++i) {
                if (i < wnum) {
                    win[i] = b32[wfirst + i];
                }
            }
            if (HALVES) {
                uint32_t out[16];
                if (part == 0) {
                    decode_half0(win, out);
                } else {
                    decode_half1(win, out);
                    dws[buf * BN + row] = __half2float(__ushort_as_half((unsigned short) (win[6] >> 16)));
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    *reinterpret_cast<uint4 *>(wb + swz(row, 4 * part + c)) =
                        make_uint4(out[4 * c], out[4 * c + 1], out[4 * c + 2], out[4 * c + 3]);
                }
            } else {
                uint32_t out[8];
                decode_unit_rt(part, win, out);
                *reinterpret_cast<uint4 *>(wb + swz(row, 2 * part)) = make_uint4(out[0], out[1], out[2], out[3]);
                *reinterpret_cast<uint4 *>(wb + swz(row, 2 * part + 1)) = make_uint4(out[4], out[5], out[6], out[7]);
                if (part == 3) {
                    dws[buf * BN + row] = __half2float(__ushort_as_half((unsigned short) (win[2] >> 16)));
                }
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
        if (ablate & 4) return;
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

    // prologue: raw group 0 with A stage 0, A stages 1 .. S-2, then block 0 decoded
    issue_raw(0, 0);
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < nkb) {
            issue_a(st, st);
        }
        cp_async_commit();
    }
    load_dx(0);
    cp_async_wait<S - 2>();
    __syncthreads();
    decode_store(0, 0);
    store_dx(0);
    __syncthreads();

    // per-lane ldmatrix row offsets (the swizzle only depends on row & 7 = lane & 7)
    const int a_row = wm * WM + (lane & 15);
    const int b_row = wn * WN + ((lane >> 4) << 3) + (lane & 7);
    const int a_hi = lane >> 4;           // A: chunk 2s + (lane >> 4)
    const int b_hi = (lane >> 3) & 1;     // B: chunk 2s + ((lane >> 3) & 1)

    for (int kb = 0; kb < nkb; ++kb) {
        const int nxt = kb + S - 1;
        if (nxt < nkb) {
            issue_a(nxt, nxt % S);
        }
        // the next raw group, one group ahead: complete (and visible) by the end of block 4G + 2,
        // its buffer last read when block 4G - 1 was decoded
        if ((kb & 3) == 0 && kb + 4 < nkb) {
            issue_raw((kb >> 2) + 1, ((kb >> 2) + 1) & 1);
        }
        cp_async_commit();
        const bool more = kb + 1 < nkb;
        if (more) {
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

        // fragments of sub-block s + 1 load while sub-block s runs
        uint32_t af[2][MT][4];
        uint32_t bf[2][NTL][2];
        auto load_frags = [&](const int s, const int slot) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ldmatrix_x4(af[slot][i], ab + swz(a_row + i * 16, 2 * s + a_hi));
            }
#pragma unroll
            for (int jp = 0; jp < NTL / 2; ++jp) {
                uint32_t tmp[4];
                ldmatrix_x4(tmp, wb + swz(b_row + jp * 16, 2 * s + b_hi));
                bf[slot][2 * jp][0] = tmp[0];
                bf[slot][2 * jp][1] = tmp[1];
                bf[slot][2 * jp + 1][0] = tmp[2];
                bf[slot][2 * jp + 1][1] = tmp[3];
            }
        };
        load_frags(0, 0);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int cur = s & 1;
            if (s < 3) {
                load_frags(s + 1, cur ^ 1);
            }
            // dx and -1.5*2^23*dx: both exact in fp32 (11 + 2 significant bits)
            float2 dxv[MT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * WM + i * 16 + g;
                const float d0 = dxb[r * 4 + s];
                const float d1 = dxb[(r + 8) * 4 + s];
                dxv[i][0] = make_float2(d0, -MAGIC_F * d0);
                dxv[i][1] = make_float2(d1, -MAGIC_F * d1);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTL; ++j) {
                    int d[4];
                    mma_s8(d, af[cur][i], bf[cur][j], MAGIC);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        // D holds the bits of 1.5*2^23 + dot; F*dx - 1.5*2^23*dx = dot*dx exactly
                        // (13 + 11 significant bits), and dot*dx*dw = dot*(dx*dw) exactly, so this is
                        // fma(float(dot), dx*dw, acc) with one rounding -- the Triton tier's value
                        const float2 dv = dxv[i][e >> 1];
                        const float gdx = __fmaf_rn(__int_as_float(d[e]), dv.x, dv.y);
                        acc[i][j][e] = __fmaf_rn(gdx, dwr[j][e & 1], acc[i][j][e]);
                    }
                }
            }
        }

        if (more) {
            decode_store(kb + 1, buf ^ 1);
            store_dx(buf ^ 1);
        }
        cp_async_wait<S - 2>();
        if (!(ablate & 8)) __syncthreads();
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
    (void) q; (void) dq; (void) w; (void) y; (void) M; (void) N; (void) K; (void) ablate;
#endif
}

// Variants for the bench / dispatch (BM, BN, warps M x N, A stages):
//   0: 128 x 128, 2 x 4 warps (64 x 32 warp tile), 2 stages
//   1: 128 x 128, 4 x 4 warps (32 x 32), 2 stages
//   2: 128 x 128, 4 x 2 warps (32 x 64), 2 stages
//   3: 64 x 128, 2 x 4 warps (32 x 32), 3 stages
//   4: 64 x 128, 1 x 4 warps (64 x 32), 3 stages
using Cfg0 = Config<128, 128, 2, 4, 2>;
using Cfg1 = Config<128, 128, 4, 4, 2>;
using Cfg2 = Config<128, 128, 4, 2, 2>;
using Cfg3 = Config<64, 128, 2, 4, 3>;
using Cfg4 = Config<64, 128, 1, 4, 3>;
#define PTQ1_0_MMA_NUM_VARIANTS 5

static inline int variant_bn(const int variant) { (void) variant; return 128; }

template <typename out_t, typename Cfg>
static void launch_gemm(const int8_t * q, const half * dq, const uint8_t * w, out_t * y, const int M, const int N,
                        const int K, cudaStream_t stream, const int ablate) {
    static bool attr_set = false;
    if (!attr_set) {
        const cudaError_t err = cudaFuncSetAttribute(ptq1_0_mma_gemm_kernel<out_t, Cfg>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::SMEM);
        STD_TORCH_CHECK(err == cudaSuccess, "ptq1_0_mma: cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
        attr_set = true;
    }
    const dim3 grid((M + Cfg::BM - 1) / Cfg::BM, N / Cfg::BN, 1);
    ptq1_0_mma_gemm_kernel<out_t, Cfg><<<grid, Cfg::NT, Cfg::SMEM, stream>>>(q, dq, w, y, M, N, K, ablate);
}

template <typename out_t>
static void ptq1_0_mma_gemm_cuda(const int8_t * q, const half * dq, const void * w, out_t * y, const int M,
                                 const int N, const int K, const int variant, cudaStream_t stream) {
    const uint8_t * wb = static_cast<const uint8_t *>(w);
    const int ablate = variant >> 8;
    switch (variant & 255) {
        case 1: launch_gemm<out_t, Cfg1>(q, dq, wb, y, M, N, K, stream, ablate); break;
        case 2: launch_gemm<out_t, Cfg2>(q, dq, wb, y, M, N, K, stream, ablate); break;
        case 3: launch_gemm<out_t, Cfg3>(q, dq, wb, y, M, N, K, stream, ablate); break;
        case 4: launch_gemm<out_t, Cfg4>(q, dq, wb, y, M, N, K, stream, ablate); break;
        default: launch_gemm<out_t, Cfg0>(q, dq, wb, y, M, N, K, stream, ablate); break;
    }
}

static inline int variant_group(const int variant) { (void) variant; return 32; }

template <typename scalar_t>
static void quantize_q8_cuda(const scalar_t * x, int8_t * q, half * d, const int M, const int K, const int group,
                             cudaStream_t stream) {
    const int64_t threads = (int64_t) M * (K / 8);
    const int block = 256;
    const int64_t grid = (threads + block - 1) / block;
    if (group == 128) {
        quantize_q8_kernel<scalar_t, 128><<<(unsigned) grid, block, 0, stream>>>(x, q, d, M, K);
    } else {
        quantize_q8_kernel<scalar_t, 32><<<(unsigned) grid, block, 0, stream>>>(x, q, d, M, K);
    }
}

#endif  // __CUDACC__

}  // namespace ptq1_0_mma
