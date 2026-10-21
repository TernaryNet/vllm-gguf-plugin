// PrismML PTQ1_0 mat-vec with a planar-transposed Q8 activation layout.
//
// The generic mmvq kernel re-reads and re-decodes every weight block once per
// activation column, so its cost grows linearly with the batch. Here a lane
// decodes its 128-weight block once and dots it against up to NCOLS columns.
// To make the per-column activation loads cheap, the quantized activations
// are stored per column as 9 planes of 16-byte pieces (after the design in
// PrismML-Eng/llama.cpp's mmvq-ptq1_0.cuh):
//   plane t (t = 0..7): nblk * 16 bytes; byte b of block kb holds the Q8 value
//                       of element kb*128 + t*16 + b
//   plane 8:            nblk * 16 bytes; block kb holds four (half d, int16 isum)
//                       pairs, one per 32-element sub-block, isum being the
//                       exact integer sum of that sub-block's quantized values
// The column stride is 9 * nblk * 16 bytes, which equals the block_q8_1 stride
// for the same padded row length, and nblk covers the row padded to 512.

#define PTQ1_0_PT_PLANES 9
#define PTQ1_0_PT_MAX_COLS 16
#define PTQ1_0_PT_ROWS_PER_BLOCK 4  // one warp per weight row

static __host__ __device__ __forceinline__ int ptq1_0_pt_nblk(const int ncols_x) {
    return ((ncols_x + 512 - 1) / 512) * (512 / QK_PTQ1_0);
}

// One warp quantizes one 128-element block of one activation row.
template <typename scalar_t>
static __global__ void quantize_q8_1_pt_kernel(const scalar_t * __restrict__ x, int8_t * __restrict__ y,
                                               const int ncols_x, const int nrows_x, const int nblk) {
    const int lane = threadIdx.x % WARP_SIZE_GGUF;
    const int kb   = blockIdx.x * (blockDim.x / WARP_SIZE_GGUF) + threadIdx.x / WARP_SIZE_GGUF;
    const int row  = blockIdx.y;
    if (kb >= nblk || row >= nrows_x) {
        return;
    }

    // lane owns elements kb*128 + 4*lane .. +3, all inside sub-block lane/8
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = kb * QK_PTQ1_0 + 4 * lane + j;
        v[j] = col < ncols_x ? static_cast<float>(x[(int64_t) row * ncols_x + col]) : 0.0f;
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 8));
    }
    // identical rounding to quantize_q8_1 so that every batch size quantizes
    // an activation the same way
    const float d = amax / 127.0f;
    int q[4];
    int isum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = amax == 0.0f ? 0 : (int) roundf(v[j] / d);
        isum += q[j];
    }
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        isum += VLLM_SHFL_XOR_SYNC_WIDTH(isum, mask, 8);
    }

    int8_t * col_base = y + (int64_t) row * PTQ1_0_PT_PLANES * nblk * 16;
    const int plane = lane / 4;              // element e = 4*lane + j -> plane e / 16
    const int byte  = (lane % 4) * 4;        // e % 16
    const int packed = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
    *(int *) (col_base + ((int64_t) plane * nblk + kb) * 16 + byte) = packed;
    if (lane % 8 == 0) {
        const int sub = lane / 8;
        half * ds = (half *) (col_base + ((int64_t) 8 * nblk + kb) * 16 + sub * 4);
        ds[0] = __float2half(d);
        ((int16_t *) ds)[1] = (int16_t) isum;
    }
}

template <typename scalar_t>
static void quantize_q8_1_pt_cuda(const scalar_t * x, void * y, const int ncols_x, const int nrows_x,
                                  const int nblk, cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    const dim3 grid((nblk + warps_per_block - 1) / warps_per_block, nrows_x, 1);
    const dim3 block(WARP_SIZE_GGUF * warps_per_block, 1, 1);
    quantize_q8_1_pt_kernel<scalar_t><<<grid, block, 0, stream>>>(x, (int8_t *) y, ncols_x, nrows_x, nblk);
}

// Digit words of one block in activation word order: word w holds the raw
// base-3 digits {0,1,2} of elements 4w .. 4w+3.
static __device__ __forceinline__ void ptq1_0_decode_block(const block_ptq1_0 * __restrict__ bq, int (&dig)[32]) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = (uint32_t) get_int_b4(bq->qs, g);
        uint32_t v_lo = (packed & 0xFF) | ((packed & 0xFF00) << 8);
        uint32_t v_hi = ((packed >> 16) & 0xFF) | ((packed >> 24) << 16);
#pragma unroll
        for (int n = 0; n < 5; ++n) {
            dig[4 * n + g] = ptq1_0_digits4(v_lo, v_hi);  // element 16n + 4g
        }
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t packed = (uint32_t) get_int_b4(bq->qs + 16, g);
        uint32_t v_lo = (packed & 0xFF) | ((packed & 0xFF00) << 8);
        uint32_t v_hi = ((packed >> 16) & 0xFF) | ((packed >> 24) << 16);
#pragma unroll
        for (int n = 0; n < 5; ++n) {
            dig[20 + 2 * n + g] = ptq1_0_digits4(v_lo, v_hi);  // element 80 + 8n + 4g
        }
    }
    uint32_t v = (uint32_t) bq->qh[0] | ((uint32_t) bq->qh[1] << 16);
#pragma unroll
    for (int n = 0; n < 4; n += 2) {
        const uint32_t w0 = v * 3;
        v = w0 & 0x00FF00FF;
        const uint32_t w1 = v * 3;
        v = w1 & 0x00FF00FF;
        dig[30 + n / 2] = (int) (((w0 >> 8) & 0xFF) | (((w0 >> 24) & 0xFF) << 8) |
                                 (((w1 >> 8) & 0xFF) << 16) | ((w1 >> 24) << 24));  // element 120 + 2n
    }
}

static __device__ __forceinline__ int ptq1_0_int4_word(const int4 & v, const int k) {
    switch (k & 3) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

template <typename scalar_t, int NCOLS>
static __global__ void mul_mat_vec_ptq1_0_pt_kernel(const void * __restrict__ vx, const void * __restrict__ vy,
                                                    scalar_t * __restrict__ dst, const int ncols_x,
                                                    const int nrows, const int nvecs, const int nblk) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const int lane = threadIdx.x % WARP_SIZE_GGUF;
    const int row  = blockIdx.x * PTQ1_0_PT_ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE_GGUF;
    if (row >= nrows) {
        return;
    }
    const int blocks_per_row = ncols_x / QK_PTQ1_0;
    const block_ptq1_0 * x = (const block_ptq1_0 *) vx + (int64_t) row * blocks_per_row;
    const int64_t col_stride_int4 = (int64_t) PTQ1_0_PT_PLANES * nblk;  // in 16-byte units

    float acc[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        acc[j] = 0.0f;
    }

    for (int kb = lane; kb < blocks_per_row; kb += WARP_SIZE_GGUF) {
        int dig[32];
        ptq1_0_decode_block(x + kb, dig);
        const float d = __half2float(x[kb].d);

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
            acc[j] += d * sum;
        }
    }

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE_GGUF / 2; mask > 0; mask >>= 1) {
            acc[j] += VLLM_SHFL_XOR_SYNC_WIDTH(acc[j], mask, WARP_SIZE_GGUF);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            if (j < nvecs) {
                dst[(int64_t) j * nrows + row] = acc[j];
            }
        }
    }
#else
    (void) vx; (void) vy; (void) dst; (void) ncols_x; (void) nrows; (void) nvecs; (void) nblk;
#endif
}

template <typename scalar_t, int NCOLS>
static void launch_mul_mat_vec_ptq1_0_pt(const void * vx, const void * vy, scalar_t * dst, const int ncols_x,
                                         const int nrows, const int nvecs, const int nblk, cudaStream_t stream) {
    const dim3 grid((nrows + PTQ1_0_PT_ROWS_PER_BLOCK - 1) / PTQ1_0_PT_ROWS_PER_BLOCK, 1, 1);
    const dim3 block(WARP_SIZE_GGUF * PTQ1_0_PT_ROWS_PER_BLOCK, 1, 1);
    mul_mat_vec_ptq1_0_pt_kernel<scalar_t, NCOLS><<<grid, block, 0, stream>>>(vx, vy, dst, ncols_x, nrows, nvecs, nblk);
}

// Number of activation columns the quantized buffer must hold for nvecs rows of X.
static inline int ptq1_0_pt_ncols(const int nvecs) {
    if (nvecs <= 1) return 1;
    if (nvecs <= 2) return 2;
    if (nvecs <= 4) return 4;
    if (nvecs <= 8) return 8;
    return PTQ1_0_PT_MAX_COLS;
}

template <typename scalar_t>
static void mul_mat_vec_ptq1_0_pt_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols_x,
                                       const int nrows, const int nvecs, cudaStream_t stream) {
    const int nblk = ptq1_0_pt_nblk(ncols_x);
    switch (ptq1_0_pt_ncols(nvecs)) {
        case 1: launch_mul_mat_vec_ptq1_0_pt<scalar_t, 1>(vx, vy, dst, ncols_x, nrows, nvecs, nblk, stream); break;
        case 2: launch_mul_mat_vec_ptq1_0_pt<scalar_t, 2>(vx, vy, dst, ncols_x, nrows, nvecs, nblk, stream); break;
        case 4: launch_mul_mat_vec_ptq1_0_pt<scalar_t, 4>(vx, vy, dst, ncols_x, nrows, nvecs, nblk, stream); break;
        case 8: launch_mul_mat_vec_ptq1_0_pt<scalar_t, 8>(vx, vy, dst, ncols_x, nrows, nvecs, nblk, stream); break;
        default: launch_mul_mat_vec_ptq1_0_pt<scalar_t, 16>(vx, vy, dst, ncols_x, nrows, nvecs, nblk, stream); break;
    }
}
