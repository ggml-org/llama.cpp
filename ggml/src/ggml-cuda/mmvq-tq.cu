/*
 * Fused mul_mat for TQ4_1S / TQ3_1S weight types.
 *
 * ne[1]≤8: dp4a multi-token kernel (weight reuse across tokens)
 * ne[1]>8: runtime TQ4_1S→q8_0 scratch + cuBLAS tensor core GEMM
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"
#include "convert.cuh"

#define MMVQ_TQ_NWARPS 4

// ============================================================================
// Pre-rotate activation to q8_1 format (for TQ4_1S dp4a path)
// ============================================================================

static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// ============================================================================
// TQ4_1S: dp4a path with fixed int8 centroid LUT + q8_1 activation
// ============================================================================

// Fixed int8 centroid table: centroid_i8[i] = round(TQ4_CENTROIDS_WEIGHT[i] * 127 / 2.733)
// Rescale factor to recover float centroids: 2.733 / 127
static constexpr float TQ4_CENTROID_I8_RESCALE = 2.733f / 127.0f;

// Register-based centroid lookup: maps 4 qs bytes (1 uint32) to 2 packed 4× centroid_i8 for dp4a.
// Processes a full uint32 at once, sharing nibble extraction across both byte pairs.
__device__ __forceinline__ void tq4_cents8_reg(uint32_t four_bytes, int &c0, int &c1) {
    // Centroid i8 values packed into 4 registers (little-endian byte order):
    // [-127,-96,-75,-58] [-44,-31,-18,-6] [6,18,31,44] [58,75,96,127]
    constexpr uint32_t CR03 = 0xC6B5A081u;
    constexpr uint32_t CR47 = 0xFAEEE1D4u;
    constexpr uint32_t CR8B = 0x2C1F1206u;
    constexpr uint32_t CRCF = 0x7F604B3Au;

    // Extract all 8 nibbles from 4 bytes at once (shared across both pairs)
    const uint32_t lo = four_bytes & 0x0F0F0F0Fu;
    const uint32_t hi = (four_bytes >> 4) & 0x0F0F0F0Fu;

    // Interleave: bytes 0-1 → sel0 [n0,n1,n2,n3], bytes 2-3 → sel1 [n4,n5,n6,n7]
    const uint32_t sel0 = __byte_perm(lo, hi, 0x5140u);
    const uint32_t sel1 = __byte_perm(lo, hi, 0x7362u);

    // Lookup centroids for sel0 (elements from qs bytes 0-1)
    {
        const uint32_t flo = __byte_perm(CR03, CR47, sel0);
        const uint32_t fhi = __byte_perm(CR8B, CRCF, sel0);
        const uint32_t msb = (sel0 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c0 = (int)__byte_perm(flo, fhi, psel);
    }

    // Lookup centroids for sel1 (elements from qs bytes 2-3)
    {
        const uint32_t flo = __byte_perm(CR03, CR47, sel1);
        const uint32_t fhi = __byte_perm(CR8B, CRCF, sel1);
        const uint32_t msb = (sel1 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c1 = (int)__byte_perm(flo, fhi, psel);
    }
}

// ============================================================================
// Pre-rotate activation to half (for TQ3_1S scalar path)
// ============================================================================

static __global__ void tq_prerotate_activation(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = __float2half(val);
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

// ============================================================================
// Multi-token TQ4_1S dp4a kernel (ncols_dst ≤ 8)
// Weight data loaded once per block, reused across all ncols_dst tokens.
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq4_1s_dp4a_multi(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        // Load weight once, reuse across all tokens
        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const block_q8_1 * a_blk = &vy_q8[j * stride_col_y + ib];
            const float d_act = __half2float((__half)a_blk->ds.x);
            const int * a_qs = (const int *)(a_blk->qs);

            const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                           ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
            const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                           ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

            sumf[j] += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
        }
    }

    // Apply centroid int8→float rescale + warp reduction
    #pragma unroll
    for (int j = 0; j < ncols_dst; j++)
        sumf[j] *= TQ4_CENTROID_I8_RESCALE;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// Multi-token TQ3_1S scalar kernel (ncols_dst ≤ 8)
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq3_1s_multi(
        const void  * __restrict__ vx,
        const half  * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    __shared__ float s_lut[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        s_lut[threadIdx.x] = TQ3_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        const float w = s_lut[idx] * d;

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = __half2float(vy_rot[j * stride_col_y + ib * QK_TQ3_0 + lane]);
            sumf[j] += act * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// TQ4_1S scalar/half kernel (AMD fallback — no dp4a)
// Same pattern as TQ3_1S: pre-rotated half activations, scalar centroid lookup.
// On RDNA4, sudot4 throughput differs from NVIDIA dp4a — this path is faster.
// ============================================================================

template <int ncols_dst>
static __global__ void mul_mat_tq4_1s_scalar_multi(
        const void  * __restrict__ vx,
        const half  * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    __shared__ float s_lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        s_lut[threadIdx.x] = TQ4_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        const float w = s_lut[idx] * d;

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = __half2float(vy_rot[j * stride_col_y + ib * QK_TQ4_1S + lane]);
            sumf[j] += act * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// ============================================================================
// Dispatch: ne[1]=1 (decode), ne[1]≤8 (multi-token dp4a / scalar)
// ne[1]>8 handled by ggml_cuda_mul_mat_tq4_1s_cublas (runtime dequant + cuBLAS)
// AMD: uses scalar half path for TQ4_1S (dp4a regresses on RDNA4)
// ============================================================================

template <int ncols_dst>
static void launch_tq4_1s_multi(
        const void * src0_d, const block_q8_1 * q8_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq4_1s_dp4a_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_tq4_1s_scalar_multi(
        const void * src0_d, const half * act_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq4_1s_scalar_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_tq3_1s_multi(
        const void * src0_d, const half * act_buf,
        float * dst_d, int ncols_x, int nrows_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_tq3_1s_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * src0,
                           const ggml_tensor * src1,
                           ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int ncols_x   = src0->ne[0];
    const int nrows_x   = src0->ne[1];
    const int ncols_dst = src1->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int n_total_elements = ncols_x * ncols_dst;
    const bool use_dp4a = !GGML_CUDA_CC_IS_AMD(cc) && src0->type == GGML_TYPE_TQ4_1S;

    if (use_dp4a) {
        // NVIDIA TQ4_1S: dp4a int8 path (optimized for Turing+ dp4a throughput)
        const int n_total_blocks = n_total_elements / 32;
        ggml_cuda_pool_alloc<block_q8_1> q8_1_buf(ctx.pool(id), n_total_blocks);

        // Phase 1: Pre-rotate all tokens → q8_1
        {
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_q8_1<<<grid, block, 0, stream>>>(src1_d, q8_1_buf.get(), n_total_elements);
        }

        // Phase 2: dispatch based on ncols_dst
        const int stride_col_y   = ncols_x / 32;  // q8_1 blocks per column
        const int stride_col_dst = nrows_x;

        switch (ncols_dst) {
            case 1: launch_tq4_1s_multi<1>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 2: launch_tq4_1s_multi<2>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 3: launch_tq4_1s_multi<3>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 4: launch_tq4_1s_multi<4>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 5: launch_tq4_1s_multi<5>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 6: launch_tq4_1s_multi<6>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 7: launch_tq4_1s_multi<7>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 8: launch_tq4_1s_multi<8>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
        }
    } else {
        // Scalar half path: TQ3_1S (all vendors) + TQ4_1S on AMD (dp4a regresses on RDNA4)
        ggml_cuda_pool_alloc<half> act_buf(ctx.pool(id), n_total_elements);

        {
            const int n_total_blocks = n_total_elements / 32;
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, act_buf.get(), n_total_elements);
        }

        const int stride_col_y   = ncols_x;  // half elements per column
        const int stride_col_dst = nrows_x;
        const bool is_tq4 = (src0->type == GGML_TYPE_TQ4_1S);

        // Macro to dispatch to the right kernel based on quant type
        #define LAUNCH_SCALAR(N, src0_ptr, act_ptr, dst_ptr) \
            if (is_tq4) { launch_tq4_1s_scalar_multi<N>(src0_ptr, act_ptr, dst_ptr, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); } \
            else        { launch_tq3_1s_multi<N>(src0_ptr, act_ptr, dst_ptr, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); }

        if (ncols_dst <= 8) {
            switch (ncols_dst) {
                case 1: LAUNCH_SCALAR(1, src0_d, act_buf.get(), dst_d); break;
                case 2: LAUNCH_SCALAR(2, src0_d, act_buf.get(), dst_d); break;
                case 3: LAUNCH_SCALAR(3, src0_d, act_buf.get(), dst_d); break;
                case 4: LAUNCH_SCALAR(4, src0_d, act_buf.get(), dst_d); break;
                case 5: LAUNCH_SCALAR(5, src0_d, act_buf.get(), dst_d); break;
                case 6: LAUNCH_SCALAR(6, src0_d, act_buf.get(), dst_d); break;
                case 7: LAUNCH_SCALAR(7, src0_d, act_buf.get(), dst_d); break;
                case 8: LAUNCH_SCALAR(8, src0_d, act_buf.get(), dst_d); break;
            }
        } else {
            // Large prefill: batch in groups of 8
            for (int j = 0; j < ncols_dst; j += 8) {
                const int batch = min(8, ncols_dst - j);
                const half * act_j = act_buf.get() + j * ncols_x;
                float * dst_j = dst_d + j * nrows_x;
                switch (batch) {
                    case 1: LAUNCH_SCALAR(1, src0_d, act_j, dst_j); break;
                    case 2: LAUNCH_SCALAR(2, src0_d, act_j, dst_j); break;
                    case 3: LAUNCH_SCALAR(3, src0_d, act_j, dst_j); break;
                    case 4: LAUNCH_SCALAR(4, src0_d, act_j, dst_j); break;
                    case 5: LAUNCH_SCALAR(5, src0_d, act_j, dst_j); break;
                    case 6: LAUNCH_SCALAR(6, src0_d, act_j, dst_j); break;
                    case 7: LAUNCH_SCALAR(7, src0_d, act_j, dst_j); break;
                    case 8: LAUNCH_SCALAR(8, src0_d, act_j, dst_j); break;
                }
            }
        }
        #undef LAUNCH_SCALAR
    }
}


// ============================================================================
// Load-time conversion: TQ4_1S → q8_0 (opt-in via GGML_TQ_CONVERT_Q8=1)
// ============================================================================

static __global__ void k_convert_tq4_1s_to_q8_0(
        const block_tq4_1s * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;
    const int lane = threadIdx.x;
    const block_tq4_1s * blk = &src[block_idx];

    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    val *= TQ_WEIGHT_SIGNS[lane];

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) dst[block_idx].d = __float2half(d);
}

void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_TQ4_1S == 0);
    const int n_blocks = n_elements / QK_TQ4_1S;
    const int wpb = 4;
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);
    k_convert_tq4_1s_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_tq4_1s *)src_tq4, (block_q8_0 *)dst_q8, n_blocks);
}

// ============================================================================
// Large prefill: runtime TQ4_1S → q8_0 scratch + q8_0→fp16 dequant + cuBLAS
// Gets tensor core throughput without permanent 1.7× VRAM cost.
// ============================================================================

void ggml_cuda_mul_mat_tq4_1s_cublas(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src0,
                                      const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];  // K (hidden dim)
    const int64_t ne01 = src0->ne[1];  // M (rows = output features)
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // N (tokens)
    GGML_ASSERT(ne00 == ne10);

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    const int64_t n_elements = ne00 * ne01;

    // Step 1: TQ4_1S → fp16 via warp-cooperative dequant (WHT in-warp)
    ggml_cuda_pool_alloc<half> src0_f16(ctx.pool(id), n_elements);
    {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_TQ4_1S);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *)src0->data, src0_f16.get(), n_elements, stream);
    }

    // Step 2: src1 f32 → fp16
    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(id), ne10 * ne11);
    {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *)src1->data, src1_f16.get(), ne10 * ne11, stream);
    }

    // Step 3: cuBLAS fp16 GEMM with fp32 compute (tensor cores)
    // dst[M×N] = src0[M×K]^T × src1[K×N]
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int64_t ldc = dst->ne[0];  // M

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    CUBLAS_CHECK(
        cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne00,
                &alpha, src0_f16.get(), CUDA_R_16F, ne00,
                        src1_f16.get(), CUDA_R_16F, ne10,
                &beta,  (float *)dst->data, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
