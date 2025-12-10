#include "quantize.cuh"
#include <cstdint>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// Helper to compute E8M0 scale from amax using fast math
__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (amax == 0.0f) {
        return 127;  // Special case: use scale of 1.0 for zero input
    }
    // log2(amax / 6.0) = log2(amax) - log2(6) ≈ log2(amax) - 2.585
    // Use __log2f for fast approximate log2
    const float log2_amax = __log2f(amax) - 2.5849625007211563f;  // log2(6)
    const int e_int = __float2int_rd(log2_amax) + 127;  // floor + bias
    return static_cast<uint8_t>(max(1, min(254, e_int)));
}

static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32

    // Multiple warps per block - each warp handles different data
    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;
    const int64_t i0_block0 = warp_start_offset + lane_id_32;
    const int64_t i0_block1 = warp_start_offset + vals_per_scale + lane_id_32;

    if (i0_block0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 4 * QK_MXFP4;  // 128 values
    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * nwarps * vals_per_warp / block_fp4_mmq_size);
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t pair_idx_in_block = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    // Precompute common values
    const int lane_id  = lane_id_32 % 4;
    const int group_id = lane_id_32 / 4;
    const int group_base = group_id * 4;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;
    const float xi0 = (i0_block0 < ne00) ? x[base_pos + i0_block0] : 0.0f;
    const float xi1 = (i0_block1 < ne00) ? x[base_pos + i0_block1] : 0.0f;

    // === Process first block (0-31) ===
    float amax0 = fabsf(xi0);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, mask, WARP_SIZE));
    }

    const uint8_t e0 = compute_e8m0_scale(amax0);
    const float inv_s0 = (amax0 == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e0));
    const uint8_t q_val0 = ggml_cuda_float_to_fp4_e2m1(xi0, inv_s0);

    // Gather 4 values from consecutive threads using shuffle
    const uint8_t q0_0 = __shfl_sync(0xFFFFFFFF, q_val0, group_base + 0, WARP_SIZE);
    const uint8_t q0_1 = __shfl_sync(0xFFFFFFFF, q_val0, group_base + 1, WARP_SIZE);
    const uint8_t q0_2 = __shfl_sync(0xFFFFFFFF, q_val0, group_base + 2, WARP_SIZE);
    const uint8_t q0_3 = __shfl_sync(0xFFFFFFFF, q_val0, group_base + 3, WARP_SIZE);

    if (lane_id == 0) {
        char2 q;
        q.x = (q0_1 << 4) | q0_0;
        q.y = (q0_3 << 4) | q0_2;
        yqs2[pair_idx_in_block * 16 + group_id] = q;
    }

    // === Process second block (32-63) ===
    float amax1 = fabsf(xi1);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, mask, WARP_SIZE));
    }

    const uint8_t e1 = compute_e8m0_scale(amax1);
    const float inv_s1 = (amax1 == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e1));
    const uint8_t q_val1 = ggml_cuda_float_to_fp4_e2m1(xi1, inv_s1);

    const uint8_t q1_0 = __shfl_sync(0xFFFFFFFF, q_val1, group_base + 0, WARP_SIZE);
    const uint8_t q1_1 = __shfl_sync(0xFFFFFFFF, q_val1, group_base + 1, WARP_SIZE);
    const uint8_t q1_2 = __shfl_sync(0xFFFFFFFF, q_val1, group_base + 2, WARP_SIZE);
    const uint8_t q1_3 = __shfl_sync(0xFFFFFFFF, q_val1, group_base + 3, WARP_SIZE);

    if (lane_id == 0) {
        char2 q;
        q.x = (q1_1 << 4) | q1_0;
        q.y = (q1_3 << 4) | q1_2;
        yqs2[pair_idx_in_block * 16 + 8 + group_id] = q;
    }

    // Write packed exponents
    if (lane_id_32 == 0) {
        y[ib].d4[pair_idx_in_block] = (e1 << 8) | e0;
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);  // Each warp processes 64 values

    constexpr int nwarps = 8;
    constexpr int vals_per_warp = 2 * QK_MXFP4;  // 64 values per warp
    constexpr int vals_per_block = nwarps * vals_per_warp;  // 512 values per block

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);  // 32 threads x 8 warps = 256 threads per block

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
