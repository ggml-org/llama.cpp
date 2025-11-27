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

    // Each warp processes 2 adjacent blocks of 32 values (64 values total)
    const int64_t warp_start_offset = blockIdx.y * vals_per_warp;
    const int64_t i0_block0         = warp_start_offset + threadIdx.x;                   // First block: 0-31
    const int64_t i0_block1         = warp_start_offset + vals_per_scale + threadIdx.x;  // Second block: 32-63

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

    const int64_t ib0 =
        blockIdx.z * ((int64_t) gridDim.x * gridDim.y * vals_per_warp / block_fp4_mmq_size);  // first block of channel
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;     // block index in channel
    const int64_t pair_idx_in_block =
        (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;  // 0-1: which pair of blocks within block_fp4_mmq

    uint8_t e_packed[2];

    // Process first block (0-31)
    {
        const int64_t global_src_pos = i03 * s03 + i02 * s02 + i01 * s01 + i0_block0;
        const float   xi             = i0_block0 < ne00 ? x[global_src_pos] : 0.0f;

        float amax = fabsf(xi);

        // Reduce max across all 32 threads in the warp
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax / 6.0f)) + 127) : 0;

        float val   = ggml_cuda_e8m0_to_fp32(e);
        float inv_s = (amax == 0.0f) ? 0.0f : 1.0f / val;

        // Quantize: each thread processes 1 value
        uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        if (e == 0) {
            e = 127;
        }

        // Pack 4 values into char2: threads 0,1,2,3 -> first char2, etc.
        const int lane_id  = threadIdx.x % 4;
        const int group_id = threadIdx.x / 4;

        // Use shuffle to gather values from 4 consecutive threads
        uint8_t q0 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 0, WARP_SIZE);
        uint8_t q1 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 1, WARP_SIZE);
        uint8_t q2 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 2, WARP_SIZE);
        uint8_t q3 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 3, WARP_SIZE);

        char2 q;
        if (lane_id == 0) {
            q.x = (q1 << 4) | q0;
            q.y = (q3 << 4) | q2;

            // Write to output: first block in pair uses positions based on pair_idx_in_block
            // Each pair has 2 blocks of 32 = 64 values = 16 char2 elements
            char2 * yqs2                            = (char2 *) y[ib].qs;
            yqs2[pair_idx_in_block * 16 + group_id] = q;
        }

        if (threadIdx.x == 0) {
            e_packed[0] = e;
        }
    }

    // Process second block (32-63)
    {
        const int64_t global_src_pos = i03 * s03 + i02 * s02 + i01 * s01 + i0_block1;
        const float   xi             = i0_block1 < ne00 ? x[global_src_pos] : 0.0f;

        float amax = fabsf(xi);

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax / 6.0f)) + 127) : 0;

        float val   = ggml_cuda_e8m0_to_fp32(e);
        float inv_s = (amax == 0.0f) ? 0.0f : 1.0f / val;

        if (e == 0) {
            e = 127;
        }

        uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const int lane_id  = threadIdx.x % 4;
        const int group_id = threadIdx.x / 4;

        // Use shuffle to gather values from 4 consecutive threads
        uint8_t q0 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 0, WARP_SIZE);
        uint8_t q1 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 1, WARP_SIZE);
        uint8_t q2 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 2, WARP_SIZE);
        uint8_t q3 = __shfl_sync(0xFFFFFFFF, q_val, (group_id * 4) + 3, WARP_SIZE);

        char2 q;
        if (lane_id == 0) {
            q.x = (q1 << 4) | q0;
            q.y = (q3 << 4) | q2;

            // Write to output: second block in pair uses positions 8-15 within the pair
            char2 * yqs2                                = (char2 *) y[ib].qs;
            yqs2[pair_idx_in_block * 16 + 8 + group_id] = q;
        }

        if (threadIdx.x == 0) {
            e_packed[1] = e;
        }
    }

    // Write packed exponents: d4[0-1] each stores 2 scales (for 2 blocks of 32)
    // pair_idx_in_block tells us which d4 entry to use (0-1)
    if (threadIdx.x == 0) {
        y[ib].d4[pair_idx_in_block] = (e_packed[1] << 8) | e_packed[0];
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

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    constexpr int vals_per_warp = 2 * QK_MXFP4;  // 64
    const int64_t block_num_y   = (ne0 + vals_per_warp - 1) / vals_per_warp;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(32, 1, 1);  // Warp size
    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
