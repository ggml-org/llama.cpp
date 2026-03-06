#pragma once

// GFX906 Warp-Cooperative Q4_1 GEMV Kernel
// Uses half-warp (32 threads) per row for better memory coalescing
// Achieves better bandwidth improvement over sequential per-thread approach, which results in faster performance for small matrixes (ncols less than 1024)
// This kernel is only included from mmvq.cu where all dependencies are available

#if defined(GGML_USE_HIP)

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q4_0_warp_coop(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x) {

    constexpr int qk_q4_0 = 32;

    const int lane_id = threadIdx.x;
    const int half_lane = lane_id % 32;
    const int row_offset = lane_id / 32;

    const int row = blockIdx.x * 2 + row_offset;

    if (row >= (int)nrows_x) return;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int blocks_per_row = ncols_x / qk_q4_0;
    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;

    const block_q4_0 * x = (const block_q4_0 *)vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q4_0 * bq4 = x + ib;
        const block_q8_1 * bq8 = y + ib;

        // Load 16 bytes of Q4_0 quantized values (32 nibbles)
        int v0, v1, v2, v3;
        memcpy(&v0, bq4->qs +  0, 4);
        memcpy(&v1, bq4->qs +  4, 4);
        memcpy(&v2, bq4->qs +  8, 4);
        memcpy(&v3, bq4->qs + 12, 4);

        // Load 32 bytes of Q8_1 quantized values
        const int * q8 = (const int *)bq8->qs;
        const int u0 = q8[0];
        const int u1 = q8[1];
        const int u2 = q8[2];
        const int u3 = q8[3];
        const int u4 = q8[4];
        const int u5 = q8[5];
        const int u6 = q8[6];
        const int u7 = q8[7];

        // Compute dot product (8 dp4a for full 32 values)
        int sumi = 0;
        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);
        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);
        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);
        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);

        // Q4_0 formula: d4 * (sumi * d8 - 8 * s8)
        // where s8 is the sum of Q8 values (stored in bq8->ds.y)
        const float d4 = bq4->d;
        const float2 ds8 = __half22float2(bq8->ds);
        sumf += d4 * (sumi * ds8.x - 8.0f * ds8.y);
    }

    // Half-warp reduction using fused DPP instructions
    sumf = warp_reduce_sum<32>(sumf);

    if (half_lane == 0) {
        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
    }
}

static void gfx906_launch_mul_mat_vec_q4_0_warp_coop(
        const void * vx, const void * vy, const int32_t * ids,
        float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y,
        const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t nrows_x,
        const uint32_t nchannels_dst, const uint32_t nsamples_dst,
        cudaStream_t stream) {

    const dim3 block_dims(64, 1, 1);
    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);

    gfx906_mul_mat_vec_q4_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst, nrows_x);
}

#endif // GGML_USE_HIP
