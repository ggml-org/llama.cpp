#pragma once

// GFX906 Warp-Cooperative Q4_1 GEMV Kernel
// Uses half-warp (32 threads) per row for better memory coalescing
// Achieves better bandwidth improvement over sequential per-thread approach, which results in faster performance for small matrixes (ncols less than 1024)
// This kernel is only included from mmvq.cu where all dependencies are available

#if defined(GGML_USE_HIP)

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q4_1_warp_coop(
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

    constexpr int qk_q4_1 = 32;

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

    const int blocks_per_row = ncols_x / qk_q4_1;
    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;

    const block_q4_1 * x = (const block_q4_1 *)vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q4_1 * bq4 = x + ib;
        const block_q8_1 * bq8 = y + ib;  

        // Load ALL 16 bytes of Q4_1 quantized values (32 nibbles = 32 values)
        int v0, v1, v2, v3;
        memcpy(&v0, bq4->qs +  0, 4);  // bytes 0-3:   values 0-3 (low), 16-19 (high)
        memcpy(&v1, bq4->qs +  4, 4);  // bytes 4-7:   values 4-7 (low), 20-23 (high)
        memcpy(&v2, bq4->qs +  8, 4);  // bytes 8-11:  values 8-11 (low), 24-27 (high)
        memcpy(&v3, bq4->qs + 12, 4);  // bytes 12-15: values 12-15 (low), 28-31 (high)

        // Load ALL 32 bytes of Q8_1 quantized values (32 int8 values)
        const int * q8 = (const int *)bq8->qs;
        const int u0 = q8[0];  // Q8 values 0-3
        const int u1 = q8[1];  // Q8 values 4-7
        const int u2 = q8[2];  // Q8 values 8-11
        const int u3 = q8[3];  // Q8 values 12-15
        const int u4 = q8[4];  // Q8 values 16-19
        const int u5 = q8[5];  // Q8 values 20-23
        const int u6 = q8[6];  // Q8 values 24-27
        const int u7 = q8[7];  // Q8 values 28-31

        // Compute dot product with nibble extraction (8 dp4a for full 32 values)
        int sumi = 0;
        sumi = ggml_cuda_dp4a((v0 >> 0) & 0x0F0F0F0F, u0, sumi);  // Q4 0-3 * Q8 0-3
        sumi = ggml_cuda_dp4a((v0 >> 4) & 0x0F0F0F0F, u4, sumi);  // Q4 16-19 * Q8 16-19
        sumi = ggml_cuda_dp4a((v1 >> 0) & 0x0F0F0F0F, u1, sumi);  // Q4 4-7 * Q8 4-7
        sumi = ggml_cuda_dp4a((v1 >> 4) & 0x0F0F0F0F, u5, sumi);  // Q4 20-23 * Q8 20-23
        sumi = ggml_cuda_dp4a((v2 >> 0) & 0x0F0F0F0F, u2, sumi);  // Q4 8-11 * Q8 8-11
        sumi = ggml_cuda_dp4a((v2 >> 4) & 0x0F0F0F0F, u6, sumi);  // Q4 24-27 * Q8 24-27
        sumi = ggml_cuda_dp4a((v3 >> 0) & 0x0F0F0F0F, u3, sumi);  // Q4 12-15 * Q8 12-15
        sumi = ggml_cuda_dp4a((v3 >> 4) & 0x0F0F0F0F, u7, sumi);  // Q4 28-31 * Q8 28-31

        // Load and apply scale/bias: Q4_1 has (d, m), Q8_1 has (d, s)
        // Full block formula: result = sumi * d4 * d8 + m4 * s8
        const float2 dm4 = __half22float2(bq4->dm);
        const float2 ds8 = __half22float2(bq8->ds);
        sumf += sumi * dm4.x * ds8.x + dm4.y * ds8.y;
    }

    // Half-warp reduction using fused DPP instructions
    sumf = warp_reduce_sum<32>(sumf);

    // First thread of each half-warp writes result
    if (half_lane == 0) {
        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
    }
}

// Host-side dispatch function for GFX906 Q4_1 warp-cooperative kernel
static void gfx906_launch_mul_mat_vec_q4_1_warp_coop(
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

    // 2 rows per block, 64 threads per block (1 warp processing 2 rows)
    const dim3 block_dims(64, 1, 1);
    const dim3 block_nums((nrows_x + 1) / 2, nchannels_dst, nsamples_dst);

    gfx906_mul_mat_vec_q4_1_warp_coop<<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst, nrows_x);
}

#endif // GGML_USE_HIP
 