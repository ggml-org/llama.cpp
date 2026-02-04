#pragma once

// GFX906 Warp-Cooperative Q8_0 GEMV Kernel
// Simpler than Q4 variants - no nibble extraction needed

#if defined(GGML_USE_HIP)

__launch_bounds__(64, 1)
static __global__ void gfx906_mul_mat_vec_q8_0_warp_coop(
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

    constexpr int qk_q8_0 = 32;

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

    const int blocks_per_row = ncols_x / qk_q8_0;
    const int kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;

    const block_q8_0 * x = (const block_q8_0 *)vx + kbx_offset;
    const block_q8_1 * y = (const block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;

    float sumf = 0.0f;

    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        const block_q8_0 * bq8_0 = x + ib;
        const block_q8_1 * bq8_1 = y + ib;

        // Load 32 bytes of Q8_0 quantized values (32 int8 values)
        const int * v = (const int *)bq8_0->qs;
        const int v0 = v[0];
        const int v1 = v[1];
        const int v2 = v[2];
        const int v3 = v[3];
        const int v4 = v[4];
        const int v5 = v[5];
        const int v6 = v[6];
        const int v7 = v[7];

        // Load 32 bytes of Q8_1 quantized values
        const int * u = (const int *)bq8_1->qs;
        const int u0 = u[0];
        const int u1 = u[1];
        const int u2 = u[2];
        const int u3 = u[3];
        const int u4 = u[4];
        const int u5 = u[5];
        const int u6 = u[6];
        const int u7 = u[7];

        // Compute dot product (8 dp4a for full 32 values)
        int sumi = 0;
        sumi = ggml_cuda_dp4a(v0, u0, sumi);
        sumi = ggml_cuda_dp4a(v1, u1, sumi);
        sumi = ggml_cuda_dp4a(v2, u2, sumi);
        sumi = ggml_cuda_dp4a(v3, u3, sumi);
        sumi = ggml_cuda_dp4a(v4, u4, sumi);
        sumi = ggml_cuda_dp4a(v5, u5, sumi);
        sumi = ggml_cuda_dp4a(v6, u6, sumi);
        sumi = ggml_cuda_dp4a(v7, u7, sumi);

        // Q8_0 formula: d0 * d1 * sumi (simple!)
        const float d0 = bq8_0->d;
        const float d1 = __low2float(bq8_1->ds);
        sumf += d0 * d1 * (float)sumi;
    }

    // Half-warp reduction using fused DPP instructions
    sumf = warp_reduce_sum<32>(sumf);

    if (half_lane == 0) {
        dst[sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row] = sumf;
    }
}

static void gfx906_launch_mul_mat_vec_q8_0_warp_coop(
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

    gfx906_mul_mat_vec_q8_0_warp_coop<<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,
        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst, nrows_x);
}

#endif // GGML_USE_HIP
