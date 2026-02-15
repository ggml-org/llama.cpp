// Fused RMS Norm + Q8_1 Quantization Kernel for GFX906

#include "norm-fused-q8.cuh"

#ifdef GGML_USE_HIP
#include "../gfx906-common.cuh"
#include "../quantize/epilogue.cuh"
#endif

template <int block_size, mmq_q8_1_ds_layout ds_layout, bool do_multiply = false>
__launch_bounds__(256, 8)
static __global__ void rms_norm_f32_to_q8_1(
    const float* __restrict__ x,
    void* __restrict__ vy,
    const int ncols,
    const int64_t stride_row,
    const int64_t stride_channel,
    const int64_t stride_sample,
    const float eps,
    const float* __restrict__ mul = nullptr,
    const int64_t mul_stride_row = 0,
    const int64_t mul_stride_channel = 0,
    const int64_t mul_stride_sample = 0,
    const uint3 mul_ncols_packed = make_uint3(0, 0, 0),
    const uint3 mul_nrows_packed = make_uint3(0, 0, 0),
    const uint3 mul_nchannels_packed = make_uint3(0, 0, 0),
    const uint3 mul_nsamples_packed = make_uint3(0, 0, 0)
) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    const float* __restrict__ x_row = x + sample * stride_sample + channel * stride_channel + row * stride_row;

    const float* __restrict__ mul_row = nullptr;
    if constexpr (do_multiply) {
        const uint32_t mul_row_idx = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel_idx = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample_idx = fastmodulo(sample, mul_nsamples_packed);
        mul_row = mul + mul_sample_idx * mul_stride_sample + mul_channel_idx * mul_stride_channel + mul_row_idx * mul_stride_row;
    }

    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const int64_t blocks_per_row = ncols_padded / (4 * QK8_1);
    const int64_t channel_idx = (int64_t)sample * nchannels + channel;
    const int64_t ib0 = channel_idx * (nrows * blocks_per_row);
    block_q8_1_mmq* __restrict__ y_base = (block_q8_1_mmq*)vy + ib0;

    extern __shared__ float4 s_data4[];
    float* s_data = (float*)s_data4;
    __shared__ float s_reduce[32];

    const float4* __restrict__ x4 = (const float4*)x_row;
    const int n_float4 = ncols / 4;

    float sum_sq = 0.0f;
    for (int idx = tid; idx < n_float4; idx += block_size) {
        const float4 v = x4[idx];
        s_data4[idx] = v;
        sum_sq = __fmaf_rn(v.x, v.x, sum_sq);
        sum_sq = __fmaf_rn(v.y, v.y, sum_sq);
        sum_sq = __fmaf_rn(v.z, v.z, sum_sq);
        sum_sq = __fmaf_rn(v.w, v.w, sum_sq);
    }

    const int remainder = ncols % 4;
    if (tid < remainder) {
        const int idx = n_float4 * 4 + tid;
        const float v = x_row[idx];
        s_data[idx] = v;
        sum_sq = __fmaf_rn(v, v, sum_sq);
    }
    __syncthreads();

    sum_sq = warp_reduce_sum(sum_sq);

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int nwarps = block_size / WARP_SIZE;

    if (lane_id == 0) s_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < nwarps) ? s_reduce[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }

    float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(sum_sq / ncols + eps);
        s_reduce[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = s_reduce[0];

    constexpr int vals_per_iter = block_size * 4;
    const float4* __restrict__ mul4 = do_multiply ? (const float4*)mul_row : nullptr;

    for (int base_col = 0; base_col < ncols_padded; base_col += vals_per_iter) {
        const int col = base_col + tid * 4;
        if (col >= ncols_padded) continue;

        float4 v;
        if (col + 3 < ncols) {
            v = s_data4[col / 4];
        } else {
            v.x = (col + 0 < ncols) ? s_data[col + 0] : 0.0f;
            v.y = (col + 1 < ncols) ? s_data[col + 1] : 0.0f;
            v.z = (col + 2 < ncols) ? s_data[col + 2] : 0.0f;
            v.w = (col + 3 < ncols) ? s_data[col + 3] : 0.0f;
        }

        v.x *= rms_scale; v.y *= rms_scale; v.z *= rms_scale; v.w *= rms_scale;

        if constexpr (do_multiply) {
            if (col + 3 < ncols) {
                const float4 m = mul4[col / 4];
                v.x *= m.x; v.y *= m.y; v.z *= m.z; v.w *= m.w;
            } else {
                if (col + 0 < ncols) v.x *= mul_row[fastmodulo(col + 0, mul_ncols_packed)];
                if (col + 1 < ncols) v.y *= mul_row[fastmodulo(col + 1, mul_ncols_packed)];
                if (col + 2 < ncols) v.z *= mul_row[fastmodulo(col + 2, mul_ncols_packed)];
                if (col + 3 < ncols) v.w *= mul_row[fastmodulo(col + 3, mul_ncols_packed)];
            }
        }

        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        float sum = v.x + v.y + v.z + v.w;

        // 8-thread group reduction using explicit shuffle
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 8));
            if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 8);
            }
        }

        constexpr float inv_127 = 1.0f / 127.0f;
        const float d = amax * inv_127;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
        const float d_inv = fast_rcp_f32(d);
#else
        const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
#endif

        const char4 q = make_char4(
            static_cast<int8_t>(__float2int_rn(v.x * d_inv)),
            static_cast<int8_t>(__float2int_rn(v.y * d_inv)),
            static_cast<int8_t>(__float2int_rn(v.z * d_inv)),
            static_cast<int8_t>(__float2int_rn(v.w * d_inv))
        );

        const int block_idx = col / 128;
        const int pos_in_block = col % 128;
        const int group_in_block = pos_in_block / 32;
        const int lane_in_group = (pos_in_block % 32) / 4;

        if (block_idx < blocks_per_row) {
            block_q8_1_mmq* __restrict__ block_out = &y_base[block_idx * nrows + row];
            reinterpret_cast<char4*>(block_out->qs)[group_in_block * 8 + lane_in_group] = q;

            if (lane_in_group == 0) {
                if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                    block_out->ds4[group_in_block] = make_half2(__float2half(d), __float2half(sum));
                } else if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    block_out->d4[group_in_block] = d;
                }
            }
        }
    }
}

void ggml_cuda_op_rms_norm_fused_q8_1(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* rms_norm,
    void* q8_output,
    mmq_q8_1_ds_layout ds_layout,
    const float* mul_weights,
    const ggml_tensor* mul_src
) {
    const ggml_tensor* src = rms_norm->src[0];
    const float* src_d = (const float*)src->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const int64_t ne00 = src->ne[0];
    const int64_t ne01 = src->ne[1];
    const int64_t ne02 = src->ne[2];
    const int64_t ne03 = src->ne[3];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src->type);
    GGML_ASSERT(src->nb[0] == ts0);
    const int64_t s01 = src->nb[1] / ts0;
    const int64_t s02 = src->nb[2] / ts0;
    const int64_t s03 = src->nb[3] / ts0;

    const dim3 blocks_num(ne01, ne02, ne03);
    const size_t smem_size = ne00 * sizeof(float);
    const dim3 block_dims(256, 1, 1);

    if (mul_weights != nullptr && mul_src != nullptr) {
        const size_t ts_mul = ggml_type_size(mul_src->type);
        GGML_ASSERT(mul_src->nb[0] == ts_mul);
        const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
        const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
        const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

        const uint3 mul_ncols_packed = init_fastdiv_values(mul_src->ne[0]);
        const uint3 mul_nrows_packed = init_fastdiv_values(mul_src->ne[1]);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_src->ne[2]);
        const uint3 mul_nsamples_packed = init_fastdiv_values(mul_src->ne[3]);

        switch (ds_layout) {
            case MMQ_Q8_1_DS_LAYOUT_D4:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_D4, true><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps, mul_weights, mul_s01, mul_s02, mul_s03,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
                break;
            case MMQ_Q8_1_DS_LAYOUT_DS4:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_DS4, true><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps, mul_weights, mul_s01, mul_s02, mul_s03,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
                break;
            case MMQ_Q8_1_DS_LAYOUT_D2S6:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps, mul_weights, mul_s01, mul_s02, mul_s03,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
                break;
        }
    } else {
        switch (ds_layout) {
            case MMQ_Q8_1_DS_LAYOUT_D4:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_D4, false><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps);
                break;
            case MMQ_Q8_1_DS_LAYOUT_DS4:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_DS4, false><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps);
                break;
            case MMQ_Q8_1_DS_LAYOUT_D2S6:
                rms_norm_f32_to_q8_1<256, MMQ_Q8_1_DS_LAYOUT_D2S6, false><<<blocks_num, block_dims, smem_size, stream>>>(
                    src_d, q8_output, ne00, s01, s02, s03, eps);
                break;
        }
    }
}

size_t ggml_cuda_get_q8_1_buffer_size(int64_t ncols, int64_t nrows, int cc) {
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const int64_t blocks_per_row = ncols_padded / (4 * QK8_1);
    return nrows * blocks_per_row * sizeof(block_q8_1_mmq) + get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
}
