#pragma once

// GFX906 Q8_1 quantization epilogue functions using DPP reductions.
// Allows producer kernels to quantize FP32 values directly to Q8_1 format.

#include "../gfx906-config.h"
#include "../gfx906-common.cuh"
#include "../../mmq.cuh"

#ifdef GGML_USE_HIP

// Quantizes 32 FP32 values across 8 threads (4 values per thread) to Q8_1.
// Thread organization: 8 threads per group, each owns 4 consecutive values.
// Lane 0 writes the scale/sum metadata.
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantize_q8_1_epilogue_32vals(
    float vals[4],
    int8_t* __restrict__ q_out,
    void* __restrict__ ds_out,
    const int lane_in_group
) {
    float amax = fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                       fmaxf(fabsf(vals[2]), fabsf(vals[3])));
    float sum = vals[0] + vals[1] + vals[2] + vals[3];

    // 8-thread DPP reduction using inline assembly for maximum performance
    // The fused DPP operations (v_max_f32_dpp/v_add_f32_dpp) are ~4x faster than separate shuffle+op
    int amax_i = __float_as_int(amax);
    int sum_i = __float_as_int(sum);

    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        int amax_tmp;
        asm volatile(
            "v_mov_b32 %0, %2\n"
            "s_nop 1\n"
            "v_mov_b32_dpp %0, %2 row_shl:4 row_mask:0xf bank_mask:0x5\n"
            "v_mov_b32_dpp %0, %2 row_shr:4 row_mask:0xf bank_mask:0xa\n"
            "v_max_f32 %1, %2, %0\n"
            "s_nop 1\n"
            "v_max_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
            "s_nop 1\n"
            "v_max_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
            : "=&v"(amax_tmp), "=v"(amax_i)
            : "v"(amax_i)
            : "memory"
        );
        amax = __int_as_float(amax_i);
    } else {
        int amax_tmp, sum_tmp;
        asm volatile(
            "v_mov_b32 %0, %4\n"
            "v_mov_b32 %1, %5\n"
            "s_nop 1\n"
            "v_mov_b32_dpp %0, %4 row_shl:4 row_mask:0xf bank_mask:0x5\n"
            "v_mov_b32_dpp %1, %5 row_shl:4 row_mask:0xf bank_mask:0x5\n"
            "v_mov_b32_dpp %0, %4 row_shr:4 row_mask:0xf bank_mask:0xa\n"
            "v_mov_b32_dpp %1, %5 row_shr:4 row_mask:0xf bank_mask:0xa\n"
            "v_max_f32 %2, %4, %0\n"
            "v_add_f32 %3, %5, %1\n"
            "s_nop 1\n"
            "v_max_f32_dpp %2, %2, %2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp %3, %3, %3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
            "s_nop 1\n"
            "v_max_f32_dpp %2, %2, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp %3, %3, %3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
            : "=&v"(amax_tmp), "=&v"(sum_tmp), "=v"(amax_i), "=v"(sum_i)
            : "v"(amax_i), "v"(sum_i)
            : "memory"
        );
        amax = __int_as_float(amax_i);
        sum = __int_as_float(sum_i);
    }

    constexpr float inv_127 = 1.0f / 127.0f;
    const float d = amax * inv_127;
    const float d_inv = fast_rcp_f32(d);

    const int q0 = __float2int_rn(vals[0] * d_inv);
    const int q1 = __float2int_rn(vals[1] * d_inv);
    const int q2 = __float2int_rn(vals[2] * d_inv);
    const int q3 = __float2int_rn(vals[3] * d_inv);

    const int offset = lane_in_group * 4;
    q_out[offset + 0] = static_cast<int8_t>(q0);
    q_out[offset + 1] = static_cast<int8_t>(q1);
    q_out[offset + 2] = static_cast<int8_t>(q2);
    q_out[offset + 3] = static_cast<int8_t>(q3);

    if (lane_in_group == 0) {
        if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            *reinterpret_cast<half2*>(ds_out) = make_half2(__float2half(d), __float2half(sum));
        } else if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
            *reinterpret_cast<float*>(ds_out) = d;
        }
    }
}

// Helper to compute Q8_1 buffer size for a given shape
static __host__ __forceinline__ size_t get_q8_1_mmq_buffer_size(
    int64_t ncols, int64_t nrows, int cc
) {
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const size_t row_size = (ncols_padded / QK8_1) * sizeof(block_q8_1) +
                            get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    return nrows * row_size;
}

// Variant for 128-value blocks (block_q8_1_mmq format).
// 32 threads process 128 values, subdivided into 4 groups of 8.
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantize_q8_1_epilogue_128vals(
    float vals[4],
    block_q8_1_mmq* __restrict__ block_out,
    const int tid_in_block
) {
    const int group_idx = tid_in_block / 8;
    const int lane_in_group = tid_in_block % 8;

    int8_t* q_out = block_out->qs + group_idx * 32;

    void* ds_out = nullptr;
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        ds_out = &block_out->ds4[group_idx];
    } else if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        ds_out = &block_out->d4[group_idx];
    }

    quantize_q8_1_epilogue_32vals<ds_layout>(vals, q_out, ds_out, lane_in_group);
}

static __device__ __forceinline__ float broadcast_scale_from_lane0(float scale, int lane_in_group) {
    return __shfl(scale, 0, 8);
}

#endif // GGML_USE_HIP
