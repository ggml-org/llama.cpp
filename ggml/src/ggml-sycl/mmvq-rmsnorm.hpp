//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Fused MMVQ with SLM-cached RMSNorm quantization
// Option 2: Quantize entire row to SLM, then use MMVQ loop
//
// This approach:
// 1. Phase 1: Parallel quantization with RMSNorm to SLM (block_q8_1 format)
// 2. Barrier
// 3. Phase 2: MMVQ loop reading Q8 from SLM instead of device memory
//
// Benefits:
// - SLM is nearly as fast as registers on Intel Arc
// - Embarrassingly parallel quantization phase
// - Only one barrier vs per-block reductions in inline approach
//

#ifndef GGML_SYCL_MMVQ_RMSNORM_HPP
#define GGML_SYCL_MMVQ_RMSNORM_HPP

#include "common.hpp"
#include "vecdotq.hpp"
#include "fused-norm-gemm.hpp"  // For compute_rms_scales_sycl
#include <sycl/sycl.hpp>

// Maximum SLM size for Q8 buffer (in bytes)
// For 4096 hidden dim: 4096 + 128*4 = 4608 bytes
// Max SLM on Intel Arc: 128KB - we're well within this
constexpr int MMVQ_RMSNORM_MAX_SLM_BYTES = 32 * 1024;  // 32KB should handle most models

// =============================================================================
// Phase 1: Parallel quantization with RMSNorm to SLM
// ALL work-items participate - use SLM for inter-block communication
// Two-pass approach: (1) compute all values, (2) quantize
// =============================================================================

// Work-group quantization: each thread handles multiple elements
// Use SLM scratch space for intermediate normalized values
static __dpct_inline__ void quantize_row_to_slm_workgroup(
    const float * __restrict__ f32_input,  // [ncols] F32 input for this row
    const float * __restrict__ gamma,       // [ncols] RMSNorm weights
    const float rms_scale,                  // Pre-computed RMS scale for this row
    int8_t * __restrict__ slm_qs,           // SLM buffer for quantized values [ncols]
    sycl::half2 * __restrict__ slm_ds,      // SLM buffer for (scale, sum) [num_blocks]
    float * __restrict__ slm_scratch,       // SLM scratch for normalized values [ncols]
    const int ncols,
    const sycl::nd_item<3> & item
) {
    const int tid = item.get_local_linear_id();
    const int wg_size = item.get_local_range(1) * item.get_local_range(2);  // 4 * 32 = 128
    const int num_blocks = ncols / QK8_1;

    // Pass 1: Compute normalized values and write to SLM scratch
    // Each thread handles ncols/wg_size elements (4096/128 = 32 elements per thread)
    for (int col = tid; col < ncols; col += wg_size) {
        float val = f32_input[col] * rms_scale * gamma[col];
        slm_scratch[col] = val;
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Pass 2: Each sub-group computes max/sum for one block
    // With 4 sub-groups and 128 blocks, each sub-group handles 32 blocks
    const int sg_id = item.get_local_id(1);  // 0-3
    const int lane = item.get_local_id(2);   // 0-31
    const int num_sg = item.get_local_range(1);  // 4

    for (int block_idx = sg_id; block_idx < num_blocks; block_idx += num_sg) {
        const int col_base = block_idx * QK8_1;

        // Each lane reads one element from scratch SLM
        float val = slm_scratch[col_base + lane];

        // Manual warp reduction for max and sum using shuffle
        float amax = sycl::fabs(val);
        float sum = val;

        // Warp reduction using sub-group shuffle
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_max = sycl::shift_group_left(item.get_sub_group(), amax, offset);
            float other_sum = sycl::shift_group_left(item.get_sub_group(), sum, offset);
            amax = sycl::fmax(amax, other_max);
            sum = sum + other_sum;
        }

        // Lane 0 broadcasts scale and writes to SLM
        float d = sycl::select_from_group(item.get_sub_group(), amax, 0);
        d = (d == 0.0f) ? 1.0f : d / 127.0f;

        // Quantize and store
        int8_t q = sycl::clamp((int)sycl::round(val / d), -128, 127);
        slm_qs[col_base + lane] = q;

        // Store block metadata (only lane 0)
        if (lane == 0) {
            float final_d = (amax == 0.0f) ? 0.0f : d;
            slm_ds[block_idx] = sycl::half2(sycl::half(final_d), sycl::half(sum));
        }
    }

    item.barrier(sycl::access::fence_space::local_space);
}

// =============================================================================
// Simplified MMVQ kernel with SLM Q8
// Each row is processed by one sub-group, simple sequential accumulation
// =============================================================================

static void mul_mat_vec_q4_0_f32_rmsnorm_slm(
    const void * __restrict__ vx,          // Q4_0 weights [N, K]
    const float * __restrict__ f32_input,  // F32 input [K] (single batch row)
    const float * __restrict__ gamma,      // RMSNorm weights [K]
    const float rms_scale,                 // Pre-computed RMS scale
    float * __restrict__ dst,              // Output [N]
    const int ncols,                       // K (input features)
    const int nrows,                       // N (output features)
    int8_t * __restrict__ slm_qs,          // SLM for Q8 values
    sycl::half2 * __restrict__ slm_ds,     // SLM for (d, sum)
    float * __restrict__ slm_scratch,      // SLM scratch for normalized values
    const sycl::nd_item<3> & item
) {
    // DEBUG: Just set output to 0 to test if kernel returns
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();

    if (row >= nrows) {
        return;
    }

    // Just set output to zero - skip all computation
    if (lane == 0) {
        dst[row] = 0.0f;
    }
    return;  // DEBUG: Skip actual kernel

    // =========================================================================
    // Phase 1: Quantize input row to SLM (ALL work-items cooperate)
    // =========================================================================
    quantize_row_to_slm_workgroup(f32_input, gamma, rms_scale, slm_qs, slm_ds, slm_scratch, ncols, item);

    // (Barrier is inside quantize_row_to_slm_workgroup)

    if (row >= nrows) {
        return;
    }

    // =========================================================================
    // Phase 2: MMVQ - each sub-group processes one row
    // =========================================================================
    const int blocks_per_row = ncols / QK4_0;
    const block_q4_0 * x = (const block_q4_0 *)vx;

    float tmp = 0.0f;

    // Each thread processes multiple Q4 blocks
    for (int block_idx = lane; block_idx < blocks_per_row; block_idx += WARP_SIZE) {
        const int ibx = row * blocks_per_row + block_idx;
        const block_q4_0 * bq4 = &x[ibx];
        const float d4 = bq4->d;

        // Get Q8 scale for this block
        const sycl::half2 ds8 = slm_ds[block_idx];
        const float d8 = float(ds8.x());
        const float scale = d4 * d8;

        // Process all 32 elements of this block
        int sumi = 0;
        const int q8_base = block_idx * QK8_1;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            // Q4_0: byte j contains element j (low nibble) and element j+16 (high nibble)
            const uint8_t q4_byte = bq4->qs[j];
            const int q4_lo = (q4_byte & 0x0F) - 8;
            const int q4_hi = ((q4_byte >> 4) & 0x0F) - 8;

            // Get Q8 values from SLM
            const int8_t q8_lo = slm_qs[q8_base + j];
            const int8_t q8_hi = slm_qs[q8_base + j + 16];

            sumi += q4_lo * q8_lo + q4_hi * q8_hi;
        }

        tmp += scale * sumi;
    }

    // Warp reduction
    tmp = sycl::reduce_over_group(sg, tmp, sycl::plus<float>());

    if (lane == 0) {
        dst[row] = tmp;
    }
}

static void mul_mat_vec_q8_0_f32_rmsnorm_slm(
    const void * __restrict__ vx,          // Q8_0 weights [N, K]
    const float * __restrict__ f32_input,  // F32 input [K] (single batch row)
    const float * __restrict__ gamma,      // RMSNorm weights [K]
    const float rms_scale,                 // Pre-computed RMS scale
    float * __restrict__ dst,              // Output [N]
    const int ncols,                       // K (input features)
    const int nrows,                       // N (output features)
    int8_t * __restrict__ slm_qs,          // SLM for Q8 values
    sycl::half2 * __restrict__ slm_ds,     // SLM for (d, sum)
    float * __restrict__ slm_scratch,      // SLM scratch for normalized values
    const sycl::nd_item<3> & item
) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();

    // =========================================================================
    // Phase 1: Quantize input row to SLM (ALL work-items cooperate)
    // =========================================================================
    quantize_row_to_slm_workgroup(f32_input, gamma, rms_scale, slm_qs, slm_ds, slm_scratch, ncols, item);

    // (Barrier is inside quantize_row_to_slm_workgroup)

    if (row >= nrows) {
        return;
    }

    // =========================================================================
    // Phase 2: MMVQ for Q8_0 weights
    // =========================================================================
    const int blocks_per_row = ncols / QK8_0;
    const block_q8_0 * x = (const block_q8_0 *)vx;

    float tmp = 0.0f;

    for (int block_idx = lane; block_idx < blocks_per_row; block_idx += WARP_SIZE) {
        const int ibx = row * blocks_per_row + block_idx;
        const block_q8_0 * bq8_w = &x[ibx];
        const float d_w = bq8_w->d;

        // Get activation Q8 scale for this block
        const sycl::half2 ds8 = slm_ds[block_idx];
        const float d_a = float(ds8.x());
        const float scale = d_w * d_a;

        // Process all 32 elements of this block
        int sumi = 0;
        const int q8_base = block_idx * QK8_0;

        #pragma unroll
        for (int j = 0; j < 32; j++) {
            sumi += bq8_w->qs[j] * slm_qs[q8_base + j];
        }

        tmp += scale * sumi;
    }

    // Warp reduction
    tmp = sycl::reduce_over_group(sg, tmp, sycl::plus<float>());

    if (lane == 0) {
        dst[row] = tmp;
    }
}

// =============================================================================
// Host dispatch functions
// =============================================================================

static void mul_mat_vec_q4_0_f32_rmsnorm_sycl(
    const void * vx,
    const float * f32_input,
    const float * gamma,
    const float * rms_scales,
    float * dst,
    const int ncols,
    const int nrows,
    const int batch_size,
    dpct::queue_ptr stream
) {
    GGML_ASSERT(ncols % QK4_0 == 0);

    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // DEBUG: Try even simpler - use sycl::memset
    stream->memset(dst, 0, nrows * batch_size * sizeof(float));
}

static void mul_mat_vec_q8_0_f32_rmsnorm_sycl(
    const void * vx,
    const float * f32_input,
    const float * gamma,
    const float * rms_scales,
    float * dst,
    const int ncols,
    const int nrows,
    const int batch_size,
    dpct::queue_ptr stream
) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    const int num_blocks = ncols / QK8_1;
    const size_t slm_qs_size = ncols * sizeof(int8_t);
    const size_t slm_ds_size = num_blocks * sizeof(sycl::half2);
    const size_t slm_scratch_size = ncols * sizeof(float);  // For normalized values
    const size_t total_slm = slm_qs_size + slm_ds_size + slm_scratch_size;

    if (total_slm > MMVQ_RMSNORM_MAX_SLM_BYTES) {
        GGML_ABORT("Input dimension %d exceeds SLM capacity for fused MMVQ\n", ncols);
    }

    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // Match standard MMVQ dimension ordering: (1, 1, block_num_y)
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    for (int b = 0; b < batch_size; b++) {
        const float * input_b = f32_input + b * ncols;
        float * dst_b = dst + b * nrows;
        const float rms_scale = rms_scales[b];

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<int8_t, 1> slm_qs_acc(sycl::range<1>(ncols), cgh);
            sycl::local_accessor<sycl::half2, 1> slm_ds_acc(sycl::range<1>(num_blocks), cgh);
            sycl::local_accessor<float, 1> slm_scratch_acc(sycl::range<1>(ncols), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q8_0_f32_rmsnorm_slm(
                        vx, input_b, gamma, rms_scale, dst_b,
                        ncols, nrows,
                        slm_qs_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        slm_ds_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        slm_scratch_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
                }
            );
        });
    }
}

// =============================================================================
// Main dispatch function for fused RMSNorm + MMVQ
// =============================================================================

static void ggml_sycl_mul_mat_vec_rmsnorm(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * x,          // F32 input (pre-normalized)
    const ggml_tensor * gamma,      // RMSNorm weights
    const ggml_tensor * W,          // Quantized GEMM weights
    ggml_tensor * dst,              // Output
    float eps                       // RMSNorm epsilon
) {
    const int64_t nrows = ggml_nrows(W);   // Output features (N)
    const int64_t ncols = W->ne[0];        // Input features (K)
    const int64_t batch_size = x->ne[1];   // Batch size (M)

    dpct::queue_ptr stream = ctx.stream();

    // DEBUG: Skip RMS scale computation and use constant 1.0
    ggml_sycl_pool_alloc<float> scales_buf(ctx.pool(), batch_size);

    std::vector<float> ones(batch_size, 1.0f);
    stream->memcpy(scales_buf.get(), ones.data(), batch_size * sizeof(float)).wait();

    // SKIP: compute_rms_scales_sycl

    // Phase 2: Fused MMVQ with SLM-cached normalization
    const float * f32_input = (const float *)x->data;
    const float * gamma_data = (const float *)gamma->data;
    float * dst_data = (float *)dst->data;

    switch (W->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_f32_rmsnorm_sycl(
                W->data, f32_input, gamma_data, scales_buf.get(),
                dst_data, ncols, nrows, batch_size, stream
            );
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_f32_rmsnorm_sycl(
                W->data, f32_input, gamma_data, scales_buf.get(),
                dst_data, ncols, nrows, batch_size, stream
            );
            break;
        default:
            GGML_ABORT("Fused RMSNorm+MMVQ not supported for weight type %d\n", W->type);
    }
}

#endif // GGML_SYCL_MMVQ_RMSNORM_HPP
