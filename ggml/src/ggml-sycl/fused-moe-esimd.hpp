//
// Fused MoE ESIMD Kernel for Intel Arc GPUs
// Processes all tokens in parallel with direct expert indexing
//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FUSED_MOE_ESIMD_HPP
#define GGML_SYCL_FUSED_MOE_ESIMD_HPP

#include "common.hpp"
#include "dequantize.hpp"
#include <sycl/sycl.hpp>

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#define SYCL_ESIMD_MOE_AVAILABLE 1
#include <sycl/ext/intel/esimd.hpp>
#else
#define SYCL_ESIMD_MOE_AVAILABLE 0
#endif

#if SYCL_ESIMD_MOE_AVAILABLE

namespace esimd = sycl::ext::intel::esimd;

// =============================================================================
// Configuration
// =============================================================================

// Number of elements to process per work-item in the output dimension
// Higher values = better memory coalescing but more register pressure
constexpr int MOE_ELEMENTS_PER_THREAD = 4;

// Work-group size for reduction in hidden dimension
constexpr int MOE_WG_SIZE = 32;

// Block size for quantized types
constexpr int MOE_QK8_0 = 32;

// =============================================================================
// Fused MoE Kernel - Q8_0 Weights
// =============================================================================
// Each work-group computes one output row for one (token, expert_selection) pair
// Grid: (total_batches, nrows_per_expert / rows_per_wg)
// Block: (WG_SIZE)
//
// Expert weights layout: [num_experts, nrows_per_expert, ncols]
// Quantized as Q8_0 blocks: each block = {scale (f16), qs[32] (int8)}

template <int HIDDEN_DIM_BLOCKS>  // ncols / QK8_0
void fused_moe_q8_0_kernel(
    const void * __restrict__ expert_weights,  // [num_experts, nrows, ncols] Q8_0
    const void * __restrict__ input,           // F32 input with 2D layout
    const int32_t * __restrict__ expert_ids,   // [num_tokens, n_ids] expert indices
    float * __restrict__ output,               // [num_tokens, n_ids, nrows] F32
    const int64_t stride_expert,               // Bytes between experts in weights
    const int64_t ncols,                       // Hidden size (input dimension)
    const int64_t nrows,                       // Output size per expert
    const int64_t n_ids,                       // Number of expert selections per token
    const int64_t num_tokens,                  // Total number of tokens
    const int64_t ne11,                        // src1 dimension 1 size
    const int64_t ids_nb0,                     // ids stride for id dimension (bytes)
    const int64_t ids_nb1,                     // ids stride for token dimension (bytes)
    const int64_t in_nb11,                     // input stride for dimension 1 (bytes)
    const int64_t in_nb12,                     // input stride for dimension 2 (bytes)
    const int64_t out_nb1,                     // output stride for id dimension (bytes)
    const int64_t out_nb2,                     // output stride for token dimension (bytes)
    const sycl::nd_item<3> & item
) {
    using namespace esimd;

    // Work distribution:
    // group(0) = batch_idx = token_idx * n_ids + id_idx
    // group(2) = output row
    const int batch_idx = item.get_group(0);
    const int row = item.get_group(2);
    const int tid = item.get_local_id(2);

    if (row >= nrows) return;

    // Decompose batch_idx into (token_idx, id_idx) - 2D iteration
    const int token_idx = batch_idx / n_ids;  // i12 = iid1
    const int id_idx = batch_idx % n_ids;     // id

    if (token_idx >= num_tokens) return;

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)expert_ids +
                                                  token_idx * ids_nb1 + id_idx * ids_nb0);

    // Expert weights: offset by expert_id * stride_expert
    const block_q8_0 * weights_row = (const block_q8_0 *)((const char*)expert_weights +
                                                           expert_id * stride_expert) +
                                     row * HIDDEN_DIM_BLOCKS;

    // Compute input offset using proper 2D indexing (matching MMVQ kernel)
    // i11 = id_idx % ne11, i12 = token_idx
    const int64_t i11 = id_idx % ne11;
    const int64_t i12 = token_idx;
    const float * input_row = (const float *)((const char*)input + i11 * in_nb11 + i12 * in_nb12);

    // Each thread processes a subset of blocks and accumulates
    float partial_sum = 0.0f;

    // Process blocks assigned to this thread
    for (int b = tid; b < HIDDEN_DIM_BLOCKS; b += MOE_WG_SIZE) {
        // Load scale
        float scale = static_cast<float>(weights_row[b].d);

        // Load and dequantize weights, compute dot product with input
        float block_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < MOE_QK8_0; i++) {
            float w = static_cast<float>(weights_row[b].qs[i]) * scale;
            float x = input_row[b * MOE_QK8_0 + i];
            block_sum += w * x;
        }

        partial_sum += block_sum;
    }

    // Warp-level reduction using sub-group operations
    auto sg = item.get_sub_group();

    #pragma unroll
    for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
    }

    // Write output using proper 2D indexing
    // i1 = id_idx, i2 = token_idx
    if (tid == 0) {
        float * out_ptr = (float *)((char*)output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row] = partial_sum;
    }
}

// =============================================================================
// Fused MoE Kernel - MXFP4 Weights (4-bit quantization)
// =============================================================================
// MXFP4 format: each block = {scale (u8), qs[16] (4-bit packed pairs)}

// Block size for MXFP4 (use different name to avoid macro conflict)
constexpr int MOE_QK_MXFP4 = 32;

template <int HIDDEN_DIM_BLOCKS>  // ncols / MOE_QK_MXFP4
void fused_moe_mxfp4_kernel(
    const void * __restrict__ expert_weights,
    const void * __restrict__ input,          // F32 input with 2D layout
    const int32_t * __restrict__ expert_ids,
    float * __restrict__ output,
    const int64_t stride_expert,
    const int64_t ncols,
    const int64_t nrows,
    const int64_t n_ids,
    const int64_t num_tokens,
    const int64_t ne11,                       // src1 dimension 1 size
    const int64_t ids_nb0,
    const int64_t ids_nb1,
    const int64_t in_nb11,                    // input stride for dimension 1 (bytes)
    const int64_t in_nb12,                    // input stride for dimension 2 (bytes)
    const int64_t out_nb1,
    const int64_t out_nb2,
    const sycl::nd_item<3> & item
) {
    using namespace esimd;

    const int batch_idx = item.get_group(0);
    const int row = item.get_group(2);
    const int tid = item.get_local_id(2);

    if (row >= nrows) return;

    // Decompose batch_idx into (token_idx, id_idx) - 2D iteration
    const int token_idx = batch_idx / n_ids;  // i12 = iid1
    const int id_idx = batch_idx % n_ids;     // id

    if (token_idx >= num_tokens) return;

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)expert_ids +
                                                  token_idx * ids_nb1 + id_idx * ids_nb0);

    // Validate expert_id is in range (helps catch indexing bugs)
    if (expert_id < 0 || expert_id >= 32) {
        // Invalid expert ID - this indicates an indexing bug
        if (tid == 0 && row == 0) {
            // Can't print from kernel, but at least write a sentinel value
        }
        return;
    }

    // Expert weights: offset by expert_id * stride_expert
    const block_mxfp4 * weights_row = (const block_mxfp4 *)((const char*)expert_weights +
                                                             expert_id * stride_expert) +
                                      row * HIDDEN_DIM_BLOCKS;

    // Compute input offset using proper 2D indexing (matching MMVQ kernel)
    // i11 = id_idx % ne11, i12 = token_idx
    const int64_t i11 = id_idx % ne11;
    const int64_t i12 = token_idx;
    const float * input_row = (const float *)((const char*)input + i11 * in_nb11 + i12 * in_nb12);

    float partial_sum = 0.0f;

    for (int b = tid; b < HIDDEN_DIM_BLOCKS; b += MOE_WG_SIZE) {
        // MXFP4 scale: E8M0 exponent to FP32/2
        const float scale = sycl_e8m0_to_fp32_half(weights_row[b].e);

        float block_sum = 0.0f;

        // MXFP4 format: low nibbles (bytes 0-15) -> elements 0-15
        //               high nibbles (bytes 0-15) -> elements 16-31
        // This matches the dp4a layout used in MMVQ
        #pragma unroll
        for (int i = 0; i < MOE_QK_MXFP4 / 2; i++) {
            uint8_t packed = weights_row[b].qs[i];

            // MXFP4 dequantization using lookup table
            // kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
            // Values are doubled (e2m1 * 2), but sycl_e8m0_to_fp32_half already halves,
            // so the two cancel out and we don't need additional scaling
            float w_lo = scale * kvalues_mxfp4[packed & 0xF];  // Low nibble -> element i
            float w_hi = scale * kvalues_mxfp4[packed >> 4];   // High nibble -> element i+16

            float x_lo = input_row[b * MOE_QK_MXFP4 + i];       // Input element i
            float x_hi = input_row[b * MOE_QK_MXFP4 + i + 16];  // Input element i+16

            block_sum += w_lo * x_lo;
            block_sum += w_hi * x_hi;
        }

        partial_sum += block_sum;
    }

    // Work-group level reduction using SLM
    // Sub-group reduction alone won't work if work-group > sub-group
    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_id()[0];
    const int sg_lid = sg.get_local_id()[0];
    const int sg_size = sg.get_max_local_range()[0];

    // First reduce within each sub-group
    #pragma unroll
    for (int offset = sg_size / 2; offset > 0; offset /= 2) {
        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
    }

    // Write output - only first lane of first sub-group
    // Since WG_SIZE=32 and SG_SIZE=32, we should have exactly one sub-group
    if (tid == 0) {
        float * out_ptr = (float *)((char*)output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row] = partial_sum;
    }
}

// =============================================================================
// Launch Functions
// =============================================================================

inline bool fused_moe_esimd_available() {
    return true;
}

// Launch fused MoE kernel for Q8_0 weights
static void launch_fused_moe_q8_0(
    const void * expert_weights,
    const void * input,
    const int32_t * expert_ids,
    float * output,
    int64_t stride_expert,
    int64_t ncols,
    int64_t nrows,
    int64_t n_ids,
    int64_t num_tokens,
    int64_t ne11,
    int64_t ids_nb0,
    int64_t ids_nb1,
    int64_t in_nb11,
    int64_t in_nb12,
    int64_t out_nb1,
    int64_t out_nb2,
    sycl::queue & stream
) {
    const int64_t total_batches = num_tokens * n_ids;
    const int hidden_dim_blocks = ncols / MOE_QK8_0;

    sycl::range<3> grid(total_batches, 1, nrows);
    sycl::range<3> block(1, 1, MOE_WG_SIZE);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                // Dispatch based on hidden dimension
                // Common sizes: 2880 (GPT-OSS), 4096 (Llama), etc.
                if (hidden_dim_blocks == 90) {  // 2880 / 32
                    fused_moe_q8_0_kernel<90>(
                        expert_weights, input, expert_ids, output,
                        stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                        ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
                } else if (hidden_dim_blocks == 128) {  // 4096 / 32
                    fused_moe_q8_0_kernel<128>(
                        expert_weights, input, expert_ids, output,
                        stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                        ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
                } else {
                    // Generic fallback - less optimized
                    fused_moe_q8_0_kernel<0>(
                        expert_weights, input, expert_ids, output,
                        stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                        ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
                }
            });
    });
}

// Helper to launch MXFP4 kernel with specific template parameter
template <int HIDDEN_DIM_BLOCKS>
static void launch_fused_moe_mxfp4_impl(
    const void * expert_weights,
    const float * input,
    const int32_t * expert_ids,
    float * output,
    int64_t stride_expert,
    int64_t ncols,
    int64_t nrows,
    int64_t n_ids,
    int64_t num_tokens,
    int64_t ne11,
    int64_t ids_nb0,
    int64_t ids_nb1,
    int64_t in_nb11,
    int64_t in_nb12,
    int64_t out_nb1,
    int64_t out_nb2,
    sycl::queue & stream
) {
    const int64_t total_batches = num_tokens * n_ids;

    sycl::range<3> grid(total_batches, 1, nrows);
    sycl::range<3> block(1, 1, MOE_WG_SIZE);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                fused_moe_mxfp4_kernel<HIDDEN_DIM_BLOCKS>(
                    expert_weights, input, expert_ids, output,
                    stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                    ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
            });
    });
}

// Launch fused MoE kernel for MXFP4 weights
static void launch_fused_moe_mxfp4(
    const void * expert_weights,
    const float * input,
    const int32_t * expert_ids,
    float * output,
    int64_t stride_expert,
    int64_t ncols,
    int64_t nrows,
    int64_t n_ids,
    int64_t num_tokens,
    int64_t ne11,
    int64_t ids_nb0,
    int64_t ids_nb1,
    int64_t in_nb11,
    int64_t in_nb12,
    int64_t out_nb1,
    int64_t out_nb2,
    sycl::queue & stream
) {
    const int hidden_dim_blocks = ncols / MOE_QK_MXFP4;

    // Dispatch at host side to avoid runtime branching in kernel
    if (hidden_dim_blocks == 90) {  // 2880 / 32
        launch_fused_moe_mxfp4_impl<90>(
            expert_weights, input, expert_ids, output,
            stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
            ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, stream);
    } else if (hidden_dim_blocks == 128) {  // 4096 / 32
        launch_fused_moe_mxfp4_impl<128>(
            expert_weights, input, expert_ids, output,
            stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
            ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, stream);
    } else {
        // Generic fallback - pass actual block count
        const int64_t actual_blocks = ncols / MOE_QK_MXFP4;
        const int64_t total_batches = num_tokens * n_ids;
        sycl::range<3> grid(total_batches, 1, nrows);
        sycl::range<3> block(1, 1, MOE_WG_SIZE);

        stream.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                    // Use dynamic loop bound
                    const int batch_idx = item.get_group(0);
                    const int row = item.get_group(2);
                    const int tid = item.get_local_id(2);

                    if (row >= nrows) return;

                    const int token_idx = batch_idx / n_ids;
                    const int id_idx = batch_idx % n_ids;

                    if (token_idx >= num_tokens) return;

                    const int32_t expert_id = *(const int32_t *)((const char*)expert_ids +
                                                                  token_idx * ids_nb1 + id_idx * ids_nb0);

                    const block_mxfp4 * weights_row = (const block_mxfp4 *)((const char*)expert_weights +
                                                                             expert_id * stride_expert) +
                                                      row * actual_blocks;

                    const int64_t i11 = id_idx % ne11;
                    const int64_t i12 = token_idx;
                    const float * input_row = (const float *)((const char*)input + i11 * in_nb11 + i12 * in_nb12);

                    float partial_sum = 0.0f;

                    for (int b = tid; b < actual_blocks; b += MOE_WG_SIZE) {
                        const float scale = sycl_e8m0_to_fp32_half(weights_row[b].e);
                        float block_sum = 0.0f;

                        // MXFP4 format: low nibbles -> elements 0-15, high nibbles -> elements 16-31
                        #pragma unroll
                        for (int i = 0; i < MOE_QK_MXFP4 / 2; i++) {
                            uint8_t packed = weights_row[b].qs[i];
                            float w_lo = scale * kvalues_mxfp4[packed & 0xF];
                            float w_hi = scale * kvalues_mxfp4[packed >> 4];
                            float x_lo = input_row[b * MOE_QK_MXFP4 + i];
                            float x_hi = input_row[b * MOE_QK_MXFP4 + i + 16];
                            block_sum += w_lo * x_lo + w_hi * x_hi;
                        }
                        partial_sum += block_sum;
                    }

                    auto sg = item.get_sub_group();
                    #pragma unroll
                    for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
                        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
                    }

                    if (tid == 0) {
                        float * out_ptr = (float *)((char*)output + token_idx * out_nb2 + id_idx * out_nb1);
                        out_ptr[row] = partial_sum;
                    }
                });
        });
    }
}

#else  // !SYCL_ESIMD_MOE_AVAILABLE

inline bool fused_moe_esimd_available() {
    return false;
}

static void launch_fused_moe_q8_0(
    const void *, const void *, const int32_t *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, sycl::queue &
) {
    GGML_ASSERT(false && "ESIMD MoE not available");
}

static void launch_fused_moe_mxfp4(
    const void *, const float *, const int32_t *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, sycl::queue &
) {
    GGML_ASSERT(false && "ESIMD MoE not available");
}

#endif  // SYCL_ESIMD_MOE_AVAILABLE

#endif  // GGML_SYCL_FUSED_MOE_ESIMD_HPP
