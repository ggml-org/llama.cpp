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
#include "quantize.hpp"

#include <sycl/sycl.hpp>

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_ESIMD_MOE_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
#else
#    define SYCL_ESIMD_MOE_AVAILABLE 0
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

template <int HIDDEN_DIM_BLOCKS>                                          // ncols / QK8_0
void fused_moe_q8_0_kernel(const void * __restrict__ expert_weights,      // [num_experts, nrows, ncols] Q8_0
                           const void * __restrict__ input,               // F32 input with 2D layout
                           const int32_t * __restrict__ expert_ids,       // [num_tokens, n_ids] expert indices
                           float * __restrict__ output,                   // [num_tokens, n_ids, nrows] F32
                           const int64_t                  stride_expert,  // Bytes between experts in weights
                           [[maybe_unused]] const int64_t ncols,          // Hidden size (input dimension)
                           const int64_t                  nrows,          // Output size per expert
                           const int64_t                  n_ids,          // Number of expert selections per token
                           const int64_t                  num_tokens,     // Total number of tokens
                           const int64_t                  ne11,           // src1 dimension 1 size
                           const int64_t                  ids_nb0,        // ids stride for id dimension (bytes)
                           const int64_t                  ids_nb1,        // ids stride for token dimension (bytes)
                           const int64_t                  in_nb11,        // input stride for dimension 1 (bytes)
                           const int64_t                  in_nb12,        // input stride for dimension 2 (bytes)
                           const int64_t                  out_nb1,        // output stride for id dimension (bytes)
                           const int64_t                  out_nb2,        // output stride for token dimension (bytes)
                           const sycl::nd_item<3> &       item) {
    using namespace esimd;

    // Work distribution:
    // group(0) = batch_idx = token_idx * n_ids + id_idx
    // group(2) = output row
    const int batch_idx = item.get_group(0);
    const int row       = item.get_group(2);
    const int tid       = item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    // Decompose batch_idx into (token_idx, id_idx) - 2D iteration
    const int token_idx = batch_idx / n_ids;  // i12 = iid1
    const int id_idx    = batch_idx % n_ids;  // id

    if (token_idx >= num_tokens) {
        return;
    }

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

    // Expert weights: offset by expert_id * stride_expert
    const block_q8_0 * weights_row =
        (const block_q8_0 *) ((const char *) expert_weights + expert_id * stride_expert) + row * HIDDEN_DIM_BLOCKS;

    // Compute input offset using proper 2D indexing (matching MMVQ kernel)
    // i11 = id_idx % ne11, i12 = token_idx
    const int64_t i11       = id_idx % ne11;
    const int64_t i12       = token_idx;
    const float * input_row = (const float *) ((const char *) input + i11 * in_nb11 + i12 * in_nb12);

    // Each thread processes a subset of blocks and accumulates
    float partial_sum = 0.0f;

    // Process blocks assigned to this thread
    for (int b = tid; b < HIDDEN_DIM_BLOCKS; b += MOE_WG_SIZE) {
        // Load scale
        float scale = static_cast<float>(weights_row[b].d);

        // Load and dequantize weights, compute dot product with input
        float block_sum = 0.0f;

#    pragma unroll
        for (int i = 0; i < MOE_QK8_0; i++) {
            float w = static_cast<float>(weights_row[b].qs[i]) * scale;
            float x = input_row[b * MOE_QK8_0 + i];
            block_sum += w * x;
        }

        partial_sum += block_sum;
    }

    // Warp-level reduction using sub-group operations
    auto sg = item.get_sub_group();

#    pragma unroll
    for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
    }

    // Write output using proper 2D indexing
    // i1 = id_idx, i2 = token_idx
    if (tid == 0) {
        float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row]    = partial_sum;
    }
}

// =============================================================================
// Fused MoE Kernel - MXFP4 Weights (4-bit quantization)
// =============================================================================
// MXFP4 format: each block = {scale (u8), qs[16] (4-bit packed pairs)}

// Block size for MXFP4 (use different name to avoid macro conflict)
constexpr int MOE_QK_MXFP4 = 32;

// =============================================================================
// DP4A Helper: Convert 4 packed MXFP4 nibbles to int8 via byte_level_permute
// =============================================================================
// This is the same function as in vecdotq.hpp but inlined here to avoid
// pulling in the full vecdotq.hpp dependency chain.
static __dpct_inline__ sycl::int2 moe_get_int_from_table_16(
    const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t         tmp[2];
    const uint32_t   low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#    pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high = dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i]              = dpct::byte_level_permute(low, high, low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

template <int HIDDEN_DIM_BLOCKS>                              // ncols / MOE_QK_MXFP4
void fused_moe_mxfp4_kernel(const void * __restrict__ expert_weights,
                            const void * __restrict__ input,  // F32 input with 2D layout
                            const int32_t * __restrict__ expert_ids,
                            float * __restrict__ output,
                            const int64_t                  stride_expert,
                            [[maybe_unused]] const int64_t ncols,
                            const int64_t                  nrows,
                            const int64_t                  n_ids,
                            const int64_t                  num_tokens,
                            const int64_t                  ne11,  // src1 dimension 1 size
                            const int64_t                  ids_nb0,
                            const int64_t                  ids_nb1,
                            const int64_t                  in_nb11,  // input stride for dimension 1 (bytes)
                            const int64_t                  in_nb12,  // input stride for dimension 2 (bytes)
                            const int64_t                  out_nb1,
                            const int64_t                  out_nb2,
                            const sycl::nd_item<3> &       item) {
    using namespace esimd;

    const int batch_idx = item.get_group(0);
    const int row       = item.get_group(2);
    const int tid       = item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    // Decompose batch_idx into (token_idx, id_idx) - 2D iteration
    const int token_idx = batch_idx / n_ids;  // i12 = iid1
    const int id_idx    = batch_idx % n_ids;  // id

    if (token_idx >= num_tokens) {
        return;
    }

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

    // Validate expert_id is in range (helps catch indexing bugs)
    if (expert_id < 0 || expert_id >= 64) {
        // Invalid expert ID - this indicates an indexing bug
        if (tid == 0 && row == 0) {
            // Can't print from kernel, but at least write a sentinel value
        }
        return;
    }

    // Expert weights: offset by expert_id * stride_expert
    const block_mxfp4 * weights_row =
        (const block_mxfp4 *) ((const char *) expert_weights + expert_id * stride_expert) + row * HIDDEN_DIM_BLOCKS;

    // Compute input offset using proper 2D indexing (matching MMVQ kernel)
    // i11 = id_idx % ne11, i12 = token_idx
    const int64_t i11       = id_idx % ne11;
    const int64_t i12       = token_idx;
    const float * input_row = (const float *) ((const char *) input + i11 * in_nb11 + i12 * in_nb12);

    float partial_sum = 0.0f;

    for (int b = tid; b < HIDDEN_DIM_BLOCKS; b += MOE_WG_SIZE) {
        // MXFP4 scale: E8M0 exponent to FP32/2
        const float scale = sycl_e8m0_to_fp32_half(weights_row[b].e);

        float block_sum = 0.0f;

// MXFP4 format: low nibbles (bytes 0-15) -> elements 0-15
//               high nibbles (bytes 0-15) -> elements 16-31
// This matches the dp4a layout used in MMVQ
#    pragma unroll
        for (int i = 0; i < MOE_QK_MXFP4 / 2; i++) {
            uint8_t packed = weights_row[b].qs[i];

            // MXFP4 dequantization using lookup table
            // kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
            // Values are doubled (e2m1 * 2), but sycl_e8m0_to_fp32_half already halves,
            // so the two cancel out and we don't need additional scaling
            float w_lo = scale * kvalues_mxfp4[packed & 0xF];   // Low nibble -> element i
            float w_hi = scale * kvalues_mxfp4[packed >> 4];    // High nibble -> element i+16

            float x_lo = input_row[b * MOE_QK_MXFP4 + i];       // Input element i
            float x_hi = input_row[b * MOE_QK_MXFP4 + i + 16];  // Input element i+16

            block_sum += w_lo * x_lo;
            block_sum += w_hi * x_hi;
        }

        partial_sum += block_sum;
    }

    // Work-group level reduction using SLM
    // Sub-group reduction alone won't work if work-group > sub-group
    auto      sg      = item.get_sub_group();
    const int sg_size = sg.get_max_local_range()[0];

// First reduce within each sub-group
#    pragma unroll
    for (int offset = sg_size / 2; offset > 0; offset /= 2) {
        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
    }

    // Write output - only first lane of first sub-group
    // Since WG_SIZE=32 and SG_SIZE=32, we should have exactly one sub-group
    if (tid == 0) {
        float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row]    = partial_sum;
    }
}

// =============================================================================
// Fused MoE Kernel - MXFP4 Weights with DP4A (int8 dot product acceleration)
// =============================================================================
// Replaces scalar FP32 dequantization with DP4A-accelerated integer arithmetic.
// Input is pre-quantized to Q8_1 format before kernel launch.
// Each DP4A instruction computes 4 int8 multiply-accumulates in one cycle,
// giving ~4x throughput improvement over scalar FP32.
//
// Data flow:
//   MXFP4 weights -> get_int_from_table_16() -> 4 int8 values
//   F32 input -> quantize_row_q8_1_sycl() -> Q8_1 blocks {d, sum, qs[32]}
//   DP4A(mxfp4_int8, q8_int8, accumulator) -> int32 partial sum
//   Final: mxfp4_scale * q8_scale * int_sum

template <int HIDDEN_DIM_BLOCKS>                              // ncols / MOE_QK_MXFP4
void fused_moe_mxfp4_dp4a_kernel(const void * __restrict__ expert_weights,
                                  const void * __restrict__ q8_input,  // Q8_1 quantized input
                                  const int32_t * __restrict__ expert_ids,
                                  float * __restrict__ output,
                                  const int64_t                  stride_expert,
                                  [[maybe_unused]] const int64_t ncols,
                                  const int64_t                  nrows,
                                  const int64_t                  n_ids,
                                  const int64_t                  num_tokens,
                                  const int64_t                  ne11,  // src1 dimension 1 size
                                  const int64_t                  ids_nb0,
                                  const int64_t                  ids_nb1,
                                  const int64_t                  q8_row_stride,   // bytes per Q8_1 row
                                  const int64_t                  q8_batch_stride, // bytes per batch of rows
                                  const int64_t                  out_nb1,
                                  const int64_t                  out_nb2,
                                  const sycl::nd_item<3> &       item) {
    using namespace esimd;

    const int batch_idx = item.get_group(0);
    const int row       = item.get_group(2);
    const int tid       = item.get_local_id(2);

    if (row >= nrows) {
        return;
    }

    // Decompose batch_idx into (token_idx, id_idx) - 2D iteration
    const int token_idx = batch_idx / n_ids;  // i12 = iid1
    const int id_idx    = batch_idx % n_ids;  // id

    if (token_idx >= num_tokens) {
        return;
    }

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

    // Validate expert_id is in range
    if (expert_id < 0 || expert_id >= 64) {
        return;
    }

    // Resolve block count (compile-time or runtime)
    const int blocks_per_row = (HIDDEN_DIM_BLOCKS > 0) ? HIDDEN_DIM_BLOCKS : (ncols / MOE_QK_MXFP4);

    // Expert weights: offset by expert_id * stride_expert
    const block_mxfp4 * weights_row =
        (const block_mxfp4 *) ((const char *) expert_weights + expert_id * stride_expert) + row * blocks_per_row;

    // Compute Q8_1 input offset using proper 2D indexing
    // i11 = id_idx % ne11, i12 = token_idx
    const int64_t i11 = id_idx % ne11;
    const int64_t i12 = token_idx;
    const char *  q8_row_base = (const char *) q8_input + i11 * q8_row_stride + i12 * q8_batch_stride;

    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += MOE_WG_SIZE) {
        // Load E8M0 scale for this MXFP4 block
        const float d_mxfp4 = ggml_sycl_e8m0_to_fp32(weights_row[b].e) * 0.5f;

        // Load Q8_1 block data (AoS layout: block_q8_1 = {ds: half2, qs[32]: int8})
        const block_q8_1 *  q8_block = (const block_q8_1 *) q8_row_base + b;
        const int *         q8_qs    = (const int *) q8_block->qs;
        const float         d_q8     = (float) q8_block->ds[0];

        // DP4A dot product: process 16 packed bytes (32 nibbles = 32 MXFP4 values)
        int sumi = 0;
#    pragma unroll
        for (int i = 0; i < MOE_QK_MXFP4 / 2; i += 4) {
            // Load 4 packed bytes at once (8 MXFP4 nibbles)
            const int        aux_q4 = *((const int *) (weights_row[b].qs + i));
            const sycl::int2 v      = moe_get_int_from_table_16(aux_q4, kvalues_mxfp4);

            // DP4A: 4-way int8 dot product
            // v.x() = low nibbles (elements i..i+3), q8_qs[i/4] = Q8_1 quants for elements i..i+3
            // v.y() = high nibbles (elements i+16..i+19), q8_qs[i/4+4] = Q8_1 quants for elements i+16..i+19
            sumi = ggml_sycl_dp4a(v.x(), q8_qs[i / 4], sumi);
            sumi = ggml_sycl_dp4a(v.y(), q8_qs[i / 4 + 4], sumi);
        }

        acc += d_mxfp4 * d_q8 * sumi;
    }

    // Work-group level reduction using sub-group shuffle
    auto      sg      = item.get_sub_group();
    const int sg_size = sg.get_max_local_range()[0];

#    pragma unroll
    for (int offset = sg_size / 2; offset > 0; offset /= 2) {
        acc += sycl::shift_group_left(sg, acc, offset);
    }

    // Write output - only first lane of first sub-group
    if (tid == 0) {
        float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row]    = acc;
    }
}

// =============================================================================
// Fused MoE Kernel - MXFP4 SoA Layout (Reordered Weights)
// =============================================================================
// After reorder_qw_mxfp4(), weights are stored as:
//   [all qs for all experts] [all scales for all experts]
// This is a flat SoA layout where expert boundaries are implicit.
//
// Expert `e`, row `r`, block `b` access:
//   block_index = e * nrows_per_expert * blocks_per_row + r * blocks_per_row + b
//   qs_offset = block_index * 16 (16 packed bytes per block)
//   scale_offset = total_qs_size + block_index

template <int HIDDEN_DIM_BLOCKS>                                              // ncols / MOE_QK_MXFP4
void fused_moe_mxfp4_soa_kernel(sycl::nd_item<3> item,
                                const uint8_t * __restrict__ weights_qs,      // Pointer to qs region start
                                const uint8_t * __restrict__ weights_scales,  // Pointer to scales region start
                                const float * __restrict__ input,
                                const int32_t * __restrict__ expert_ids,
                                float * __restrict__ output,
                                const int64_t nrows_per_expert,  // Rows per expert
                                const int64_t ncols,             // Hidden dimension
                                const int64_t n_ids,
                                const int64_t num_tokens,
                                const int64_t ne11,
                                const int64_t ids_nb0,
                                const int64_t ids_nb1,
                                const int64_t in_nb11,
                                const int64_t in_nb12,
                                const int64_t out_nb1,
                                const int64_t out_nb2) {
    const int batch_idx = item.get_group(0);
    const int row       = item.get_group(2);
    const int tid       = item.get_local_id(2);

    if (row >= nrows_per_expert) {
        return;
    }

    const int token_idx = batch_idx / n_ids;
    const int id_idx    = batch_idx % n_ids;

    if (token_idx >= num_tokens) {
        return;
    }

    // Get expert ID for this token
    const int32_t expert_id = *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

    if (expert_id < 0 || expert_id >= 64) {
        return;  // Validate (up to 64 experts)
    }

    // Compute number of blocks per row
    const int hidden_dim_blocks = (HIDDEN_DIM_BLOCKS > 0) ? HIDDEN_DIM_BLOCKS : (ncols / MOE_QK_MXFP4);

    // SoA layout: compute absolute block offset for this expert's row
    // block_offset = expert_id * nrows_per_expert * hidden_dim_blocks + row * hidden_dim_blocks
    const int64_t row_block_offset = (expert_id * nrows_per_expert + row) * hidden_dim_blocks;

    // qs region: each block contributes 16 bytes (32 4-bit values packed into 16 bytes)
    const uint8_t * qs_row    = weights_qs + row_block_offset * (MOE_QK_MXFP4 / 2);
    // scales region: each block contributes 1 byte (E8M0 scale)
    const uint8_t * scale_row = weights_scales + row_block_offset;

    // Input row with proper 2D indexing
    const int64_t i11       = id_idx % ne11;
    const float * input_row = (const float *) ((const char *) input + i11 * in_nb11 + token_idx * in_nb12);

    float partial_sum = 0.0f;

    for (int b = tid; b < hidden_dim_blocks; b += MOE_WG_SIZE) {
        // Load scale from SoA scales region
        const float scale = sycl_e8m0_to_fp32_half(scale_row[b]);

        float block_sum = 0.0f;

#    pragma unroll
        for (int i = 0; i < MOE_QK_MXFP4 / 2; i++) {
            uint8_t packed = qs_row[b * (MOE_QK_MXFP4 / 2) + i];

            float w_lo = scale * kvalues_mxfp4[packed & 0xF];
            float w_hi = scale * kvalues_mxfp4[packed >> 4];

            float x_lo = input_row[b * MOE_QK_MXFP4 + i];
            float x_hi = input_row[b * MOE_QK_MXFP4 + i + 16];

            block_sum += w_lo * x_lo + w_hi * x_hi;
        }

        partial_sum += block_sum;
    }

    // Subgroup reduction
    auto sg = item.get_sub_group();
#    pragma unroll
    for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
        partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
    }

    if (tid == 0) {
        float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
        out_ptr[row]    = partial_sum;
    }
}

// =============================================================================
// Optimized Persistent MoE Kernel with ESIMD and SLM
// =============================================================================
// Key optimizations:
// 1. Persistent kernel - fixed work-groups loop over all work (reduces launch overhead)
// 2. SLM input caching - input row cached in SLM, shared across all experts
// 3. ESIMD vectorized loads - 16-byte aligned loads for weights
// 4. Fused expert reduction - all experts processed and summed in one kernel
// 5. LUT caching - kvalues_mxfp4 cached in SLM

// Number of persistent work-groups (2 per XeCore, B50 has 20 XeCores)
constexpr int MOE_PERSISTENT_GROUPS  = 40;
// Work-group size for persistent kernel (must match sub-group size)
constexpr int MOE_PERSISTENT_WG_SIZE = 32;
// Maximum input dimension we can cache in SLM (128KB limit, using 32KB for input)
constexpr int MOE_MAX_CACHED_COLS    = 8192;  // 8192 * 4 = 32KB

// Persistent MXFP4 kernel with SLM caching
// Key difference from original: processes all experts for one (token, row) together,
// caching input in SLM. Writes separate outputs per expert (API compatible).
template <int HIDDEN_DIM_BLOCKS>
void persistent_moe_mxfp4_kernel(const void * __restrict__ expert_weights,
                                 const float * __restrict__ input,
                                 const int32_t * __restrict__ expert_ids,
                                 float * __restrict__ output,
                                 const int64_t stride_expert,
                                 const int64_t ncols,
                                 const int64_t nrows,
                                 const int64_t n_ids,
                                 const int64_t num_tokens,
                                 const int64_t ne11,
                                 const int64_t ids_nb0,
                                 const int64_t ids_nb1,
                                 const int64_t in_nb11,
                                 const int64_t in_nb12,
                                 const int64_t out_nb1,
                                 const int64_t out_nb2,
                                 const int64_t num_groups,
                                 float * __restrict__ slm_input,    // SLM for input caching
                                 float * __restrict__ slm_kvalues,  // SLM for LUT
                                 const sycl::nd_item<1> & item) {
    const int group_id = item.get_group_linear_id();
    const int tid      = item.get_local_id(0);
    auto      sg       = item.get_sub_group();

    // Cache kvalues_mxfp4 in SLM (16 floats = 64 bytes)
    if (tid < 16) {
        slm_kvalues[tid] = kvalues_mxfp4[tid];
    }
    sycl::group_barrier(item.get_group());

    // Total work: one (token, row) pair per work item
    // Each work-group processes all n_ids experts for that pair
    const int64_t total_work = num_tokens * nrows;

    const int actual_blocks = (HIDDEN_DIM_BLOCKS > 0) ? HIDDEN_DIM_BLOCKS : (ncols / MOE_QK_MXFP4);

    // Persistent loop - each work-group handles multiple work items
    for (int64_t work_idx = group_id; work_idx < total_work; work_idx += num_groups) {
        const int row       = work_idx % nrows;
        const int token_idx = work_idx / nrows;

        // Load input to SLM (collaborative load across work-group)
        const int64_t i11         = 0;  // Default dimension
        const int64_t i12         = token_idx;
        const float * token_input = (const float *) ((const char *) input + i11 * in_nb11 + i12 * in_nb12);

        // Collaborative load: each thread loads multiple elements
        for (int i = tid; i < ncols && i < MOE_MAX_CACHED_COLS; i += MOE_PERSISTENT_WG_SIZE) {
            slm_input[i] = token_input[i];
        }
        sycl::group_barrier(item.get_group());

        // Process all experts for this (token, row), writing separate outputs
        for (int id_idx = 0; id_idx < n_ids; id_idx++) {
            // Read expert ID
            const int32_t expert_id =
                *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

            if (expert_id < 0 || expert_id >= 64) {
                // Write zero for invalid expert
                if (tid == 0) {
                    float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
                    out_ptr[row]    = 0.0f;
                }
                continue;
            }

            // Expert weights for this row
            const block_mxfp4 * weights_row =
                (const block_mxfp4 *) ((const char *) expert_weights + expert_id * stride_expert) + row * actual_blocks;

            // Each thread processes subset of blocks
            float partial_sum = 0.0f;

            for (int b = tid; b < actual_blocks; b += MOE_PERSISTENT_WG_SIZE) {
                // MXFP4 scale
                const float scale = sycl_e8m0_to_fp32_half(weights_row[b].e);

                float block_sum = 0.0f;

// Process 32 elements per block (16 bytes of packed 4-bit values)
#    pragma unroll
                for (int i = 0; i < MOE_QK_MXFP4 / 2; i++) {
                    uint8_t packed = weights_row[b].qs[i];

                    // Use SLM-cached lookup table
                    float w_lo = scale * slm_kvalues[packed & 0xF];
                    float w_hi = scale * slm_kvalues[packed >> 4];

                    // Read from SLM-cached input
                    float x_lo = slm_input[b * MOE_QK_MXFP4 + i];
                    float x_hi = slm_input[b * MOE_QK_MXFP4 + i + 16];

                    block_sum += w_lo * x_lo + w_hi * x_hi;
                }

                partial_sum += block_sum;
            }

// Sub-group reduction
#    pragma unroll
            for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
                partial_sum += sycl::shift_group_left(sg, partial_sum, offset);
            }

            // Write output for this expert (maintains original API)
            if (tid == 0) {
                float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
                out_ptr[row]    = partial_sum;
            }
        }

        sycl::group_barrier(item.get_group());
    }
}

// =============================================================================
// Persistent MoE Kernel - MXFP4 with DP4A (int8 dot product acceleration)
// =============================================================================
// Same persistent loop pattern as persistent_moe_mxfp4_kernel but uses DP4A
// for the inner dot product instead of scalar FP32 dequantization.
// Input is pre-quantized to Q8_1 and cached in SLM as Q8_1 blocks.
template <int HIDDEN_DIM_BLOCKS>
void persistent_moe_mxfp4_dp4a_kernel(const void * __restrict__ expert_weights,
                                       const void * __restrict__ q8_input,     // Q8_1 quantized input
                                       const int32_t * __restrict__ expert_ids,
                                       float * __restrict__ output,
                                       const int64_t stride_expert,
                                       const int64_t ncols,
                                       const int64_t nrows,
                                       const int64_t n_ids,
                                       const int64_t num_tokens,
                                       const int64_t ne11,
                                       const int64_t ids_nb0,
                                       const int64_t ids_nb1,
                                       const int64_t q8_row_stride,
                                       const int64_t q8_batch_stride,
                                       const int64_t out_nb1,
                                       const int64_t out_nb2,
                                       const int64_t num_groups,
                                       const sycl::nd_item<1> & item) {
    const int group_id = item.get_group_linear_id();
    const int tid      = item.get_local_id(0);
    auto      sg       = item.get_sub_group();

    // Total work: one (token, row) pair per work item
    const int64_t total_work = num_tokens * nrows;

    const int actual_blocks = (HIDDEN_DIM_BLOCKS > 0) ? HIDDEN_DIM_BLOCKS : (ncols / MOE_QK_MXFP4);

    // Persistent loop - each work-group handles multiple work items
    for (int64_t work_idx = group_id; work_idx < total_work; work_idx += num_groups) {
        const int row       = work_idx % nrows;
        const int token_idx = work_idx / nrows;

        // Q8_1 input for this token (already quantized)
        const int64_t i11           = 0;  // Default dimension
        const int64_t i12           = token_idx;
        const char *  q8_row_base   = (const char *) q8_input + i11 * q8_row_stride + i12 * q8_batch_stride;

        // Process all experts for this (token, row), writing separate outputs
        for (int id_idx = 0; id_idx < n_ids; id_idx++) {
            // Read expert ID
            const int32_t expert_id =
                *(const int32_t *) ((const char *) expert_ids + token_idx * ids_nb1 + id_idx * ids_nb0);

            if (expert_id < 0 || expert_id >= 64) {
                if (tid == 0) {
                    float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
                    out_ptr[row]    = 0.0f;
                }
                continue;
            }

            // Expert weights for this row
            const block_mxfp4 * weights_row =
                (const block_mxfp4 *) ((const char *) expert_weights + expert_id * stride_expert) + row * actual_blocks;

            // Each thread processes subset of blocks using DP4A
            float acc = 0.0f;

            for (int b = tid; b < actual_blocks; b += MOE_PERSISTENT_WG_SIZE) {
                // Load E8M0 scale
                const float d_mxfp4 = ggml_sycl_e8m0_to_fp32(weights_row[b].e) * 0.5f;

                // Load Q8_1 block
                const block_q8_1 * q8_block = (const block_q8_1 *) q8_row_base + b;
                const int *        q8_qs    = (const int *) q8_block->qs;
                const float        d_q8     = (float) q8_block->ds[0];

                // DP4A dot product
                int sumi = 0;
#    pragma unroll
                for (int i = 0; i < MOE_QK_MXFP4 / 2; i += 4) {
                    const int        aux_q4 = *((const int *) (weights_row[b].qs + i));
                    const sycl::int2 v      = moe_get_int_from_table_16(aux_q4, kvalues_mxfp4);

                    sumi = ggml_sycl_dp4a(v.x(), q8_qs[i / 4], sumi);
                    sumi = ggml_sycl_dp4a(v.y(), q8_qs[i / 4 + 4], sumi);
                }

                acc += d_mxfp4 * d_q8 * sumi;
            }

// Sub-group reduction
#    pragma unroll
            for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
                acc += sycl::shift_group_left(sg, acc, offset);
            }

            // Write output
            if (tid == 0) {
                float * out_ptr = (float *) ((char *) output + token_idx * out_nb2 + id_idx * out_nb1);
                out_ptr[row]    = acc;
            }
        }
    }
}

// Launch persistent MXFP4 kernel
template <int HIDDEN_DIM_BLOCKS>
static void launch_persistent_moe_mxfp4_impl(const void *    expert_weights,
                                             const float *   input,
                                             const int32_t * expert_ids,
                                             float *         output,
                                             int64_t         stride_expert,
                                             int64_t         ncols,
                                             int64_t         nrows,
                                             int64_t         n_ids,
                                             int64_t         num_tokens,
                                             int64_t         ne11,
                                             int64_t         ids_nb0,
                                             int64_t         ids_nb1,
                                             int64_t         in_nb11,
                                             int64_t         in_nb12,
                                             int64_t         out_nb1,
                                             int64_t         out_nb2,
                                             sycl::queue &   stream) {
    // --- Q8_1 quantization of F32 input (amortized across all experts) ---
    const int64_t ncols_padded    = GGML_PAD(ncols, QK8_1);
    const int64_t q8_1_row_size   = ncols_padded * sizeof(block_q8_1) / QK8_1;
    const int64_t total_src1_rows = ne11 * num_tokens;
    const size_t  q8_buf_size     = total_src1_rows * q8_1_row_size;

    // Allocate device buffer for Q8_1 quantized input
    void * q8_buffer = sycl::malloc_device(q8_buf_size, stream);

    // Quantize F32 input to Q8_1 (AoS layout)
    quantize_row_q8_1_sycl<quantize_q8_1>(input, (char *) q8_buffer, ncols, total_src1_rows, ncols_padded, &stream);

    // Q8_1 strides
    const int64_t q8_row_stride   = q8_1_row_size;
    const int64_t q8_batch_stride = ne11 * q8_1_row_size;

    // Use fewer groups for small workloads
    const int64_t total_work = num_tokens * nrows;
    const int     num_groups = std::min((int64_t) MOE_PERSISTENT_GROUPS, total_work);

    sycl::range<1> grid(num_groups * MOE_PERSISTENT_WG_SIZE);
    sycl::range<1> block(MOE_PERSISTENT_WG_SIZE);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(grid, block), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            persistent_moe_mxfp4_dp4a_kernel<HIDDEN_DIM_BLOCKS>(expert_weights, q8_buffer, expert_ids, output,
                                                                 stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                                                                 ids_nb0, ids_nb1, q8_row_stride, q8_batch_stride,
                                                                 out_nb1, out_nb2, num_groups, item);
        });
    });

    // Free Q8_1 buffer after kernel completes (host_task runs after all prior submissions)
    stream.submit([&](sycl::handler & cgh) {
        cgh.host_task([=, &stream]() { sycl::free(q8_buffer, stream); });
    });
}

// Main launch function for persistent MXFP4 kernel
static void launch_persistent_moe_mxfp4(const void *    expert_weights,
                                        const float *   input,
                                        const int32_t * expert_ids,
                                        float *         output,
                                        int64_t         stride_expert,
                                        int64_t         ncols,
                                        int64_t         nrows,
                                        int64_t         n_ids,
                                        int64_t         num_tokens,
                                        int64_t         ne11,
                                        int64_t         ids_nb0,
                                        int64_t         ids_nb1,
                                        int64_t         in_nb11,
                                        int64_t         in_nb12,
                                        int64_t         out_nb1,
                                        int64_t         out_nb2,
                                        sycl::queue &   stream) {
    const int hidden_dim_blocks = ncols / MOE_QK_MXFP4;

    // Dispatch based on common hidden dimensions
    if (hidden_dim_blocks == 90) {  // 2880 / 32 (GPT-OSS)
        launch_persistent_moe_mxfp4_impl<90>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows,
                                             n_ids, num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1,
                                             out_nb2, stream);
    } else if (hidden_dim_blocks == 128) {  // 4096 / 32
        launch_persistent_moe_mxfp4_impl<128>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows,
                                              n_ids, num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1,
                                              out_nb2, stream);
    } else {
        // Generic fallback
        launch_persistent_moe_mxfp4_impl<0>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows,
                                            n_ids, num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1,
                                            out_nb2, stream);
    }
}

// Check if persistent kernel should be used (controlled by env var)
// NOTE: Disabled by default - benchmarking showed no benefit over parallel kernel
// (SLM caching doesn't help when each token has different input in pp)
inline bool use_persistent_moe_kernel() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_SYCL_MOE_PERSISTENT");
        enabled          = (env && atoi(env) == 1) ? 1 : 0;  // Default OFF
    }
    return enabled != 0;
}

// =============================================================================
// Launch Functions
// =============================================================================

inline bool fused_moe_esimd_available() {
    return true;
}

// Launch fused MoE kernel for Q8_0 weights
static void launch_fused_moe_q8_0(const void *    expert_weights,
                                  const void *    input,
                                  const int32_t * expert_ids,
                                  float *         output,
                                  int64_t         stride_expert,
                                  int64_t         ncols,
                                  int64_t         nrows,
                                  int64_t         n_ids,
                                  int64_t         num_tokens,
                                  int64_t         ne11,
                                  int64_t         ids_nb0,
                                  int64_t         ids_nb1,
                                  int64_t         in_nb11,
                                  int64_t         in_nb12,
                                  int64_t         out_nb1,
                                  int64_t         out_nb2,
                                  sycl::queue &   stream) {
    const int64_t total_batches     = num_tokens * n_ids;
    const int     hidden_dim_blocks = ncols / MOE_QK8_0;

    sycl::range<3> grid(total_batches, 1, nrows);
    sycl::range<3> block(1, 1, MOE_WG_SIZE);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(
                                                                     32)]] {
            // Dispatch based on hidden dimension
            // Common sizes: 2880 (GPT-OSS), 4096 (Llama), etc.
            if (hidden_dim_blocks == 90) {  // 2880 / 32
                fused_moe_q8_0_kernel<90>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows, n_ids,
                                          num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
            } else if (hidden_dim_blocks == 128) {  // 4096 / 32
                fused_moe_q8_0_kernel<128>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows,
                                           n_ids, num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1,
                                           out_nb2, item);
            } else {
                // Generic fallback - less optimized
                fused_moe_q8_0_kernel<0>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows, n_ids,
                                         num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, item);
            }
        });
    });
}

// Helper to launch MXFP4 DP4A kernel with specific template parameter.
// Quantizes F32 input to Q8_1 before launching the DP4A kernel.
// Falls back to original scalar kernel during graph recording (malloc_device is incompatible).
template <int HIDDEN_DIM_BLOCKS>
static void launch_fused_moe_mxfp4_impl(const void *    expert_weights,
                                        const float *   input,
                                        const int32_t * expert_ids,
                                        float *         output,
                                        int64_t         stride_expert,
                                        int64_t         ncols,
                                        int64_t         nrows,
                                        int64_t         n_ids,
                                        int64_t         num_tokens,
                                        int64_t         ne11,
                                        int64_t         ids_nb0,
                                        int64_t         ids_nb1,
                                        int64_t         in_nb11,
                                        int64_t         in_nb12,
                                        int64_t         out_nb1,
                                        int64_t         out_nb2,
                                        sycl::queue &   stream) {
    const int64_t total_batches = num_tokens * n_ids;
    sycl::range<3> grid(total_batches, 1, nrows);
    sycl::range<3> block(1, 1, MOE_WG_SIZE);

    // During graph recording, sycl::malloc_device and host_task are forbidden.
    // Fall back to the original scalar kernel which operates directly on F32 input.
    if (g_ggml_sycl_graph_recording) {
        stream.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                    fused_moe_mxfp4_kernel<HIDDEN_DIM_BLOCKS>(expert_weights, input, expert_ids, output, stride_expert,
                                                              ncols, nrows, n_ids, num_tokens, ne11, ids_nb0, ids_nb1,
                                                              in_nb11, in_nb12, out_nb1, out_nb2, item);
                });
        });
        return;
    }

    // --- Q8_1 quantization of F32 input (amortized across all experts) ---
    const int64_t ncols_padded    = GGML_PAD(ncols, QK8_1);
    const int64_t q8_1_row_size   = ncols_padded * sizeof(block_q8_1) / QK8_1;
    const int64_t total_src1_rows = ne11 * num_tokens;
    const size_t  q8_buf_size     = total_src1_rows * q8_1_row_size;

    // Allocate device buffer for Q8_1 quantized input
    void * q8_buffer = sycl::malloc_device(q8_buf_size, stream);

    // Quantize F32 input to Q8_1 (AoS layout)
    quantize_row_q8_1_sycl<quantize_q8_1>(input, (char *) q8_buffer, ncols, total_src1_rows, ncols_padded, &stream);

    // Q8_1 strides (matching the original F32 input layout)
    const int64_t q8_row_stride   = q8_1_row_size;
    const int64_t q8_batch_stride = ne11 * q8_1_row_size;

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                fused_moe_mxfp4_dp4a_kernel<HIDDEN_DIM_BLOCKS>(expert_weights, q8_buffer, expert_ids, output,
                                                                stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                                                                ids_nb0, ids_nb1, q8_row_stride, q8_batch_stride,
                                                                out_nb1, out_nb2, item);
            });
    });

    // Free Q8_1 buffer after kernel completes (host_task runs after all prior submissions)
    stream.submit([&](sycl::handler & cgh) {
        cgh.host_task([=, &stream]() { sycl::free(q8_buffer, stream); });
    });
}

// Launch fused MoE kernel for MXFP4 weights
static void launch_fused_moe_mxfp4(const void *    expert_weights,
                                   const float *   input,
                                   const int32_t * expert_ids,
                                   float *         output,
                                   int64_t         stride_expert,
                                   int64_t         ncols,
                                   int64_t         nrows,
                                   int64_t         n_ids,
                                   int64_t         num_tokens,
                                   int64_t         ne11,
                                   int64_t         ids_nb0,
                                   int64_t         ids_nb1,
                                   int64_t         in_nb11,
                                   int64_t         in_nb12,
                                   int64_t         out_nb1,
                                   int64_t         out_nb2,
                                   sycl::queue &   stream) {
    const int hidden_dim_blocks = ncols / MOE_QK_MXFP4;

    // Dispatch at host side to avoid runtime branching in kernel
    if (hidden_dim_blocks == 90) {  // 2880 / 32
        launch_fused_moe_mxfp4_impl<90>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows, n_ids,
                                        num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, stream);
    } else if (hidden_dim_blocks == 128) {  // 4096 / 32
        launch_fused_moe_mxfp4_impl<128>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows, n_ids,
                                         num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2,
                                         stream);
    } else {
        // Generic fallback - uses dynamic block count via template parameter 0
        launch_fused_moe_mxfp4_impl<0>(expert_weights, input, expert_ids, output, stride_expert, ncols, nrows, n_ids,
                                       num_tokens, ne11, ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2, stream);
    }
}

// Launch fused MoE kernel for MXFP4 weights in SoA (reordered) layout
// After reorder_qw_mxfp4: [all qs for all experts][all scales for all experts]
static void launch_fused_moe_mxfp4_soa(const void *    expert_weights,  // Base pointer to reordered tensor
                                       const float *   input,
                                       const int32_t * expert_ids,
                                       float *         output,
                                       int64_t         total_qs_size,  // Size of qs region = (ncols / 2) * total_rows
                                       int64_t         ncols,          // Hidden dimension
                                       int64_t         nrows_per_expert,  // Rows per expert (not total rows)
                                       int64_t         n_ids,
                                       int64_t         num_tokens,
                                       int64_t         ne11,
                                       int64_t         ids_nb0,
                                       int64_t         ids_nb1,
                                       int64_t         in_nb11,
                                       int64_t         in_nb12,
                                       int64_t         out_nb1,
                                       int64_t         out_nb2,
                                       sycl::queue &   stream) {
    const int     hidden_dim_blocks = ncols / MOE_QK_MXFP4;
    const int64_t total_batches     = num_tokens * n_ids;

    // SoA layout: [all qs] [all scales]
    const uint8_t * weights_qs     = (const uint8_t *) expert_weights;
    const uint8_t * weights_scales = weights_qs + total_qs_size;

    sycl::range<3> block(1, 1, MOE_WG_SIZE);
    sycl::range<3> grid(total_batches, 1, nrows_per_expert);

    // Dispatch based on hidden_dim_blocks (compile-time optimization)
    if (hidden_dim_blocks == 90) {  // 2880 / 32
        stream.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                    fused_moe_mxfp4_soa_kernel<90>(item, weights_qs, weights_scales, input, expert_ids, output,
                                                   nrows_per_expert, ncols, n_ids, num_tokens, ne11, ids_nb0, ids_nb1,
                                                   in_nb11, in_nb12, out_nb1, out_nb2);
                });
        });
    } else if (hidden_dim_blocks == 128) {  // 4096 / 32
        stream.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                    fused_moe_mxfp4_soa_kernel<128>(item, weights_qs, weights_scales, input, expert_ids, output,
                                                    nrows_per_expert, ncols, n_ids, num_tokens, ne11, ids_nb0, ids_nb1,
                                                    in_nb11, in_nb12, out_nb1, out_nb2);
                });
        });
    } else {
        // Fallback for other hidden dimensions - use runtime parameter
        stream.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=, hdb = hidden_dim_blocks](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                                 fused_moe_mxfp4_soa_kernel<0>(item, weights_qs, weights_scales, input, expert_ids,
                                                               output, nrows_per_expert, ncols, n_ids, num_tokens, ne11,
                                                               ids_nb0, ids_nb1, in_nb11, in_nb12, out_nb1, out_nb2);
                             });
        });
    }
}

#else   // !SYCL_ESIMD_MOE_AVAILABLE

inline bool fused_moe_esimd_available() {
    return false;
}

static void launch_fused_moe_q8_0(const void *,
                                  const void *,
                                  const int32_t *,
                                  float *,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  int64_t,
                                  sycl::queue &) {
    GGML_ASSERT(false && "ESIMD MoE not available");
}

static void launch_fused_moe_mxfp4(const void *,
                                   const float *,
                                   const int32_t *,
                                   float *,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   sycl::queue &) {
    GGML_ASSERT(false && "ESIMD MoE not available");
}

static void launch_fused_moe_mxfp4_soa(const void *,
                                       const float *,
                                       const int32_t *,
                                       float *,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       sycl::queue &) {
    GGML_ASSERT(false && "ESIMD MoE SoA not available");
}

#endif  // SYCL_ESIMD_MOE_AVAILABLE

#endif  // GGML_SYCL_FUSED_MOE_ESIMD_HPP
