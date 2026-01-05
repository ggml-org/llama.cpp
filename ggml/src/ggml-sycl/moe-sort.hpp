//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// moe-sort.hpp - GPU-side token sorting for MoE kernels
#ifndef GGML_SYCL_MOE_SORT_HPP
#define GGML_SYCL_MOE_SORT_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Stores original position for scatter-back after GEMM
struct MoETokenMapping {
    int32_t original_idx;  // Original token index
    int32_t expert_idx;    // Which expert this goes to
};

// Per-expert batch info
struct MoEExpertBatch {
    int32_t offset;  // Start index in sorted buffer
    int32_t count;   // Number of tokens for this expert
};

// Convert F32 tokens to F16 for XMX processing
// Handles 2D non-contiguous token layouts (ne11 x n_tokens rows)
// Input layout: [in_dim, ne11, n_tokens] with strides [nb0, nb1, nb2]
// Output layout: [n_input_rows, hidden_dim] contiguous, where n_input_rows = ne11 * n_tokens
// Row mapping: output row r -> input at (r % ne11) * nb1 + (r / ne11) * nb2
inline void moe_convert_f32_to_f16(const char *  tokens_f32,  // Raw byte pointer to input tensor data
                                   sycl::half *  tokens_f16,  // [n_input_rows, hidden_dim] F16 output (contiguous)
                                   int64_t       n_tokens,
                                   int64_t       hidden_dim,
                                   int64_t       ne11,  // Broadcast dimension (1 or n_ids)
                                   int64_t       nb1,   // Byte stride between id slots (src1->nb[1])
                                   int64_t       nb2,   // Byte stride between tokens (src1->nb[2])
                                   sycl::queue & queue) {
    constexpr int SG_SIZE        = 16;
    int64_t       n_input_rows   = ne11 * n_tokens;
    int64_t       total_elements = n_input_rows * hidden_dim;

    queue
        .parallel_for(sycl::nd_range<1>(((total_elements + SG_SIZE - 1) / SG_SIZE) * SG_SIZE, SG_SIZE),
                      [=](sycl::nd_item<1> item) {
                          int64_t idx = item.get_global_id(0);
                          if (idx < total_elements) {
                              int64_t       row_idx   = idx / hidden_dim;
                              int64_t       dim_idx   = idx % hidden_dim;
                              // Decompose row index into token and id_slot
                              int64_t       token_idx = row_idx / ne11;
                              int64_t       id_slot   = row_idx % ne11;
                              // Access input using 2D byte strides
                              const float * input_row =
                                  reinterpret_cast<const float *>(tokens_f32 + token_idx * nb2 + id_slot * nb1);
                              tokens_f16[idx] = sycl::half(input_row[dim_idx]);
                          }
                      })
        .wait();
}

// Sort tokens by expert ID for efficient batched GEMM
// Returns total number of (token, expert) pairs processed
template <int MAX_EXPERTS = 64>
void moe_count_tokens_per_expert(const int32_t * expert_ids,     // [n_tokens * n_ids] expert assignments
                                 int32_t *       expert_counts,  // [MAX_EXPERTS] output counts
                                 int64_t         n_tokens,
                                 int64_t         n_ids,
                                 sycl::queue &   queue) {
    // Zero counts
    queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t)).wait();

    // Parallel histogram
    queue
        .parallel_for(sycl::range<1>(n_tokens * n_ids),
                      [=](sycl::id<1> idx) {
                          int expert = expert_ids[idx];
                          if (expert >= 0 && expert < MAX_EXPERTS) {
                              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                               sycl::access::address_space::global_space>(expert_counts[expert])
                                  .fetch_add(1);
                          }
                      })
        .wait();
}

// Exclusive prefix sum to compute write offsets
// Output: expert_offsets[i] = sum of counts[0..i-1] (exclusive prefix sum)
//         expert_offsets[n_experts] = total_pairs (sum of all counts)
// This allows computing expert token range as [offsets[e], offsets[e+1])
inline void moe_compute_expert_offsets(
    const int32_t * expert_counts,   // [n_experts] input counts
    int32_t *       expert_offsets,  // [n_experts + 1] output offsets (includes total at end)
    int64_t         n_experts,
    sycl::queue &   queue) {
    // Simple sequential scan on host for now
    // TODO: GPU parallel scan for large n_experts
    std::vector<int32_t> counts(n_experts);
    std::vector<int32_t> offsets(n_experts + 1);  // +1 for total at end

    queue.memcpy(counts.data(), expert_counts, n_experts * sizeof(int32_t)).wait();

    int32_t sum = 0;
    for (int64_t i = 0; i < n_experts; i++) {
        offsets[i] = sum;
        sum += counts[i];
    }
    offsets[n_experts] = sum;  // Store total_pairs at end for fused kernel

    queue.memcpy(expert_offsets, offsets.data(),
                 (n_experts + 1) * sizeof(int32_t))
        .wait();  // Copy all n_experts + 1 elements
}

// Gather tokens into expert-contiguous layout
//
// PRECONDITION: expert_write_pos must be initialized with expert_offsets values
//               before calling (typically via memcpy from moe_compute_expert_offsets output).
//               Each element will be atomically incremented as tokens are written.
//
// Input layout: tokens_in[n_tokens * ne11, hidden_dim] - pre-converted F16 rows
// Row mapping: For pair (token_idx, id_slot), input row is token_idx * ne11 + (id_slot % ne11)
// This matches ESIMD's i11 = id_idx % ne11 broadcast pattern
template <typename T>
void moe_sort_tokens_by_expert(const T *         tokens_in,         // [n_tokens * ne11, hidden_dim] pre-converted rows
                               T *               tokens_sorted,     // [total_pairs, hidden_dim]
                               const int32_t *   expert_ids,        // [n_tokens * n_ids]
                               int32_t *         expert_write_pos,  // [n_experts] atomic write positions
                               MoETokenMapping * token_map,         // [total_pairs] for scatter-back
                               int64_t           n_tokens,
                               int64_t           n_ids,
                               int64_t           ne11,  // Broadcast dimension (1 or n_ids)
                               int64_t           hidden_dim,
                               int64_t           n_experts,
                               sycl::queue &     queue) {
    // Each work-item handles one (token, expert_slot) pair
    queue
        .parallel_for(sycl::range<1>(n_tokens * n_ids),
                      [=](sycl::id<1> idx) {
                          int64_t token_idx = idx / n_ids;
                          int64_t id_slot   = idx % n_ids;
                          int     expert    = expert_ids[idx];

                          if (expert < 0 || expert >= n_experts) {
                              return;
                          }

                          // Atomically claim a slot for this expert
                          int32_t write_pos =
                              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                               sycl::access::address_space::global_space>(expert_write_pos[expert])
                                  .fetch_add(1);

                          // Compute input row index with broadcast (matches ESIMD's i11 = id_idx % ne11)
                          int64_t input_row = token_idx * ne11 + (id_slot % ne11);

                          // Copy token data
                          for (int64_t d = 0; d < hidden_dim; d++) {
                              tokens_sorted[write_pos * hidden_dim + d] = tokens_in[input_row * hidden_dim + d];
                          }

                          // Record mapping for scatter-back
                          token_map[write_pos].original_idx = static_cast<int32_t>(idx);
                          token_map[write_pos].expert_idx   = expert;
                      })
        .wait();
}

// Scatter results back to original positions
template <typename T>
void moe_scatter_results(const T *               sorted_output,  // [total_pairs, output_dim]
                         T *                     final_output,   // [n_tokens * n_ids, output_dim]
                         const MoETokenMapping * token_map,
                         int64_t                 total_pairs,
                         int64_t                 output_dim,
                         sycl::queue &           queue) {
    queue
        .parallel_for(sycl::range<1>(total_pairs),
                      [=](sycl::id<1> idx) {
                          int32_t original_pos = token_map[idx].original_idx;

                          for (int64_t d = 0; d < output_dim; d++) {
                              final_output[original_pos * output_dim + d] = sorted_output[idx * output_dim + d];
                          }
                      })
        .wait();
}

// Scatter results back with F16→F32 conversion
// Use when sorted output is F16 but final output must be F32
// Uses byte strides to handle non-contiguous output tensor layouts
inline void moe_scatter_results_f16_to_f32(const sycl::half * sorted_output,  // [total_pairs, output_dim] F16
                                           char *             final_output,  // Output tensor with byte-addressed layout
                                           const MoETokenMapping * token_map,
                                           int64_t                 total_pairs,
                                           int64_t                 output_dim,
                                           int64_t                 n_ids,  // Number of expert slots per token
                                           int64_t       out_nb1,          // Byte stride between id slots (dst->nb[1])
                                           int64_t       out_nb2,          // Byte stride between tokens (dst->nb[2])
                                           sycl::queue & queue) {
    queue
        .parallel_for(sycl::range<1>(total_pairs),
                      [=](sycl::id<1> idx) {
                          int32_t original_pos = token_map[idx].original_idx;

                          // Decompose original_pos into token and id_slot indices
                          int32_t token_idx = original_pos / n_ids;
                          int32_t id_slot   = original_pos % n_ids;

                          // Calculate output pointer using byte strides (ESIMD-compatible layout)
                          float * out_ptr =
                              reinterpret_cast<float *>(final_output + token_idx * out_nb2 + id_slot * out_nb1);

                          for (int64_t d = 0; d < output_dim; d++) {
                              out_ptr[d] = static_cast<float>(sorted_output[idx * output_dim + d]);
                          }
                      })
        .wait();
}

#endif  // GGML_SYCL_MOE_SORT_HPP
