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

// Sort tokens by expert ID for efficient batched GEMM
// Returns total number of (token, expert) pairs processed
template<int MAX_EXPERTS = 64>
void moe_count_tokens_per_expert(
    const int32_t* expert_ids,    // [n_tokens * n_ids] expert assignments
    int32_t* expert_counts,       // [MAX_EXPERTS] output counts
    int64_t n_tokens,
    int64_t n_ids,
    sycl::queue& queue)
{
    // Zero counts
    queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t)).wait();

    // Parallel histogram
    queue.parallel_for(
        sycl::range<1>(n_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int expert = expert_ids[idx];
            if (expert >= 0 && expert < MAX_EXPERTS) {
                sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(
                        expert_counts[expert]).fetch_add(1);
            }
        }).wait();
}

// Exclusive prefix sum to compute write offsets
inline void moe_compute_expert_offsets(
    const int32_t* expert_counts,  // [n_experts] input counts
    int32_t* expert_offsets,       // [n_experts] output offsets
    int64_t n_experts,
    sycl::queue& queue)
{
    // Simple sequential scan on host for now
    // TODO: GPU parallel scan for large n_experts
    std::vector<int32_t> counts(n_experts);
    std::vector<int32_t> offsets(n_experts);

    queue.memcpy(counts.data(), expert_counts,
                 n_experts * sizeof(int32_t)).wait();

    int32_t sum = 0;
    for (int64_t i = 0; i < n_experts; i++) {
        offsets[i] = sum;
        sum += counts[i];
    }

    queue.memcpy(expert_offsets, offsets.data(),
                 n_experts * sizeof(int32_t)).wait();
}

// Gather tokens into expert-contiguous layout
//
// PRECONDITION: expert_write_pos must be initialized with expert_offsets values
//               before calling (typically via memcpy from moe_compute_expert_offsets output).
//               Each element will be atomically incremented as tokens are written.
template<typename T>
void moe_sort_tokens_by_expert(
    const T* tokens_in,           // [n_tokens, hidden_dim]
    T* tokens_sorted,             // [total_pairs, hidden_dim]
    const int32_t* expert_ids,    // [n_tokens * n_ids]
    int32_t* expert_write_pos,    // [n_experts] atomic write positions
    MoETokenMapping* token_map,   // [total_pairs] for scatter-back
    int64_t n_tokens,
    int64_t n_ids,
    int64_t hidden_dim,
    int64_t n_experts,
    sycl::queue& queue)
{
    // Each work-item handles one (token, expert_slot) pair
    queue.parallel_for(
        sycl::range<1>(n_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int64_t token_idx = idx / n_ids;
            int expert = expert_ids[idx];

            if (expert < 0 || expert >= n_experts) return;

            // Atomically claim a slot for this expert
            int32_t write_pos = sycl::atomic_ref<int32_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(
                    expert_write_pos[expert]).fetch_add(1);

            // Copy token data
            for (int64_t d = 0; d < hidden_dim; d++) {
                tokens_sorted[write_pos * hidden_dim + d] =
                    tokens_in[token_idx * hidden_dim + d];
            }

            // Record mapping for scatter-back
            token_map[write_pos].original_idx = static_cast<int32_t>(idx);
            token_map[write_pos].expert_idx = expert;
        }).wait();
}

// Scatter results back to original positions
template<typename T>
void moe_scatter_results(
    const T* sorted_output,       // [total_pairs, output_dim]
    T* final_output,              // [n_tokens * n_ids, output_dim]
    const MoETokenMapping* token_map,
    int64_t total_pairs,
    int64_t output_dim,
    sycl::queue& queue)
{
    queue.parallel_for(
        sycl::range<1>(total_pairs),
        [=](sycl::id<1> idx) {
            int32_t original_pos = token_map[idx].original_idx;

            for (int64_t d = 0; d < output_dim; d++) {
                final_output[original_pos * output_dim + d] =
                    sorted_output[idx * output_dim + d];
            }
        }).wait();
}

#endif // GGML_SYCL_MOE_SORT_HPP
