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
static_assert(sizeof(MoETokenMapping) == kMoETokenMappingBytes,
              "MoETokenMapping size changed; update kMoETokenMappingBytes in common.hpp");

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
void moe_count_tokens_per_expert(const char *  ids_base,       // Raw ids base pointer
                                 size_t        ids_nb0,        // Byte stride between id slots
                                 size_t        ids_nb1,        // Byte stride between tokens
                                 int32_t *     expert_counts,  // [MAX_EXPERTS] output counts
                                 int64_t       n_tokens,
                                 int64_t       n_ids,
                                 sycl::queue & queue) {
    // Zero counts
    queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t)).wait();

    // Parallel histogram
    queue
        .parallel_for(sycl::range<1>(n_tokens * n_ids),
                      [=](sycl::id<1> idx) {
                          const int64_t token_idx = idx[0] / n_ids;
                          const int64_t id_slot   = idx[0] % n_ids;
                          const int32_t * id_ptr =
                              reinterpret_cast<const int32_t *>(ids_base + token_idx * ids_nb1 + id_slot * ids_nb0);
                          int expert = *id_ptr;
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
                               const char *      ids_base,          // Raw ids base pointer
                               size_t            ids_nb0,           // Byte stride between id slots
                               size_t            ids_nb1,           // Byte stride between tokens
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
                          const int32_t * id_ptr =
                              reinterpret_cast<const int32_t *>(ids_base + token_idx * ids_nb1 + id_slot * ids_nb0);
                          int     expert    = *id_ptr;

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

// ============================================================================
// Graph-compatible async versions (no .wait() calls)
// These functions return sycl::event for dependency chaining
// ============================================================================

// Async F32 to F16 conversion - returns event for chaining
inline sycl::event moe_convert_f32_to_f16_async(const char *  tokens_f32,
                                                 sycl::half *  tokens_f16,
                                                 int64_t       n_tokens,
                                                 int64_t       hidden_dim,
                                                 int64_t       ne11,
                                                 int64_t       nb1,
                                                 int64_t       nb2,
                                                 sycl::queue & queue) {
    constexpr int SG_SIZE        = 16;
    int64_t       n_input_rows   = ne11 * n_tokens;
    int64_t       total_elements = n_input_rows * hidden_dim;

    return queue.parallel_for(sycl::nd_range<1>(((total_elements + SG_SIZE - 1) / SG_SIZE) * SG_SIZE, SG_SIZE),
                              [=](sycl::nd_item<1> item) {
                                  int64_t idx = item.get_global_id(0);
                                  if (idx < total_elements) {
                                      int64_t       row_idx   = idx / hidden_dim;
                                      int64_t       dim_idx   = idx % hidden_dim;
                                      int64_t       token_idx = row_idx / ne11;
                                      int64_t       id_slot   = row_idx % ne11;
                                      const float * input_row =
                                          reinterpret_cast<const float *>(tokens_f32 + token_idx * nb2 + id_slot * nb1);
                                      tokens_f16[idx] = sycl::half(input_row[dim_idx]);
                                  }
                              });
}

// Async token counting - returns event for chaining
template <int MAX_EXPERTS = 64>
sycl::event moe_count_tokens_per_expert_async(const char *  ids_base,
                                               size_t        ids_nb0,
                                               size_t        ids_nb1,
                                               int32_t *     expert_counts,
                                               int64_t       n_tokens,
                                               int64_t       n_ids,
                                               sycl::queue & queue,
                                               sycl::event   dep_event = {}) {
    // Zero counts (with dependency on previous event if provided)
    sycl::event memset_event;
    if (ggml_sycl_should_add_dependency(dep_event)) {
        memset_event = queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t), { dep_event });
    } else {
        memset_event = queue.memset(expert_counts, 0, MAX_EXPERTS * sizeof(int32_t));
    }

    // Parallel histogram
    return queue.submit([&](sycl::handler & cgh) {
        cgh.depends_on(memset_event);
        cgh.parallel_for(sycl::range<1>(n_tokens * n_ids), [=](sycl::id<1> idx) {
            const int64_t token_idx = idx[0] / n_ids;
            const int64_t id_slot   = idx[0] % n_ids;
            const int32_t * id_ptr =
                reinterpret_cast<const int32_t *>(ids_base + token_idx * ids_nb1 + id_slot * ids_nb0);
            int expert = *id_ptr;
            if (expert >= 0 && expert < MAX_EXPERTS) {
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(expert_counts[expert])
                    .fetch_add(1);
            }
        });
    });
}

// GPU-side exclusive prefix sum (parallel scan) for small arrays
// Works well for n_experts <= 64 (typical MoE size)
inline sycl::event moe_compute_expert_offsets_gpu(const int32_t * expert_counts,
                                                   int32_t *       expert_offsets,
                                                   int64_t         n_experts,
                                                   sycl::queue &   queue,
                                                   sycl::event     dep_event = {}) {
    // For small n_experts, use single work-group prefix sum
    // Output: expert_offsets[i] = sum of counts[0..i-1] (exclusive)
    //         expert_offsets[n_experts] = total (inclusive sum)
    return queue.submit([&](sycl::handler & cgh) {
        if (ggml_sycl_should_add_dependency(dep_event)) {
            cgh.depends_on(dep_event);
        }

        // Use single work-group with local memory for prefix sum
        constexpr int WG_SIZE = 64;  // Must be >= n_experts
        sycl::local_accessor<int32_t, 1> local_data(WG_SIZE * 2, cgh);

        cgh.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), [=](sycl::nd_item<1> item) {
            int tid = item.get_local_id(0);

            // Load data into local memory (with bounds check)
            local_data[tid]            = (tid < n_experts) ? expert_counts[tid] : 0;
            local_data[tid + WG_SIZE]  = 0;
            item.barrier(sycl::access::fence_space::local_space);

            // Hillis-Steele parallel inclusive scan
            int offset = 1;
            for (int d = WG_SIZE; d > 0; d >>= 1) {
                if (tid < d && tid >= offset) {
                    local_data[tid + WG_SIZE] = local_data[tid - offset] + local_data[tid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Swap buffers
                if (tid < WG_SIZE) {
                    int temp                  = local_data[tid];
                    local_data[tid]           = local_data[tid + WG_SIZE];
                    local_data[tid + WG_SIZE] = temp;
                }
                item.barrier(sycl::access::fence_space::local_space);
                offset <<= 1;
            }

            // Convert inclusive scan to exclusive scan and write output
            // exclusive[i] = inclusive[i-1], exclusive[0] = 0
            if (tid == 0) {
                expert_offsets[0] = 0;
            }
            if (tid < n_experts - 1) {
                expert_offsets[tid + 1] = local_data[tid];
            }
            // Write total at the end
            if (tid == n_experts - 1) {
                expert_offsets[n_experts] = local_data[tid];
            }
        });
    });
}

// Simpler GPU-side prefix sum using sequential single-thread (reliable for n_experts <= 64)
inline sycl::event moe_compute_expert_offsets_gpu_simple(const int32_t * expert_counts,
                                                          int32_t *       expert_offsets,
                                                          int64_t         n_experts,
                                                          sycl::queue &   queue,
                                                          sycl::event     dep_event = {}) {
    return queue.submit([&](sycl::handler & cgh) {
        if (ggml_sycl_should_add_dependency(dep_event)) {
            cgh.depends_on(dep_event);
        }

        // Single work-item does sequential prefix sum (simple and correct for small n_experts)
        cgh.single_task([=]() {
            int32_t sum       = 0;
            expert_offsets[0] = 0;
            for (int64_t i = 0; i < n_experts; i++) {
                sum += expert_counts[i];
                expert_offsets[i + 1] = sum;
            }
        });
    });
}

// Async token sorting - returns event for chaining
template <typename T>
sycl::event moe_sort_tokens_by_expert_async(const T *         tokens_in,
                                             T *               tokens_sorted,
                                             const char *      ids_base,
                                             size_t            ids_nb0,
                                             size_t            ids_nb1,
                                             int32_t *         expert_write_pos,
                                             MoETokenMapping * token_map,
                                             int64_t           n_tokens,
                                             int64_t           n_ids,
                                             int64_t           ne11,
                                             int64_t           hidden_dim,
                                             int64_t           n_experts,
                                             sycl::queue &     queue,
                                             sycl::event       dep_event = {}) {
    return queue.submit([&](sycl::handler & cgh) {
        if (ggml_sycl_should_add_dependency(dep_event)) {
            cgh.depends_on(dep_event);
        }

        cgh.parallel_for(sycl::range<1>(n_tokens * n_ids), [=](sycl::id<1> idx) {
            int64_t token_idx = idx / n_ids;
            int64_t id_slot   = idx % n_ids;
            const int32_t * id_ptr =
                reinterpret_cast<const int32_t *>(ids_base + token_idx * ids_nb1 + id_slot * ids_nb0);
            int     expert    = *id_ptr;

            if (expert < 0 || expert >= n_experts) {
                return;
            }

            // Atomically claim a slot for this expert
            int32_t write_pos = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(expert_write_pos[expert])
                                    .fetch_add(1);

            // Compute input row index with broadcast
            int64_t input_row = token_idx * ne11 + (id_slot % ne11);

            // Copy token data
            for (int64_t d = 0; d < hidden_dim; d++) {
                tokens_sorted[write_pos * hidden_dim + d] = tokens_in[input_row * hidden_dim + d];
            }

            // Record mapping for scatter-back
            token_map[write_pos].original_idx = static_cast<int32_t>(idx);
            token_map[write_pos].expert_idx   = expert;
        });
    });
}

// Async scatter results with F16->F32 conversion - returns event for chaining
inline sycl::event moe_scatter_results_f16_to_f32_async(const sycl::half *      sorted_output,
                                                         char *                  final_output,
                                                         const MoETokenMapping * token_map,
                                                         int64_t                 total_pairs,
                                                         int64_t                 output_dim,
                                                         int64_t                 n_ids,
                                                         int64_t                 out_nb1,
                                                         int64_t                 out_nb2,
                                                         sycl::queue &           queue,
                                                         sycl::event             dep_event = {}) {
    return queue.submit([&](sycl::handler & cgh) {
        if (ggml_sycl_should_add_dependency(dep_event)) {
            cgh.depends_on(dep_event);
        }

        cgh.parallel_for(sycl::range<1>(total_pairs), [=](sycl::id<1> idx) {
            int32_t original_pos = token_map[idx].original_idx;
            int32_t token_idx    = original_pos / n_ids;
            int32_t id_slot      = original_pos % n_ids;

            float * out_ptr = reinterpret_cast<float *>(final_output + token_idx * out_nb2 + id_slot * out_nb1);

            for (int64_t d = 0; d < output_dim; d++) {
                out_ptr[d] = static_cast<float>(sorted_output[idx * output_dim + d]);
            }
        });
    });
}

// Compute tile mapping for fused XMX kernel
// Converts expert counts to tile counts and computes cumulative tile offsets
// This enables work-groups to self-assign to experts via binary search
//
// Output:
//   expert_tile_offsets[i] = cumulative tiles for experts 0..i-1
//   expert_tile_offsets[n_experts] = total_tiles (for grid launch size)
//
// Formula: tiles_for_expert[e] = ceil(expert_counts[e] / tile_M)
inline sycl::event moe_compute_tile_mapping(
    const int32_t * expert_counts,       // [n_experts] token counts per expert
    int32_t *       expert_tile_offsets, // [n_experts + 1] output tile offsets
    int32_t *       total_tiles_out,     // [1] output: total tiles for grid launch
    int64_t         n_experts,
    int64_t         tile_M,              // XMX tile size in M dimension (typically 32)
    sycl::queue &   queue,
    sycl::event     dep_event = {}) {

    return queue.submit([&](sycl::handler & cgh) {
        if (ggml_sycl_should_add_dependency(dep_event)) {
            cgh.depends_on(dep_event);
        }

        // Single work-item computes prefix sum of tile counts
        cgh.single_task([=]() {
            int32_t cumulative = 0;
            expert_tile_offsets[0] = 0;

            for (int64_t e = 0; e < n_experts; e++) {
                int32_t count = expert_counts[e];
                int32_t tiles = (count + tile_M - 1) / tile_M;  // ceil division
                cumulative += tiles;
                expert_tile_offsets[e + 1] = cumulative;
            }

            *total_tiles_out = cumulative;
        });
    });
}

#endif  // GGML_SYCL_MOE_SORT_HPP
