//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_V2_PARTITION_HPP
#define GGML_SYCL_FATTN_V2_PARTITION_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>

// =============================================================================
// Paged Attention V2 - Multi-Partition Architecture
// =============================================================================
//
// For long sequences (>PARTITION_SIZE tokens), we split attention computation
// across multiple partitions. Each partition computes partial attention over
// PARTITION_SIZE KV positions and stores:
//   - exp_sum: sum of softmax exponentials for this partition
//   - max_logit: maximum attention score in this partition
//   - tmp_out: weighted sum of V (unnormalized) for this partition
//
// A separate reduction kernel then:
//   1. Finds global max across all partitions
//   2. Rescales exp_sums using the global max
//   3. Computes weighted average of partial outputs
//
// This enables O(n) memory complexity for O(n²) attention!
//
// Memory layout:
//   K/V:        [n_kv, num_kv_heads, head_size] (3D contiguous, block_tables for logical addressing)
//   exp_sums:   [num_seqs, num_heads, max_num_partitions]
//   max_logits: [num_seqs, num_heads, max_num_partitions]
//   tmp_out:    [num_seqs, num_heads, max_num_partitions, head_size]
//
// The block_tables map logical block indices to physical positions in the contiguous K/V cache.
// This enables efficient paged attention without requiring a 4D physical memory layout.
//

// =============================================================================
// V2 Configuration
// =============================================================================

// Partition size - number of KV positions per partition
// 512 is optimal: large enough for efficiency, small enough for many partitions
constexpr int V2_PARTITION_SIZE = 512;

// Block size for KV cache (used in paged attention)
constexpr int V2_BLOCK_SIZE = 16;

// Work-group configuration
constexpr int V2_NTHREADS = 256;
constexpr int V2_WARP_SIZE = 32;
constexpr int V2_NUM_WARPS = V2_NTHREADS / V2_WARP_SIZE;

// =============================================================================
// Partition Kernel - Computes partial attention for one partition
// =============================================================================
//
// Grid: (num_heads, num_seqs, max_num_partitions)
// Each work-group handles one (seq, head, partition) combination
//

template <int D, int PARTITION_SIZE>
static void paged_attention_v2_partition_kernel(
    float * __restrict__ exp_sums,      // [num_seqs, num_heads, max_num_partitions]
    float * __restrict__ max_logits,    // [num_seqs, num_heads, max_num_partitions]
    float * __restrict__ tmp_out,       // [num_seqs, num_heads, max_num_partitions, D]
    const sycl::half * __restrict__ Q,  // [num_seqs, num_heads, D]
    const sycl::half * __restrict__ K,  // [n_kv, num_kv_heads, D] (3D contiguous, block_tables for logical addressing)
    const sycl::half * __restrict__ V,  // [n_kv, num_kv_heads, D] (3D contiguous, block_tables for logical addressing)
    const float scale,
    const int * __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int * __restrict__ context_lens,  // [num_seqs]
    const int num_heads,
    const int num_kv_heads,
    const int max_num_blocks_per_seq,
    const int max_num_partitions,
    const int block_size,
    const sycl::nd_item<3> & item,
    float * shared_mem) {

    const int head_idx = item.get_group(2);
    const int seq_idx = item.get_group(1);
    const int partition_idx = item.get_group(0);
    const int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    const int warp_id = tid / V2_WARP_SIZE;
    const int lane_id = tid % V2_WARP_SIZE;

    // GQA: map query head to KV head
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Check if this partition is within bounds
    const int context_len = context_lens[seq_idx];
    const int partition_start = partition_idx * PARTITION_SIZE;
    if (partition_start >= context_len) {
        // This partition is beyond the sequence length
        // Write sentinel values
        if (tid == 0) {
            const int64_t idx = seq_idx * num_heads * max_num_partitions +
                               head_idx * max_num_partitions + partition_idx;
            exp_sums[idx] = 0.0f;
            max_logits[idx] = -FLT_MAX;
        }
        return;
    }

    const int partition_end = sycl::min(partition_start + PARTITION_SIZE, context_len);
    const int num_tokens_in_partition = partition_end - partition_start;

    // Shared memory layout:
    // - Q tile: [D] floats (loaded once, reused)
    // - logits: [PARTITION_SIZE] floats (QK scores)
    // - output_acc: [D] floats (partial V accumulation)
    // - warp_reduce: [NUM_WARPS] floats
    float * q_shared = shared_mem;
    float * logits = q_shared + D;
    float * output_acc = logits + PARTITION_SIZE;
    float * warp_reduce = output_acc + D;

    // Load Q into shared memory (scaled)
    const sycl::half * q_ptr = Q + seq_idx * num_heads * D + head_idx * D;
    for (int d = tid; d < D; d += V2_NTHREADS) {
        q_shared[d] = static_cast<float>(q_ptr[d]) * scale;
    }

    // Initialize output accumulator
    for (int d = tid; d < D; d += V2_NTHREADS) {
        output_acc[d] = 0.0f;
    }

    sycl::group_barrier(item.get_group());

    // =======================================================================
    // Phase 1: Compute Q @ K^T for this partition
    // =======================================================================
    float qk_max = -FLT_MAX;

    for (int token_idx = tid; token_idx < num_tokens_in_partition; token_idx += V2_NTHREADS) {
        const int kv_pos = partition_start + token_idx;

        // Compute physical block address from block table
        const int logical_block = kv_pos / block_size;
        const int offset_in_block = kv_pos % block_size;
        const int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + logical_block];

        // K layout: [n_kv, num_kv_heads, D] (3D contiguous after permute)
        // token_pos is the physical position in the contiguous KV cache
        const int token_pos = physical_block * block_size + offset_in_block;
        const sycl::half * k_ptr = K +
            token_pos * num_kv_heads * D +
            kv_head_idx * D;

        // Compute dot product Q · K
        float qk = 0.0f;
        for (int d = 0; d < D; ++d) {
            qk += q_shared[d] * static_cast<float>(k_ptr[d]);
        }

        logits[token_idx] = qk;
        qk_max = sycl::fmax(qk_max, qk);
    }

    // Warp-level reduction for max
    #pragma unroll
    for (int mask = V2_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        qk_max = sycl::fmax(qk_max,
            sycl::permute_group_by_xor(sg, qk_max, mask));
    }

    // Store warp max to shared memory
    if (lane_id == 0) {
        warp_reduce[warp_id] = qk_max;
    }
    sycl::group_barrier(item.get_group());

    // Final reduction across warps
    if (tid < V2_NUM_WARPS) {
        qk_max = warp_reduce[tid];
    } else {
        qk_max = -FLT_MAX;
    }
    #pragma unroll
    for (int mask = V2_NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max = sycl::fmax(qk_max,
            sycl::permute_group_by_xor(sg, qk_max, mask));
    }
    qk_max = sycl::select_from_group(sg, qk_max, 0);

    sycl::group_barrier(item.get_group());

    // =======================================================================
    // Phase 2: Compute softmax and accumulate weighted V
    // =======================================================================
    float exp_sum = 0.0f;

    for (int token_idx = tid; token_idx < num_tokens_in_partition; token_idx += V2_NTHREADS) {
        const float qk = logits[token_idx];
        // Use IEEE-compliant exp instead of native::exp for determinism
        const float exp_qk = sycl::exp(qk - qk_max);
        logits[token_idx] = exp_qk;  // Store for V weighting
        exp_sum += exp_qk;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int mask = V2_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        exp_sum += sycl::permute_group_by_xor(sg, exp_sum, mask);
    }

    if (lane_id == 0) {
        warp_reduce[warp_id] = exp_sum;
    }
    sycl::group_barrier(item.get_group());

    // Final reduction across warps
    if (tid < V2_NUM_WARPS) {
        exp_sum = warp_reduce[tid];
    } else {
        exp_sum = 0.0f;
    }
    #pragma unroll
    for (int mask = V2_NUM_WARPS / 2; mask >= 1; mask /= 2) {
        exp_sum += sycl::permute_group_by_xor(sg, exp_sum, mask);
    }
    exp_sum = sycl::select_from_group(sg, exp_sum, 0);

    sycl::group_barrier(item.get_group());

    // =======================================================================
    // Phase 3: Compute weighted sum of V
    // =======================================================================
    // Each thread handles a subset of D dimensions
    // Loop over all tokens, accumulating weighted V values

    for (int d = tid; d < D; d += V2_NTHREADS) {
        float acc = 0.0f;

        for (int token_idx = 0; token_idx < num_tokens_in_partition; ++token_idx) {
            const int kv_pos = partition_start + token_idx;
            const int logical_block = kv_pos / block_size;
            const int offset_in_block = kv_pos % block_size;
            const int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + logical_block];

            // V layout: [n_kv, num_kv_heads, D] (3D contiguous after permute)
            // token_pos is the physical position in the contiguous KV cache
            const int token_pos = physical_block * block_size + offset_in_block;
            const sycl::half * v_ptr = V +
                token_pos * num_kv_heads * D +
                kv_head_idx * D;

            const float weight = logits[token_idx];
            acc += weight * static_cast<float>(v_ptr[d]);
        }

        output_acc[d] = acc;
    }

    sycl::group_barrier(item.get_group());

    // =======================================================================
    // Phase 4: Write outputs
    // =======================================================================
    const int64_t out_offset = seq_idx * num_heads * max_num_partitions * D +
                               head_idx * max_num_partitions * D +
                               partition_idx * D;

    for (int d = tid; d < D; d += V2_NTHREADS) {
        tmp_out[out_offset + d] = output_acc[d];
    }

    if (tid == 0) {
        const int64_t idx = seq_idx * num_heads * max_num_partitions +
                           head_idx * max_num_partitions + partition_idx;
        exp_sums[idx] = exp_sum;
        max_logits[idx] = qk_max;
    }
}

// =============================================================================
// Reduction Kernel - Combines results from all partitions
// =============================================================================
//
// Grid: (num_heads, num_seqs)
// Each work-group handles one (seq, head) and reduces all its partitions
//

template <int D, int PARTITION_SIZE>
static void paged_attention_v2_reduce_kernel(
    float * __restrict__ out,           // [num_seqs, num_heads, D]
    const float * __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float * __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const float * __restrict__ tmp_out,     // [num_seqs, num_heads, max_num_partitions, D]
    const int * __restrict__ context_lens,  // [num_seqs]
    const int num_heads,
    const int max_num_partitions,
    const sycl::nd_item<3> & item,
    float * shared_mem) {

    const int head_idx = item.get_group(2);
    const int seq_idx = item.get_group(1);
    const int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    const int warp_id = tid / V2_WARP_SIZE;
    const int lane_id = tid % V2_WARP_SIZE;

    const int context_len = context_lens[seq_idx];
    const int num_partitions = (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

    // If only one partition, just copy tmp_out to out
    if (num_partitions == 1) {
        const int64_t in_offset = seq_idx * num_heads * max_num_partitions * D +
                                  head_idx * max_num_partitions * D;
        const int64_t out_offset = seq_idx * num_heads * D + head_idx * D;

        for (int d = tid; d < D; d += V2_NTHREADS) {
            out[out_offset + d] = tmp_out[in_offset + d];
        }
        return;
    }

    // Shared memory layout:
    // - max_logits_shared: [num_partitions] floats
    // - exp_sums_shared: [num_partitions] floats
    // - warp_reduce: [NUM_WARPS] floats
    float * max_logits_shared = shared_mem;
    float * exp_sums_shared = max_logits_shared + num_partitions;
    float * warp_reduce = exp_sums_shared + num_partitions;

    // Load max_logits into shared memory
    const int64_t base_idx = seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions;

    for (int p = tid; p < num_partitions; p += V2_NTHREADS) {
        max_logits_shared[p] = max_logits[base_idx + p];
    }
    sycl::group_barrier(item.get_group());

    // Find global max across all partitions
    float global_max = -FLT_MAX;
    for (int p = tid; p < num_partitions; p += V2_NTHREADS) {
        global_max = sycl::fmax(global_max, max_logits_shared[p]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int mask = V2_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        global_max = sycl::fmax(global_max,
            sycl::permute_group_by_xor(sg, global_max, mask));
    }

    if (lane_id == 0) {
        warp_reduce[warp_id] = global_max;
    }
    sycl::group_barrier(item.get_group());

    // Final reduction
    if (tid < V2_NUM_WARPS) {
        global_max = warp_reduce[tid];
    } else {
        global_max = -FLT_MAX;
    }
    #pragma unroll
    for (int mask = V2_NUM_WARPS / 2; mask >= 1; mask /= 2) {
        global_max = sycl::fmax(global_max,
            sycl::permute_group_by_xor(sg, global_max, mask));
    }
    global_max = sycl::select_from_group(sg, global_max, 0);

    sycl::group_barrier(item.get_group());

    // Rescale exp_sums and compute global sum
    float global_exp_sum = 0.0f;
    for (int p = tid; p < num_partitions; p += V2_NTHREADS) {
        const float local_max = max_logits_shared[p];
        const float local_exp_sum = exp_sums[base_idx + p];
        // Use IEEE-compliant exp instead of native::exp for determinism
        const float rescaled = local_exp_sum * sycl::exp(local_max - global_max);
        exp_sums_shared[p] = rescaled;
        global_exp_sum += rescaled;
    }

    // Reduce global_exp_sum
    #pragma unroll
    for (int mask = V2_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        global_exp_sum += sycl::permute_group_by_xor(sg, global_exp_sum, mask);
    }

    if (lane_id == 0) {
        warp_reduce[warp_id] = global_exp_sum;
    }
    sycl::group_barrier(item.get_group());

    if (tid < V2_NUM_WARPS) {
        global_exp_sum = warp_reduce[tid];
    } else {
        global_exp_sum = 0.0f;
    }
    #pragma unroll
    for (int mask = V2_NUM_WARPS / 2; mask >= 1; mask /= 2) {
        global_exp_sum += sycl::permute_group_by_xor(sg, global_exp_sum, mask);
    }
    global_exp_sum = sycl::select_from_group(sg, global_exp_sum, 0);

    const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

    sycl::group_barrier(item.get_group());

    // Compute weighted average of partial outputs
    const int64_t out_offset = seq_idx * num_heads * D + head_idx * D;
    const int64_t tmp_base = seq_idx * num_heads * max_num_partitions * D +
                            head_idx * max_num_partitions * D;

    for (int d = tid; d < D; d += V2_NTHREADS) {
        float acc = 0.0f;
        for (int p = 0; p < num_partitions; ++p) {
            const float weight = exp_sums_shared[p] * inv_global_exp_sum;
            acc += tmp_out[tmp_base + p * D + d] * weight;
        }
        out[out_offset + d] = acc;
    }
}

// =============================================================================
// Launch Functions
// =============================================================================

template <int D>
void launch_paged_attention_v2(
    float * out,                 // [num_seqs, num_heads, D]
    float * exp_sums,            // [num_seqs, num_heads, max_num_partitions]
    float * max_logits,          // [num_seqs, num_heads, max_num_partitions]
    float * tmp_out,             // [num_seqs, num_heads, max_num_partitions, D]
    const sycl::half * Q,        // [num_seqs, num_heads, D]
    const sycl::half * K,        // [n_kv, num_kv_heads, D] (3D contiguous, block_tables for logical addressing)
    const sycl::half * V,        // [n_kv, num_kv_heads, D] (3D contiguous, block_tables for logical addressing)
    const float scale,
    const int * block_tables,    // [num_seqs, max_blocks_per_seq]
    const int * context_lens,    // [num_seqs]
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int max_num_blocks_per_seq,
    const int max_context_len,
    const int block_size,
    sycl::queue * stream) {

    const int max_num_partitions = (max_context_len + V2_PARTITION_SIZE - 1) / V2_PARTITION_SIZE;

    // Partition kernel - one work-group per (seq, head, partition)
    {
        constexpr size_t shared_size = D + V2_PARTITION_SIZE + D + V2_NUM_WARPS;  // floats

        sycl::range<3> grid(max_num_partitions, num_seqs, num_heads);
        sycl::range<3> block(1, 1, V2_NTHREADS);

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_acc(sycl::range<1>(shared_size), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(V2_WARP_SIZE)]] {
                    float * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                    paged_attention_v2_partition_kernel<D, V2_PARTITION_SIZE>(
                        exp_sums, max_logits, tmp_out,
                        Q, K, V, scale,
                        block_tables, context_lens,
                        num_heads, num_kv_heads,
                        max_num_blocks_per_seq, max_num_partitions,
                        block_size, item, shared);
                });
        });
    }

    // Reduction kernel - one work-group per (seq, head)
    {
        const size_t shared_size = max_num_partitions * 2 + V2_NUM_WARPS;  // floats

        sycl::range<3> grid(1, num_seqs, num_heads);
        sycl::range<3> block(1, 1, V2_NTHREADS);

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_acc(sycl::range<1>(shared_size), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(V2_WARP_SIZE)]] {
                    float * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                    paged_attention_v2_reduce_kernel<D, V2_PARTITION_SIZE>(
                        out, exp_sums, max_logits, tmp_out,
                        context_lens, num_heads, max_num_partitions,
                        item, shared);
                });
        });
    }
}

// Helper to check if V2 partition-based attention should be used
inline bool should_use_paged_attention_v2(int max_context_len) {
    // Use V2 when sequence length exceeds partition size
    // This gives significant memory savings for long sequences
    return max_context_len > V2_PARTITION_SIZE;
}

// Get memory requirements for V2 temporary buffers
inline size_t paged_attention_v2_temp_size(
    int num_seqs, int num_heads, int max_context_len, int D) {

    const int max_num_partitions = (max_context_len + V2_PARTITION_SIZE - 1) / V2_PARTITION_SIZE;

    // exp_sums: [num_seqs, num_heads, max_num_partitions]
    // max_logits: [num_seqs, num_heads, max_num_partitions]
    // tmp_out: [num_seqs, num_heads, max_num_partitions, D]
    const size_t exp_sums_size = num_seqs * num_heads * max_num_partitions * sizeof(float);
    const size_t max_logits_size = exp_sums_size;
    const size_t tmp_out_size = num_seqs * num_heads * max_num_partitions * D * sizeof(float);

    return exp_sums_size + max_logits_size + tmp_out_size;
}

#endif // GGML_SYCL_FATTN_V2_PARTITION_HPP
