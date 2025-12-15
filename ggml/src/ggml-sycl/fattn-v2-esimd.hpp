//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_V2_ESIMD_HPP
#define GGML_SYCL_FATTN_V2_ESIMD_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>
#include <cmath>
#include <type_traits>  // For std::is_same_v

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#define SYCL_V2_ESIMD_AVAILABLE 1
#include <sycl/ext/intel/esimd.hpp>
#else
#define SYCL_V2_ESIMD_AVAILABLE 0
#endif

#if SYCL_V2_ESIMD_AVAILABLE

namespace esimd = sycl::ext::intel::esimd;

// =============================================================================
// V2 ESIMD Configuration for Intel Arc GPUs
// =============================================================================
// This is an optimized version of the V2 paged attention kernel using ESIMD
// vectorization for improved performance on Intel Arc GPUs.
//
// Key optimizations over scalar V2:
// 1. Vectorized Q@K^T dot products using block_load and SIMD reduction
// 2. Vectorized V accumulation with online softmax
// 3. SLM-based inter-partition reduction
// 4. D=64 workaround (split into two 32-element halves)
// 5. Element-index based addressing (same as working ESIMD kernel)

// Number of partitions (work-items) per query head
// Each partition handles a portion of the KV sequence
// Benchmarked: 32 partitions is optimal
constexpr int V2_ESIMD_PARTITIONS = 32;

// =============================================================================
// V2 ESIMD Flash Attention - Fused Partition + Reduce Kernel
// =============================================================================
// Handles long sequences (>512 tokens) with paged attention support.
// Uses auto-V2 mode: identity block table with block_size=1 for contiguous KV.
//
// Grid: V2_ESIMD_PARTITIONS work-items per (batch, head, query_token)
// Each work-item processes a different KV range, then thread 0 merges via SLM.

template <int D, typename Q_type>
void launch_v2_esimd_attention(
    const fattn_params & params,           // Use params struct like working ESIMD kernel
    const int32_t * __restrict__ block_table,  // Block table: [num_seqs, max_blocks]
    const int32_t * __restrict__ seq_lens,     // Sequence lengths: [num_seqs]
    const int num_seqs,
    const int max_blocks_per_seq,
    const int max_context_len,
    const int block_size,
    sycl::queue * stream) {

    // Extract pointers from params - Q can be F16 or F32
    const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(params.Q);
    const sycl::half * K_ptr = reinterpret_cast<const sycl::half*>(params.K);
    const sycl::half * V_ptr = reinterpret_cast<const sycl::half*>(params.V);
    const sycl::half * mask_ptr = reinterpret_cast<const sycl::half*>(params.mask);
    float * dst_ptr = params.dst;
    const float scale = params.scale;

    // Mask strides
    const int ne31 = params.ne31;  // Mask query dimension
    const int nb31 = params.nb31;  // Mask stride between queries

    // Extract dimensions from params
    const int num_heads = params.ne02;     // Number of query heads
    const int num_kv_heads = params.ne12;  // Number of KV heads
    const int ne00 = params.ne00;          // Head dimension (should equal D)

    // Extract strides (convert to element counts, not bytes)
    // Q uses Q_type element size (F16 or F32), K/V use sycl::half
    const int nb01 = params.nb01;  // Q: stride between queries (bytes)
    const int nb02 = params.nb02;  // Q: stride between heads (bytes)
    const int nb03 = params.nb03;  // Q: stride between sequences (bytes)
    const int nb11 = params.nb11;  // K: stride between KV positions (bytes)
    const int nb12 = params.nb12;  // K: stride between KV heads (bytes)
    const int64_t nb13 = params.nb13;  // K: stride between sequences (bytes)
    const int nb21 = params.nb21;  // V: stride between KV positions (bytes)
    const int nb22 = params.nb22;  // V: stride between KV heads (bytes)
    const int64_t nb23 = params.nb23;  // V: stride between sequences (bytes)

    // Grid: V2_ESIMD_PARTITIONS work-items per (seq, head)
    sycl::range<3> grid(num_seqs, num_heads, 1);
    sycl::range<3> block(1, 1, V2_ESIMD_PARTITIONS);

    // SLM size for reduction:
    // - partial_acc: PARTITIONS * D * sizeof(float)
    // - partial_max: PARTITIONS * sizeof(float)
    // - partial_sum: PARTITIONS * sizeof(float)
    constexpr size_t slm_acc_offset = 0;
    constexpr size_t slm_max_offset = V2_ESIMD_PARTITIONS * D * sizeof(float);
    constexpr size_t slm_sum_offset = slm_max_offset + V2_ESIMD_PARTITIONS * sizeof(float);
    constexpr size_t slm_size = slm_sum_offset + V2_ESIMD_PARTITIONS * sizeof(float);

    // GQA ratio
    const int gqa_ratio = num_heads / num_kv_heads;

    // For unified KV mode (ne13=1), all sequences share the same K/V buffer
    const int ne13 = params.ne13;

    // Debug output for V2 ESIMD kernel launch
    static bool v2_debug_shown = false;
    if (!v2_debug_shown) {
        fprintf(stderr, "[V2-ESIMD-DEBUG] Launch params:\n");
        fprintf(stderr, "  D=%d, num_seqs=%d, num_heads=%d, num_kv_heads=%d\n",
                D, num_seqs, num_heads, num_kv_heads);
        fprintf(stderr, "  ne00=%d (head_dim), ne01=%d (n_queries), ne02=%d (n_heads), ne03=%d (batch)\n",
                ne00, params.ne01, params.ne02, params.ne03);
        fprintf(stderr, "  nb01=%d, nb02=%d, nb03=%d (Q strides bytes)\n", nb01, nb02, nb03);
        fprintf(stderr, "  nb11=%d, nb12=%d, nb13=%ld (K strides bytes)\n", nb11, nb12, (long)nb13);
        fprintf(stderr, "  nb21=%d, nb22=%d, nb23=%ld (V strides bytes)\n", nb21, nb22, (long)nb23);
        fprintf(stderr, "  block_size=%d, max_blocks=%d, max_ctx=%d, ne13=%d, gqa_ratio=%d\n",
                block_size, max_blocks_per_seq, max_context_len, ne13, gqa_ratio);
        fprintf(stderr, "  scale=%.6f, Q_type=%s\n", scale, std::is_same_v<Q_type, float> ? "F32" : "F16");
        fprintf(stderr, "  mask_ptr=%p, ne31=%d, nb31=%d\n", (void*)mask_ptr, ne31, nb31);
        v2_debug_shown = true;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace esimd;

                // Initialize SLM for reduction
                slm_init<slm_size>();

                // Work distribution
                const int seq_idx = item.get_group(0);
                const int head_idx = item.get_group(1);
                const int partition_id = item.get_local_id(2);

                // GQA: map query head to KV head
                const int kv_head_idx = head_idx / gqa_ratio;

                // For unified KV mode (ne13=1), all sequences share the same K/V buffer
                const int kv_sequence = (ne13 == 1) ? 0 : seq_idx;

                // Get sequence length for this sequence
                const int context_len = seq_lens[seq_idx];

                // Compute this partition's KV range
                const int kv_per_partition = (context_len + V2_ESIMD_PARTITIONS - 1) / V2_ESIMD_PARTITIONS;
                const int kv_start = partition_id * kv_per_partition;
                const int kv_end = std::min(kv_start + kv_per_partition, context_len);

                // Compute base pointers using element-index addressing (same as working ESIMD kernel)
                // Q: [num_seqs, num_heads, D] - uses sizeof(Q_type) for stride calculation
                const Q_type * Q_base = Q_ptr + (nb03 / sizeof(Q_type)) * seq_idx
                                              + (nb02 / sizeof(Q_type)) * head_idx;

                // K: [num_seqs, num_kv, num_kv_heads, D] - note: sequence stride uses nb13
                const sycl::half * K_base = K_ptr + (nb13 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb12 / sizeof(sycl::half)) * kv_head_idx;

                // V: [num_seqs, num_kv, num_kv_heads, D]
                const sycl::half * V_base = V_ptr + (nb23 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb22 / sizeof(sycl::half)) * kv_head_idx;

                // Output pointer
                float * out_base = dst_ptr + (ne00 * params.ne01 * num_heads) * seq_idx
                                           + (ne00 * params.ne01) * head_idx
                                           + ne00 * 0;  // query_idx = 0 for V2 (single query per seq)

                // Mask base pointer (query_idx = 0 for V2)
                const int stride_mask = nb31 / sizeof(sycl::half);
                const sycl::half * mask_base = mask_ptr ? mask_ptr + 0 * stride_mask : nullptr;

                // Stride for K/V rows (in elements, not bytes)
                const int k_stride = nb11 / sizeof(sycl::half);
                const int v_stride = nb21 / sizeof(sycl::half);

                // Load query vector once and apply scale
                // For D=64, we load and process in two 32-element halves to work around block_load issues
                // Q can be either F16 (sycl::half) or F32 (float) - handle both like working ESIMD kernel
                simd<float, 32> query_row_1, query_row_2;  // For D=64 split case
                simd<float, D> query_row;  // For D!=64 case

                if constexpr (D == 64 && std::is_same_v<Q_type, sycl::half>) {
                    // Q is half, load in two halves and convert to float
                    simd<sycl::half, 32> q_row_h1 = block_load<sycl::half, 32>(Q_base);
                    simd<sycl::half, 32> q_row_h2 = block_load<sycl::half, 32>(Q_base + 32);
                    query_row_1 = convert<float>(q_row_h1) * scale;
                    query_row_2 = convert<float>(q_row_h2) * scale;
                } else if constexpr (D == 64) {
                    // Q is float, load in two halves (float is 4 bytes so 32 floats = 128 bytes)
                    query_row_1 = block_load<float, 32>(Q_base) * scale;
                    query_row_2 = block_load<float, 32>(Q_base + 32) * scale;
                } else if constexpr (std::is_same_v<Q_type, sycl::half>) {
                    // D != 64, Q is half
                    simd<sycl::half, D> query_row_raw = block_load<sycl::half, D>(Q_base);
                    query_row = convert<float>(query_row_raw) * scale;
                } else {
                    // D != 64, Q is float
                    query_row = block_load<float, D>(Q_base) * scale;
                }

                // Initialize accumulators for this partition
                // For D=64, we use two 32-element halves for the accumulator
                simd<float, 32> acc_v_1 = 0.0f, acc_v_2 = 0.0f;  // For D=64
                simd<float, D> acc_v = 0.0f;  // For D!=64
                float softmax_sum = 0.0f;
                float max_score = -FLT_MAX;

                // Process this partition's KV positions
                // For auto-V2 mode (block_size=1, identity block table), directly use kv_pos
                // For paged mode (block_size>1), use block table lookup
                for (int kv_pos = kv_start; kv_pos < kv_end; ++kv_pos) {
                    int token_pos;
                    if (block_size == 1) {
                        // Auto-V2 mode: identity mapping, no block table lookup needed
                        token_pos = kv_pos;
                    } else {
                        // Paged mode: look up physical position from block table
                        const int logical_block = kv_pos / block_size;
                        const int offset_in_block = kv_pos % block_size;
                        const int physical_block = block_table[seq_idx * max_blocks_per_seq + logical_block];
                        token_pos = physical_block * block_size + offset_in_block;
                    }

                    // Use element-index addressing (same as working ESIMD kernel)
                    const sycl::half * K_row = K_base + k_stride * token_pos;
                    const sycl::half * V_row = V_base + v_stride * token_pos;

                    // Load K row and compute dot product with Q
                    // For D=64, we load and compute in two 32-element halves
                    float score;
                    if constexpr (D == 64) {
                        // Load K in two halves
                        simd<sycl::half, 32> k_row_h1 = block_load<sycl::half, 32>(K_row);
                        simd<sycl::half, 32> k_row_h2 = block_load<sycl::half, 32>(K_row + 32);
                        simd<float, 32> k1 = convert<float>(k_row_h1);
                        simd<float, 32> k2 = convert<float>(k_row_h2);

                        // Compute dot product in halves and sum
                        simd<float, 32> prod1 = query_row_1 * k1;
                        simd<float, 32> prod2 = query_row_2 * k2;
                        float sum1 = esimd::detail::sum<float, float, 32>(prod1);
                        float sum2 = esimd::detail::sum<float, float, 32>(prod2);
                        score = sum1 + sum2;
                    } else {
                        simd<sycl::half, D> k_row_h = block_load<sycl::half, D>(K_row);
                        simd<float, D> k_row = convert<float>(k_row_h);
                        simd<float, D> prod = query_row * k_row;
                        score = esimd::detail::sum<float, float, D>(prod);
                    }

                    // Apply mask if present (causal attention)
                    if (mask_base) {
                        score += static_cast<float>(mask_base[kv_pos]);
                    }

                    // Online softmax update with vectorized V accumulation
                    // For D=64, we process V in two 32-element halves
                    if constexpr (D == 64) {
                        // Load V in two halves
                        simd<sycl::half, 32> v_row_h1 = block_load<sycl::half, 32>(V_row);
                        simd<sycl::half, 32> v_row_h2 = block_load<sycl::half, 32>(V_row + 32);
                        simd<float, 32> v1 = convert<float>(v_row_h1);
                        simd<float, 32> v2 = convert<float>(v_row_h2);

                        if (score <= max_score) {
                            float exp_score = esimd::exp(score - max_score);
                            acc_v_1 = acc_v_1 + v1 * exp_score;
                            acc_v_2 = acc_v_2 + v2 * exp_score;
                            softmax_sum += exp_score;
                        } else {
                            float exp_factor = esimd::exp(max_score - score);
                            acc_v_1 = acc_v_1 * exp_factor + v1;
                            acc_v_2 = acc_v_2 * exp_factor + v2;
                            softmax_sum = softmax_sum * exp_factor + 1.0f;
                            max_score = score;
                        }
                    } else {
                        // D != 64: use single D-element vectors
                        simd<sycl::half, D> v_row_h = block_load<sycl::half, D>(V_row);
                        simd<float, D> v_row = convert<float>(v_row_h);

                        if (score <= max_score) {
                            float exp_score = esimd::exp(score - max_score);
                            acc_v = acc_v + v_row * exp_score;
                            softmax_sum += exp_score;
                        } else {
                            float exp_factor = esimd::exp(max_score - score);
                            acc_v = acc_v * exp_factor + v_row;
                            softmax_sum = softmax_sum * exp_factor + 1.0f;
                            max_score = score;
                        }
                    }
                }

                // Store partial results to SLM
                // For D=64, store in two halves
                if constexpr (D == 64) {
                    slm_block_store<float, 32>(slm_acc_offset + partition_id * D * sizeof(float), acc_v_1);
                    slm_block_store<float, 32>(slm_acc_offset + partition_id * D * sizeof(float) + 32 * sizeof(float), acc_v_2);
                } else {
                    slm_block_store(slm_acc_offset + partition_id * D * sizeof(float), acc_v);
                }
                slm_scalar_store<float>(slm_max_offset + partition_id * sizeof(float), max_score);
                slm_scalar_store<float>(slm_sum_offset + partition_id * sizeof(float), softmax_sum);

                barrier();

                // Thread 0 performs final reduction
                if (partition_id == 0) {
                    // For D=64, use two halves throughout the reduction
                    if constexpr (D == 64) {
                        // Load first partition's results
                        simd<float, 32> final_acc_1 = slm_block_load<float, 32>(slm_acc_offset);
                        simd<float, 32> final_acc_2 = slm_block_load<float, 32>(slm_acc_offset + 32 * sizeof(float));
                        float final_max = slm_scalar_load<float>(slm_max_offset);
                        float final_sum = slm_scalar_load<float>(slm_sum_offset);

                        // Merge remaining partitions using online softmax
                        for (int p = 1; p < V2_ESIMD_PARTITIONS; ++p) {
                            simd<float, 32> p_acc_1 = slm_block_load<float, 32>(slm_acc_offset + p * D * sizeof(float));
                            simd<float, 32> p_acc_2 = slm_block_load<float, 32>(slm_acc_offset + p * D * sizeof(float) + 32 * sizeof(float));
                            float p_max = slm_scalar_load<float>(slm_max_offset + p * sizeof(float));
                            float p_sum = slm_scalar_load<float>(slm_sum_offset + p * sizeof(float));

                            // Skip empty partitions
                            if (p_sum == 0.0f) continue;

                            // Online softmax merge
                            if (p_max <= final_max) {
                                float exp_factor = esimd::exp(p_max - final_max);
                                final_acc_1 = final_acc_1 + p_acc_1 * exp_factor;
                                final_acc_2 = final_acc_2 + p_acc_2 * exp_factor;
                                final_sum += p_sum * exp_factor;
                            } else {
                                float exp_factor = esimd::exp(final_max - p_max);
                                final_acc_1 = final_acc_1 * exp_factor + p_acc_1;
                                final_acc_2 = final_acc_2 * exp_factor + p_acc_2;
                                final_sum = final_sum * exp_factor + p_sum;
                                final_max = p_max;
                            }
                        }

                        // Final normalization and output
                        if (final_sum > 0.0f) {
                            simd<float, 32> result_1 = final_acc_1 / final_sum;
                            simd<float, 32> result_2 = final_acc_2 / final_sum;
                            block_store<float, 32>(out_base, result_1);
                            block_store<float, 32>(out_base + 32, result_2);
                        }
                    } else {
                        // D != 64: use single D-element vectors
                        // Load first partition's results
                        simd<float, D> final_acc = slm_block_load<float, D>(slm_acc_offset);
                        float final_max = slm_scalar_load<float>(slm_max_offset);
                        float final_sum = slm_scalar_load<float>(slm_sum_offset);

                        // Merge remaining partitions using online softmax
                        for (int p = 1; p < V2_ESIMD_PARTITIONS; ++p) {
                            simd<float, D> p_acc = slm_block_load<float, D>(slm_acc_offset + p * D * sizeof(float));
                            float p_max = slm_scalar_load<float>(slm_max_offset + p * sizeof(float));
                            float p_sum = slm_scalar_load<float>(slm_sum_offset + p * sizeof(float));

                            // Skip empty partitions
                            if (p_sum == 0.0f) continue;

                            // Online softmax merge
                            if (p_max <= final_max) {
                                float exp_factor = esimd::exp(p_max - final_max);
                                final_acc = final_acc + p_acc * exp_factor;
                                final_sum += p_sum * exp_factor;
                            } else {
                                float exp_factor = esimd::exp(final_max - p_max);
                                final_acc = final_acc * exp_factor + p_acc;
                                final_sum = final_sum * exp_factor + p_sum;
                                final_max = p_max;
                            }
                        }

                        // Final normalization and output
                        if (final_sum > 0.0f) {
                            simd<float, D> result = final_acc / final_sum;
                            block_store(out_base, result);
                        }
                    }
                }
            });
    });
}

// Check if V2 ESIMD kernel is available
inline bool v2_esimd_available() {
    return true;  // Available when this header is included (ESIMD support checked at compile time)
}

#else // SYCL_V2_ESIMD_AVAILABLE == 0

// Stub when ESIMD is not available
inline bool v2_esimd_available() {
    return false;
}

#endif // SYCL_V2_ESIMD_AVAILABLE

#endif // GGML_SYCL_FATTN_V2_ESIMD_HPP
