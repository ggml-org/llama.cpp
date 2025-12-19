//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_ESIMD_F16_HPP
#define GGML_SYCL_FATTN_ESIMD_F16_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>
#include <cmath>  // For std::max, std::tanh

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#define SYCL_ESIMD_AVAILABLE 1
#include <sycl/ext/intel/esimd.hpp>
#else
#define SYCL_ESIMD_AVAILABLE 0
#endif

#if SYCL_ESIMD_AVAILABLE

namespace esimd = sycl::ext::intel::esimd;

// ESIMD-compatible tanh implementation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
inline float esimd_tanh(float x) {
    float e2x = esimd::exp(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// ESIMD-compatible ALiBi slope computation
// Uses exp(exph * log(base)) instead of dpct::pow which isn't supported in ESIMD
inline float esimd_get_alibi_slope(const float max_bias,
                                   const uint32_t h,
                                   const uint32_t n_head_log2,
                                   const float m0,
                                   const float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;

    // Use exp(exph * log(base)) instead of pow(base, exph)
    return esimd::exp(static_cast<float>(exph) * esimd::log(base));
}

// =============================================================================
// ESIMD Configuration for Intel Arc GPUs
// =============================================================================

// ESIMD group size - number of KV positions processed per iteration (SLM version)
// Each work-item in the group loads one K/V row to SLM
constexpr int ESIMD_GS = 32;

// KV batch size for optimized single work-item kernel (no SLM)
// Process this many KV positions per loop iteration
constexpr int ESIMD_KV_BATCH = 8;

// SLM layout (for SLM version):
// Key cache:   GS * D * sizeof(half) bytes
// Value cache: GS * D * sizeof(half) bytes
// Total: 2 * GS * D * sizeof(half) bytes

// =============================================================================
// OPTIMIZED ESIMD Flash Attention - Partitioned Version
// =============================================================================
// This version partitions the KV sequence across multiple work-items for parallelism.
// Each work-item computes partial attention over its KV partition.
// Final reduction combines partial results using online softmax merge.
// Uses SLM for efficient inter-thread reduction.

// Number of partitions (work-items) per query head
// Benchmarked: 32 partitions is optimal (8=44.3, 16=45.2, 32=45.5, 64=44.5)
constexpr int ESIMD_PARTITIONS = 32;

template <int D, bool use_logit_softcap, typename Q_type>
void launch_fattn_esimd_f16_optimized(
    const fattn_params & params,
    sycl::queue & stream) {

    const int ne01 = params.ne01;  // Number of query tokens
    const int ne02 = params.ne02;  // Number of heads
    const int ne03 = params.ne03;  // Batch size

    // Grid: ESIMD_PARTITIONS work-items per (batch, head, query_token)
    sycl::range<3> grid(ne03 * ne02, 1, ne01);
    sycl::range<3> block(1, 1, ESIMD_PARTITIONS);

    // Extract parameters
    const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(params.Q);
    const sycl::half * K_ptr = reinterpret_cast<const sycl::half*>(params.K);
    const sycl::half * V_ptr = reinterpret_cast<const sycl::half*>(params.V);
    const sycl::half * mask_ptr = reinterpret_cast<const sycl::half*>(params.mask);
    float * dst_ptr = params.dst;

    const float scale_val = params.scale;
    const float logit_softcap_val = params.logit_softcap;

    // ALiBi parameters
    const float max_bias = params.max_bias;
    const float m0 = params.m0;
    const float m1 = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;

    const int ne00 = params.ne00;
    const int nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int ne10 = params.ne10, ne11 = params.ne11, ne12 = params.ne12, ne13 = params.ne13;
    const int nb11 = params.nb11, nb12 = params.nb12;
    const int64_t nb13 = params.nb13;
    const int nb21 = params.nb21, nb22 = params.nb22;
    const int64_t nb23 = params.nb23;
    const int ne30 = params.ne30, ne31 = params.ne31;
    const int nb31 = params.nb31;

    // PagedAttention parameters
    const bool use_paged_attn = params.use_paged_attn;
    const int32_t pa_block_size = params.block_size;
    const int32_t pa_max_blocks = params.max_blocks_per_seq;
    const int32_t * pa_block_table = params.block_table;
    const int32_t * pa_seq_lens = params.seq_lens;

    // Multi-token decode parameters (speculative decoding / multi-step generation)
    const bool multi_token_decode = params.multi_token_decode;
    const int32_t * q_positions = params.q_positions;
    const int32_t kv_base_pos = params.kv_base_pos;

    // Sequence ID masking parameters (continuous batching with multi-sequence support)
    const int n_seqs = params.n_seqs;
    const int32_t * q_seq_ids = params.q_seq_ids;
    const int32_t * kv_seq_ids = params.kv_seq_ids;

    // Attention sinks parameter
    const float * sinks_ptr = reinterpret_cast<const float *>(params.sinks);

    // SLM size for reduction:
    // - partial_acc: PARTITIONS * D * sizeof(float)
    // - partial_max: PARTITIONS * sizeof(float)
    // - partial_sum: PARTITIONS * sizeof(float)
    constexpr size_t slm_acc_offset = 0;
    constexpr size_t slm_max_offset = ESIMD_PARTITIONS * D * sizeof(float);
    constexpr size_t slm_sum_offset = slm_max_offset + ESIMD_PARTITIONS * sizeof(float);
    constexpr size_t slm_size = slm_sum_offset + ESIMD_PARTITIONS * sizeof(float);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace esimd;

                // Initialize SLM for reduction
                slm_init<slm_size>();

                // Work distribution
                const int sequence = item.get_group(0) / ne02;
                const int head = item.get_group(0) % ne02;
                const int query_idx = item.get_group(2);
                const int partition_id = item.get_local_id(2);

                // GQA ratio
                const int gqa_ratio = ne02 / ne12;
                const int kv_head = head / gqa_ratio;

                // For unified KV mode (ne13=1), all sequences share the same K/V buffer
                const int kv_sequence = (ne13 == 1) ? 0 : sequence;

                // Compute base pointers
                const Q_type * Q_base = Q_ptr + (nb03 / sizeof(Q_type)) * sequence
                                              + (nb02 / sizeof(Q_type)) * head
                                              + (nb01 / sizeof(Q_type)) * query_idx;

                const sycl::half * K_base = K_ptr + (nb13 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb12 / sizeof(sycl::half)) * kv_head;

                const sycl::half * V_base = V_ptr + (nb23 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb22 / sizeof(sycl::half)) * kv_head;

                // Mask and output pointers
                const int stride_mask = nb31 / sizeof(sycl::half);
                const sycl::half * mask_base = mask_ptr ? mask_ptr + query_idx * stride_mask : nullptr;

                // ALiBi slope computation
                // For single-query mode (ne01 == 1) and non-unified KV (ne13 != 1): use ALiBi slope
                // Otherwise: slope = 1.0f (mask already contains proper values)
                // This matches the XMX kernel condition to avoid incorrectly applying ALiBi during prompt processing
                float slope = 1.0f;
                if (ne01 == 1 && ne13 != 1) {
                    slope = esimd_get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
                }

                // Multi-token decode: get this query's position for per-query causal masking
                // Each query can only attend to KV positions <= its own position
                const int32_t q_pos = (multi_token_decode && q_positions) ? q_positions[query_idx] : INT32_MAX;

                // Sequence ID masking: get this query's sequence ID for cross-sequence masking
                // Used in continuous batching to ensure queries only attend to KV from same sequence
                const int32_t q_seq = (n_seqs > 1 && q_seq_ids) ? q_seq_ids[query_idx] : -1;

                float * out_base = dst_ptr + (ne00 * ne01 * ne02) * sequence
                                           + (ne00 * ne01) * head
                                           + ne00 * query_idx;

                // Determine KV sequence length
                int kv_len = ne11;
                if (use_paged_attn && pa_seq_lens) {
                    kv_len = pa_seq_lens[sequence];
                }

                // Compute this partition's KV range
                const int kv_per_partition = (kv_len + ESIMD_PARTITIONS - 1) / ESIMD_PARTITIONS;
                const int kv_start = partition_id * kv_per_partition;
                const int kv_end = std::min(kv_start + kv_per_partition, kv_len);

                // Load query vector once and apply scale
                // Native D-element vectors for all head dimensions (including D=64)
                simd<float, D> query_row;
                if constexpr (std::is_same_v<Q_type, sycl::half>) {
                    simd<sycl::half, D> query_row_raw = block_load<sycl::half, D>(Q_base);
                    query_row = convert<float>(query_row_raw) * scale_val;
                } else {
                    simd<float, D> query_row_raw = block_load<float, D>(Q_base);
                    query_row = query_row_raw * scale_val;
                }

                // Initialize accumulator for this partition
                simd<float, D> acc_v = 0.0f;
                float softmax_sum = 0.0f;
                float max_score = -FLT_MAX;

                // Stride for K/V rows (in elements)
                const int k_stride = nb11 / sizeof(sycl::half);
                const int v_stride = nb21 / sizeof(sycl::half);

                // =============================================================================
                // OPTIMIZATION: Double-buffered K loading
                // Prefetch next K while computing current score to hide memory latency
                // =============================================================================

                // Macro to compute K/V pointers (lambdas don't work in ESIMD kernels)
                // Results are stored in K_ptr and V_ptr variables
                #define COMPUTE_KV_PTRS(pos, K_ptr, V_ptr) \
                    do { \
                        if (use_paged_attn && pa_block_table) { \
                            const int logical_block_tmp = (pos) / pa_block_size; \
                            const int offset_in_block_tmp = (pos) % pa_block_size; \
                            const int physical_block_tmp = pa_block_table[sequence * pa_max_blocks + logical_block_tmp]; \
                            const int physical_pos_tmp = physical_block_tmp * pa_block_size + offset_in_block_tmp; \
                            K_ptr = K_base + k_stride * physical_pos_tmp; \
                            V_ptr = V_base + v_stride * physical_pos_tmp; \
                        } else { \
                            K_ptr = K_base + k_stride * (pos); \
                            V_ptr = V_base + v_stride * (pos); \
                        } \
                    } while(0)

                // Handle empty partition vs main computation
                if (kv_start >= kv_end) {
                    // Empty partition - store zeros
                    simd<float, D> zeros = 0.0f;
                    slm_block_store(slm_acc_offset + partition_id * D * sizeof(float), zeros);
                    slm_scalar_store<float>(slm_max_offset + partition_id * sizeof(float), -FLT_MAX);
                    slm_scalar_store<float>(slm_sum_offset + partition_id * sizeof(float), 0.0f);
                } else {
                    // =============================================================================
                    // Main partition processing with double-buffered K and V loading
                    // Prefetch next K and V while computing current score and accumulator update
                    // =============================================================================

                    // Prefetch variables for double-buffering (native D-element vectors)
                    simd<sycl::half, D> k_prefetch;
                    simd<sycl::half, D> v_prefetch;

                    // Prefetch first K and V rows
                    const sycl::half* K_first = nullptr;
                    const sycl::half* V_first = nullptr;
                    COMPUTE_KV_PTRS(kv_start, K_first, V_first);
                    k_prefetch = block_load<sycl::half, D>(K_first);
                    v_prefetch = block_load<sycl::half, D>(V_first);

                    // Process this partition's KV positions with double-buffered K and V loading
                    for (int kv_pos = kv_start; kv_pos < kv_end; ++kv_pos) {
                        // Use prefetched K and V (already loaded from previous iteration)
                        simd<float, D> k_row = convert<float>(k_prefetch);
                        simd<sycl::half, D> v_row_h = v_prefetch;

                        // Prefetch next K and V while we compute (if there is a next position)
                        if (kv_pos + 1 < kv_end) {
                            const sycl::half* K_next = nullptr;
                            const sycl::half* V_next = nullptr;
                            COMPUTE_KV_PTRS(kv_pos + 1, K_next, V_next);
                            k_prefetch = block_load<sycl::half, D>(K_next);
                            v_prefetch = block_load<sycl::half, D>(V_next);
                        }

                        // Compute dot product
                        simd<float, D> prod = query_row * k_row;
                        float score = esimd::detail::sum<float, float, D>(prod);

                        // Apply logit softcap if enabled
                        if constexpr (use_logit_softcap) {
                            score = logit_softcap_val * esimd_tanh(score);
                        }

                        // Apply mask if present (with ALiBi slope)
                        if (mask_base) {
                            score += slope * static_cast<float>(mask_base[kv_pos]);
                        }

                        // Multi-token decode: per-query position-based causal masking
                        // Each query can only attend to KV positions <= its own position
                        if (kv_base_pos + kv_pos > q_pos) {
                            score = -FLT_MAX;
                        }

                        // Sequence ID masking: mask out KV positions from different sequences
                        // Used in continuous batching to ensure queries only attend to KV from same sequence
                        if (q_seq >= 0 && kv_seq_ids) {
                            const int32_t kv_seq = kv_seq_ids[kv_pos];
                            if (kv_seq >= 0 && kv_seq != q_seq) {
                                score = -FLT_MAX;
                            }
                        }

                        // Online softmax update - V was prefetched above
                        simd<float, D> v_row = convert<float>(v_row_h);

                        if (score <= max_score) {
                            // Use FTZ threshold to avoid numerical issues
                            float diff = score - max_score;
                            float exp_score = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            acc_v = acc_v + v_row * exp_score;
                            softmax_sum += exp_score;
                        } else {
                            // Use FTZ threshold to avoid numerical issues
                            float diff = max_score - score;
                            float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            acc_v = acc_v * exp_factor + v_row;
                            softmax_sum = softmax_sum * exp_factor + 1.0f;
                            max_score = score;
                        }
                    }

                    // Store partial results to SLM (native D-element store)
                    slm_block_store(slm_acc_offset + partition_id * D * sizeof(float), acc_v);
                    slm_scalar_store<float>(slm_max_offset + partition_id * sizeof(float), max_score);
                    slm_scalar_store<float>(slm_sum_offset + partition_id * sizeof(float), softmax_sum);
                } // end if (kv_start >= kv_end) else
                barrier();

                // Thread 0 performs final reduction (native D-element vectors for all dimensions)
                if (partition_id == 0) {
                    // Load first partition's results
                    simd<float, D> final_acc = slm_block_load<float, D>(slm_acc_offset);
                    float final_max = slm_scalar_load<float>(slm_max_offset);
                    float final_sum = slm_scalar_load<float>(slm_sum_offset);

                    // Merge remaining partitions using online softmax
                    for (int p = 1; p < ESIMD_PARTITIONS; ++p) {
                        simd<float, D> p_acc = slm_block_load<float, D>(slm_acc_offset + p * D * sizeof(float));
                        float p_max = slm_scalar_load<float>(slm_max_offset + p * sizeof(float));
                        float p_sum = slm_scalar_load<float>(slm_sum_offset + p * sizeof(float));

                        // Online softmax merge with FTZ threshold
                        if (p_max <= final_max) {
                            float diff = p_max - final_max;
                            float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            final_acc = final_acc + p_acc * exp_factor;
                            final_sum += p_sum * exp_factor;
                        } else {
                            float diff = final_max - p_max;
                            float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            final_acc = final_acc * exp_factor + p_acc;
                            final_sum = final_sum * exp_factor + p_sum;
                            final_max = p_max;
                        }
                    }

                    // Apply attention sinks if present
                    if (sinks_ptr) {
                        const float sink = sinks_ptr[head];
                        const float new_max = std::max(sink, final_max);
                        float diff = final_max - new_max;
                        float max_scale = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                        float sink_softmax = esimd::exp(sink - new_max);
                        final_acc = final_acc * max_scale;
                        final_sum = final_sum * max_scale + sink_softmax;
                        final_max = new_max;
                    }

                    // Final normalization and output
                    if (final_sum > 0.0f) {
                        simd<float, D> result = final_acc / final_sum;
                        block_store(out_base, result);
                    }
                }

                #undef COMPUTE_KV_PTRS
            });
    });
}

// =============================================================================
// ORIGINAL ESIMD Flash Attention - Multi Work-Item SLM Version
// =============================================================================
// Note: ESIMD kernels require all code to be inlined into the kernel lambda.
// We cannot call separate device functions from ESIMD kernels.

template <int D, bool use_logit_softcap, typename Q_type>
void launch_fattn_esimd_f16(
    const fattn_params & params,
    sycl::queue & stream) {

    const int ne01 = params.ne01;  // Number of query tokens
    const int ne02 = params.ne02;  // Number of heads
    const int ne03 = params.ne03;  // Batch size

    // Grid: one work-group per (batch, head, query_token)
    sycl::range<3> grid(ne03 * ne02, 1, ne01);
    sycl::range<3> block(1, 1, ESIMD_GS);

    // Extract parameters - Q can be F16 or F32
    const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(params.Q);
    const sycl::half * K_ptr = reinterpret_cast<const sycl::half*>(params.K);
    const sycl::half * V_ptr = reinterpret_cast<const sycl::half*>(params.V);
    const sycl::half * mask_ptr = reinterpret_cast<const sycl::half*>(params.mask);
    float * dst_ptr = params.dst;

    const float scale_val = params.scale;
    const float logit_softcap_val = params.logit_softcap;

    const int ne00 = params.ne00;
    const int nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int ne10 = params.ne10, ne11 = params.ne11, ne12 = params.ne12, ne13 = params.ne13;
    const int nb11 = params.nb11, nb12 = params.nb12;
    const int64_t nb13 = params.nb13;
    const int nb21 = params.nb21, nb22 = params.nb22;
    const int64_t nb23 = params.nb23;
    const int ne30 = params.ne30, ne31 = params.ne31;
    const int nb31 = params.nb31;

    // PagedAttention parameters
    const bool use_paged_attn = params.use_paged_attn;
    const int32_t pa_block_size = params.block_size;
    const int32_t pa_max_blocks = params.max_blocks_per_seq;
    const int32_t * pa_block_table = params.block_table;
    const int32_t * pa_seq_lens = params.seq_lens;

    // ALiBi parameters
    const float max_bias = params.max_bias;
    const float m0 = params.m0;
    const float m1 = params.m1;
    const uint32_t n_head_log2 = params.n_head_log2;

    // Multi-token decode parameters (speculative decoding / multi-step generation)
    const bool multi_token_decode = params.multi_token_decode;
    const int32_t * q_positions = params.q_positions;
    const int32_t kv_base_pos = params.kv_base_pos;

    // Sequence ID masking parameters (continuous batching with multi-sequence support)
    const int n_seqs = params.n_seqs;
    const int32_t * q_seq_ids = params.q_seq_ids;
    const int32_t * kv_seq_ids = params.kv_seq_ids;

    // Attention sinks parameter
    const float * sinks_ptr = reinterpret_cast<const float *>(params.sinks);

    // SLM size
    constexpr size_t slm_size = ESIMD_GS * D * sizeof(sycl::half) * 2;
    constexpr size_t key_slm_offset = 0;
    constexpr size_t value_slm_offset = ESIMD_GS * D * sizeof(sycl::half);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace esimd;

                // Initialize SLM
                slm_init<slm_size>();

                // Work distribution
                const int sequence = item.get_group(0) / ne02;
                const int head = item.get_group(0) % ne02;
                const int query_idx = item.get_group(2);
                const int tid = item.get_local_id(2);

                // GQA ratio
                const int gqa_ratio = ne02 / ne12;
                const int kv_head = head / gqa_ratio;

                // For unified KV mode (ne13=1), all sequences share the same K/V buffer
                const int kv_sequence = (ne13 == 1) ? 0 : sequence;

                // Compute base pointers - Q uses Q_type (F16 or F32)
                const Q_type * Q_base = Q_ptr + (nb03 / sizeof(Q_type)) * sequence
                                              + (nb02 / sizeof(Q_type)) * head
                                              + (nb01 / sizeof(Q_type)) * query_idx;

                const sycl::half * K_base = K_ptr + (nb13 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb12 / sizeof(sycl::half)) * kv_head;

                const sycl::half * V_base = V_ptr + (nb23 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb22 / sizeof(sycl::half)) * kv_head;

                // Mask pointer
                const int stride_mask = nb31 / sizeof(sycl::half);
                const sycl::half * mask_base = mask_ptr ? mask_ptr + query_idx * stride_mask : nullptr;

                // ALiBi slope computation
                // For single-query mode (ne01 == 1) and non-unified KV (ne13 != 1): use ALiBi slope
                // Otherwise: slope = 1.0f (mask already contains proper values)
                // This matches the XMX kernel condition to avoid incorrectly applying ALiBi during prompt processing
                float slope = 1.0f;
                if (ne01 == 1 && ne13 != 1) {
                    slope = esimd_get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
                }

                // Multi-token decode: get this query's position for per-query causal masking
                // Each query can only attend to KV positions <= its own position
                const int32_t q_pos = (multi_token_decode && q_positions) ? q_positions[query_idx] : INT32_MAX;

                // Sequence ID masking: get this query's sequence ID for cross-sequence masking
                // Used in continuous batching to ensure queries only attend to KV from same sequence
                const int32_t q_seq = (n_seqs > 1 && q_seq_ids) ? q_seq_ids[query_idx] : -1;

                // Output pointer
                float * out_base = dst_ptr + (ne00 * ne01 * ne02) * sequence
                                           + (ne00 * ne01) * head
                                           + ne00 * query_idx;

                // Determine KV sequence length
                int kv_len = ne11;
                if (use_paged_attn && pa_seq_lens) {
                    kv_len = pa_seq_lens[sequence];
                }

                // Load query vector and apply scale
                // Native D-element vectors work for all head dimensions (including D=64)
                simd<float, D> query_row;
                if constexpr (std::is_same_v<Q_type, sycl::half>) {
                    simd<sycl::half, D> query_row_raw = block_load<sycl::half, D>(Q_base);
                    query_row = convert<float>(query_row_raw) * scale_val;
                } else {
                    simd<float, D> query_row_raw = block_load<float, D>(Q_base);
                    query_row = query_row_raw * scale_val;
                }

                // Initialize accumulators for online softmax
                simd<float, D> acc_v = 0.0f;
                float softmax_sum = 0.0f;
                float max_score = -FLT_MAX;

                // Number of full groups
                const int n_groups = kv_len / ESIMD_GS;
                const int remainder = kv_len % ESIMD_GS;

                // ================================================================
                // Main loop: Process full groups of ESIMD_GS KV positions
                // ================================================================
                for (int group = 0; group < n_groups; ++group) {
                    const int kv_start = group * ESIMD_GS;
                    int kv_pos = kv_start + tid;

                    const sycl::half * K_row;
                    const sycl::half * V_row;

                    if (use_paged_attn && pa_block_table) {
                        const int logical_block = kv_pos / pa_block_size;
                        const int offset_in_block = kv_pos % pa_block_size;
                        const int physical_block = pa_block_table[sequence * pa_max_blocks + logical_block];
                        const int physical_pos = physical_block * pa_block_size + offset_in_block;
                        K_row = K_base + (nb11 / sizeof(sycl::half)) * physical_pos;
                        V_row = V_base + (nb21 / sizeof(sycl::half)) * physical_pos;
                    } else {
                        K_row = K_base + (nb11 / sizeof(sycl::half)) * kv_pos;
                        V_row = V_base + (nb21 / sizeof(sycl::half)) * kv_pos;
                    }

                    // Load K and V rows to SLM (native D-element vectors for all dimensions)
                    simd<sycl::half, D> key_row = block_load<sycl::half, D>(K_row);
                    simd<sycl::half, D> value_row = block_load<sycl::half, D>(V_row);
                    slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), key_row);
                    slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), value_row);

                    barrier();

                    // Compute attention scores for all GS positions
                    simd<float, ESIMD_GS> scores;

                    #pragma unroll
                    for (int r = 0; r < ESIMD_GS; ++r) {
                        simd<sycl::half, D> k_row_h = slm_block_load<sycl::half, D>(
                            key_slm_offset + r * D * sizeof(sycl::half));

                        // Dot product: Q @ K^T (both in float)
                        simd<float, D> k_row = convert<float>(k_row_h);
                        simd<float, D> prod_f = query_row * k_row;
                        float score = esimd::detail::sum<float, float, D>(prod_f);

                        // Apply logit softcap if enabled
                        if constexpr (use_logit_softcap) {
                            score = logit_softcap_val * esimd_tanh(score);
                        }

                        // Apply mask if present (with ALiBi slope)
                        if (mask_base) {
                            score += slope * static_cast<float>(mask_base[kv_start + r]);
                        }

                        // Multi-token decode: per-query position-based causal masking
                        if (kv_base_pos + kv_start + r > q_pos) {
                            score = -FLT_MAX;
                        }

                        // Sequence ID masking: mask out KV positions from different sequences
                        if (q_seq >= 0 && kv_seq_ids) {
                            const int32_t kv_seq = kv_seq_ids[kv_start + r];
                            if (kv_seq >= 0 && kv_seq != q_seq) {
                                score = -FLT_MAX;
                            }
                        }

                        scores[r] = score;
                    }

                    // Online softmax update with KQ max offset for numerical stability
                    float new_max = esimd::hmax<float>(scores) + FATTN_KQ_MAX_OFFSET;
                    new_max = std::max(new_max, max_score);

                    // Rescale previous accumulator with FTZ threshold
                    float diff_val = max_score - new_max;
                    float exp_factor = diff_val >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff_val) : 0.0f;
                    acc_v = acc_v * exp_factor;
                    softmax_sum *= exp_factor;
                    max_score = new_max;

                    // Compute exp(scores - max) with FTZ threshold and accumulate weighted V
                    simd<float, ESIMD_GS> scores_diff = scores - max_score;
                    simd<float, ESIMD_GS> exp_scores;
                    #pragma unroll
                    for (int r = 0; r < ESIMD_GS; ++r) {
                        exp_scores[r] = scores_diff[r] >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(scores_diff[r]) : 0.0f;
                    }

                    #pragma unroll
                    for (int r = 0; r < ESIMD_GS; ++r) {
                        simd<sycl::half, D> v_row = slm_block_load<sycl::half, D>(
                            value_slm_offset + r * D * sizeof(sycl::half));
                        simd<float, D> v_float = convert<float>(v_row);
                        float exp_val = exp_scores[r];
                        acc_v = acc_v + v_float * exp_val;
                    }

                    softmax_sum += esimd::detail::sum<float, float, ESIMD_GS>(exp_scores);

                    barrier();
                }

                // ================================================================
                // Handle remainder KV positions (kv_len % ESIMD_GS)
                // ================================================================
                if (remainder > 0) {
                    const int kv_start = n_groups * ESIMD_GS;

                    // Only threads with tid < remainder load data
                    if (tid < remainder) {
                        int kv_pos = kv_start + tid;

                        const sycl::half * K_row;
                        const sycl::half * V_row;

                        if (use_paged_attn && pa_block_table) {
                            const int logical_block = kv_pos / pa_block_size;
                            const int offset_in_block = kv_pos % pa_block_size;
                            const int physical_block = pa_block_table[sequence * pa_max_blocks + logical_block];
                            const int physical_pos = physical_block * pa_block_size + offset_in_block;
                            K_row = K_base + (nb11 / sizeof(sycl::half)) * physical_pos;
                            V_row = V_base + (nb21 / sizeof(sycl::half)) * physical_pos;
                        } else {
                            K_row = K_base + (nb11 / sizeof(sycl::half)) * kv_pos;
                            V_row = V_base + (nb21 / sizeof(sycl::half)) * kv_pos;
                        }

                        // Load K and V rows to SLM (native D-element vectors for all dimensions)
                        simd<sycl::half, D> key_row = block_load<sycl::half, D>(K_row);
                        simd<sycl::half, D> value_row = block_load<sycl::half, D>(V_row);
                        slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), key_row);
                        slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), value_row);
                    }

                    barrier();

                    // Process remainder positions one by one
                    for (int r = 0; r < remainder; ++r) {
                        simd<sycl::half, D> k_row_h = slm_block_load<sycl::half, D>(
                            key_slm_offset + r * D * sizeof(sycl::half));
                        simd<sycl::half, D> v_row = slm_block_load<sycl::half, D>(
                            value_slm_offset + r * D * sizeof(sycl::half));

                        // Compute attention score (both in float)
                        simd<float, D> k_row = convert<float>(k_row_h);
                        simd<float, D> prod_f = query_row * k_row;
                        float score = esimd::detail::sum<float, float, D>(prod_f);

                        if constexpr (use_logit_softcap) {
                            score = logit_softcap_val * esimd_tanh(score);
                        }

                        // Apply mask if present (with ALiBi slope)
                        if (mask_base) {
                            score += slope * static_cast<float>(mask_base[kv_start + r]);
                        }

                        // Multi-token decode: per-query position-based causal masking
                        if (kv_base_pos + kv_start + r > q_pos) {
                            score = -FLT_MAX;
                        }

                        // Sequence ID masking: mask out KV positions from different sequences
                        if (q_seq >= 0 && kv_seq_ids) {
                            const int32_t kv_seq = kv_seq_ids[kv_start + r];
                            if (kv_seq >= 0 && kv_seq != q_seq) {
                                score = -FLT_MAX;
                            }
                        }

                        // Online softmax update with FTZ threshold
                        simd<float, D> v_float = convert<float>(v_row);

                        if (score <= max_score) {
                            float diff = score - max_score;
                            float exp_score = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            acc_v = acc_v + v_float * exp_score;
                            softmax_sum += exp_score;
                        } else {
                            float diff = max_score - score;
                            float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            acc_v = acc_v * exp_factor + v_float;
                            softmax_sum = softmax_sum * exp_factor + 1.0f;
                            max_score = score;
                        }
                    }
                }

                // ================================================================
                // Apply attention sinks if present (only thread 0)
                // ================================================================
                if (tid == 0 && sinks_ptr) {
                    const float sink = sinks_ptr[head];
                    const float new_max = std::max(sink, max_score);
                    float diff = max_score - new_max;
                    float max_scale = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                    float sink_softmax = esimd::exp(sink - new_max);
                    acc_v = acc_v * max_scale;
                    softmax_sum = softmax_sum * max_scale + sink_softmax;
                    max_score = new_max;
                }

                // ================================================================
                // Final normalization and output (only thread 0 writes)
                // ================================================================
                if (tid == 0 && softmax_sum > 0.0f) {
                    simd<float, D> result = acc_v / softmax_sum;
                    block_store(out_base, result);
                }
            });
    });
}

// =============================================================================
// ESIMD Flash Attention - Batched Version for Prompt Processing
// =============================================================================
// This version processes multiple queries (ncols) per work-group.
// Each thread handles all ncols queries over its KV partition.
// Designed for prompt processing where ne01 > 1.

// Number of queries per work-group for batched ESIMD
constexpr int ESIMD_NCOLS = 8;

// Number of threads per work-group for batched version
// Each thread processes ESIMD_NCOLS queries over its KV partition
constexpr int ESIMD_BATCHED_PARTITIONS = 32;

// Debug flag for batched ESIMD kernel
#define ESIMD_BATCHED_DEBUG 0

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_esimd_f16_batched(
    const fattn_params & params,
    sycl::queue & stream) {

    const int ne01 = params.ne01;  // Number of query tokens
    const int ne02 = params.ne02;  // Number of heads
    const int ne03 = params.ne03;  // Batch size

    // Grid: one work-group per (batch, head, query_block)
    const int n_query_blocks = (ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(ne03 * ne02, 1, n_query_blocks);
    sycl::range<3> block(1, 1, ESIMD_BATCHED_PARTITIONS);

    // Extract parameters
    const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(params.Q);
    const sycl::half * K_ptr = reinterpret_cast<const sycl::half*>(params.K);
    const sycl::half * V_ptr = reinterpret_cast<const sycl::half*>(params.V);
    const sycl::half * mask_ptr = reinterpret_cast<const sycl::half*>(params.mask);
    float * dst_ptr = params.dst;

    const float scale_val = params.scale;
    const float logit_softcap_val = params.logit_softcap;

    const int ne00 = params.ne00;
    const int nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
    const int ne10 = params.ne10, ne11 = params.ne11, ne12 = params.ne12, ne13 = params.ne13;
    const int nb11 = params.nb11, nb12 = params.nb12;
    const int64_t nb13 = params.nb13;
    const int nb21 = params.nb21, nb22 = params.nb22;
    const int64_t nb23 = params.nb23;
    // Mask dimensions and strides (critical for proper indexing across batch/head)
    const int ne30 = params.ne30, ne31 = params.ne31, ne32 = params.ne32, ne33 = params.ne33;
    const int nb31 = params.nb31, nb32 = params.nb32;
    const int64_t nb33 = params.nb33;

    // PagedAttention parameters
    const bool use_paged_attn = params.use_paged_attn;
    const int32_t pa_block_size = params.block_size;
    const int32_t pa_max_blocks = params.max_blocks_per_seq;
    const int32_t * pa_block_table = params.block_table;
    const int32_t * pa_seq_lens = params.seq_lens;

    // Attention sinks parameter
    const float * sinks_ptr = reinterpret_cast<const float *>(params.sinks);

    // SLM size for reduction:
    // - partial_acc: PARTITIONS * ncols * D * sizeof(float)
    // - partial_max: PARTITIONS * ncols * sizeof(float)
    // - partial_sum: PARTITIONS * ncols * sizeof(float)
    constexpr size_t slm_acc_offset = 0;
    constexpr size_t slm_max_offset = ESIMD_BATCHED_PARTITIONS * ncols * D * sizeof(float);
    constexpr size_t slm_sum_offset = slm_max_offset + ESIMD_BATCHED_PARTITIONS * ncols * sizeof(float);
    constexpr size_t slm_size = slm_sum_offset + ESIMD_BATCHED_PARTITIONS * ncols * sizeof(float);

    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                using namespace esimd;

                // Initialize SLM for reduction
                slm_init<slm_size>();

                // Work distribution
                const int sequence = item.get_group(0) / ne02;
                const int head = item.get_group(0) % ne02;
                const int query_block = item.get_group(2);
                const int ic0 = query_block * ncols;  // First query index for this work-group
                const int partition_id = item.get_local_id(2);

                // GQA ratio
                const int gqa_ratio = ne02 / ne12;
                const int kv_head = head / gqa_ratio;

                // For unified KV mode (ne13=1), all sequences share the same K/V buffer
                const int kv_sequence = (ne13 == 1) ? 0 : sequence;

                // Base pointers for K/V
                const sycl::half * K_base = K_ptr + (nb13 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb12 / sizeof(sycl::half)) * kv_head;
                const sycl::half * V_base = V_ptr + (nb23 / sizeof(sycl::half)) * kv_sequence
                                                  + (nb22 / sizeof(sycl::half)) * kv_head;

                // Mask setup - match XMX kernel's stride calculation (critical for padding)
                const int stride_mask = nb31 / sizeof(sycl::half);
                // Multi-head mask support: ne32 > 1 means each head has its own mask
                const int mask_head = ne32 > 1 ? head % ne32 : 0;
                // Base mask pointer for this (sequence, head, query_block)
                // Layout: mask[batch % ne33][head % ne32][query][kv_pos]
                const sycl::half * mask_base = mask_ptr ?
                    mask_ptr + (nb33 / sizeof(sycl::half)) * (sequence % ne33)
                             + (nb32 / sizeof(sycl::half)) * mask_head
                             + (nb31 / sizeof(sycl::half)) * ic0 : nullptr;

                // Determine KV sequence length
                int kv_len = ne11;
                if (use_paged_attn && pa_seq_lens) {
                    kv_len = pa_seq_lens[sequence];
                }

                // Compute this partition's KV range
                const int kv_per_partition = (kv_len + ESIMD_BATCHED_PARTITIONS - 1) / ESIMD_BATCHED_PARTITIONS;
                const int kv_start = partition_id * kv_per_partition;
                // Use std::min instead of sycl::min (sycl::min not supported in ESIMD context)
                const int kv_end = (kv_start + kv_per_partition < kv_len) ? (kv_start + kv_per_partition) : kv_len;

                // Load all query vectors for this work-group
                // Each thread loads the same queries (redundant but avoids SLM for Q)
                // D=128 uses separate halves (64+64), D=64 and other D use native vectors
                simd<float, 64> query_rows_h1[ncols]; // First 64 for D=128
                simd<float, 64> query_rows_h2[ncols]; // Second 64 for D=128
                simd<float, D> query_rows[ncols];     // Full for D != 128

                #pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const int q_idx = ic0 + j;
                    if (q_idx < ne01) {
                        const Q_type * Q_base = Q_ptr + (nb03 / sizeof(Q_type)) * sequence
                                                      + (nb02 / sizeof(Q_type)) * head
                                                      + (nb01 / sizeof(Q_type)) * q_idx;
                        if constexpr (D == 128 && std::is_same_v<Q_type, sycl::half>) {
                            simd<sycl::half, 64> q_h1 = block_load<sycl::half, 64>(Q_base);
                            simd<sycl::half, 64> q_h2 = block_load<sycl::half, 64>(Q_base + 64);
                            query_rows_h1[j] = convert<float>(q_h1) * scale_val;
                            query_rows_h2[j] = convert<float>(q_h2) * scale_val;
                        } else if constexpr (D == 128) {
                            // Q is float for D=128
                            query_rows_h1[j] = block_load<float, 64>(Q_base) * scale_val;
                            query_rows_h2[j] = block_load<float, 64>(Q_base + 64) * scale_val;
                        } else if constexpr (std::is_same_v<Q_type, sycl::half>) {
                            // Native D-element vector for D=64 and all other D
                            simd<sycl::half, D> q_raw = block_load<sycl::half, D>(Q_base);
                            query_rows[j] = convert<float>(q_raw) * scale_val;
                        } else {
                            simd<float, D> q_raw = block_load<float, D>(Q_base);
                            query_rows[j] = q_raw * scale_val;
                        }
                    } else {
                        if constexpr (D == 128) {
                            query_rows_h1[j] = 0.0f;
                            query_rows_h2[j] = 0.0f;
                        } else {
                            query_rows[j] = 0.0f;
                        }
                    }
                }

                // Initialize per-query accumulators
                // D=128 uses separate halves (64+64), D=64 and other D use native vectors
                simd<float, 64> acc_v_h1[ncols], acc_v_h2[ncols]; // For D=128
                simd<float, D> acc_v[ncols];  // For D != 128
                float softmax_sum[ncols];
                float max_score[ncols];
                #pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    if constexpr (D == 128) {
                        acc_v_h1[j] = 0.0f;
                        acc_v_h2[j] = 0.0f;
                    } else {
                        acc_v[j] = 0.0f;
                    }
                    softmax_sum[j] = 0.0f;
                    max_score[j] = -FLT_MAX;
                }

                // Stride for K/V rows (in elements)
                const int k_stride = nb11 / sizeof(sycl::half);
                const int v_stride = nb21 / sizeof(sycl::half);

                // Process this partition's KV positions
                for (int kv_pos = kv_start; kv_pos < kv_end; ++kv_pos) {
                    const sycl::half * K_row;
                    const sycl::half * V_row;

                    if (use_paged_attn && pa_block_table) {
                        const int logical_block = kv_pos / pa_block_size;
                        const int offset_in_block = kv_pos % pa_block_size;
                        const int physical_block = pa_block_table[sequence * pa_max_blocks + logical_block];
                        const int physical_pos = physical_block * pa_block_size + offset_in_block;
                        K_row = K_base + k_stride * physical_pos;
                        V_row = V_base + v_stride * physical_pos;
                    } else {
                        K_row = K_base + k_stride * kv_pos;
                        V_row = V_base + v_stride * kv_pos;
                    }

                    // Load K and V vectors once, use for all queries
                    // D=128 uses separate halves (64+64), D=64 and other D use native vectors
                    simd<float, 64> k_vec_h1, k_vec_h2, v_vec_h1, v_vec_h2;  // For D=128
                    simd<float, D> k_vec, v_vec;  // For D != 128

                    if constexpr (D == 128) {
                        simd<sycl::half, 64> k_h1 = block_load<sycl::half, 64>(K_row);
                        simd<sycl::half, 64> k_h2 = block_load<sycl::half, 64>(K_row + 64);
                        simd<sycl::half, 64> v_h1 = block_load<sycl::half, 64>(V_row);
                        simd<sycl::half, 64> v_h2 = block_load<sycl::half, 64>(V_row + 64);
                        k_vec_h1 = convert<float>(k_h1);
                        k_vec_h2 = convert<float>(k_h2);
                        v_vec_h1 = convert<float>(v_h1);
                        v_vec_h2 = convert<float>(v_h2);
                    } else {
                        // Native D-element vectors for D=64 and all other D
                        simd<sycl::half, D> k_h = block_load<sycl::half, D>(K_row);
                        simd<sycl::half, D> v_h = block_load<sycl::half, D>(V_row);
                        k_vec = convert<float>(k_h);
                        v_vec = convert<float>(v_h);
                    }

                    // Process all queries against this KV position
                    #pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        const int q_idx = ic0 + j;
                        if (q_idx >= ne01) continue;

                        // Compute Q @ K dot product
                        float score;
                        if constexpr (D == 128) {
                            simd<float, 64> prod1 = query_rows_h1[j] * k_vec_h1;
                            simd<float, 64> prod2 = query_rows_h2[j] * k_vec_h2;
                            float sum1 = esimd::detail::sum<float, float, 64>(prod1);
                            float sum2 = esimd::detail::sum<float, float, 64>(prod2);
                            score = sum1 + sum2;
                        } else {
                            // Native D-element dot product for D=64 and all other D
                            simd<float, D> prod = query_rows[j] * k_vec;
                            score = esimd::detail::sum<float, float, D>(prod);
                        }

                        // Apply logit softcap if enabled
                        if constexpr (use_logit_softcap) {
                            score = logit_softcap_val * esimd_tanh(score);
                        }

                        // Apply mask if present
                        // mask_base points to (sequence, head, ic0), so use j (local query offset) not q_idx
                        if (mask_base) {
                            const sycl::half * mask_row = mask_base + j * stride_mask;
                            score += static_cast<float>(mask_row[kv_pos]);
                        }

                        // Online softmax update
                        if constexpr (D == 128) {
                            if (score <= max_score[j]) {
                                float diff = score - max_score[j];
                                float exp_score = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                acc_v_h1[j] = acc_v_h1[j] + v_vec_h1 * exp_score;
                                acc_v_h2[j] = acc_v_h2[j] + v_vec_h2 * exp_score;
                                softmax_sum[j] += exp_score;
                            } else {
                                float diff = max_score[j] - score;
                                float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                acc_v_h1[j] = acc_v_h1[j] * exp_factor + v_vec_h1;
                                acc_v_h2[j] = acc_v_h2[j] * exp_factor + v_vec_h2;
                                softmax_sum[j] = softmax_sum[j] * exp_factor + 1.0f;
                                max_score[j] = score;
                            }
                        } else {
                            if (score <= max_score[j]) {
                                float diff = score - max_score[j];
                                float exp_score = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                acc_v[j] = acc_v[j] + v_vec * exp_score;
                                softmax_sum[j] += exp_score;
                            } else {
                                float diff = max_score[j] - score;
                                float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                acc_v[j] = acc_v[j] * exp_factor + v_vec;
                                softmax_sum[j] = softmax_sum[j] * exp_factor + 1.0f;
                                max_score[j] = score;
                            }
                        }
                    }
                }

                // Store partial results to SLM for all queries
#if ESIMD_BATCHED_DEBUG
                if (head == 0 && sequence == 0 && partition_id == 0) {
                    for (int dbg_j = 0; dbg_j < ncols && dbg_j < 4; ++dbg_j) {
                        sycl::ext::oneapi::experimental::printf(
                            "[PART0] j=%d ic0=%d ne01=%d kv_end=%d max=%.4f sum=%.4f\n",
                            dbg_j, ic0, ne01, kv_end, max_score[dbg_j], softmax_sum[dbg_j]);
                    }
                }
#endif
                #pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const size_t acc_base = slm_acc_offset + (partition_id * ncols + j) * D * sizeof(float);
                    if constexpr (D == 128) {
                        slm_block_store<float, 64>(acc_base, acc_v_h1[j]);
                        slm_block_store<float, 64>(acc_base + 64 * sizeof(float), acc_v_h2[j]);
                    } else {
                        // Native D-element store for D=64 and all other D
                        slm_block_store(acc_base, acc_v[j]);
                    }
                    slm_scalar_store<float>(slm_max_offset + (partition_id * ncols + j) * sizeof(float), max_score[j]);
                    slm_scalar_store<float>(slm_sum_offset + (partition_id * ncols + j) * sizeof(float), softmax_sum[j]);
                }

                barrier();

                // Thread 0 performs final reduction for all queries
                if (partition_id == 0) {
                    for (int j = 0; j < ncols; ++j) {
                        const int q_idx = ic0 + j;
                        if (q_idx >= ne01) break;

                        // Load first partition's results
                        // D=128 uses split 64+64, D=64 and other D use native vectors
                        const size_t acc_base_0 = slm_acc_offset + j * D * sizeof(float);

                        simd<float, 64> final_acc_h1, final_acc_h2;    // For D=128
                        simd<float, D> final_acc;                       // For D != 128

                        if constexpr (D == 128) {
                            final_acc_h1 = slm_block_load<float, 64>(acc_base_0);
                            final_acc_h2 = slm_block_load<float, 64>(acc_base_0 + 64 * sizeof(float));
                        } else {
                            final_acc = slm_block_load<float, D>(acc_base_0);
                        }
                        float final_max = slm_scalar_load<float>(slm_max_offset + j * sizeof(float));
                        float final_sum = slm_scalar_load<float>(slm_sum_offset + j * sizeof(float));

                        // Merge remaining partitions
                        for (int p = 1; p < ESIMD_BATCHED_PARTITIONS; ++p) {
                            const size_t acc_base_p = slm_acc_offset + (p * ncols + j) * D * sizeof(float);

                            simd<float, 64> p_acc_h1, p_acc_h2;    // For D=128
                            simd<float, D> p_acc;                   // For D != 128

                            if constexpr (D == 128) {
                                p_acc_h1 = slm_block_load<float, 64>(acc_base_p);
                                p_acc_h2 = slm_block_load<float, 64>(acc_base_p + 64 * sizeof(float));
                            } else {
                                p_acc = slm_block_load<float, D>(acc_base_p);
                            }
                            float p_max = slm_scalar_load<float>(slm_max_offset + (p * ncols + j) * sizeof(float));
                            float p_sum = slm_scalar_load<float>(slm_sum_offset + (p * ncols + j) * sizeof(float));

                            // Online softmax merge
                            if constexpr (D == 128) {
                                if (p_max <= final_max) {
                                    float diff = p_max - final_max;
                                    float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                    final_acc_h1 = final_acc_h1 + p_acc_h1 * exp_factor;
                                    final_acc_h2 = final_acc_h2 + p_acc_h2 * exp_factor;
                                    final_sum += p_sum * exp_factor;
                                } else {
                                    float diff = final_max - p_max;
                                    float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                    final_acc_h1 = final_acc_h1 * exp_factor + p_acc_h1;
                                    final_acc_h2 = final_acc_h2 * exp_factor + p_acc_h2;
                                    final_sum = final_sum * exp_factor + p_sum;
                                    final_max = p_max;
                                }
                            } else {
                                // Native D-element merge for D=64 and all other D
                                if (p_max <= final_max) {
                                    float diff = p_max - final_max;
                                    float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                    final_acc = final_acc + p_acc * exp_factor;
                                    final_sum += p_sum * exp_factor;
                                } else {
                                    float diff = final_max - p_max;
                                    float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                    final_acc = final_acc * exp_factor + p_acc;
                                    final_sum = final_sum * exp_factor + p_sum;
                                    final_max = p_max;
                                }
                            }
                        }

                        // Apply attention sinks if present
                        if (sinks_ptr) {
                            const float sink = sinks_ptr[head];
                            const float new_max = std::max(sink, final_max);
                            float diff = final_max - new_max;
                            float max_scale = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            float sink_softmax = esimd::exp(sink - new_max);
                            if constexpr (D == 128) {
                                final_acc_h1 = final_acc_h1 * max_scale;
                                final_acc_h2 = final_acc_h2 * max_scale;
                            } else {
                                final_acc = final_acc * max_scale;
                            }
                            final_sum = final_sum * max_scale + sink_softmax;
                            final_max = new_max;
                        }

                        // Final normalization and output
                        // Layout: [sequence][head][query][D] - matches single-query ESIMD kernel
                        // Offset = (ne00*ne01*ne02)*sequence + (ne00*ne01)*head + ne00*query
                        float * out_ptr = dst_ptr + (ne00 * ne01 * ne02) * sequence
                                                  + (ne00 * ne01) * head
                                                  + ne00 * q_idx;

#if ESIMD_BATCHED_DEBUG
                        if (head == 0 && sequence == 0) {
                            sycl::ext::oneapi::experimental::printf(
                                "[REDUCE] j=%d q_idx=%d ic0=%d ne01=%d ncols=%d final_max=%.4f final_sum=%.4f\n",
                                j, q_idx, ic0, ne01, ncols, final_max, final_sum);
                        }
#endif

                        if (final_sum > 0.0f) {
                            if constexpr (D == 128) {
                                simd<float, 64> result_h1 = final_acc_h1 / final_sum;
                                simd<float, 64> result_h2 = final_acc_h2 / final_sum;
                                block_store<float, 64>(out_ptr, result_h1);
                                block_store<float, 64>(out_ptr + 64, result_h2);
                            } else {
                                // Native D-element store for D=64 and all other D
                                simd<float, D> result = final_acc / final_sum;
                                block_store(out_ptr, result);
#if ESIMD_BATCHED_DEBUG
                                if (head == 0 && sequence == 0) {
                                    sycl::ext::oneapi::experimental::printf("[STORE_DONE] q_idx=%d\n", q_idx);
                                }
#endif
                            }
                        }
                    }
                }
            });
    });
}

// =============================================================================
// ESIMD Flash Attention Dispatch Function
// =============================================================================

template <int D, typename Q_type>
void fattn_esimd_f16(
    const fattn_params & params,
    sycl::queue & stream) {

    const bool use_logit_softcap = params.logit_softcap != 0.0f;

    // Use optimized single work-item version by default
    // It has better thread utilization (no redundant computation)
    if (use_logit_softcap) {
        launch_fattn_esimd_f16_optimized<D, true, Q_type>(params, stream);
    } else {
        launch_fattn_esimd_f16_optimized<D, false, Q_type>(params, stream);
    }
}

// Check if batched ESIMD kernel fits in SLM budget
// SLM layout:
//   - partial_acc: PARTITIONS * ncols * D * sizeof(float)
//   - partial_max: PARTITIONS * ncols * sizeof(float)
//   - partial_sum: PARTITIONS * ncols * sizeof(float)
// With PARTITIONS=32, ncols=8:
//   D=64:  32*8*64*4 + 32*8*4 + 32*8*4 = 65536 + 1024 + 1024 = 67584 (66 KB)
//   D=128: 32*8*128*4 + 2048 = 133120 (130 KB)
// Intel Arc has 128 KB SLM, so D=64 fits but is tight
template <int D>
constexpr bool fattn_esimd_batched_fits_slm() {
    constexpr size_t slm_budget = 128 * 1024;  // Intel Arc has 128 KB SLM
    constexpr size_t slm_acc = ESIMD_BATCHED_PARTITIONS * ESIMD_NCOLS * D * sizeof(float);
    constexpr size_t slm_max = ESIMD_BATCHED_PARTITIONS * ESIMD_NCOLS * sizeof(float);
    constexpr size_t slm_sum = ESIMD_BATCHED_PARTITIONS * ESIMD_NCOLS * sizeof(float);
    constexpr size_t slm_needed = slm_acc + slm_max + slm_sum;
    return slm_needed <= slm_budget;
}

// Batched ESIMD dispatch for prompt processing (ne01 > 1)
// Only instantiates the batched kernel if it fits in SLM budget
template <int D, typename Q_type>
void fattn_esimd_f16_batched(
    const fattn_params & params,
    sycl::queue & stream) {

    const bool use_logit_softcap = params.logit_softcap != 0.0f;
    const int ne01 = params.ne01;

    // Only use batched kernel if it fits in SLM budget
    if constexpr (fattn_esimd_batched_fits_slm<D>()) {
        // Choose ncols based on number of queries
        if (ne01 <= 1) {
            // Single query - use partitioned version
            if (use_logit_softcap) {
                launch_fattn_esimd_f16_optimized<D, true, Q_type>(params, stream);
            } else {
                launch_fattn_esimd_f16_optimized<D, false, Q_type>(params, stream);
            }
        } else if (ne01 <= 4) {
            // Small batch - use 4 queries per work-group
            if (use_logit_softcap) {
                launch_fattn_esimd_f16_batched<D, 4, true, Q_type>(params, stream);
            } else {
                launch_fattn_esimd_f16_batched<D, 4, false, Q_type>(params, stream);
            }
        } else {
            // Large batch - use 8 queries per work-group
            if (use_logit_softcap) {
                launch_fattn_esimd_f16_batched<D, 8, true, Q_type>(params, stream);
            } else {
                launch_fattn_esimd_f16_batched<D, 8, false, Q_type>(params, stream);
            }
        }
    } else {
        // D too large for batched kernel SLM, fall back to partitioned kernel
        // This path should not be reached if dispatch logic is correct
        if (use_logit_softcap) {
            launch_fattn_esimd_f16_optimized<D, true, Q_type>(params, stream);
        } else {
            launch_fattn_esimd_f16_optimized<D, false, Q_type>(params, stream);
        }
    }
}

// Legacy SLM-based version (kept for reference/comparison)
template <int D, typename Q_type>
void fattn_esimd_f16_slm(
    const fattn_params & params,
    sycl::queue & stream) {

    const bool use_logit_softcap = params.logit_softcap != 0.0f;

    if (use_logit_softcap) {
        launch_fattn_esimd_f16<D, true, Q_type>(params, stream);
    } else {
        launch_fattn_esimd_f16<D, false, Q_type>(params, stream);
    }
}

// Check if ESIMD F16 kernel is available
inline bool fattn_esimd_f16_available() {
#if SYCL_ESIMD_AVAILABLE
    return true;
#else
    return false;
#endif
}

#else // !SYCL_ESIMD_AVAILABLE

// Stub implementations when ESIMD is not available
template <int D, typename Q_type>
void fattn_esimd_f16(
    const fattn_params & params,
    sycl::queue & stream) {
    GGML_UNUSED(params);
    GGML_UNUSED(stream);
    GGML_ASSERT(false && "SYCL ESIMD not available");
}

inline bool fattn_esimd_f16_available() {
    return false;
}

#endif // SYCL_ESIMD_AVAILABLE

#endif // GGML_SYCL_FATTN_ESIMD_F16_HPP
