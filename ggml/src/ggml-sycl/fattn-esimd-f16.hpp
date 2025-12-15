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
                // For D=64, we load and process in two 32-element halves to work around block_load issues
                simd<float, 32> query_row_1, query_row_2;  // For D=64 split case
                simd<float, D> query_row;  // For D!=64 case
                if constexpr (D == 64 && std::is_same_v<Q_type, sycl::half>) {
                    simd<sycl::half, 32> q_row_h1 = block_load<sycl::half, 32>(Q_base);
                    simd<sycl::half, 32> q_row_h2 = block_load<sycl::half, 32>(Q_base + 32);
                    query_row_1 = convert<float>(q_row_h1) * scale_val;
                    query_row_2 = convert<float>(q_row_h2) * scale_val;
                } else if constexpr (D == 64) {
                    // Q is float, also split load (float is 4 bytes so 32 floats = 128 bytes)
                    query_row_1 = block_load<float, 32>(Q_base) * scale_val;
                    query_row_2 = block_load<float, 32>(Q_base + 32) * scale_val;
                } else {
                    simd<Q_type, D> query_row_raw = block_load<Q_type, D>(Q_base);
                    query_row = convert<float>(query_row_raw) * scale_val;
                }

                // Initialize accumulators for this partition
                // For D=64, we use two 32-element halves for the accumulator
                simd<float, 32> acc_v_1 = 0.0f, acc_v_2 = 0.0f;  // For D=64
                simd<float, D> acc_v = 0.0f;  // For D!=64
                float softmax_sum = 0.0f;
                float max_score = -FLT_MAX;

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

                    // Apply logit softcap if enabled
                    if constexpr (use_logit_softcap) {
                        score = logit_softcap_val * esimd_tanh(score);
                    }

                    // Apply mask if present
                    if (mask_base) {
                        score += static_cast<float>(mask_base[kv_pos]);
                    }

                    // Online softmax update
                    // For D=64, we process V in two 32-element halves to avoid block_load<64> issues
                    if constexpr (D == 64) {
                        // Load V in two halves
                        simd<sycl::half, 32> v_row_h1 = block_load<sycl::half, 32>(V_row);
                        simd<sycl::half, 32> v_row_h2 = block_load<sycl::half, 32>(V_row + 32);
                        simd<float, 32> v1 = convert<float>(v_row_h1);
                        simd<float, 32> v2 = convert<float>(v_row_h2);

                        if (score <= max_score) {
                            // Use FTZ threshold to avoid numerical issues
                            float diff = score - max_score;
                            float exp_score = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                            acc_v_1 = acc_v_1 + v1 * exp_score;
                            acc_v_2 = acc_v_2 + v2 * exp_score;
                            softmax_sum += exp_score;
                        } else {
                            // Use FTZ threshold to avoid numerical issues
                            float diff = max_score - score;
                            float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
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
                        for (int p = 1; p < ESIMD_PARTITIONS; ++p) {
                            simd<float, 32> p_acc_1 = slm_block_load<float, 32>(slm_acc_offset + p * D * sizeof(float));
                            simd<float, 32> p_acc_2 = slm_block_load<float, 32>(slm_acc_offset + p * D * sizeof(float) + 32 * sizeof(float));
                            float p_max = slm_scalar_load<float>(slm_max_offset + p * sizeof(float));
                            float p_sum = slm_scalar_load<float>(slm_sum_offset + p * sizeof(float));

                            // Online softmax merge with FTZ threshold
                            if (p_max <= final_max) {
                                float diff = p_max - final_max;
                                float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
                                final_acc_1 = final_acc_1 + p_acc_1 * exp_factor;
                                final_acc_2 = final_acc_2 + p_acc_2 * exp_factor;
                                final_sum += p_sum * exp_factor;
                            } else {
                                float diff = final_max - p_max;
                                float exp_factor = diff >= SOFTMAX_FTZ_THRESHOLD ? esimd::exp(diff) : 0.0f;
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
                // Load Q as Q_type, then convert to float for computation
                // For D=64, block_load may not work correctly, so load in halves
                simd<float, D> query_row;
                if constexpr (D == 64 && std::is_same_v<Q_type, sycl::half>) {
                    simd<sycl::half, 32> q_row_h1 = block_load<sycl::half, 32>(Q_base);
                    simd<sycl::half, 32> q_row_h2 = block_load<sycl::half, 32>(Q_base + 32);
                    simd<float, 32> q1 = convert<float>(q_row_h1) * scale_val;
                    simd<float, 32> q2 = convert<float>(q_row_h2) * scale_val;
                    query_row = simd<float, D>(q1, q2);
                } else if constexpr (D == 64) {
                    simd<float, 32> q_row_1 = block_load<float, 32>(Q_base) * scale_val;
                    simd<float, 32> q_row_2 = block_load<float, 32>(Q_base + 32) * scale_val;
                    query_row = simd<float, D>(q_row_1, q_row_2);
                } else {
                    simd<Q_type, D> query_row_raw = block_load<Q_type, D>(Q_base);
                    query_row = convert<float>(query_row_raw) * scale_val;
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

                    // Load K and V rows to SLM
                    // For D=64, block_load may not work correctly, so load in halves
                    // and store each half separately to SLM
                    if constexpr (D == 64) {
                        simd<sycl::half, 32> k_h1 = block_load<sycl::half, 32>(K_row);
                        simd<sycl::half, 32> k_h2 = block_load<sycl::half, 32>(K_row + 32);
                        slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), k_h1);
                        slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half) + 32 * sizeof(sycl::half), k_h2);
                        simd<sycl::half, 32> v_h1 = block_load<sycl::half, 32>(V_row);
                        simd<sycl::half, 32> v_h2 = block_load<sycl::half, 32>(V_row + 32);
                        slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), v_h1);
                        slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half) + 32 * sizeof(sycl::half), v_h2);
                    } else {
                        simd<sycl::half, D> key_row = block_load<sycl::half, D>(K_row);
                        simd<sycl::half, D> value_row = block_load<sycl::half, D>(V_row);
                        slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), key_row);
                        slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), value_row);
                    }

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

                        // Apply mask if present
                        if (mask_base) {
                            score += static_cast<float>(mask_base[kv_start + r]);
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

                        // For D=64, block_load may not work correctly, so load in halves
                        // and store each half separately to SLM
                        if constexpr (D == 64) {
                            simd<sycl::half, 32> k_h1 = block_load<sycl::half, 32>(K_row);
                            simd<sycl::half, 32> k_h2 = block_load<sycl::half, 32>(K_row + 32);
                            slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), k_h1);
                            slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half) + 32 * sizeof(sycl::half), k_h2);
                            simd<sycl::half, 32> v_h1 = block_load<sycl::half, 32>(V_row);
                            simd<sycl::half, 32> v_h2 = block_load<sycl::half, 32>(V_row + 32);
                            slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), v_h1);
                            slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half) + 32 * sizeof(sycl::half), v_h2);
                        } else {
                            simd<sycl::half, D> key_row = block_load<sycl::half, D>(K_row);
                            simd<sycl::half, D> value_row = block_load<sycl::half, D>(V_row);
                            slm_block_store(key_slm_offset + tid * D * sizeof(sycl::half), key_row);
                            slm_block_store(value_slm_offset + tid * D * sizeof(sycl::half), value_row);
                        }
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

                        if (mask_base) {
                            score += static_cast<float>(mask_base[kv_start + r]);
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
