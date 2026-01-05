//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// moe-xmx-fused.hpp - Fused XMX MoE GEMM kernel with persistent work-groups
#pragma once

#include "common.hpp"
#include "moe-xmx.hpp"  // For MoEXMXConfig and preprocessing

#include <sycl/sycl.hpp>
#include <vector>

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx_fused {

using namespace sycl::ext::oneapi::experimental::matrix;

// Fused kernel configuration
struct FusedMoEConfig {
    int    num_persistent_wgs = 0;      // nsm * 2 (from device info)
    int    wg_size            = 256;    // 256 default
    int    tiles_m            = 4;      // 4 (from XMXCapabilities)
    int    tiles_n            = 4;      // 4 (from XMXCapabilities)
    size_t slm_size           = 65536;  // Device SLM budget

    static FusedMoEConfig from_device(int device_id) {
        const auto & dev_info = ggml_sycl_info().devices[device_id];
        const auto & xmx      = dev_info.xmx_caps;

        FusedMoEConfig cfg;
        cfg.num_persistent_wgs = dev_info.nsm * 2;  // 2 WGs per XeCore
        cfg.wg_size            = std::min(256, ggml_sycl_info().max_work_group_sizes[device_id]);
        cfg.tiles_m            = xmx.optimal_tiles_m > 0 ? xmx.optimal_tiles_m : 4;
        cfg.tiles_n            = xmx.optimal_tiles_n > 0 ? xmx.optimal_tiles_n : 4;
        cfg.slm_size           = xmx.slm_size > 0 ? xmx.slm_size : 65536;
        return cfg;
    }
};

// Fused XMX MoE GEMM for Q8_0 weights
// Processes ALL experts in a single kernel launch using persistent work-groups
template <int TILES_M = 4, int TILES_N = 4>
void fused_xmx_moe_gemm_q8_0(
    // Expert weight data (Q8_0 format)
    const int8_t *     all_expert_qs,  // [n_experts * out_dim * in_dim/32, 32] int8
    const sycl::half * all_expert_d,   // [n_experts * out_dim * in_dim/32] scales

    // Pre-quantized tokens
    const int8_t *     q_tokens,      // [num_tokens, in_dim]
    const sycl::half * token_scales,  // [num_tokens, in_dim/32]

    // Sorted indices from moe_sort_tokens_by_expert
    const int32_t * sorted_token_ids,  // [total_sorted] original token indices
    const int32_t * expert_offsets,    // [n_experts + 1] cumulative offsets

    // Output
    sycl::half * output,  // [total_sorted, out_dim]

    // Dimensions
    int     num_tokens,
    int     n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t expert_stride,  // Bytes between expert weight blocks

    // Configuration
    const FusedMoEConfig & cfg,
    sycl::queue &          queue) {
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int TILE_M  = TILES_M * XMX_M;  // 32
    constexpr int TILE_N  = TILES_N * XMX_N;  // 64

    const int num_k_blocks   = in_dim / XMX_K;
    const int n_output_tiles = (out_dim + TILE_N - 1) / TILE_N;
    const int num_sgs        = cfg.wg_size / SG_SIZE;

    // Copy expert_offsets to host ONCE before kernel launch
    std::vector<int32_t> h_offsets(n_experts + 1);
    queue.copy(expert_offsets, h_offsets.data(), n_experts + 1).wait();
    int total_sorted = h_offsets[n_experts];

    if (total_sorted == 0) {
        return;
    }

    // Compute total work items across ALL experts using host-side offsets
    int64_t total_work = 0;
    for (int e = 0; e < n_experts; e++) {
        int expert_tokens = h_offsets[e + 1] - h_offsets[e];
        total_work += static_cast<int64_t>(expert_tokens) * n_output_tiles;
    }

    if (total_work == 0) {
        return;
    }

    // Suppress unused variable warnings
    (void) TILE_M;
    (void) num_sgs;
    (void) num_tokens;
    (void) expert_stride;
    (void) total_work;

    queue.submit([&](sycl::handler & cgh) {
        // SLM allocations
        // Token data for current K-block only (not full in_dim - loaded per K-iteration)
        sycl::local_accessor<int8_t, 1>  slm_token(sycl::range<1>(XMX_K), cgh);
        // Token scale for current K-block (single value per token, but we process 1 token/WG)
        sycl::local_accessor<float, 1>   slm_token_scale(sycl::range<1>(1), cgh);
        // Weight tile for current K-block
        sycl::local_accessor<int8_t, 1>  slm_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        // Per-column weight scales for current K-block
        sycl::local_accessor<float, 1>   slm_weight_scales(sycl::range<1>(TILE_N), cgh);
        // Per-sub-group accumulator extraction buffer
        sycl::local_accessor<int32_t, 1> slm_acc(sycl::range<1>(cfg.wg_size / SG_SIZE * XMX_M * XMX_N), cgh);

        const int wg_size_captured            = cfg.wg_size;
        const int num_persistent_wgs_captured = cfg.num_persistent_wgs;

        cgh.parallel_for(
            sycl::nd_range<1>(cfg.num_persistent_wgs * cfg.wg_size, cfg.wg_size),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg       = item.get_sub_group();
                int  group_id = item.get_group_linear_id();
                int  tid      = item.get_local_linear_id();
                int  sg_id    = sg.get_group_linear_id();
                int  lane     = sg.get_local_linear_id();

                // Persistent loop - compute work item offset
                for (int expert = 0; expert < n_experts; expert++) {
                    int expert_start  = expert_offsets[expert];
                    int expert_end    = expert_offsets[expert + 1];
                    int expert_tokens = expert_end - expert_start;

                    if (expert_tokens == 0) {
                        continue;
                    }

                    int64_t expert_work = static_cast<int64_t>(expert_tokens) * n_output_tiles;

                    for (int64_t local_work = group_id; local_work < expert_work;
                         local_work += num_persistent_wgs_captured) {
                        int tile_idx        = local_work % n_output_tiles;
                        int local_token_idx = local_work / n_output_tiles;
                        int sorted_idx      = expert_start + local_token_idx;
                        int token_idx       = sorted_token_ids[sorted_idx];

                        // Expert weight pointers
                        const int8_t *     expert_qs = all_expert_qs + expert * (out_dim * num_k_blocks * XMX_K);
                        const sycl::half * expert_d  = all_expert_d + expert * (out_dim * num_k_blocks);

                        int col_start = tile_idx * TILE_N;

                        // Initialize int32 accumulators (reset per K-block)
                        joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_N];
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tn], 0);
                        }

                        // Float accumulators for precision across K-blocks
                        // For decode (single token), we only need TILES_N * XMX_N outputs
                        float float_acc[TILES_N * XMX_N] = { 0.0f };

                        // K-dimension reduction with per-K-block scale application
                        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                            // Load token data for this K-block only
                            // NOTE: q_tokens was quantized from tokens_sorted (sorted order),
                            // so index with sorted_idx (not token_idx which is original ID)
                            for (int i = tid; i < XMX_K; i += wg_size_captured) {
                                slm_token[i] = q_tokens[sorted_idx * in_dim + k_block * XMX_K + i];
                            }
                            // Load token scale for this K-block (single value)
                            if (tid == 0) {
                                slm_token_scale[0] =
                                    static_cast<float>(token_scales[sorted_idx * num_k_blocks + k_block]);
                            }

                            // Load weight tile for this K-block
                            for (int i = tid; i < TILE_N * XMX_K; i += wg_size_captured) {
                                int col     = i / XMX_K;
                                int k       = i % XMX_K;
                                int out_col = col_start + col;
                                if (out_col < static_cast<int>(out_dim)) {
                                    slm_weights[i] = expert_qs[out_col * num_k_blocks * XMX_K + k_block * XMX_K + k];
                                } else {
                                    slm_weights[i] = 0;
                                }
                            }
                            // Load per-column weight scales for this K-block
                            for (int i = tid; i < TILE_N; i += wg_size_captured) {
                                int out_col = col_start + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    slm_weight_scales[i] =
                                        static_cast<float>(expert_d[out_col * num_k_blocks + k_block]);
                                } else {
                                    slm_weight_scales[i] = 0.0f;
                                }
                            }
                            sycl::group_barrier(item.get_group());

                            // XMX multiply-accumulate
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            // Load mat_a ONCE outside TILES_N loop (FIX #3)
                            auto slm_token_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                          sycl::access::decorated::no>(&slm_token[0]);
                            joint_matrix_load(sg, mat_a, slm_token_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto slm_weights_ptr =
                                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                                             sycl::access::decorated::no>(
                                        &slm_weights[tn * XMX_N * XMX_K]);
                                joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                joint_matrix_mad(sg, acc[tn], mat_a, mat_b, acc[tn]);
                            }

                            // Per-K-block scale application (FIX #1)
                            // Extract int32 accumulator, apply scales, accumulate to float_acc
                            int32_t * sg_acc_ptr = &slm_acc[sg_id * XMX_M * XMX_N];
                            float     t_scale    = slm_token_scale[0];

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto acc_slm_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                            sycl::access::decorated::no>(sg_acc_ptr);
                                joint_matrix_store(sg, acc[tn], acc_slm_ptr, XMX_N, layout::row_major);
                                sycl::group_barrier(sg);

                                // For decode (single token), only row 0 matters (FIX #2)
                                // Extract XMX_N values from row 0 of the 8x16 accumulator
                                for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                    int     tile_col = i;
                                    float   w_scale  = slm_weight_scales[tn * XMX_N + tile_col];
                                    int32_t raw      = sg_acc_ptr[i];  // Row 0, col i
                                    float_acc[tn * XMX_N + tile_col] += raw * t_scale * w_scale;
                                }

                                // Reset int32 accumulator for next K-block
                                joint_matrix_fill(sg, acc[tn], 0);
                            }

                            sycl::group_barrier(item.get_group());
                        }

                        // Store final results
                        for (int tn = 0; tn < TILES_N; tn++) {
                            for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                int out_col = col_start + tn * XMX_N + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    output[sorted_idx * out_dim + out_col] = sycl::half(float_acc[tn * XMX_N + i]);
                                }
                            }
                        }

                        sycl::group_barrier(item.get_group());
                    }
                }
            });
    });
}

// ============================================================================
// MXFP4 Fused XMX MoE GEMM Kernel
// ============================================================================

// MXFP4 format constants
constexpr int MXFP4_PACKED_BYTES = 16;  // 16 packed bytes = 32 elements (4-bit each)

// Note: MXFP4_BLOCK_STRIDE (17) is only used in AoS format; SoA stores qs and e separately

// Fused XMX MoE GEMM for MXFP4 weights (SoA layout)
// Processes ALL experts in a single kernel launch using persistent work-groups
//
// IMPORTANT: This kernel is optimized for DECODE mode (batch=1, single token per expert).
// For prompt processing (batch > 1), use the per-expert launch path instead.
//
// SoA layout (per expert):
//   - qs: [nblocks * 16] packed nibble bytes contiguously
//   - e:  [nblocks] E8M0 exponents contiguously
// Where nblocks = out_dim * (in_dim / 32)
//
// Block indexing: block_idx = out_col * num_k_blocks + k_block
//   - qs offset: block_idx * 16
//   - e offset: block_idx
//
template <int TILES_M = 4, int TILES_N = 4>
void fused_xmx_moe_gemm_mxfp4_soa(
    // Expert weight data (MXFP4 SoA format)
    const uint8_t * all_expert_qs,  // [n_experts, nblocks, 16] packed nibbles
    const uint8_t * all_expert_e,   // [n_experts, nblocks] E8M0 exponents

    // Pre-quantized tokens (int8)
    const int8_t *     q_tokens,      // [num_tokens, in_dim]
    const sycl::half * token_scales,  // [num_tokens, in_dim/32]

    // Sorted indices from moe_sort_tokens_by_expert
    const int32_t * sorted_token_ids,  // [total_sorted] original token indices
    const int32_t * expert_offsets,    // [n_experts + 1] cumulative offsets

    // Output
    sycl::half * output,  // [total_sorted, out_dim]

    // Dimensions
    int     num_tokens,
    int     n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t expert_qs_stride,  // Bytes between expert packed nibble arrays
    int64_t expert_e_stride,   // Bytes between expert exponent arrays

    // Configuration
    const FusedMoEConfig & cfg,
    sycl::queue &          queue) {
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int TILE_M  = TILES_M * XMX_M;  // 32
    constexpr int TILE_N  = TILES_N * XMX_N;  // 64

    const int num_k_blocks   = in_dim / XMX_K;
    const int n_output_tiles = (out_dim + TILE_N - 1) / TILE_N;

    // Copy expert_offsets to host ONCE before kernel launch
    std::vector<int32_t> h_offsets(n_experts + 1);
    queue.copy(expert_offsets, h_offsets.data(), n_experts + 1).wait();
    int total_sorted = h_offsets[n_experts];

    if (total_sorted == 0) {
        return;
    }

    // Compute total work items across ALL experts using host-side offsets
    int64_t total_work = 0;
    for (int e = 0; e < n_experts; e++) {
        int expert_tokens = h_offsets[e + 1] - h_offsets[e];
        total_work += static_cast<int64_t>(expert_tokens) * n_output_tiles;
    }

    if (total_work == 0) {
        return;
    }

    // Suppress unused variable warnings
    (void) TILE_M;
    (void) num_tokens;
    (void) total_work;

    queue.submit([&](sycl::handler & cgh) {
        // SLM allocations
        // Token data for current K-block - needs full 8x32 matrix for joint_matrix_load
        sycl::local_accessor<int8_t, 1>  slm_token(sycl::range<1>(XMX_M * XMX_K), cgh);
        // Token scale for current K-block
        sycl::local_accessor<float, 1>   slm_token_scale(sycl::range<1>(1), cgh);
        // Dequantized weight tile (MXFP4 unpacked to int8 via LUT)
        sycl::local_accessor<int8_t, 1>  slm_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        // Per-column weight scales from E8M0 exponents
        sycl::local_accessor<float, 1>   slm_weight_scales(sycl::range<1>(TILE_N), cgh);
        // kvalues_mxfp4 LUT cached in SLM
        sycl::local_accessor<int8_t, 1>  slm_kvalues(sycl::range<1>(16), cgh);
        // Per-sub-group accumulator extraction buffer
        sycl::local_accessor<int32_t, 1> slm_acc(sycl::range<1>(cfg.wg_size / SG_SIZE * XMX_M * XMX_N), cgh);

        const int num_persistent_wgs_captured = cfg.num_persistent_wgs;

        cgh.parallel_for(
            sycl::nd_range<1>(cfg.num_persistent_wgs * cfg.wg_size, cfg.wg_size),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg       = item.get_sub_group();
                int  group_id = item.get_group_linear_id();
                int  sg_id    = sg.get_group_linear_id();
                int  lane     = sg.get_local_linear_id();

                // Load kvalues_mxfp4 LUT into SLM (once per work-group)
                if (sg_id == 0 && lane < 16) {
                    slm_kvalues[lane] = kvalues_mxfp4[lane];
                }
                sycl::group_barrier(item.get_group());

                // Early exit for non-zero sub-groups (they have no unique work in decode mode)
                // All sub-groups would compute and write to the same output location,
                // causing a race condition. Only sub-group 0 does the actual work.
                if (sg_id != 0) {
                    return;
                }

                // Persistent loop over all experts
                for (int expert = 0; expert < n_experts; expert++) {
                    int expert_start  = expert_offsets[expert];
                    int expert_end    = expert_offsets[expert + 1];
                    int expert_tokens = expert_end - expert_start;

                    if (expert_tokens == 0) {
                        continue;
                    }

                    int64_t expert_work = static_cast<int64_t>(expert_tokens) * n_output_tiles;

                    // Expert weight pointers (SoA layout)
                    const uint8_t * expert_qs = all_expert_qs + expert * expert_qs_stride;
                    const uint8_t * expert_e  = all_expert_e + expert * expert_e_stride;

                    for (int64_t local_work = group_id; local_work < expert_work;
                         local_work += num_persistent_wgs_captured) {
                        int tile_idx        = local_work % n_output_tiles;
                        int local_token_idx = local_work / n_output_tiles;
                        int sorted_idx      = expert_start + local_token_idx;
                        int token_idx       = sorted_token_ids[sorted_idx];

                        int col_start = tile_idx * TILE_N;

                        // Initialize int32 accumulators (reset per K-block)
                        joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_N];
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tn], 0);
                        }

                        // Float accumulators for precision across K-blocks
                        float float_acc[TILES_N * XMX_N] = { 0.0f };

                        // K-dimension reduction with per-K-block scale application
                        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                            // Load token data for this K-block into 8x32 matrix
                            // For decode (batch=1), we only have 1 token row but XMX needs 8 rows
                            // Row 0 gets the actual token data, rows 1-7 are zero-padded
                            // NOTE: q_tokens was quantized from tokens_sorted (sorted order),
                            // so index with sorted_idx (not token_idx which is original ID)
                            for (int i = lane; i < XMX_M * XMX_K; i += SG_SIZE) {
                                int row = i / XMX_K;
                                int col = i % XMX_K;
                                if (row == 0) {
                                    slm_token[i] = q_tokens[sorted_idx * in_dim + k_block * XMX_K + col];
                                } else {
                                    slm_token[i] = 0;  // Pad other rows with zeros
                                }
                            }
                            // Load token scale for this K-block
                            if (lane == 0) {
                                slm_token_scale[0] =
                                    static_cast<float>(token_scales[sorted_idx * num_k_blocks + k_block]);
                            }
                            sycl::group_barrier(sg);

                            // Load and dequantize MXFP4 weight tile for this K-block
                            // Only sub-group 0 is active, so use lane and SG_SIZE
                            // Each lane handles multiple columns to unpack (TILE_N=64, SG_SIZE=16)
                            for (int i = lane; i < TILE_N; i += SG_SIZE) {
                                int out_col = col_start + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    // SoA block index: out_col * num_k_blocks + k_block
                                    int64_t block_idx = static_cast<int64_t>(out_col) * num_k_blocks + k_block;

                                    // E8M0 exponent -> scale (includes 0.5 factor for kvalues)
                                    uint8_t e8m0         = expert_e[block_idx];
                                    slm_weight_scales[i] = sycl_e8m0_to_fp32_half(e8m0);

                                    // Unpack 32 MXFP4 values from 16 packed bytes
                                    const uint8_t * packed = expert_qs + block_idx * MXFP4_PACKED_BYTES;
                                    for (int k = 0; k < XMX_K / 2; k++) {
                                        uint8_t byte                    = packed[k];
                                        // Low nibble -> element k
                                        slm_weights[i * XMX_K + k]      = slm_kvalues[byte & 0xF];
                                        // High nibble -> element k + 16
                                        slm_weights[i * XMX_K + k + 16] = slm_kvalues[byte >> 4];
                                    }
                                } else {
                                    slm_weight_scales[i] = 0.0f;
                                    for (int k = 0; k < XMX_K; k++) {
                                        slm_weights[i * XMX_K + k] = 0;
                                    }
                                }
                            }
                            sycl::group_barrier(sg);

                            // XMX multiply-accumulate
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            // Load mat_a ONCE outside TILES_N loop
                            auto slm_token_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                          sycl::access::decorated::no>(&slm_token[0]);
                            joint_matrix_load(sg, mat_a, slm_token_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto slm_weights_ptr =
                                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                                             sycl::access::decorated::no>(
                                        &slm_weights[tn * XMX_N * XMX_K]);
                                joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                joint_matrix_mad(sg, acc[tn], mat_a, mat_b, acc[tn]);
                            }

                            // Per-K-block scale application
                            // Extract int32 accumulator, apply scales, accumulate to float_acc
                            int32_t * sg_acc_ptr = &slm_acc[sg_id * XMX_M * XMX_N];
                            float     t_scale    = slm_token_scale[0];

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto acc_slm_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                            sycl::access::decorated::no>(sg_acc_ptr);
                                joint_matrix_store(sg, acc[tn], acc_slm_ptr, XMX_N, layout::row_major);
                                sycl::group_barrier(sg);

                                // For decode (single token), only row 0 matters
                                for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                    int     tile_col = i;
                                    float   w_scale  = slm_weight_scales[tn * XMX_N + tile_col];
                                    int32_t raw      = sg_acc_ptr[i];  // Row 0, col i
                                    float_acc[tn * XMX_N + tile_col] += raw * t_scale * w_scale;
                                }

                                // Reset int32 accumulator for next K-block
                                joint_matrix_fill(sg, acc[tn], 0);
                            }

                            sycl::group_barrier(sg);
                        }

                        // Store final results
                        for (int tn = 0; tn < TILES_N; tn++) {
                            for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                int out_col = col_start + tn * XMX_N + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    output[sorted_idx * out_dim + out_col] = sycl::half(float_acc[tn * XMX_N + i]);
                                }
                            }
                        }

                        sycl::group_barrier(sg);
                    }
                }
            });
    });
}

}  // namespace moe_xmx_fused

// Entry point for fused XMX MoE dispatch
// Returns true if fused path was used, false to fallback
inline bool try_fused_xmx_moe_q8_0(const int8_t *     all_expert_qs,
                                   const sycl::half * all_expert_d,
                                   const int8_t *     q_tokens,
                                   const sycl::half * token_scales,
                                   const int32_t *    sorted_token_ids,
                                   const int32_t *    expert_offsets,
                                   sycl::half *       output,
                                   int                num_tokens,
                                   int                n_experts,
                                   int64_t            out_dim,
                                   int64_t            in_dim,
                                   int64_t            expert_stride,
                                   int                device_id,
                                   sycl::queue &      queue) {
    // Get device config
    const auto & dev_info = ggml_sycl_info().devices[device_id];
    if (!dev_info.xmx_caps.supported) {
        return false;
    }

    moe_xmx_fused::FusedMoEConfig cfg = moe_xmx_fused::FusedMoEConfig::from_device(device_id);

    GGML_SYCL_DEBUG(
        "[MoE-Fused] Launching fused Q8_0 kernel: "
        "tokens=%d experts=%d out=%ld in=%ld wgs=%d\n",
        num_tokens, n_experts, out_dim, in_dim, cfg.num_persistent_wgs);

    moe_xmx_fused::fused_xmx_moe_gemm_q8_0<4, 4>(all_expert_qs, all_expert_d, q_tokens, token_scales, sorted_token_ids,
                                                 expert_offsets, output, num_tokens, n_experts, out_dim, in_dim,
                                                 expert_stride, cfg, queue);

    return true;
}

// Entry point for fused XMX MoE dispatch (MXFP4 SoA layout)
// Returns true if fused path was used, false to fallback
inline bool try_fused_xmx_moe_mxfp4_soa(const uint8_t *    all_expert_qs,
                                        const uint8_t *    all_expert_e,
                                        const int8_t *     q_tokens,
                                        const sycl::half * token_scales,
                                        const int32_t *    sorted_token_ids,
                                        const int32_t *    expert_offsets,
                                        sycl::half *       output,
                                        int                num_tokens,
                                        int                n_experts,
                                        int64_t            out_dim,
                                        int64_t            in_dim,
                                        int64_t            expert_qs_stride,
                                        int64_t            expert_e_stride,
                                        int                device_id,
                                        sycl::queue &      queue) {
    const auto & dev_info = ggml_sycl_info().devices[device_id];
    if (!dev_info.xmx_caps.supported) {
        return false;
    }

    moe_xmx_fused::FusedMoEConfig cfg = moe_xmx_fused::FusedMoEConfig::from_device(device_id);

    GGML_SYCL_DEBUG(
        "[MoE-Fused] Launching fused MXFP4 SoA kernel: "
        "tokens=%d experts=%d out=%ld in=%ld wgs=%d qs_stride=%ld e_stride=%ld\n",
        num_tokens, n_experts, out_dim, in_dim, cfg.num_persistent_wgs, expert_qs_stride, expert_e_stride);

    moe_xmx_fused::fused_xmx_moe_gemm_mxfp4_soa<4, 4>(all_expert_qs, all_expert_e, q_tokens, token_scales,
                                                      sorted_token_ids, expert_offsets, output, num_tokens, n_experts,
                                                      out_dim, in_dim, expert_qs_stride, expert_e_stride, cfg, queue);

    return true;
}

#endif  // SYCL_XMX_MOE_AVAILABLE
