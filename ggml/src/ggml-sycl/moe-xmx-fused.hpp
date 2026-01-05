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

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx_fused {

using namespace sycl::ext::oneapi::experimental::matrix;

// Fused kernel configuration
struct FusedMoEConfig {
    int num_persistent_wgs = 0;       // nsm * 2 (from device info)
    int wg_size = 256;                // 256 default
    int tiles_m = 4;                  // 4 (from XMXCapabilities)
    int tiles_n = 4;                  // 4 (from XMXCapabilities)
    size_t slm_size = 65536;          // Device SLM budget

    static FusedMoEConfig from_device(int device_id) {
        const auto& dev_info = ggml_sycl_info().devices[device_id];
        const auto& xmx = dev_info.xmx_caps;

        FusedMoEConfig cfg;
        cfg.num_persistent_wgs = dev_info.nsm * 2;  // 2 WGs per XeCore
        cfg.wg_size = std::min(256, ggml_sycl_info().max_work_group_sizes[device_id]);
        cfg.tiles_m = xmx.optimal_tiles_m > 0 ? xmx.optimal_tiles_m : 4;
        cfg.tiles_n = xmx.optimal_tiles_n > 0 ? xmx.optimal_tiles_n : 4;
        cfg.slm_size = xmx.slm_size > 0 ? xmx.slm_size : 65536;
        return cfg;
    }
};

// Fused XMX MoE GEMM for Q8_0 weights
// Processes ALL experts in a single kernel launch using persistent work-groups
template<int TILES_M = 4, int TILES_N = 4>
void fused_xmx_moe_gemm_q8_0(
    // Expert weight data (Q8_0 format)
    const int8_t* all_expert_qs,      // [n_experts * out_dim * in_dim/32, 32] int8
    const sycl::half* all_expert_d,   // [n_experts * out_dim * in_dim/32] scales

    // Pre-quantized tokens
    const int8_t* q_tokens,           // [num_tokens, in_dim]
    const sycl::half* token_scales,   // [num_tokens, in_dim/32]

    // Sorted indices from moe_sort_tokens_by_expert
    const int32_t* sorted_token_ids,  // [total_sorted] original token indices
    const int32_t* expert_offsets,    // [n_experts + 1] cumulative offsets

    // Output
    sycl::half* output,               // [total_sorted, out_dim]

    // Dimensions
    int num_tokens,
    int n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t expert_stride,            // Bytes between expert weight blocks

    // Configuration
    const FusedMoEConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int TILE_M = TILES_M * XMX_M;  // 32
    constexpr int TILE_N = TILES_N * XMX_N;  // 64

    const int num_k_blocks = in_dim / XMX_K;
    const int n_output_tiles = (out_dim + TILE_N - 1) / TILE_N;
    const int num_sgs = cfg.wg_size / SG_SIZE;

    // Total sorted tokens (sum of all expert counts)
    // Read from expert_offsets[n_experts] on host
    int total_sorted = 0;
    queue.copy(&expert_offsets[n_experts], &total_sorted, 1).wait();

    if (total_sorted == 0) return;

    // Compute total work items across ALL experts
    int64_t total_work = 0;
    for (int e = 0; e < n_experts; e++) {
        int expert_start = 0, expert_end = 0;
        queue.copy(&expert_offsets[e], &expert_start, 1);
        queue.copy(&expert_offsets[e + 1], &expert_end, 1);
        queue.wait();
        int expert_tokens = expert_end - expert_start;
        total_work += static_cast<int64_t>(expert_tokens) * n_output_tiles;
    }

    if (total_work == 0) return;

    // Suppress unused variable warnings for TILE_M and num_sgs
    (void)TILE_M;
    (void)num_sgs;
    (void)num_tokens;
    (void)expert_stride;
    (void)total_work;

    queue.submit([&](sycl::handler& cgh) {
        // SLM allocations
        sycl::local_accessor<int8_t, 1> slm_token(sycl::range<1>(in_dim), cgh);
        sycl::local_accessor<sycl::half, 1> slm_token_scales(sycl::range<1>(num_k_blocks), cgh);
        sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        sycl::local_accessor<sycl::half, 1> slm_weight_scales(sycl::range<1>(TILE_N), cgh);
        sycl::local_accessor<int32_t, 1> slm_acc(sycl::range<1>(cfg.wg_size / SG_SIZE * XMX_M * XMX_N), cgh);

        const int wg_size_captured = cfg.wg_size;
        const int num_persistent_wgs_captured = cfg.num_persistent_wgs;

        cgh.parallel_for(
            sycl::nd_range<1>(cfg.num_persistent_wgs * cfg.wg_size, cfg.wg_size),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int group_id = item.get_group_linear_id();
                int tid = item.get_local_linear_id();
                int sg_id = sg.get_group_linear_id();
                int lane = sg.get_local_linear_id();

                // Persistent loop - compute work item offset
                for (int expert = 0; expert < n_experts; expert++) {
                    int expert_start = expert_offsets[expert];
                    int expert_end = expert_offsets[expert + 1];
                    int expert_tokens = expert_end - expert_start;

                    if (expert_tokens == 0) continue;

                    int64_t expert_work = static_cast<int64_t>(expert_tokens) * n_output_tiles;

                    for (int64_t local_work = group_id;
                         local_work < expert_work;
                         local_work += num_persistent_wgs_captured) {

                        int tile_idx = local_work % n_output_tiles;
                        int local_token_idx = local_work / n_output_tiles;
                        int sorted_idx = expert_start + local_token_idx;
                        int token_idx = sorted_token_ids[sorted_idx];

                        // Collaborative load of token into SLM
                        for (int i = tid; i < static_cast<int>(in_dim); i += wg_size_captured) {
                            slm_token[i] = q_tokens[token_idx * in_dim + i];
                        }
                        for (int i = tid; i < num_k_blocks; i += wg_size_captured) {
                            slm_token_scales[i] = token_scales[token_idx * num_k_blocks + i];
                        }
                        sycl::group_barrier(item.get_group());

                        // Expert weight pointers
                        const int8_t* expert_qs = all_expert_qs +
                            expert * (out_dim * num_k_blocks * XMX_K);
                        const sycl::half* expert_d = all_expert_d +
                            expert * (out_dim * num_k_blocks);

                        int col_start = tile_idx * TILE_N;

                        // Initialize accumulators
                        joint_matrix<sycl::sub_group, int32_t, use::accumulator,
                                     XMX_M, XMX_N> acc[TILES_M][TILES_N];
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                joint_matrix_fill(sg, acc[tm][tn], 0);
                            }
                        }

                        // K-dimension reduction
                        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                            // Load weight tile for this K-block
                            for (int i = tid; i < TILE_N * XMX_K; i += wg_size_captured) {
                                int col = i / XMX_K;
                                int k = i % XMX_K;
                                int out_col = col_start + col;
                                if (out_col < static_cast<int>(out_dim)) {
                                    slm_weights[i] = expert_qs[
                                        out_col * num_k_blocks * XMX_K + k_block * XMX_K + k];
                                }
                            }
                            for (int i = tid; i < TILE_N; i += wg_size_captured) {
                                int out_col = col_start + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    slm_weight_scales[i] = expert_d[
                                        out_col * num_k_blocks + k_block];
                                }
                            }
                            sycl::group_barrier(item.get_group());

                            // XMX multiply-accumulate
                            joint_matrix<sycl::sub_group, int8_t, use::a,
                                         XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b,
                                         XMX_K, XMX_N, layout::row_major> mat_b;

                            joint_matrix_load(sg, mat_a, &slm_token[k_block * XMX_K], XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                joint_matrix_load(sg, mat_b,
                                    &slm_weights[tn * XMX_N * XMX_K], XMX_K);

                                for (int tm = 0; tm < TILES_M; tm++) {
                                    joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                                }
                            }

                            sycl::group_barrier(item.get_group());
                        }

                        // Store results with scale application
                        int32_t* sg_acc_ptr = &slm_acc[sg_id * XMX_M * XMX_N];

                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                joint_matrix_store(sg, acc[tm][tn], sg_acc_ptr, XMX_N,
                                    layout::row_major);
                                sycl::group_barrier(sg);

                                if (tm == 0 && lane < XMX_N) {
                                    int out_col = col_start + tn * XMX_N + lane;
                                    if (out_col < static_cast<int>(out_dim)) {
                                        int32_t acc_val = sg_acc_ptr[lane];
                                        // TODO: Proper scale multiplication
                                        float result = static_cast<float>(acc_val);
                                        output[sorted_idx * out_dim + out_col] =
                                            sycl::half(result);
                                    }
                                }
                            }
                        }

                        sycl::group_barrier(item.get_group());
                    }
                }
            });
    });
}

} // namespace moe_xmx_fused

#endif // SYCL_XMX_MOE_AVAILABLE
