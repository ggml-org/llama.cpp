// mmq-xmx.hpp - XMX-accelerated MMQ kernel for Q8_0
//
// Uses Intel joint_matrix API for XMX tensor core acceleration.
// Gated by GGML_SYCL_XMX_MMQ=1 environment variable.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
#pragma once

#include "common.hpp"

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define SYCL_XMX_MMQ_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define SYCL_XMX_MMQ_AVAILABLE 0
#endif

#if SYCL_XMX_MMQ_AVAILABLE

namespace mmq_xmx {

using namespace sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// Environment Gate
// =============================================================================

inline bool use_xmx_mmq_kernel() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_MMQ");
        enabled          = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// MMQ XMX Configuration
// =============================================================================

struct MMQXMXConfig {
    // Hardware parameters (from XMXCapabilities)
    int M = 8;   // Tile rows
    int N = 16;  // Tile cols
    int K = 32;  // Reduction dim (matches Q8_0 block size)

    // Tunable parameters
    int tiles_m = 4;     // Tiles per WG in M dimension (output rows)
    int tiles_n = 4;     // Tiles per WG in N dimension (output cols)
    int nwarps  = 8;     // Number of warps (sub-groups) per work-group
    int sg_size = 32;    // Sub-group size (WARP_SIZE)

    static MMQXMXConfig from_capabilities(const XMXCapabilities & caps) {
        MMQXMXConfig cfg;
        if (caps.M > 0) {
            cfg.M = static_cast<int>(caps.M);
        }
        if (caps.N > 0) {
            cfg.N = static_cast<int>(caps.N);
        }
        if (caps.K > 0) {
            cfg.K = static_cast<int>(caps.K);
        }
        cfg.tiles_m = caps.optimal_tiles_m;
        cfg.tiles_n = caps.optimal_tiles_n;
        return cfg;
    }
};

// =============================================================================
// Q8_0/Q8_1 Block Constants
// =============================================================================

// Note: QK8_0 and QK8_1 are defined as macros in ggml-common.h
// We use the macros directly and define block size constants here
constexpr int XMX_Q8_0_BLOCK_SIZE = 34;   // 2 bytes fp16 scale + 32 int8 values
constexpr int XMX_Q8_0_SCALE_SIZE = 2;    // fp16 scale at start of block

constexpr int XMX_Q8_1_BLOCK_SIZE = 36;   // 2 bytes scale + 2 bytes sum + 32 int8
constexpr int XMX_Q8_1_QS_OFFSET  = 4;    // int8 values start at offset 4

// =============================================================================
// XMX Q8_0 Kernel
//
// Computes: dst[nrows_x, ncols_y] = vx[nrows_x, ncols_x] @ vy[ncols_y, ncols_x]^T
//
// Where:
//   vx: Q8_0 weights [nrows_x, ncols_x/32 blocks] - quantized weights
//   vy: Q8_1 activations [ncols_y, ncols_x/32 blocks] - quantized activations
//   dst: float output [nrows_x, ncols_y] stored column-major
//
// Mapping to GEMM: C[M,N] = A[M,K] @ B[N,K]^T
//   M = nrows_x (output rows = weight rows)
//   N = ncols_y (batch size = activation rows)
//   K = ncols_x (reduction dim = weight cols = activation cols)
//
// Grid layout (matches existing MMQ kernels):
//   block_nums(1, block_num_y, block_num_x) where:
//     block_num_x = (nrows_x + mmq_y - 1) / mmq_y  (M dimension tiles)
//     block_num_y = (ncols_y + mmq_x - 1) / mmq_x  (N dimension tiles)
//
// =============================================================================

template <int mmq_x = 64, int mmq_y = 32, int nwarps = 8>
void launch_mmq_xmx_q8_0(const void * __restrict__ vx,  // Q8_0 weights
                         const void * __restrict__ vy,  // Q8_1 activations
                         float * __restrict__ dst,      // Output
                         int                    ncols_x,
                         int                    nrows_x,
                         int                    ncols_y,
                         int                    nrows_y,
                         int                    nrows_dst,
                         sycl::queue &          stream) {
    // XMX tile dimensions (INT8: 8x16x32)
    constexpr int XMX_M   = 8;
    constexpr int XMX_N   = 16;
    constexpr int XMX_K   = 32;
    constexpr int SG_SIZE = 16;  // Sub-group size for joint_matrix

    // Tiles per work-group
    constexpr int TILES_M = mmq_y / XMX_M;  // 32/8 = 4
    constexpr int TILES_N = mmq_x / XMX_N;  // 64/16 = 4

    // Elements per XMX tile
    constexpr int TILE_ELEMS = XMX_M * XMX_N;  // 8 * 16 = 128

    // Grid dimensions (match existing MMQ pattern)
    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;  // M tiles
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;  // N tiles

    sycl::range<3> block_nums(1, block_num_y, block_num_x);
    sycl::range<3> block_dims(1, nwarps, SG_SIZE);

    const int num_k_blocks = ncols_x / XMX_K;

    stream.submit([&](sycl::handler & cgh) {
        // SLM for weight tiles (Q8_0 int8 values)
        // Size: TILES_M * XMX_M * XMX_K = 4 * 8 * 32 = 1024 bytes
        constexpr int                   slm_weights_size = TILES_M * XMX_M * XMX_K;
        sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

        // SLM for activation tiles (Q8_1 int8 values)
        // Size: TILES_N * XMX_N * XMX_K = 4 * 16 * 32 = 2048 bytes
        constexpr int                   slm_tokens_size = TILES_N * XMX_N * XMX_K;
        sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(slm_tokens_size), cgh);

        // SLM for scales
        sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILES_M * XMX_M), cgh);
        sycl::local_accessor<float, 1> slm_token_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

        // SLM for accumulator extraction (per sub-group to avoid conflicts)
        constexpr int                    slm_acc_per_sg = TILE_ELEMS;
        const int                        slm_acc_size   = nwarps * slm_acc_per_sg;
        sycl::local_accessor<int32_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg    = item_ct1.get_sub_group();
                int  sg_id = item_ct1.get_local_id(1);  // warp index
                int  lane  = sg.get_local_linear_id();  // lane within warp

                // Work-group position (matches existing MMQ pattern)
                const int row_dst_0 = item_ct1.get_group(2) * mmq_y;  // M dimension
                const int col_dst_0 = item_ct1.get_group(1) * mmq_x;  // N dimension

                // Cast input pointers
                const uint8_t * vx_ptr = static_cast<const uint8_t *>(vx);
                const uint8_t * vy_ptr = static_cast<const uint8_t *>(vy);

                // Float accumulators (per-thread, for all elements in all tiles)
                float float_acc[TILES_M][TILES_N][TILE_ELEMS];
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        for (int i = 0; i < TILE_ELEMS; i++) {
                            float_acc[tm][tn][i] = 0.0f;
                        }
                    }
                }

                // K-dimension reduction loop
                for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                    // === Cooperative Loading: Q8_0 weights to SLM ===
                    {
                        const int load_per_warp = slm_weights_size / nwarps;
                        const int warp_offset   = sg_id * load_per_warp;
                        for (int i = lane; i < load_per_warp; i += SG_SIZE) {
                            int idx = warp_offset + i;
                            if (idx < slm_weights_size) {
                                int tile_row = idx / XMX_K;          // 0..31 within mmq_y
                                int tile_k   = idx % XMX_K;          // 0..31
                                int row      = row_dst_0 + tile_row;

                                int8_t val = 0;
                                if (row < nrows_x) {
                                    int64_t         block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                                    const uint8_t * block_ptr = vx_ptr + block_idx * XMX_Q8_0_BLOCK_SIZE;
                                    val                       = static_cast<int8_t>(block_ptr[XMX_Q8_0_SCALE_SIZE + tile_k]);
                                }
                                slm_weights[idx] = val;
                            }
                        }
                    }

                    // === Load Q8_0 weight scales ===
                    if (sg_id == 0) {
                        for (int i = lane; i < TILES_M * XMX_M; i += SG_SIZE) {
                            int   row   = row_dst_0 + i;
                            float scale = 0.0f;
                            if (row < nrows_x) {
                                int64_t         block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                                const uint8_t * block_ptr = vx_ptr + block_idx * XMX_Q8_0_BLOCK_SIZE;
                                uint16_t        bits      = block_ptr[0] | (static_cast<uint16_t>(block_ptr[1]) << 8);
                                sycl::half      h;
                                std::memcpy(&h, &bits, sizeof(h));
                                scale = static_cast<float>(h);
                            }
                            slm_weight_scales[i] = scale;
                        }
                    }

                    // === Cooperative Loading: Q8_1 activations to SLM ===
                    {
                        const int load_per_warp = slm_tokens_size / nwarps;
                        const int warp_offset   = sg_id * load_per_warp;
                        for (int i = lane; i < load_per_warp; i += SG_SIZE) {
                            int idx = warp_offset + i;
                            if (idx < slm_tokens_size) {
                                int tile_col = idx / XMX_K;          // 0..63 within mmq_x
                                int tile_k   = idx % XMX_K;          // 0..31
                                int col      = col_dst_0 + tile_col; // batch index

                                int8_t val = 0;
                                if (col < ncols_y) {
                                    int64_t         block_idx = static_cast<int64_t>(col) * num_k_blocks + k_block;
                                    const uint8_t * block_ptr = vy_ptr + block_idx * XMX_Q8_1_BLOCK_SIZE;
                                    val                       = static_cast<int8_t>(block_ptr[XMX_Q8_1_QS_OFFSET + tile_k]);
                                }
                                slm_tokens[idx] = val;
                            }
                        }
                    }

                    // === Load Q8_1 activation scales ===
                    if (sg_id == 1 % nwarps) {
                        for (int i = lane; i < TILES_N * XMX_N; i += SG_SIZE) {
                            int   col   = col_dst_0 + i;
                            float scale = 0.0f;
                            if (col < ncols_y) {
                                int64_t         block_idx = static_cast<int64_t>(col) * num_k_blocks + k_block;
                                const uint8_t * block_ptr = vy_ptr + block_idx * XMX_Q8_1_BLOCK_SIZE;
                                uint16_t        bits      = block_ptr[0] | (static_cast<uint16_t>(block_ptr[1]) << 8);
                                sycl::half      h;
                                std::memcpy(&h, &bits, sizeof(h));
                                scale = static_cast<float>(h);
                            }
                            slm_token_scales[i] = scale;
                        }
                    }

                    // Barrier: ensure all data is loaded to SLM
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    // === XMX Computation (sub-group 0 only computes) ===
                    if (sg_id == 0) {
                        // Declare joint matrices
                        joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major>  mat_a;
                        joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major>  mat_b;
                        joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N>          acc;

                        // Pointer for this sub-group's accumulator SLM region
                        int32_t * acc_slm_raw =
                            const_cast<int32_t *>(&slm_acc[sg_id * slm_acc_per_sg]);
                        auto acc_slm_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                    sycl::access::decorated::no>(acc_slm_raw);

                        // Iterate over tile positions
                        for (int tm = 0; tm < TILES_M; tm++) {
                            int row = row_dst_0 + tm * XMX_M;
                            if (row >= nrows_x) continue;

                            // Load mat_a (weights) from SLM - row-major
                            auto slm_weights_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                            sycl::access::decorated::no>(
                                &slm_weights[tm * XMX_M * XMX_K]);
                            joint_matrix_load(sg, mat_a, slm_weights_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                int col = col_dst_0 + tn * XMX_N;
                                if (col >= ncols_y) continue;

                                // Load mat_b (activations) from SLM - col-major
                                auto slm_tokens_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                                               sycl::access::decorated::no>(
                                    &slm_tokens[tn * XMX_N * XMX_K]);
                                joint_matrix_load(sg, mat_b, slm_tokens_ptr, XMX_K);

                                // Initialize and compute
                                joint_matrix_fill(sg, acc, 0);
                                joint_matrix_mad(sg, acc, mat_a, mat_b, acc);

                                // Store accumulator to SLM for extraction
                                joint_matrix_store(sg, acc, acc_slm_ptr, XMX_N, layout::row_major);

                                // Sub-group barrier to ensure store is complete
                                sycl::group_barrier(sg);

                                // Extract all elements and apply scales
                                for (int i = lane; i < TILE_ELEMS; i += SG_SIZE) {
                                    int   elem_row = i / XMX_N;
                                    int   elem_col = i % XMX_N;
                                    float d_x      = slm_weight_scales[tm * XMX_M + elem_row];
                                    float d_y      = slm_token_scales[tn * XMX_N + elem_col];

                                    float_acc[tm][tn][i] += static_cast<float>(acc_slm_raw[i]) * d_x * d_y;
                                }

                                sycl::group_barrier(sg);
                            }
                        }
                    }

                    // Barrier before next K-block
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }

                // === Store Results ===
                // Only sg_id 0 has the computed results
                if (sg_id == 0) {
                    for (int tm = 0; tm < TILES_M; tm++) {
                        int row_base = row_dst_0 + tm * XMX_M;
                        if (row_base >= nrows_x) continue;

                        for (int tn = 0; tn < TILES_N; tn++) {
                            int col_base = col_dst_0 + tn * XMX_N;
                            if (col_base >= ncols_y) continue;

                            // Store all elements from this tile
                            for (int i = lane; i < TILE_ELEMS; i += SG_SIZE) {
                                int tile_row = i / XMX_N;
                                int tile_col = i % XMX_N;
                                int row      = row_base + tile_row;
                                int col      = col_base + tile_col;

                                if (row < nrows_x && col < ncols_y) {
                                    // Output: dst[nrows_x, ncols_y] column-major (matches existing MMQ kernels)
                                    dst[static_cast<int64_t>(col) * nrows_dst + row] = float_acc[tm][tn][i];
                                }
                            }
                        }
                    }
                }
            });
    });

    GGML_UNUSED(nrows_y);
}

// =============================================================================
// Dispatch Helper
// =============================================================================

inline bool can_use_xmx_q8_0(const XMXCapabilities & caps, int ncols_x) {
    // Requirements for XMX path:
    // 1. XMX hardware support
    // 2. INT8 type support
    // 3. K dimension aligns with XMX_K (32)
    // 4. Environment gate enabled
    return caps.supported && caps.supports_int8 && (ncols_x % 32 == 0) && use_xmx_mmq_kernel();
}

}  // namespace mmq_xmx

#endif  // SYCL_XMX_MMQ_AVAILABLE
