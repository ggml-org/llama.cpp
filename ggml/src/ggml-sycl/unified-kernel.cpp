//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Implementation for SYCL Matmul
//
// This file implements the unified matmul kernel supporting:
// - Q4_0 quantization (scalar and XMX paths)
// - Tile-based computation with SLM staging
// - Boundary handling for non-aligned dimensions
// - XMX dpas acceleration via Intel joint_matrix extensions
//
// XMX Path:
// - Enabled when args.use_xmx=true and hardware supports it
// - Uses 8x16x8 tile dimensions for half precision dpas
// - Falls back to scalar path when XMX is unavailable or dimensions don't fit
//

#include "unified-kernel.hpp"
#include "common.hpp"  // For ggml_sycl_info() and GGML_SYCL_DEBUG

#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace ggml_sycl_unified {

// =============================================================================
// XMXConfig::from_device() Implementation
// =============================================================================
// Queries hardware-specific XMX capabilities with robust edge case handling.

XMXConfig XMXConfig::from_device(int device_id) {
    XMXConfig cfg;  // Start with safe defaults

    // Edge case: device_id < 0 returns default config
    if (device_id < 0) {
        GGML_SYCL_DEBUG("[XMXConfig] device_id=%d < 0, returning default config\n", device_id);
        return cfg;
    }

    // Edge case: device_id >= device_count returns default config
    // Note: ggml_sycl_info() is in global namespace (defined in common.hpp)
    const auto & info = ::ggml_sycl_info();
    if (device_id >= info.device_count) {
        GGML_SYCL_DEBUG("[XMXConfig] device_id=%d >= device_count=%d, returning default config\n",
                        device_id, info.device_count);
        return cfg;
    }

    // Safe to access device info now
    const auto & dev = info.devices[device_id];
    const auto & xmx = dev.xmx_caps;

    // Copy hardware capability flags
    cfg.supported     = xmx.supported;
    cfg.supports_int8 = xmx.supports_int8;
    cfg.supports_fp16 = xmx.supports_fp16;

    // Copy nsm (compute units)
    cfg.nsm = dev.nsm > 0 ? dev.nsm : 20;  // Fallback to 20 if 0

    // Edge case: slm_size = 0 should use default
    cfg.slm_size = xmx.slm_size > 0 ? xmx.slm_size : 65536;

    // Edge case: M/N/K = 0 should use fallback defaults
    // XMX dimensions: Use queried values if valid, otherwise defaults
    cfg.xmx_m = (xmx.M > 0) ? xmx.M : 8;
    cfg.xmx_n = (xmx.N > 0) ? xmx.N : 16;

    // K dimension depends on data type:
    // - For INT8: Use queried K if valid (expected: 32)
    // - For FP16: Always 16 (SystolicDepth(8) x OpsPerChannel(2))
    cfg.xmx_k_int8 = (xmx.K > 0) ? xmx.K : 32;
    cfg.xmx_k_fp16 = 16;  // Fixed for FP16

    // Derived: double buffer feasibility
    // Double buffer if SLM can hold 2x tile buffers (conservative: 50% of SLM)
    // Tile buffer = M x K x sizeof(half) for activations + N x K x sizeof(half) for weights
    size_t tile_size = cfg.xmx_m * cfg.xmx_k_int8 * sizeof(sycl::half) +
                       cfg.xmx_n * cfg.xmx_k_int8 * sizeof(sycl::half);
    cfg.use_double_buffer = (2 * tile_size) < (cfg.slm_size / 2);

    // Default tiles per work-item (can be tuned later)
    cfg.tiles_per_workitem = 1;

    GGML_SYCL_DEBUG("[XMXConfig] device=%d: M=%zu N=%zu K_INT8=%zu K_FP16=%zu SLM=%zu nsm=%d "
                    "supported=%d int8=%d fp16=%d double_buf=%d\n",
                    device_id, cfg.xmx_m, cfg.xmx_n, cfg.xmx_k_int8, cfg.xmx_k_fp16,
                    cfg.slm_size, cfg.nsm, cfg.supported, cfg.supports_int8,
                    cfg.supports_fp16, cfg.use_double_buffer);

    return cfg;
}

static bool ggml_sycl_unified_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_DEBUG");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// Kernel Class Names for Profiling
// =============================================================================

template <int TILE_M, int TILE_N, int TILE_K, bool USE_XMX>
class unified_matmul_kernel_name;

// Separate name for fallback to avoid ODR violation
class unified_matmul_kernel_fallback;

// =============================================================================
// Q4_0 Dequantization Helper
// =============================================================================

/**
 * Dequantize a single Q4_0 weight value.
 *
 * Q4_0 block layout:
 * - qs[0..15]: 16 bytes containing 32 4-bit values
 * - For index i < 16: value = (qs[i] & 0xF) - 8
 * - For index i >= 16: value = (qs[i-16] >> 4) - 8
 *
 * @param block Pointer to Q4_0 block
 * @param i     Index within block (0..31)
 * @return Dequantized float value
 */
SYCL_EXTERNAL inline float dequant_q4_0(const block_q4_0_unified * block, int i) {
    const float d = static_cast<float>(block->d);
    int         qs_val;
    if (i < 16) {
        qs_val = block->qs[i] & 0x0F;
    } else {
        qs_val = block->qs[i - 16] >> 4;
    }
    return static_cast<float>(qs_val - 8) * d;
}

/**
 * Dequantize Q4_0 weight to half precision for XMX.
 *
 * @param block Pointer to Q4_0 block
 * @param i     Index within block (0..31)
 * @return Dequantized half value
 */
SYCL_EXTERNAL inline sycl::half dequant_q4_0_half(const block_q4_0_unified * block, int i) {
    const sycl::half d = block->d;
    int              qs_val;
    if (i < 16) {
        qs_val = block->qs[i] & 0x0F;
    } else {
        qs_val = block->qs[i - 16] >> 4;
    }
    return static_cast<sycl::half>(qs_val - 8) * d;
}

// =============================================================================
// XMX Kernel Class Names
// =============================================================================

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
template <int TILE_M, int TILE_N, int TILE_K>
class unified_matmul_xmx_kernel_name;
#endif

// =============================================================================
// Unified Matmul Kernel - Optimized XMX Path
// =============================================================================

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// Namespace alias for brevity
namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

/**
 * Optimized XMX-accelerated matmul kernel using joint_matrix.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * OPTIMIZATION: Direct joint_matrix accumulation
 * ==============================================
 * Key optimizations over the previous version:
 * 1. Larger work-groups with multiple sub-groups for better occupancy
 * 2. Direct joint_matrix_store to output without intermediate SLM extraction
 * 3. Streaming loads with reduced synchronization
 * 4. Full K-dimension processed with single accumulator
 *
 * Work distribution:
 * - Each work-group covers a TILE_M x TILE_N output region
 * - Sub-groups process XMX tiles within the work-group tile
 * - All sub-groups cooperate on loading, compute is distributed
 *
 * @tparam TILE_M  M tile size (must be multiple of 8)
 * @tparam TILE_N  N tile size (must be multiple of 16)
 * @tparam TILE_K  K tile size (must be multiple of XMX_TILE_K=16)
 */
template <int TILE_M, int TILE_N, int TILE_K>
SYCL_EXTERNAL void unified_matmul_xmx_kernel_impl(sycl::nd_item<2>                    item,
                                                   const UnifiedKernelArgs             args,
                                                   sycl::local_accessor<sycl::half, 1> slm_weights,
                                                   sycl::local_accessor<sycl::half, 1> slm_activations,
                                                   sycl::local_accessor<float, 1>      slm_acc_out) {
    auto sg = item.get_sub_group();

    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension (output rows)
    const int tile_col = item.get_group(1);  // N dimension (output columns)

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);
    const int local_linear = local_row * local_size_col + local_col;
    const int local_total = local_size_row * local_size_col;

    // Sub-group info
    const int sg_id = sg.get_group_linear_id();
    const int lane = sg.get_local_linear_id();

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;  // Starting output row
    const int64_t n_start = tile_col * TILE_N;  // Starting output column

    // Number of K tiles
    const int k_tiles = (args.K + TILE_K - 1) / TILE_K;
    const int k_blocks_per_row = args.K / UNIFIED_QK4_0;

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // XMX tile dimensions and counts
    constexpr int NUM_TILES_M = TILE_M / XMX_TILE_M;
    constexpr int NUM_TILES_N = TILE_N / XMX_TILE_N;
    constexpr int NUM_K_STEPS = TILE_K / XMX_TILE_K;

    // Each sub-group handles one (tm, tn) output tile
    // Assign sub-groups to output tiles in row-major order
    const int num_output_tiles = NUM_TILES_M * NUM_TILES_N;
    const int num_subgroups = (local_total + XMX_SUBGROUP_SIZE - 1) / XMX_SUBGROUP_SIZE;

    // Joint matrix declarations
    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half,
                           sycl_xmx::use::a, XMX_TILE_M, XMX_TILE_K,
                           sycl_xmx::layout::row_major> mat_a;
    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half,
                           sycl_xmx::use::b, XMX_TILE_K, XMX_TILE_N,
                           sycl_xmx::layout::col_major> mat_b;
    sycl_xmx::joint_matrix<sycl::sub_group, float,
                           sycl_xmx::use::accumulator, XMX_TILE_M, XMX_TILE_N> acc;

    // Initialize accumulator to zero
    sycl_xmx::joint_matrix_fill(sg, acc, 0.0f);

    // Determine which output tile this sub-group handles
    // Each sub-group processes one XMX output tile (8x16)
    const int my_tile_idx = sg_id % num_output_tiles;
    const int my_tm = my_tile_idx / NUM_TILES_N;
    const int my_tn = my_tile_idx % NUM_TILES_N;
    const int m_base = my_tm * XMX_TILE_M;
    const int n_base = my_tn * XMX_TILE_N;

    // K-loop: iterate over K dimension in tiles
    for (int kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * TILE_K;
        const int64_t k_end = sycl::min(k_start + TILE_K, args.K);
        const int     k_len = static_cast<int>(k_end - k_start);

        // ==== Cooperative Load: All threads load data to SLM ====
        // Load weights [TILE_N x TILE_K], but only up to k_len valid K elements
        for (int idx = local_linear; idx < TILE_N * TILE_K; idx += local_total) {
            const int n_off = idx / TILE_K;
            const int k_off = idx % TILE_K;
            const int64_t n_global = n_start + n_off;

            sycl::half w = sycl::half(0.0f);
            // Load only valid N/K combinations:
            // - n_off < TILE_N (always true due to loop structure)
            // - n_global < args.N (boundary check on N)
            // - k_off < k_len (only load actual K data for this tile)
            if (n_global < args.N && k_off < k_len) {
                const int64_t k_global = k_start + k_off;
                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);
                w = dequant_q4_0_half(&weights[block_idx], idx_in_block);
            }
            slm_weights[n_off * TILE_K + k_off] = w;
        }

        // Load activations [TILE_M x TILE_K], but only up to k_len valid K elements
        for (int idx = local_linear; idx < TILE_M * TILE_K; idx += local_total) {
            const int m_off = idx / TILE_K;
            const int k_off = idx % TILE_K;
            const int64_t m_global = m_start + m_off;

            sycl::half a = sycl::half(0.0f);
            // Load only valid M/K combinations:
            // - m_off < TILE_M (always true due to loop structure)
            // - m_global < args.M (boundary check on M)
            // - k_off < k_len (only load actual K data for this tile)
            if (m_global < args.M && k_off < k_len) {
                const int64_t k_global = k_start + k_off;
                a = static_cast<sycl::half>(args.activations[m_global * args.K + k_global]);
            }
            slm_activations[m_off * TILE_K + k_off] = a;
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== XMX Compute: Each sub-group computes its assigned tile ====
        if (sg_id < num_output_tiles) {
            // K-dimension loop within this K-tile
            // NOTE: TILE_K is always a full tile (32 for Q4_0)
            // Partial K only happens at last k_tile, handled by k_len check during load
            constexpr int NUM_K_TILE_STEPS = TILE_K / XMX_TILE_K;
            for (int tk = 0; tk < NUM_K_TILE_STEPS; tk++) {
                const int k_base = tk * XMX_TILE_K;

                // Load activations tile (row-major: activations[m, k])
                auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(&slm_activations[m_base * TILE_K + k_base]));
                sycl_xmx::joint_matrix_load(sg, mat_a, a_ptr, TILE_K);

                // Load weights tile (col-major for transposed access)
                auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(&slm_weights[n_base * TILE_K + k_base]));
                sycl_xmx::joint_matrix_load(sg, mat_b, b_ptr, TILE_K);

                // Compute: acc += A * B
                sycl_xmx::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            }
        }

        // Barrier before next K-tile (only if there are more tiles)
        if (kt + 1 < k_tiles) {
            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    // ==== Write output: Each sub-group stores its result ====
    if (sg_id < num_output_tiles) {
        const int64_t m_global_base = m_start + m_base;
        const int64_t n_global_base = n_start + n_base;

        // Check if ANY part of this tile is within bounds
        if (m_global_base < args.M && n_global_base < args.N) {
            // Store result directly to global memory
            // Need to handle boundary cases where tile extends beyond matrix
            const bool fully_in_bounds =
                (m_global_base + XMX_TILE_M <= args.M) &&
                (n_global_base + XMX_TILE_N <= args.N);

            if (fully_in_bounds && (args.N % XMX_TILE_N == 0)) {
                // Direct store to global memory for fully-in-bounds tiles
                // NOTE: Only use direct store when N is a multiple of XMX_TILE_N (16)
                // because joint_matrix_store with non-aligned stride causes data corruption
                auto out_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(
                    &args.output[m_global_base * args.N + n_global_base]);
                sycl_xmx::joint_matrix_store(sg, acc, out_ptr, args.N, sycl_xmx::layout::row_major);
            } else {
                // Boundary case: Store to dedicated float SLM buffer,
                // then write valid elements to global memory with per-element bounds checking
                constexpr int ACC_SIZE = XMX_TILE_M * XMX_TILE_N;

                // Use the dedicated float SLM accessor
                auto slm_acc_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                            sycl::access::decorated::no>(&slm_acc_out[0]);

                sycl_xmx::joint_matrix_store(sg, acc, slm_acc_ptr, XMX_TILE_N, sycl_xmx::layout::row_major);
                sycl::group_barrier(sg);

                // Write valid elements with explicit bounds checking
                for (int i = lane; i < ACC_SIZE; i += XMX_SUBGROUP_SIZE) {
                    const int row = i / XMX_TILE_N;
                    const int col = i % XMX_TILE_N;
                    const int64_t m_global = m_global_base + row;
                    const int64_t n_global = n_global_base + col;

                    // Only write if BOTH indices are within bounds
                    if (m_global < args.M && n_global < args.N) {
                        args.output[m_global * args.N + n_global] = slm_acc_out[i];
                    }
                }
            }
        }
    }
}

#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// =============================================================================
// Unified Matmul Kernel - Scalar Path
// =============================================================================

/**
 * Unified matmul kernel template.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Where:
 * - weights (src0) has shape [N, K] - indexed by output column n
 * - activations (src1) has shape [M, K] - indexed by output row m
 * - output (dst) has shape [M, N]
 *
 * Template parameters control tile sizes and compute path.
 * Implements scalar path; XMX path is in unified_matmul_xmx_kernel_impl.
 *
 * Work distribution:
 * - 2D grid: [ceil(M/TILE_M), ceil(N/TILE_N)]
 * - Each work-group computes one output tile of size TILE_M x TILE_N
 * - K dimension is iterated within each work-group
 *
 * Memory access pattern:
 * - Weights: load tile_n rows of K weights (one per output column), dequantize on-the-fly
 * - Activations: load tile_m rows of K activations (one per output row)
 * - Output: write tile_m x tile_n results
 */
template <int TILE_M, int TILE_N, int TILE_K, bool USE_XMX>
void unified_matmul_kernel_impl(sycl::nd_item<2>                   item,
                                const UnifiedKernelArgs            args,
                                sycl::local_accessor<float, 1>     slm_weights,
                                sycl::local_accessor<float, 1>     slm_activations) {
    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension (output rows)
    const int tile_col = item.get_group(1);  // N dimension (output columns)

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;  // Starting output row
    const int64_t n_start = tile_col * TILE_N;  // Starting output column

    // Number of K tiles
    const int k_tiles = (args.K + TILE_K - 1) / TILE_K;
    const int k_blocks_per_row = args.K / UNIFIED_QK4_0;

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize accumulator for each thread's output elements
    // Each thread computes multiple output elements
    float acc[4][4] = {{0.0f}};  // Thread computes up to 4x4 outputs

    // Determine how many outputs each thread computes
    const int outputs_per_thread_m = (TILE_M + local_size_row - 1) / local_size_row;
    const int outputs_per_thread_n = (TILE_N + local_size_col - 1) / local_size_col;

    // K-loop: iterate over K dimension in tiles
    for (int kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * TILE_K;
        const int64_t k_end = sycl::min(k_start + TILE_K, args.K);
        const int     k_len = static_cast<int>(k_end - k_start);

        // Barrier before loading new tile data
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Load weights to SLM ====
        // GGML: weights[N, K] - indexed by output column n
        // Each thread loads multiple weight elements
        // SLM layout: [TILE_N x TILE_K] indexed as [n_off * TILE_K + k_off]
        for (int n_off = local_row; n_off < TILE_N; n_off += local_size_row) {
            const int64_t n_global = n_start + n_off;  // Output column = weight row
            if (n_global >= args.N) continue;

            for (int k_off = local_col; k_off < k_len; k_off += local_size_col) {
                const int64_t k_global = k_start + k_off;

                // GGML: weights[n_global, k_global]
                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                // Dequantize and store to SLM
                float w = dequant_q4_0(&weights[block_idx], idx_in_block);
                slm_weights[n_off * TILE_K + k_off] = w;
            }
        }

        // ==== Load activations to SLM ====
        // GGML: activations[M, K] - indexed by output row m
        // SLM layout: [TILE_M x TILE_K] indexed as [m_off * TILE_K + k_off]
        for (int m_off = local_row; m_off < TILE_M; m_off += local_size_row) {
            const int64_t m_global = m_start + m_off;  // Output row
            if (m_global >= args.M) continue;

            for (int k_off = local_col; k_off < k_len; k_off += local_size_col) {
                const int64_t k_global = k_start + k_off;

                // GGML: activations[m_global, k_global]
                float a = args.activations[m_global * args.K + k_global];
                slm_activations[m_off * TILE_K + k_off] = a;
            }
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Compute: accumulate partial products ====
        // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
        // Each thread computes its assigned outputs
        for (int mo = 0; mo < outputs_per_thread_m; mo++) {
            const int m_off = local_row + mo * local_size_row;
            if (m_off >= TILE_M) continue;
            if (m_start + m_off >= args.M) continue;

            for (int no = 0; no < outputs_per_thread_n; no++) {
                const int n_off = local_col + no * local_size_col;
                if (n_off >= TILE_N) continue;
                if (n_start + n_off >= args.N) continue;

                // Dot product over K tile
                // dst[m,n] = sum_k(weights[n,k] * activations[m,k])
                float sum = 0.0f;
                for (int k = 0; k < k_len; k++) {
                    float w = slm_weights[n_off * TILE_K + k];     // weights[n, k]
                    float a = slm_activations[m_off * TILE_K + k]; // activations[m, k]
                    sum += w * a;
                }

                // Accumulate to thread-local storage
                if (mo < 4 && no < 4) {
                    acc[mo][no] += sum;
                }
            }
        }
    }

    // ==== Write output ====
    // Barrier not strictly needed here since we're writing to global memory
    for (int mo = 0; mo < outputs_per_thread_m && mo < 4; mo++) {
        const int m_off = local_row + mo * local_size_row;
        if (m_off >= TILE_M) continue;
        const int64_t m_global = m_start + m_off;
        if (m_global >= args.M) continue;

        for (int no = 0; no < outputs_per_thread_n && no < 4; no++) {
            const int n_off = local_col + no * local_size_col;
            if (n_off >= TILE_N) continue;
            const int64_t n_global = n_start + n_off;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = acc[mo][no];
        }
    }
}

// =============================================================================
// Kernel Launcher
// =============================================================================

void launch_unified_matmul(sycl::queue & q, const UnifiedKernelArgs & args) {
    // Validate arguments
    if (!validate_args(args)) {
        fprintf(stderr, "[unified-kernel] Invalid arguments\n");
        return;
    }

    // Only Q4_0 supported currently
    if (args.quant_type != QUANT_TYPE_Q4_0) {
        fprintf(stderr, "[unified-kernel] Only Q4_0 quantization supported (got type=%d, expected=%d)\n",
                args.quant_type, QUANT_TYPE_Q4_0);
        return;
    }

    // Debug: Print launch parameters (opt-in to avoid log spam in production)
    if (ggml_sycl_unified_debug_enabled()) {
        fprintf(stderr, "[unified-kernel] LAUNCH: M=%lld N=%lld K=%lld type=%d tile=(%d,%d,%d) xmx=%d\n",
                static_cast<long long>(args.M), static_cast<long long>(args.N),
                static_cast<long long>(args.K), args.quant_type,
                args.tile_m, args.tile_n, args.tile_k, args.use_xmx ? 1 : 0);
        fflush(stderr);
    }

    // Calculate grid dimensions
    const int tile_m = args.tile_m;
    const int tile_n = args.tile_n;
    const int tile_k = args.tile_k;

    const int grid_m = (static_cast<int>(args.M) + tile_m - 1) / tile_m;
    const int grid_n = (static_cast<int>(args.N) + tile_n - 1) / tile_n;

    // Work-group size: use square-ish shape
    // Limit to reasonable size that divides well
    const int wg_size_m = std::min(tile_m, 8);
    const int wg_size_n = std::min(tile_n, 16);

    // Determine if XMX path should be used
    bool use_xmx_path = false;
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (args.use_xmx && can_use_xmx(args.M, args.N, args.K)) {
        // Check if device supports XMX
        sycl::device dev = q.get_device();
        use_xmx_path = dev.has(sycl::aspect::ext_intel_matrix);
    }
#endif

    // ==========================================================================
    // XMX Path: Use joint_matrix for dpas acceleration (optimized)
    // ==========================================================================
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (use_xmx_path) {
        // XMX path uses fixed tile sizes aligned to XMX dimensions
        constexpr int XMX_TM = 8;   // Must be multiple of XMX_TILE_M=8
        constexpr int XMX_TN = 16;  // Must be multiple of XMX_TILE_N=16
        constexpr int XMX_TK = 32;  // Must be multiple of XMX_TILE_K=16

        const int xmx_grid_m = (static_cast<int>(args.M) + XMX_TM - 1) / XMX_TM;
        const int xmx_grid_n = (static_cast<int>(args.N) + XMX_TN - 1) / XMX_TN;

        // Work-group size: 1 sub-group (16 threads)
        // Each work-group handles one XMX tile (8x16 output)
        constexpr int xmx_wg_m = 1;
        constexpr int xmx_wg_n = XMX_SUBGROUP_SIZE;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM for half-precision data
            sycl::local_accessor<sycl::half, 1> slm_w(XMX_TN * XMX_TK, cgh);  // Weights
            sycl::local_accessor<sycl::half, 1> slm_a(XMX_TM * XMX_TK, cgh);  // Activations
            // Float SLM for boundary case accumulator output
            sycl::local_accessor<float, 1> slm_acc_out(XMX_TILE_M * XMX_TILE_N, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(xmx_grid_m * xmx_wg_m, xmx_grid_n * xmx_wg_n),
                sycl::range<2>(xmx_wg_m, xmx_wg_n)
            );

            cgh.parallel_for<unified_matmul_xmx_kernel_name<XMX_TM, XMX_TN, XMX_TK>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]] {
                    unified_matmul_xmx_kernel_impl<XMX_TM, XMX_TN, XMX_TK>(
                        item, args, slm_w, slm_a, slm_acc_out);
                }
            );
        });
        return;
    }
#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

    // ==========================================================================
    // Scalar Path: Standard matmul with dequantization
    // ==========================================================================

    // Launch based on tile sizes
    // For simplicity, dispatch to fixed tile sizes initially
    // A more sophisticated version would use template instantiation for common sizes

    if (tile_m == 1) {
        // Decode path: M=1. Use a wider N tile for better throughput.
        // Important: we must actually submit a kernel here; otherwise the
        // output buffer is left unchanged and sampling goes off the rails.
        constexpr int TM = 1;
        constexpr int TN = 64;
        constexpr int TK = 32;

        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = 1;
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    } else if (tile_m <= 8 && tile_n <= 16 && tile_k <= 32) {
        // Small tiles: 8x16x32
        constexpr int TM = 8;
        constexpr int TN = 16;
        constexpr int TK = 32;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(grid_m * wg_size_m, grid_n * wg_size_n),
                sycl::range<2>(wg_size_m, wg_size_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    } else if (tile_m <= 16 && tile_n <= 32 && tile_k <= 32) {
        // Medium tiles: 16x32x32
        constexpr int TM = 16;
        constexpr int TN = 32;
        constexpr int TK = 32;

        // Recalculate grid dimensions using ACTUAL template tile sizes
        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] MEDIUM path: TM=%d TN=%d TK=%d grid=(%d,%d) wg=(%d,%d)\n",
                    TM, TN, TK, tm_grid_m, tm_grid_n, wg_m, wg_n);
            fflush(stderr);
        }

        q.submit([&](sycl::handler & cgh) {
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
        // NOTE: Don't call q.wait_and_throw() here - incompatible with SYCL command graphs.
        // Errors will be caught when the queue is synchronized elsewhere.
    } else {
        // Fallback: use 32x32x32 tiles with dynamic SLM allocation
        // This path handles larger tile sizes
        constexpr int TM = 32;
        constexpr int TN = 32;
        constexpr int TK = 32;

        // IMPORTANT: grid dimensions must match the ACTUAL template tile sizes.
        // Using the caller-provided tile sizes here can under-cover the output.
        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_fallback>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    }
}

}  // namespace ggml_sycl_unified
