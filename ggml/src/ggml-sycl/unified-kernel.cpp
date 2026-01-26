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

#include <algorithm>
#include <cstdio>

namespace ggml_sycl_unified {

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
// Unified Matmul Kernel - XMX Path
// =============================================================================

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

/**
 * XMX-accelerated matmul kernel using joint_matrix.
 *
 * Computes: output[M,N] = dequant(weights[M,K]) @ activations[K,N]
 *
 * This kernel dequantizes Q4_0 weights to half precision and uses
 * Intel XMX (joint_matrix) for dpas acceleration.
 *
 * Work distribution:
 * - Each sub-group computes one XMX tile (8x16 output)
 * - Work-group cooperatively loads data to SLM
 * - K dimension iterated with XMX_TILE_K=8 step size
 *
 * @tparam TILE_M  M tile size (must be multiple of 8)
 * @tparam TILE_N  N tile size (must be multiple of 16)
 * @tparam TILE_K  K tile size (must be multiple of 8)
 */
template <int TILE_M, int TILE_N, int TILE_K>
void unified_matmul_xmx_kernel_impl(sycl::nd_item<2>                   item,
                                    const UnifiedKernelArgs            args,
                                    sycl::local_accessor<sycl::half, 1> slm_weights,
                                    sycl::local_accessor<sycl::half, 1> slm_activations,
                                    sycl::local_accessor<float, 1>      slm_acc) {
    auto sg = item.get_sub_group();

    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension
    const int tile_col = item.get_group(1);  // N dimension

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);
    const int local_linear = local_row * local_size_col + local_col;
    const int local_total = local_size_row * local_size_col;

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;
    const int64_t n_start = tile_col * TILE_N;

    // Number of K tiles
    const int k_tiles = (args.K + TILE_K - 1) / TILE_K;
    const int k_blocks_per_row = args.K / UNIFIED_QK4_0;

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize accumulator (in registers)
    // Each thread stores its portion of the output tile
    constexpr int ACC_SIZE = TILE_M * TILE_N;
    float acc_regs[ACC_SIZE / 16] = {0.0f};  // Distributed across 16 threads per sub-group

    // Clear accumulator SLM
    for (int i = local_linear; i < XMX_TILE_M * XMX_TILE_N; i += local_total) {
        slm_acc[i] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // K-loop: iterate over K dimension in tiles
    for (int kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * TILE_K;
        const int64_t k_end = sycl::min(k_start + TILE_K, args.K);
        const int     k_len = static_cast<int>(k_end - k_start);

        // Barrier before loading new tile data
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Load and dequantize weights to SLM (as half) ====
        // Layout: row-major [TILE_M x TILE_K]
        for (int idx = local_linear; idx < TILE_M * k_len; idx += local_total) {
            const int m_off = idx / k_len;
            const int k_off = idx % k_len;
            const int64_t m_global = m_start + m_off;
            const int64_t k_global = k_start + k_off;

            sycl::half w = sycl::half(0.0f);
            if (m_global < args.M) {
                // Determine which Q4_0 block and index within block
                const int block_idx = static_cast<int>(m_global * k_blocks_per_row + k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);
                w = dequant_q4_0_half(&weights[block_idx], idx_in_block);
            }
            slm_weights[m_off * TILE_K + k_off] = w;
        }

        // ==== Load activations to SLM (convert to half, col-major for dpas) ====
        // Layout: col-major [TILE_N x TILE_K] for optimal B matrix access
        for (int idx = local_linear; idx < k_len * TILE_N; idx += local_total) {
            const int k_off = idx / TILE_N;
            const int n_off = idx % TILE_N;
            const int64_t k_global = k_start + k_off;
            const int64_t n_global = n_start + n_off;

            sycl::half a = sycl::half(0.0f);
            if (n_global < args.N) {
                a = static_cast<sycl::half>(args.activations[k_global * args.N + n_global]);
            }
            // Store in col-major: [n_off][k_off] for B matrix
            slm_activations[n_off * TILE_K + k_off] = a;
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== XMX Compute using joint_matrix ====
        // Only sub-group 0 in the first row performs the XMX computation
        // for this simple implementation
        if (local_row == 0 && local_col < XMX_SUBGROUP_SIZE) {
            // Call the XMX compute function from header
            compute_tile_xmx<TILE_M, TILE_N, TILE_K>(
                sg,
                &slm_weights[0],
                &slm_activations[0],
                acc_regs,
                slm_acc,
                item
            );
        }
    }

    // ==== Write output ====
    item.barrier(sycl::access::fence_space::local_space);

    // Scatter results from accumulator registers to global output
    // Each thread writes its assigned output elements
    if (local_row == 0) {
        const int lane = sg.get_local_linear_id();
        for (int i = lane; i < TILE_M * TILE_N; i += XMX_SUBGROUP_SIZE) {
            const int m_off = i / TILE_N;
            const int n_off = i % TILE_N;
            const int64_t m_global = m_start + m_off;
            const int64_t n_global = n_start + n_off;

            if (m_global < args.M && n_global < args.N) {
                // Read from appropriate register
                int reg_idx = i / XMX_SUBGROUP_SIZE;
                if (reg_idx < static_cast<int>(sizeof(acc_regs) / sizeof(float))) {
                    args.output[m_global * args.N + n_global] = acc_regs[reg_idx];
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
 * Computes: output[M,N] = dequant(weights[M,K]) @ activations[K,N]
 *
 * Template parameters control tile sizes and compute path.
 * Implements scalar path; XMX path is in unified_matmul_xmx_kernel_impl.
 *
 * Work distribution:
 * - 2D grid: [ceil(N/TILE_N), ceil(M/TILE_M)]
 * - Each work-group computes one output tile of size TILE_M x TILE_N
 * - K dimension is iterated within each work-group
 *
 * Memory access pattern:
 * - Weights: load tile_m rows of K weights, dequantize on-the-fly
 * - Activations: load tile_k x tile_n from global memory
 * - Output: write tile_m x tile_n results
 */
template <int TILE_M, int TILE_N, int TILE_K, bool USE_XMX>
void unified_matmul_kernel_impl(sycl::nd_item<2>                   item,
                                const UnifiedKernelArgs            args,
                                sycl::local_accessor<float, 1>     slm_weights,
                                sycl::local_accessor<float, 1>     slm_activations) {
    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension
    const int tile_col = item.get_group(1);  // N dimension

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;
    const int64_t n_start = tile_col * TILE_N;

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
        // Each thread loads multiple weight elements
        for (int m_off = local_row; m_off < TILE_M; m_off += local_size_row) {
            const int64_t m_global = m_start + m_off;
            if (m_global >= args.M) continue;

            for (int k_off = local_col; k_off < k_len; k_off += local_size_col) {
                const int64_t k_global = k_start + k_off;

                // Determine which Q4_0 block and index within block
                const int block_idx = static_cast<int>(m_global * k_blocks_per_row + k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                // Dequantize and store to SLM
                float w = dequant_q4_0(&weights[block_idx], idx_in_block);
                slm_weights[m_off * TILE_K + k_off] = w;
            }
        }

        // ==== Load activations to SLM ====
        // Activations layout: [K, N] row-major
        for (int k_off = local_row; k_off < k_len; k_off += local_size_row) {
            const int64_t k_global = k_start + k_off;

            for (int n_off = local_col; n_off < TILE_N; n_off += local_size_col) {
                const int64_t n_global = n_start + n_off;
                if (n_global >= args.N) continue;

                float a = args.activations[k_global * args.N + n_global];
                slm_activations[k_off * TILE_N + n_off] = a;
            }
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Compute: accumulate partial products ====
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
                float sum = 0.0f;
                for (int k = 0; k < k_len; k++) {
                    float w = slm_weights[m_off * TILE_K + k];
                    float a = slm_activations[k * TILE_N + n_off];
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
        fprintf(stderr, "[unified-kernel] Only Q4_0 quantization supported currently\n");
        return;
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
    // XMX Path: Use joint_matrix for dpas acceleration
    // ==========================================================================
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (use_xmx_path) {
        // XMX path uses fixed tile sizes aligned to XMX dimensions
        constexpr int XMX_TM = 8;   // Must be multiple of XMX_TILE_M=8
        constexpr int XMX_TN = 16;  // Must be multiple of XMX_TILE_N=16
        constexpr int XMX_TK = 32;  // Must be multiple of XMX_TILE_K=8

        const int xmx_grid_m = (static_cast<int>(args.M) + XMX_TM - 1) / XMX_TM;
        const int xmx_grid_n = (static_cast<int>(args.N) + XMX_TN - 1) / XMX_TN;

        // Work-group size for XMX: one sub-group of 16 threads
        constexpr int xmx_wg_m = 1;
        constexpr int xmx_wg_n = XMX_SUBGROUP_SIZE;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM for half-precision data
            sycl::local_accessor<sycl::half, 1> slm_w(XMX_TM * XMX_TK, cgh);
            sycl::local_accessor<sycl::half, 1> slm_a(XMX_TK * XMX_TN, cgh);
            sycl::local_accessor<float, 1> slm_acc(XMX_TILE_M * XMX_TILE_N, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(xmx_grid_m * xmx_wg_m, xmx_grid_n * xmx_wg_n),
                sycl::range<2>(xmx_wg_m, xmx_wg_n)
            );

            cgh.parallel_for<unified_matmul_xmx_kernel_name<XMX_TM, XMX_TN, XMX_TK>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]] {
                    unified_matmul_xmx_kernel_impl<XMX_TM, XMX_TN, XMX_TK>(
                        item, args, slm_w, slm_a, slm_acc);
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

    if (tile_m <= 8 && tile_n <= 16 && tile_k <= 32) {
        // Small tiles: 8x16x32
        constexpr int TM = 8;
        constexpr int TN = 16;
        constexpr int TK = 32;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM
            sycl::local_accessor<float, 1> slm_w(TM * TK, cgh);
            sycl::local_accessor<float, 1> slm_a(TK * TN, cgh);

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

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm_w(TM * TK, cgh);
            sycl::local_accessor<float, 1> slm_a(TK * TN, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(grid_m * wg_m, grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    } else {
        // Fallback: use 32x32x32 tiles with dynamic SLM allocation
        // This path handles larger tile sizes
        constexpr int TM = 32;
        constexpr int TN = 32;
        constexpr int TK = 32;

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm_w(TM * TK, cgh);
            sycl::local_accessor<float, 1> slm_a(TK * TN, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(grid_m * wg_m, grid_n * wg_n),
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
