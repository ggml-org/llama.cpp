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
// ESIMD dpas Kernel - FP16 Path (Phase 2)
// =============================================================================
// Uses Intel ESIMD xmx::dpas for explicit SIMD control on XMX hardware.
// This path provides better XVE utilization than joint_matrix for some cases.
//
// ESIMD dpas characteristics:
// - dpas operand order: dpas(accumulator, B_tile, A_tile) - B before A!
// - K-tile for FP16 = 16 (SystolicDepth=8 x OpsPerChannel=2)
// - RepeatCount = 1-8, determines M dimension (we use 8)
// - ExecutionSize = 16 for Arc (determines N dimension)
// - Output tile: 8x16 (M x N)

#if GGML_SYCL_ESIMD_AVAILABLE

// Kernel class names for SYCL naming and profiling
template <int TILE_M, int TILE_N>
class esimd_fp16_kernel;

template <int TILE_M, int TILE_N>
class esimd_int8_kernel;

// =============================================================================
// ESIMD dpas Constants - FP16
// =============================================================================
// Hardware-defined parameters for FP16 dpas on Intel Arc
constexpr int ESIMD_SYSTOLIC_DEPTH = 8;   // Always 8 for dpas
constexpr int ESIMD_REPEAT_COUNT = 8;     // M dimension = RepeatCount
constexpr int ESIMD_EXEC_SIZE = 16;       // N dimension = ExecutionSize
constexpr int ESIMD_K_PER_DPAS = 16;      // K elements per dpas for FP16

// Operand sizes for FP16 dpas
// A (activations): Repeat * K_per = 8 * 16 = 128 half elements
// B (weights): K_per * ExecSize = 16 * 16 = 256 half elements
// Acc (output): Repeat * ExecSize = 8 * 16 = 128 float elements
constexpr int ESIMD_A_SIZE = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS;   // 128
constexpr int ESIMD_B_SIZE = ESIMD_K_PER_DPAS * ESIMD_EXEC_SIZE;     // 256
constexpr int ESIMD_ACC_SIZE = ESIMD_REPEAT_COUNT * ESIMD_EXEC_SIZE; // 128

// =============================================================================
// ESIMD dpas Constants - INT8
// =============================================================================
// Hardware-defined parameters for INT8 dpas on Intel Arc
// INT8 has K=32 per dpas (SystolicDepth=8 x OpsPerChannel=4)
constexpr int ESIMD_K_PER_DPAS_INT8 = 32;  // K elements per dpas for INT8

// Operand sizes for INT8 dpas
// A (activations): Repeat * K_per = 8 * 32 = 256 int8 elements
// B (weights): K_per * ExecSize = 32 * 16 = 512 int8 elements
// Acc (output): Repeat * ExecSize = 8 * 16 = 128 int32 elements
constexpr int ESIMD_A_SIZE_INT8 = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS_INT8;   // 256
constexpr int ESIMD_B_SIZE_INT8 = ESIMD_K_PER_DPAS_INT8 * ESIMD_EXEC_SIZE;     // 512
constexpr int ESIMD_ACC_SIZE_INT8 = ESIMD_REPEAT_COUNT * ESIMD_EXEC_SIZE;      // 128

/**
 * ESIMD FP16 matmul kernel using xmx::dpas instruction.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Uses ESIMD xmx::dpas for hardware-accelerated matrix multiplication.
 * Each work-item processes one 8x16 output tile using dpas instruction.
 *
 * dpas layout requirements:
 * - A matrix (activations): [Repeat x K] = [8 x 16] half, row-major packed
 * - B matrix (weights): [K x ExecSize] = [16 x 16] half, VNNI-like layout
 * - Output: [Repeat x ExecSize] = [8 x 16] float accumulator
 *
 * Work distribution:
 * - 2D grid: [ceil(M/8), ceil(N/16)]
 * - Each work-item handles one 8x16 output tile
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_kernel_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start) {

    // Validate tile sizes match dpas requirements
    static_assert(TILE_M == ESIMD_REPEAT_COUNT, "TILE_M must be 8 for dpas");
    static_assert(TILE_N == ESIMD_EXEC_SIZE, "TILE_N must be 16 for dpas");

    // Boundary checking: return early if entire tile is out of bounds
    if (m_start >= args.M || n_start >= args.N) {
        return;
    }

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize accumulator: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Number of K tiles (each dpas processes 16 K elements)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // K-loop: iterate over K dimension in tiles of 16
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * ESIMD_K_PER_DPAS;
        // Calculate remaining K elements (avoid sycl::min which is not supported in ESIMD)
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // ============================================================
        // Load and dequantize weights into B matrix with VNNI packing
        // dpas computes: C[m,n] += sum_k(A[m,k] * B[k,n])
        // GGML wants: dst[m,n] = sum_k(activations[m,k] * weights[n,k])
        // So B[k,n] = weights[n,k] (transpose)
        //
        // For FP16 dpas, B matrix needs VNNI-like layout:
        // B_vnni[k/2 * N * 2 + n * 2 + k%2] = B[k,n]
        // This groups consecutive K values together for systolic array
        // ============================================================
        esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec = sycl::half(0.0f);

        // Iterate n first, then k pairs for VNNI packing
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                if (k >= k_len) break;
                const int64_t k_global = k_start + k;
                const int k_block_idx_part = static_cast<int>(k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx_part);
                const block_q4_0_unified * blk = &weights[block_idx];
                const sycl::half d = blk->d;
                int qs_val;
                if (idx_in_block < 16) {
                    qs_val = blk->qs[idx_in_block] & 0x0F;
                } else {
                    qs_val = blk->qs[idx_in_block - 16] >> 4;
                }

                // VNNI layout for FP16: b[(k/2) * N * 2 + n * 2 + (k%2)]
                // = b[k/2 * 32 + n * 2 + k%2]
                const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
                b_vec[vnni_idx] = static_cast<sycl::half>(qs_val - 8) * d;
            }
        }

        // ============================================================
        // Load activations into A matrix [Repeat x K] = [8 x 16]
        // A is stored row-major: a[m * K_per + k]
        // ============================================================
        esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec = sycl::half(0.0f);

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const int64_t m_global = m_start + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                if (k >= k_len) break;

                const int64_t k_global = k_start + k;
                const float act_f32 = args.activations[m_global * args.K + k_global];
                // A layout for dpas: a[m * K_per + k]
                a_vec[m * ESIMD_K_PER_DPAS + k] = static_cast<sycl::half>(act_f32);
            }
        }

        // ============================================================
        // Execute dpas: acc += A @ B (computes C[m,n] += sum_k(A[m,k] * B[k,n]))
        //
        // dpas<SystolicDepth, RepeatCount, AccType, CType, BType, AType>
        // Note: operand order is dpas(acc, B, A) - B before A!
        //
        // dpas computes: C[m,n] += sum_k(A[m,k] * B[k,n])
        // A layout: row-major, a[m * K + k] where m=0..7, k=0..15
        // B layout: VNNI-packed, b[(k/2) * N * 2 + n * 2 + (k%2)] for k=0..15, n=0..15
        // ============================================================
        acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                        float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
    }

    // Write output with boundary checking
    // Output layout: acc[m * TILE_N + n] = [8 x 16] row-major
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = acc[m * TILE_N + n];
        }
    }
}

// =============================================================================
// ESIMD dpas Kernel - INT8 Path (Phase 3)
// =============================================================================
// Uses INT8 dpas with dynamic activation quantization.
// Q4_0 weights are unpacked to INT8 (nibble - 8 for signed [-8, +7]).
// Activations are dynamically quantized per-row using max-abs scaling.
//
// Key differences from FP16:
// - K-tile = 32 (not 16)
// - dpas outputs INT32 accumulator (scaled back to FP32)
// - Requires per-row activation quantization
// - Weight scales stored per N column, activation scales per M row
//
// IMPORTANT: INT8 is LOSSY - not bit-exact with FP16/FP32 path!
// The dynamic quantization introduces noise. Test for coherent output,
// not exact numerical match.

/**
 * ESIMD INT8 matmul kernel using xmx::dpas instruction.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Uses INT8 dpas with dynamic activation quantization:
 * - Q4_0 weights unpacked to INT8: (nibble - 8) for signed range [-8, +7]
 * - Activations quantized per-row: scale = max_abs, qact = act * (127 / scale)
 * - Final result: result_fp32 = int32_acc * weight_scale * act_scale / 127
 *
 * dpas layout requirements for INT8:
 * - A matrix (activations): [Repeat x K] = [8 x 32] int8, row-major packed
 * - B matrix (weights): [K x ExecSize] = [32 x 16] int8, VNNI-like layout
 * - Output: [Repeat x ExecSize] = [8 x 16] int32 accumulator
 *
 * Work distribution:
 * - 2D grid: [ceil(M/8), ceil(N/16)]
 * - Each work-item handles one 8x16 output tile
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_int8_kernel_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start) {

    // Validate tile sizes match dpas requirements
    static_assert(TILE_M == ESIMD_REPEAT_COUNT, "TILE_M must be 8 for dpas");
    static_assert(TILE_N == ESIMD_EXEC_SIZE, "TILE_N must be 16 for dpas");

    // Boundary checking: return early if entire tile is out of bounds
    if (m_start >= args.M || n_start >= args.N) {
        return;
    }

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize INT32 accumulator: [8 x 16]
    esimd::simd<int32_t, ESIMD_ACC_SIZE_INT8> acc = 0;

    // Store weight scales per N column for this tile
    // Will be accumulated across K tiles
    esimd::simd<float, TILE_N> weight_scale_accum = 0.0f;

    // Number of K tiles (each dpas processes 32 K elements for INT8)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS_INT8 - 1) / ESIMD_K_PER_DPAS_INT8;

    // ========================================================================
    // Step 1: Compute per-row activation scales (max-abs for each M row)
    // This needs to scan all K elements for each of the 8 rows in this tile
    // ========================================================================
    esimd::simd<float, TILE_M> act_scales = 0.0f;

    // Scan activations to find max-abs per row
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) {
            act_scales[m] = 1.0f;  // Dummy scale for out-of-bounds rows
            continue;
        }

        float max_abs = 0.0f;
        // Scan all K elements for this row
        for (int64_t k = 0; k < args.K; k++) {
            float val = args.activations[m_global * args.K + k];
            float abs_val = (val >= 0.0f) ? val : -val;
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        // Edge case: All zeros - avoid division by zero
        // Use scale_inv = 0 if scale is too small (results in 0 quantized values)
        if (max_abs > 1e-10f) {
            act_scales[m] = max_abs / 127.0f;  // scale to convert back
        } else {
            act_scales[m] = 1.0f;  // Dummy scale (will produce zeros anyway)
        }
    }

    // ========================================================================
    // K-loop: iterate over K dimension in tiles of 32
    // ========================================================================
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * ESIMD_K_PER_DPAS_INT8;
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS_INT8 ? k_remaining : ESIMD_K_PER_DPAS_INT8);

        // ====================================================================
        // Load and dequantize weights into B matrix as INT8 with VNNI packing
        // Q4_0 unpacking: (nibble - 8) for signed range [-8, +7]
        //
        // For INT8 dpas, B matrix needs VNNI-like layout:
        // B_vnni[k/4 * N * 4 + n * 4 + k%4] = B[k,n]
        // This groups 4 consecutive K values together for systolic array
        // ====================================================================
        esimd::simd<int8_t, ESIMD_B_SIZE_INT8> b_vec = 0;

        // Track weight scales for this K-tile (per N column)
        esimd::simd<float, TILE_N> weight_scales_tile = 0.0f;
        esimd::simd<int, TILE_N> weight_scale_count = 0;

        // Iterate n first, then k for VNNI packing
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (k >= k_len) break;
                const int64_t k_global = k_start + k;
                const int k_block_idx_part = static_cast<int>(k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx_part);
                const block_q4_0_unified * blk = &weights[block_idx];

                // Accumulate scale (we'll average later)
                // Only count unique blocks
                if (idx_in_block == 0) {
                    weight_scales_tile[n] += static_cast<float>(blk->d);
                    weight_scale_count[n] += 1;
                }

                // Unpack Q4_0 nibble to signed INT8: (nibble - 8) gives [-8, +7]
                int qs_val;
                if (idx_in_block < 16) {
                    qs_val = blk->qs[idx_in_block] & 0x0F;
                } else {
                    qs_val = blk->qs[idx_in_block - 16] >> 4;
                }
                const int8_t w_int8 = static_cast<int8_t>(qs_val - 8);

                // VNNI layout for INT8: b[(k/4) * N * 4 + n * 4 + (k%4)]
                const int vnni_idx = (k / 4) * (TILE_N * 4) + n * 4 + (k % 4);
                b_vec[vnni_idx] = w_int8;
            }
        }

        // Average weight scales for this tile (per column)
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            if (weight_scale_count[n] > 0) {
                weight_scale_accum[n] += weight_scales_tile[n];
            }
        }

        // ====================================================================
        // Load and quantize activations into A matrix as INT8 [Repeat x K] = [8 x 32]
        // Dynamic quantization: qact = act * (127 / max_abs)
        // ====================================================================
        esimd::simd<int8_t, ESIMD_A_SIZE_INT8> a_vec = 0;

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const int64_t m_global = m_start + m;
            if (m_global >= args.M) continue;

            // Recover max_abs from act_scale (act_scale = max_abs / 127)
            const float max_abs = act_scales[m] * 127.0f;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (k >= k_len) break;

                const int64_t k_global = k_start + k;
                const float act_f32 = args.activations[m_global * args.K + k_global];

                // Quantize to INT8: qval = act * 127 / max_abs
                // Clamp to [-127, 127]
                float qval = (max_abs > 1e-10f) ? (act_f32 * 127.0f / max_abs) : 0.0f;
                if (qval > 127.0f) qval = 127.0f;
                if (qval < -127.0f) qval = -127.0f;

                const int8_t a_int8 = static_cast<int8_t>(qval);

                // A layout for dpas: a[m * K_per + k]
                a_vec[m * ESIMD_K_PER_DPAS_INT8 + k] = a_int8;
            }
        }

        // ====================================================================
        // Execute dpas: acc += A @ B (INT8 x INT8 -> INT32)
        //
        // dpas<SystolicDepth, RepeatCount, AccType, CType, BType, AType>
        // Note: operand order is dpas(acc, B, A) - B before A!
        //
        // For INT8: SystolicDepth=8, RepeatCount=8, K=32
        // ====================================================================
        acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                        int32_t, int32_t, int8_t, int8_t>(acc, b_vec, a_vec);
    }

    // ========================================================================
    // Step 3: Convert INT32 accumulator to FP32 and apply scales
    // result_fp32 = int32_acc * weight_scale * act_scale / 127
    //
    // The INT32 accumulator contains: sum_k(q_weight * q_act)
    // where q_weight = (nibble - 8) and q_act = act * 127 / max_abs
    //
    // To recover FP32: result = int32_acc * weight_scale * (max_abs / 127)
    //                        = int32_acc * weight_scale * act_scale
    // ========================================================================

    // Average weight scales per column (across all K tiles)
    // For Q4_0, each block covers 32 elements, so we have K/32 blocks per K-tile
    const int total_k_blocks = static_cast<int>(args.K / UNIFIED_QK4_0);
    esimd::simd<float, TILE_N> avg_weight_scales = weight_scale_accum / static_cast<float>(total_k_blocks);

    // Write output with boundary checking
    // Apply combined scales: result = int32_acc * weight_scale * act_scale
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        const float act_scale = act_scales[m];

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            // Get INT32 accumulator value
            const int32_t int32_val = acc[m * TILE_N + n];

            // Convert to FP32 and apply scales
            // int32_val = sum_k((nibble-8) * (act*127/max_abs))
            // result = int32_val * weight_scale * max_abs / 127
            //        = int32_val * weight_scale * act_scale
            const float weight_scale = avg_weight_scales[n];
            const float result = static_cast<float>(int32_val) * weight_scale * act_scale;

            args.output[m_global * args.N + n_global] = result;
        }
    }
}

/**
 * Check if INT8 ESIMD dpas path can be used for given dimensions.
 *
 * INT8 ESIMD dpas requires:
 * - ESIMD enabled via GGML_SYCL_XMX_ESIMD=1
 * - INT8 enabled via GGML_SYCL_XMX_INT8=1
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if INT8 ESIMD dpas can be used
 */
inline bool can_use_esimd_int8_dpas(int64_t M, int64_t N, int64_t K) {
    // Both ESIMD and INT8 must be enabled
    if (!use_esimd_dpas() || !use_int8_dpas()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Must have at least some work to do
    return M >= 1 && N >= 1 && K >= 1;
}

/**
 * Check if ESIMD dpas path can be used for given dimensions.
 *
 * ESIMD dpas requires:
 * - ESIMD enabled via GGML_SYCL_XMX_ESIMD=1
 * - M >= 1 (we handle partial tiles)
 * - N >= 1 (we handle partial tiles)
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if ESIMD dpas can be used
 */
inline bool can_use_esimd_dpas(int64_t M, int64_t N, int64_t K) {
    // ESIMD path disabled by default, enable with GGML_SYCL_XMX_ESIMD=1
    if (!use_esimd_dpas()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Must have at least some work to do
    return M >= 1 && N >= 1 && K >= 1;
}

#endif  // GGML_SYCL_ESIMD_AVAILABLE

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
    // ESIMD dpas Path: Use ESIMD xmx::dpas for explicit SIMD control
    // ==========================================================================
    // Enabled via GGML_SYCL_XMX_ESIMD=1 environment variable.
    // Each work-item processes one 8x16 output tile using ESIMD dpas instruction.
    //
    // Two variants:
    // 1. INT8 (GGML_SYCL_XMX_INT8=1): Dynamic activation quantization, K=32 per dpas
    // 2. FP16 (default when ESIMD enabled): K=16 per dpas
    //
    // NOTE: INT8 is LOSSY - not bit-exact with FP16 path!

#if GGML_SYCL_ESIMD_AVAILABLE
    // Try INT8 path first (requires both ESIMD and INT8 flags)
    if (can_use_esimd_int8_dpas(args.M, args.N, args.K)) {
        // ESIMD INT8 dpas tile sizes (fixed by hardware)
        constexpr int ESIMD_TM = 8;   // RepeatCount = 8
        constexpr int ESIMD_TN = 16;  // ExecutionSize = 16

        const int esimd_grid_m = (static_cast<int>(args.M) + ESIMD_TM - 1) / ESIMD_TM;
        const int esimd_grid_n = (static_cast<int>(args.N) + ESIMD_TN - 1) / ESIMD_TN;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD INT8 path: M=%lld N=%lld K=%lld grid=(%d,%d)\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), esimd_grid_m, esimd_grid_n);
            fflush(stderr);
        }

        q.submit([&](sycl::handler & cgh) {
            // ESIMD kernel: one work-item per output tile (no work-group cooperation)
            // Total work items = grid_m * grid_n
            sycl::range<2> global(esimd_grid_m, esimd_grid_n);
            sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

            cgh.parallel_for<esimd_int8_kernel<ESIMD_TM, ESIMD_TN>>(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    // Calculate tile coordinates
                    const int tile_row = item.get_global_id(0);  // M tile index
                    const int tile_col = item.get_global_id(1);  // N tile index

                    const int64_t m_start = tile_row * ESIMD_TM;
                    const int64_t n_start = tile_col * ESIMD_TN;

                    // Call ESIMD INT8 kernel implementation
                    esimd_matmul_int8_kernel_impl<ESIMD_TM, ESIMD_TN>(args, m_start, n_start);
                }
            );
        });
        return;
    }

    // FP16 path (ESIMD enabled but INT8 not enabled)
    if (can_use_esimd_dpas(args.M, args.N, args.K)) {
        // ESIMD FP16 dpas tile sizes (fixed by hardware)
        constexpr int ESIMD_TM = 8;   // RepeatCount = 8
        constexpr int ESIMD_TN = 16;  // ExecutionSize = 16

        const int esimd_grid_m = (static_cast<int>(args.M) + ESIMD_TM - 1) / ESIMD_TM;
        const int esimd_grid_n = (static_cast<int>(args.N) + ESIMD_TN - 1) / ESIMD_TN;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD FP16 path: M=%lld N=%lld K=%lld grid=(%d,%d)\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), esimd_grid_m, esimd_grid_n);
            fflush(stderr);
        }

        q.submit([&](sycl::handler & cgh) {
            // ESIMD kernel: one work-item per output tile (no work-group cooperation)
            // Total work items = grid_m * grid_n
            sycl::range<2> global(esimd_grid_m, esimd_grid_n);
            sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

            cgh.parallel_for<esimd_fp16_kernel<ESIMD_TM, ESIMD_TN>>(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    // Calculate tile coordinates
                    const int tile_row = item.get_global_id(0);  // M tile index
                    const int tile_col = item.get_global_id(1);  // N tile index

                    const int64_t m_start = tile_row * ESIMD_TM;
                    const int64_t n_start = tile_col * ESIMD_TN;

                    // Call ESIMD FP16 kernel implementation
                    esimd_matmul_fp16_kernel_impl<ESIMD_TM, ESIMD_TN>(args, m_start, n_start);
                }
            );
        });
        return;
    }
#endif  // GGML_SYCL_ESIMD_AVAILABLE

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
