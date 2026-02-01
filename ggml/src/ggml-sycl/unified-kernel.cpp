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

// =============================================================================
// Standalone Test Mode Support
// =============================================================================
// When UNIFIED_KERNEL_TEST_STANDALONE is defined, provide stub implementations
// for symbols normally provided by common.cpp. This allows standalone testing
// without linking the entire ggml-sycl library.
#ifdef UNIFIED_KERNEL_TEST_STANDALONE

// Stub global debug flags
bool g_ggml_sycl_debug = false;
bool g_ggml_sycl_debug_forced_off = false;

// Stub device info structure (minimal for testing)
struct ggml_sycl_device_info_stub {
    int nsm = 20;
    struct {
        bool supported = true;
        bool supports_int8 = true;
        bool supports_fp16 = true;
        int slm_size = 65536;
        int max_wg_size = 1024;
        int sg_size_preferred = 16;
        int M = 8;   // XMX tile M dimension
        int N = 16;  // XMX tile N dimension
        int K = 32;  // XMX tile K dimension (for INT8)
    } xmx_caps;
};

struct ggml_sycl_info_stub {
    int device_count = 1;
    ggml_sycl_device_info_stub devices[1] = {};
};

static ggml_sycl_info_stub g_stub_sycl_info;

const ggml_sycl_info_stub & ggml_sycl_info() {
    return g_stub_sycl_info;
}

// Stub GGML_SYCL_DEBUG macro (no-op in standalone mode)
#define GGML_SYCL_DEBUG(...) do {} while(0)

#else  // !UNIFIED_KERNEL_TEST_STANDALONE

#include "common.hpp"  // For ggml_sycl_info() and GGML_SYCL_DEBUG

#endif  // UNIFIED_KERNEL_TEST_STANDALONE

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
class esimd_fp16_double_buffered_kernel;

template <int TILE_M, int TILE_N>
class esimd_int8_kernel;

// Cooperative ESIMD kernel: multi-work-item with named barriers
template <int WG_SIZE>
class esimd_fp16_cooperative_kernel;

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

// =============================================================================
// SLM Double-Buffering Constants for ESIMD FP16 Path (Phase 4)
// =============================================================================
// SLM layout for double-buffering memory/compute overlap.
// Each buffer holds one K-tile of weights and activations in FP16.
//
// Buffer sizes (FP16):
// - Weights: TILE_N × K_TILE × sizeof(half) = 16 × 16 × 2 = 512 bytes
// - Activations: TILE_M × K_TILE × sizeof(half) = 8 × 16 × 2 = 256 bytes
// - Total per buffer: 768 bytes
// - Double-buffer total: 1536 bytes
//
// Layout in SLM:
// Buffer 0: [0, 512): weights_0 [16×16 half]
//           [512, 768): activations_0 [8×16 half]
// Buffer 1: [768, 1280): weights_1 [16×16 half]
//           [1280, 1536): activations_1 [8×16 half]

constexpr int ESIMD_SLM_WEIGHTS_SIZE = ESIMD_EXEC_SIZE * ESIMD_K_PER_DPAS;      // 16 × 16 = 256 half elements
constexpr int ESIMD_SLM_ACTS_SIZE = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS;      // 8 × 16 = 128 half elements
constexpr int ESIMD_SLM_BUFFER_SIZE = ESIMD_SLM_WEIGHTS_SIZE + ESIMD_SLM_ACTS_SIZE;  // 384 half elements
constexpr int ESIMD_SLM_TOTAL_SIZE = 2 * ESIMD_SLM_BUFFER_SIZE;                 // 768 half elements for double-buffer

// SLM byte offsets for double-buffering
constexpr uint32_t ESIMD_SLM_BUF0_WEIGHTS = 0;
constexpr uint32_t ESIMD_SLM_BUF0_ACTS = ESIMD_SLM_WEIGHTS_SIZE * sizeof(sycl::half);                          // 512 bytes
constexpr uint32_t ESIMD_SLM_BUF1_WEIGHTS = ESIMD_SLM_BUFFER_SIZE * sizeof(sycl::half);                        // 768 bytes
constexpr uint32_t ESIMD_SLM_BUF1_ACTS = (ESIMD_SLM_BUFFER_SIZE + ESIMD_SLM_WEIGHTS_SIZE) * sizeof(sycl::half); // 1280 bytes
constexpr uint32_t ESIMD_SLM_TOTAL_BYTES = ESIMD_SLM_TOTAL_SIZE * sizeof(sycl::half);                          // 1536 bytes

// =============================================================================
// Cooperative ESIMD Constants (Multi-work-item with named barriers)
// =============================================================================
// Work-group configuration for cooperative loading:
// - 32 work-items per work-group (2 sub-groups of 16)
// - Each sub-group owns one 8x16 output tile
// - All work-items cooperate on loading larger tiles to SLM
//
// SLM layout for cooperative kernel:
// - Weights: [16] x [16] half values (raw, row-major)
// - Activations: [16] x [16] half values (raw, row-major)
// - Each sub-group reads its portion and packs to VNNI in registers
//
// Work-group size constraints:
// - Fixed at 32 (2 sub-groups of 16 work-items)
// - TODO: Add WG_SIZE=64 support in future (4 sub-groups)

constexpr int COOP_SUBGROUP_SIZE = 16;       // Sub-group size for XMX (fixed by hardware)

// Default constants for WG_SIZE=32 (compile-time constants for kernel instantiation)
constexpr int COOP_WG_SIZE_DEFAULT = 32;     // Default work-group size
constexpr int COOP_NUM_SUBGROUPS_DEFAULT = COOP_WG_SIZE_DEFAULT / COOP_SUBGROUP_SIZE;  // 2 sub-groups

// Output tile dimensions per work-group (for WG_SIZE=32)
constexpr int COOP_WG_TILES_M = 2;           // 2 M-tiles per work-group
constexpr int COOP_WG_TILES_N = 1;           // 1 N-tile per work-group
constexpr int COOP_WG_M = COOP_WG_TILES_M * ESIMD_REPEAT_COUNT;   // 16 output rows
constexpr int COOP_WG_N = COOP_WG_TILES_N * ESIMD_EXEC_SIZE;      // 16 output columns

// SLM sizes for cooperative kernel (single buffer for simplicity)
// Note: SLM usage stays under 64KB (actual: 1024 bytes) for all supported WG sizes
constexpr int COOP_SLM_WEIGHTS_SIZE = COOP_WG_N * ESIMD_K_PER_DPAS;  // 16 * 16 = 256 half
constexpr int COOP_SLM_ACTS_SIZE = COOP_WG_M * ESIMD_K_PER_DPAS;     // 16 * 16 = 256 half
constexpr int COOP_SLM_TOTAL_HALF = COOP_SLM_WEIGHTS_SIZE + COOP_SLM_ACTS_SIZE;  // 512 half
constexpr uint32_t COOP_SLM_TOTAL_BYTES = COOP_SLM_TOTAL_HALF * sizeof(sycl::half);  // 1024 bytes

// SLM byte offsets for cooperative kernel
constexpr uint32_t COOP_SLM_WEIGHTS_OFFSET = 0;
constexpr uint32_t COOP_SLM_ACTS_OFFSET = COOP_SLM_WEIGHTS_SIZE * sizeof(sycl::half);  // 512 bytes

// Legacy constant for backwards compatibility
constexpr int COOP_WG_SIZE = COOP_WG_SIZE_DEFAULT;

// =============================================================================
// Vectorized Q4_0 Dequantization using ESIMD SIMD Operations
// =============================================================================
// These functions provide high-throughput dequantization for Q4_0 weights
// using ESIMD vector operations instead of scalar loops.
//
// Q4_0 block layout (18 bytes total):
// - d: sycl::half scale factor (2 bytes)
// - qs[16]: 16 packed bytes containing 32 nibbles (16 bytes)
//   - Low nibble:  qs[i] & 0x0F  -> value - 8 for signed range [-8, +7]
//   - High nibble: qs[i] >> 4   -> value - 8 for signed range [-8, +7]
//
// Weight order in memory (after unpacking):
// - Positions 0-15:  low nibbles from qs[0..15]
// - Positions 16-31: high nibbles from qs[0..15]
// =============================================================================

/**
 * Vectorized dequantization of a full Q4_0 block (32 weights) to FP16.
 *
 * Loads 16 packed bytes and unpacks to 32 half-precision values using
 * ESIMD SIMD operations. This eliminates scalar loops in the hot path.
 *
 * @param block  Pointer to Q4_0 block (must be valid, no bounds check)
 * @return simd<sycl::half, 32> containing dequantized weights
 *
 * Performance: ~16 bytes loaded, 32 weights output = 2:1 expansion ratio
 * Target throughput: >100 GB/s for Q4_0 on Intel Arc
 */
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, UNIFIED_QK4_0>
dequant_q4_0_block_vectorized(const block_q4_0_unified * block) {
    // Load scale factor
    const sycl::half d = block->d;

    // Load all 16 packed bytes at once using pointer cast
    // Note: block->qs is 16 bytes, we load them into a simd vector
    esimd::simd<uint8_t, 16> packed;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        packed[i] = block->qs[i];
    }

    // Extract low nibbles: (packed & 0x0F) - 8
    // Use bitwise AND and subtraction on simd vectors
    esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;

    // Extract high nibbles: (packed >> 4) - 8
    esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

    // Convert to half precision and apply scale
    // Result layout: [lo_0, lo_1, ..., lo_15, hi_0, hi_1, ..., hi_15]
    esimd::simd<sycl::half, UNIFIED_QK4_0> result;

    // Low nibbles go to positions 0-15
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = static_cast<sycl::half>(lo_nibbles[i]) * d;
    }

    // High nibbles go to positions 16-31
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
    }

    return result;
}

/**
 * Vectorized dequantization of a partial Q4_0 block to FP16.
 *
 * For tiles that don't align to full 32-weight blocks, this function
 * handles partial block extraction efficiently.
 *
 * @param block       Pointer to Q4_0 block
 * @param start_idx   Starting index within block (0..31)
 * @param count       Number of weights to extract (1..32-start_idx)
 * @param output      Output array to fill with dequantized weights
 *
 * Note: For best performance, prefer dequant_q4_0_block_vectorized()
 * when processing full blocks.
 */
template <int MAX_COUNT>
SYCL_ESIMD_FUNCTION void dequant_q4_0_partial_vectorized(
    const block_q4_0_unified * block,
    int                        start_idx,
    int                        count,
    esimd::simd<sycl::half, MAX_COUNT> & output) {

    // Get full block dequantization
    esimd::simd<sycl::half, UNIFIED_QK4_0> full = dequant_q4_0_block_vectorized(block);

    // Copy requested portion
    #pragma unroll
    for (int i = 0; i < MAX_COUNT; i++) {
        if (i < count) {
            output[i] = full[start_idx + i];
        }
    }
}

/**
 * Vectorized dequantization for a tile of weights spanning multiple blocks.
 *
 * This is the main entry point for tile-based dequantization. Handles:
 * - Full blocks within the tile (vectorized)
 * - Partial blocks at tile boundaries (vectorized with masking)
 * - VNNI format output for dpas compatibility
 *
 * @tparam TILE_K  K dimension tile size (should be multiple of ESIMD_K_PER_DPAS)
 * @tparam TILE_N  N dimension tile size (typically 16 for dpas)
 * @param weights  Pointer to Q4_0 weight blocks
 * @param n_global Global N index for this weight row
 * @param k_start  Starting K index for this tile
 * @param K        Total K dimension
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param k_len    Valid K elements in this tile
 * @return simd containing dequantized weights in row-major order
 */
template <int TILE_K>
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, TILE_K>
dequant_q4_0_tile_vectorized(
    const block_q4_0_unified * weights,
    int64_t                    n_global,
    int64_t                    k_start,
    int64_t                    K,
    int                        k_blocks_per_row,
    int                        k_len) {

    esimd::simd<sycl::half, TILE_K> result = sycl::half(0.0f);

    // For TILE_K=16 (ESIMD_K_PER_DPAS), we may span at most 2 Q4_0 blocks
    // since each block has 32 weights and TILE_K=16

    // Calculate which block(s) we need
    const int first_block_idx = static_cast<int>(k_start / UNIFIED_QK4_0);
    const int start_in_block = static_cast<int>(k_start % UNIFIED_QK4_0);

    // Get the first block
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + first_block_idx);
    const block_q4_0_unified * blk = &weights[global_block_idx];

    // Dequantize the full block
    esimd::simd<sycl::half, UNIFIED_QK4_0> full_block = dequant_q4_0_block_vectorized(blk);

    // Copy weights from the block to result
    // Handle the case where we start mid-block
    // Note: Using ternary instead of sycl::min which is not supported in ESIMD context
    const int remaining_in_block = UNIFIED_QK4_0 - start_in_block;
    const int weights_from_first_block = (remaining_in_block < k_len) ? remaining_in_block : k_len;

    #pragma unroll
    for (int i = 0; i < TILE_K; i++) {
        if (i < weights_from_first_block) {
            result[i] = full_block[start_in_block + i];
        }
    }

    // If we need weights from a second block (tile spans block boundary)
    if (weights_from_first_block < k_len) {
        const int second_block_idx = first_block_idx + 1;
        if (second_block_idx < k_blocks_per_row) {
            const int global_block_idx_2 = static_cast<int>(n_global * k_blocks_per_row + second_block_idx);
            const block_q4_0_unified * blk2 = &weights[global_block_idx_2];

            esimd::simd<sycl::half, UNIFIED_QK4_0> full_block_2 = dequant_q4_0_block_vectorized(blk2);

            const int weights_from_second = k_len - weights_from_first_block;
            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                if (i >= weights_from_first_block && i < k_len) {
                    result[i] = full_block_2[i - weights_from_first_block];
                }
            }
        }
    }

    return result;
}

// =============================================================================
// Vectorized Q8_0 Dequantization using ESIMD SIMD Operations
// =============================================================================
// Q8_0 is simpler than Q4_0 since each weight is a full byte (no nibble packing).
//
// Q8_0 block layout (34 bytes total):
// - d: sycl::half scale factor (2 bytes)
// - qs[32]: 32 signed int8 values (32 bytes)
//
// No unpacking needed - just load, cast to half, and multiply by scale.
// =============================================================================

/**
 * Vectorized dequantization of a full Q8_0 block (32 weights) to FP16.
 *
 * Q8_0 is simpler than Q4_0 - no nibble unpacking required.
 * Just load 32 int8 values, convert to half, and multiply by scale.
 *
 * @param qs     Pointer to 32 int8 quantized values
 * @param scale  Scale factor to apply
 * @return simd<sycl::half, 32> containing dequantized weights
 *
 * Target throughput: >200 GB/s for Q8_0 on Intel Arc (simpler path)
 */
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, 32>
dequant_q8_0_block_vectorized(const int8_t * qs, sycl::half scale) {
    // Load all 32 int8 values
    esimd::simd<int32_t, 32> weights_int;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        weights_int[i] = static_cast<int32_t>(qs[i]);
    }

    // Convert to half and apply scale
    esimd::simd<sycl::half, 32> result;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        result[i] = static_cast<sycl::half>(weights_int[i]) * scale;
    }

    return result;
}

/**
 * Vectorized dequantization for a K-tile of Q8_0 weights.
 *
 * @tparam TILE_K  K dimension tile size
 * @param qs       Pointer to int8 quantized values
 * @param scale    Scale factor to apply
 * @param k_start  Starting offset within the block
 * @param k_len    Number of weights to extract
 * @return simd containing dequantized weights
 */
template <int TILE_K>
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, TILE_K>
dequant_q8_0_tile_vectorized(
    const int8_t * qs,
    sycl::half     scale,
    int            k_start,
    int            k_len) {

    esimd::simd<sycl::half, TILE_K> result = sycl::half(0.0f);

    // For Q8_0, weights are stored directly as int8
    // Just load, convert, and scale
    #pragma unroll
    for (int i = 0; i < TILE_K; i++) {
        if (i < k_len) {
            const int8_t q = qs[k_start + i];
            result[i] = static_cast<sycl::half>(static_cast<int32_t>(q)) * scale;
        }
    }

    return result;
}

// =============================================================================
// Prefetch Support for Memory/Compute Overlap (Phase 4 - Task llama.cpp-attk)
// =============================================================================
// Prefetching future K-tiles while computing current tiles improves throughput.
//
// Cache hint strategy:
// - Weights: use once per output element, minimal caching benefit
// - Activations: reused across N columns, benefits from caching
//
// Note: ESIMD prefetch uses different APIs depending on compiler version.
// We use esimd::prefetch for basic prefetch functionality.
// The prefetch distance is passed in via args.prefetch_depth from the host.

/**
 * Prefetch Q4_0 weights for a future K-tile using LSC prefetch with cache hints.
 *
 * Prefetches weight data asynchronously using Intel LSC (Load/Store Cache)
 * prefetch intrinsics. Data will be in cache when the load is executed
 * later in the K-loop.
 *
 * Cache hint strategy for weights:
 * - L1: streaming - weights are used once per output element, don't pollute L1
 * - L2: uncached  - weights have poor temporal locality, don't cache in L2
 *
 * @param weights          Pointer to Q4_0 weight blocks
 * @param n_global         Global N index for this weight row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param N                Total N dimension (for bounds checking)
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_weights_block(
    const block_q4_0_unified * weights,
    int64_t                    n_global,
    int64_t                    k_tile_start,
    int64_t                    K,
    int                        k_blocks_per_row,
    int64_t                    N) {

    // Bounds check: don't prefetch beyond array
    if (n_global >= N || k_tile_start >= K) {
        return;
    }

    // Calculate block address for this K-tile
    // Q4_0 blocks have 32 weights each
    const int k_block_idx = static_cast<int>(k_tile_start / UNIFIED_QK4_0);
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx);

    // Prefetch the Q4_0 block using LSC prefetch with streaming cache hints
    // This brings the block into cache ahead of the actual load without
    // occupying registers or polluting cache with data that's only used once.
    const block_q4_0_unified * block_ptr = &weights[global_block_idx];

    // Use LSC prefetch with streaming hints for weights:
    // - L1 streaming: evict-first policy to minimize cache pollution
    // - L2 uncached: don't cache in L2 since weights have poor temporal locality
    //
    // NOTE: Alignment is critical for LSC prefetch:
    // - block_q4_0_unified is 18 bytes (2 bytes scale + 16 bytes quants)
    // - Array of 18-byte blocks is at offsets 0, 18, 36, ... (≡ 2 mod 4)
    // - This violates DWORD (4-byte) alignment required by uint32_t prefetch
    //
    // Solution: Align to 16-byte boundary and prefetch that aligned portion.
    // This covers the entire 18-byte block within a 32-byte range.
    // Calculate 16-byte aligned address (align down)
    constexpr int ALIGN_SIZE = 16;  // 16-byte alignment for proper DWORD-aligned access
    const uint8_t * byte_ptr = reinterpret_cast<const uint8_t *>(block_ptr);
    const uint64_t addr = reinterpret_cast<uint64_t>(byte_ptr);
    const uint64_t aligned_addr = (addr / ALIGN_SIZE) * ALIGN_SIZE;
    const uint32_t * aligned_ptr = reinterpret_cast<const uint32_t *>(aligned_addr);

    constexpr auto props = esimd::properties{
        esimd::cache_hint_L1<esimd::cache_hint::streaming>,
        esimd::cache_hint_L2<esimd::cache_hint::uncached>
    };
    // Prefetch 4 uint32_t values (16 bytes, covers the entire 18-byte block within aligned boundary)
    esimd::prefetch<uint32_t, 4>(
        aligned_ptr, 0, esimd::simd_mask<1>(1), props);
}

/**
 * Prefetch activations for a future K-tile using LSC prefetch with cache hints.
 *
 * Prefetches activation data asynchronously using Intel LSC (Load/Store Cache)
 * prefetch intrinsics. Activations benefit more from caching as they are
 * reused across N columns.
 *
 * Cache hint strategy for activations:
 * - L1: cached - activations are reused across multiple output columns
 * - L2: cached - activations have good temporal locality
 *
 * @param activations      Pointer to activation matrix
 * @param m_global         Global M index for this activation row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension (for indexing and bounds)
 * @param M                Total M dimension (for bounds checking)
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_activations_block(
    const float * activations,
    int64_t       m_global,
    int64_t       k_tile_start,
    int64_t       K,
    int64_t       M) {

    // Bounds check: don't prefetch beyond array
    if (m_global >= M || k_tile_start >= K) {
        return;
    }

    // Calculate address for this K-tile
    const float * tile_ptr = activations + m_global * K + k_tile_start;

    // Use LSC prefetch with cached hints for activations:
    // - L1 cached: activations are reused across N columns
    // - L2 cached: activations have good temporal locality
    // Prefetch K_TILE_SIZE floats (typically 16 floats = 64 bytes)
    constexpr auto props = esimd::properties{
        esimd::cache_hint_L1<esimd::cache_hint::cached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>
    };
    esimd::prefetch<float, K_TILE_SIZE>(tile_ptr, 0, esimd::simd_mask<1>(1), props);
}

/**
 * Prefetch weights for a K-tile (cooperative version).
 *
 * Called by work-items in cooperative kernel. Each work-item prefetches
 * one weight row if local_id < TILE_N.
 *
 * @param weights          Pointer to Q4_0 weight blocks
 * @param n_wg_start       Starting N index for this work-group
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param args             Kernel arguments (for dimensions)
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param local_id         Work-item local ID [0..WG_SIZE)
 * @param tile_n           N tile size (COOP_WG_N)
 */
template <int K_TILE_SIZE, int TILE_N>
SYCL_ESIMD_FUNCTION void prefetch_weights_cooperative(
    const block_q4_0_unified *  weights,
    int64_t                     n_wg_start,
    int64_t                     k_tile_start,
    const UnifiedKernelArgs &   args,
    int                         k_blocks_per_row,
    int                         local_id,
    int                         tile_n) {

    // Only work-items [0..TILE_N) prefetch weights
    if (local_id < tile_n) {
        const int64_t n_global = n_wg_start + local_id;
        prefetch_weights_block<K_TILE_SIZE>(
            weights, n_global, k_tile_start, args.K, k_blocks_per_row, args.N);
    }
}

/**
 * Prefetch activations for a K-tile (cooperative version).
 *
 * Called by work-items in cooperative kernel. Each work-item prefetches
 * one activation row if local_id >= TILE_N.
 *
 * @param activations      Pointer to activation matrix
 * @param m_wg_start       Starting M index for this work-group
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param args             Kernel arguments (for dimensions)
 * @param local_id         Work-item local ID [0..WG_SIZE)
 * @param tile_n           N tile size (COOP_WG_N) - work-items >= this prefetch activations
 * @param tile_m           M tile size (COOP_WG_M)
 */
template <int K_TILE_SIZE, int TILE_M>
SYCL_ESIMD_FUNCTION void prefetch_activations_cooperative(
    const float *             activations,
    int64_t                   m_wg_start,
    int64_t                   k_tile_start,
    const UnifiedKernelArgs & args,
    int                       local_id,
    int                       tile_n,
    int                       tile_m) {

    // Work-items [TILE_N..TILE_N+TILE_M) prefetch activations
    const int act_id = local_id - tile_n;
    if (act_id >= 0 && act_id < tile_m) {
        const int64_t m_global = m_wg_start + act_id;
        prefetch_activations_block<K_TILE_SIZE>(
            activations, m_global, k_tile_start, args.K, args.M);
    }
}

/**
 * Check if cooperative ESIMD dpas path is enabled via environment.
 *
 * Cooperative path uses multiple work-items with named barriers for
 * work-group level loading. Enabled by default while optimizing;
 * set GGML_SYCL_XMX_COOPERATIVE=0 to disable.
 *
 * @return true if cooperative ESIMD path is enabled
 */
inline bool use_cooperative_esimd() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_COOPERATIVE");
        if (!env) {
            enabled = 1;
        } else {
            enabled = (std::string(env) == "0") ? 0 : 1;
        }
    }
    return enabled != 0;
}

/**
 * Get the configured work-group size for cooperative ESIMD kernel.
 *
 * Currently only WG_SIZE=32 is supported (2 sub-groups of 16 work-items).
 * TODO: Add WG_SIZE=64 support in future (4 sub-groups covering 4 output tiles)
 *
 * @return Work-group size (always 32)
 */
inline int get_cooperative_wg_size() {
    return 32;
}

/**
 * Check if cooperative ESIMD dpas path can be used for given dimensions.
 *
 * Cooperative ESIMD dpas requires:
 * - ESIMD enabled via GGML_SYCL_XMX_ESIMD=1
 * - Cooperative enabled via GGML_SYCL_XMX_COOPERATIVE=1
 * - M >= 8 (at least one dpas M-tile)
 * - N >= 16 (at least one dpas N-tile)
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if cooperative ESIMD dpas can be used
 */
inline bool can_use_cooperative_esimd(int64_t M, int64_t N, int64_t K) {
    // Both ESIMD and cooperative must be enabled
    if (!use_esimd_dpas() || !use_cooperative_esimd()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Need enough work for cooperative loading to be beneficial
    // At least 8 M-rows for one dpas tile per sub-group
    return M >= ESIMD_REPEAT_COUNT && N >= ESIMD_EXEC_SIZE;
}

/**
 * Load weights for a K-tile to SLM with VNNI packing for ESIMD dpas.
 *
 * Loads and dequantizes Q4_0 weights from global memory to SLM in VNNI format.
 * VNNI layout for FP16: b[(k/2) * N * 2 + n * 2 + (k%2)]
 *
 * @param weights       Pointer to Q4_0 weight blocks
 * @param slm_offset    SLM byte offset for weights buffer
 * @param n_start       Starting N index
 * @param k_start       Starting K index for this tile
 * @param N             Total N dimension
 * @param K             Total K dimension
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param k_len         Valid K elements in this tile (may be < ESIMD_K_PER_DPAS)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_weights_to_slm_vnni(
    const block_q4_0_unified * weights,
    uint32_t                   slm_offset,
    int64_t                    n_start,
    int64_t                    k_start,
    int64_t                    N,
    int64_t                    K,
    int                        k_blocks_per_row,
    int                        k_len) {

    // Load and dequantize weights with VNNI packing using vectorized dequantization
    esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> w_vec = sycl::half(0.0f);

    #pragma unroll
    for (int n = 0; n < TILE_N; n++) {
        const int64_t n_global = n_start + n;
        if (n_global >= N) continue;

        // Use vectorized tile dequantization for this row
        esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_weights =
            dequant_q4_0_tile_vectorized<ESIMD_K_PER_DPAS>(
                weights, n_global, k_start, K, k_blocks_per_row, k_len);

        // Repack to VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
        #pragma unroll
        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
            const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
            w_vec[vnni_idx] = (k < k_len) ? row_weights[k] : sycl::half(0.0f);
        }
    }

    // Store to SLM
    esimd::slm_block_store<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(slm_offset, w_vec);
}

/**
 * Load activations for a K-tile to SLM in row-major format.
 *
 * @param activations   Pointer to activation matrix
 * @param slm_offset    SLM byte offset for activations buffer
 * @param m_start       Starting M index
 * @param k_start       Starting K index for this tile
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param k_len         Valid K elements in this tile (may be < ESIMD_K_PER_DPAS)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_activations_to_slm(
    const float * activations,
    uint32_t      slm_offset,
    int64_t       m_start,
    int64_t       k_start,
    int64_t       M,
    int64_t       K,   // Total K dimension (used to index activations)
    int           k_len) {

    esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec = sycl::half(0.0f);

    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= M) continue;

        #pragma unroll
        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
            if (k >= k_len) break;

            const int64_t k_global = k_start + k;
            const float act_f32 = activations[m_global * K + k_global];
            // Row-major: a[m * K_per + k]
            a_vec[m * ESIMD_K_PER_DPAS + k] = static_cast<sycl::half>(act_f32);
        }
    }

    // Store to SLM
    esimd::slm_block_store<sycl::half, ESIMD_SLM_ACTS_SIZE>(slm_offset, a_vec);
}

/**
 * ESIMD FP16 matmul kernel with double-buffering (Phase 4).
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Uses double-buffering to overlap memory loads with compute:
 * 1. Pre-load first K-tile into buffer 0
 * 2. For each K-tile:
 *    - Load next K-tile into alternate buffer (if not last)
 *    - Execute dpas on current buffer
 *    - Swap buffers
 * 3. Write final accumulator to output
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 * @param args     Kernel arguments
 * @param m_start  Starting M index for this work-item's tile
 * @param n_start  Starting N index for this work-item's tile
 * @param cfg      XMX configuration (use_double_buffer flag)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_double_buffered_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start,
    const XMXConfig &       /* cfg - reserved for future tuning */) {

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

    // Initialize SLM for double-buffering
    esimd::slm_init<ESIMD_SLM_TOTAL_BYTES>();

    // Initialize accumulator: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Number of K tiles (each dpas processes 16 K elements)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // Edge case: Single K-tile (K <= K_TILE)
    // No overlap opportunity, but still works correctly
    if (k_tiles <= 1) {
        // Just load and compute directly without double-buffering
        const int64_t k_start = 0;
        const int64_t k_remaining = args.K;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // Load weights and activations to buffer 0
        load_weights_to_slm_vnni<TILE_M, TILE_N>(
            weights, ESIMD_SLM_BUF0_WEIGHTS,
            n_start, k_start, args.N, args.K, k_blocks_per_row, k_len);
        load_activations_to_slm<TILE_M, TILE_N>(
            args.activations, ESIMD_SLM_BUF0_ACTS,
            m_start, k_start, args.M, args.K, k_len);

        // Fence to ensure SLM writes complete before reads
        esimd::fence<esimd::fence_mask::local_barrier>();

        // Load from SLM to registers
        esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> b_vec =
            esimd::slm_block_load<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(ESIMD_SLM_BUF0_WEIGHTS);
        esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec =
            esimd::slm_block_load<sycl::half, ESIMD_SLM_ACTS_SIZE>(ESIMD_SLM_BUF0_ACTS);

        // Execute dpas
        acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                        float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
    } else {
        // ========================================================================
        // Double-buffered K-loop for memory/compute overlap
        // ========================================================================

        // Pre-load first K-tile into buffer 0
        {
            const int64_t k_start = 0;
            const int64_t k_remaining = args.K;
            const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

            load_weights_to_slm_vnni<TILE_M, TILE_N>(
                weights, ESIMD_SLM_BUF0_WEIGHTS,
                n_start, k_start, args.N, args.K, k_blocks_per_row, k_len);
            load_activations_to_slm<TILE_M, TILE_N>(
                args.activations, ESIMD_SLM_BUF0_ACTS,
                m_start, k_start, args.M, args.K, k_len);
        }

        // Fence after pre-loading buffer 0
        esimd::fence<esimd::fence_mask::local_barrier>();

        int buf_compute = 0;  // Start with buffer 0

        // Get prefetch distance from kernel args (set by host-side launch_unified_matmul)
        // Double-buffering loads kt+1, so prefetch targets kt+prefetch_depth
        const int prefetch_distance = args.prefetch_depth;

        // Main K-loop with double-buffering
        for (int64_t kt = 0; kt < k_tiles; kt++) {
            // ================================================================
            // Prefetch: Look ahead beyond double-buffer distance
            // ================================================================
            // Double-buffering loads kt+1. Prefetch targets kt+prefetch_depth.
            // This gets data into cache before the load is needed.
            if (prefetch_distance > 1 && kt + prefetch_distance < k_tiles) {
                const int64_t prefetch_k_start = (kt + prefetch_distance) * ESIMD_K_PER_DPAS;

                // Prefetch weights for future K-tile
                #pragma unroll
                for (int n = 0; n < TILE_N; n++) {
                    prefetch_weights_block<ESIMD_K_PER_DPAS>(
                        weights, n_start + n, prefetch_k_start, args.K, k_blocks_per_row, args.N);
                }

                // Prefetch activations for future K-tile
                #pragma unroll
                for (int m = 0; m < TILE_M; m++) {
                    prefetch_activations_block<ESIMD_K_PER_DPAS>(
                        args.activations, m_start + m, prefetch_k_start, args.K, args.M);
                }
            }

            // Determine current buffer offsets
            const uint32_t compute_w_off = (buf_compute == 0) ? ESIMD_SLM_BUF0_WEIGHTS : ESIMD_SLM_BUF1_WEIGHTS;
            const uint32_t compute_a_off = (buf_compute == 0) ? ESIMD_SLM_BUF0_ACTS : ESIMD_SLM_BUF1_ACTS;

            // Determine load buffer offsets (alternate buffer)
            const uint32_t load_w_off = (buf_compute == 0) ? ESIMD_SLM_BUF1_WEIGHTS : ESIMD_SLM_BUF0_WEIGHTS;
            const uint32_t load_a_off = (buf_compute == 0) ? ESIMD_SLM_BUF1_ACTS : ESIMD_SLM_BUF0_ACTS;

            // Load from SLM to registers (from compute buffer)
            esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> b_vec =
                esimd::slm_block_load<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(compute_w_off);
            esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec =
                esimd::slm_block_load<sycl::half, ESIMD_SLM_ACTS_SIZE>(compute_a_off);

            // Prefetch/load next K-tile into alternate buffer (if not last iteration)
            if (kt + 1 < k_tiles) {
                const int64_t next_k_start = (kt + 1) * ESIMD_K_PER_DPAS;
                const int64_t next_k_remaining = args.K - next_k_start;
                const int next_k_len = static_cast<int>(next_k_remaining < ESIMD_K_PER_DPAS ? next_k_remaining : ESIMD_K_PER_DPAS);

                load_weights_to_slm_vnni<TILE_M, TILE_N>(
                    weights, load_w_off,
                    n_start, next_k_start, args.N, args.K, k_blocks_per_row, next_k_len);
                load_activations_to_slm<TILE_M, TILE_N>(
                    args.activations, load_a_off,
                    m_start, next_k_start, args.M, args.K, next_k_len);
            }

            // Execute dpas on current buffer's data
            acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                            float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);

            // Fence after SLM writes (ensure next iteration's loads are visible)
            if (kt + 1 < k_tiles) {
                esimd::fence<esimd::fence_mask::local_barrier>();
            }

            // Swap buffers
            buf_compute = 1 - buf_compute;
        }
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
        // Using vectorized dequantization for improved throughput.
        //
        // dpas computes: C[m,n] += sum_k(A[m,k] * B[k,n])
        // GGML wants: dst[m,n] = sum_k(activations[m,k] * weights[n,k])
        // So B[k,n] = weights[n,k] (transpose)
        //
        // For FP16 dpas, B matrix needs VNNI-like layout:
        // B_vnni[k/2 * N * 2 + n * 2 + k%2] = B[k,n]
        // This groups consecutive K values together for systolic array
        // ============================================================
        esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec = sycl::half(0.0f);

        // Use vectorized dequantization for each weight row, then repack to VNNI
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            // Vectorized tile dequantization for this row
            esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_weights =
                dequant_q4_0_tile_vectorized<ESIMD_K_PER_DPAS>(
                    weights, n_global, k_start, args.K, k_blocks_per_row, k_len);

            // Repack to VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
                b_vec[vnni_idx] = (k < k_len) ? row_weights[k] : sycl::half(0.0f);
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
// Uses INT8 dpas with dynamic quantization for both weights and activations.
//
// Quantization approach:
// 1. Dequantize Q4_0 weights to FP values: w_fp = (nibble - 8) * d
// 2. Find max-abs per N column per K-tile for weights
// 3. Quantize weights to INT8: w_int8 = w_fp * 127 / w_max_abs
// 4. Find max-abs per M row for activations (across all K)
// 5. Quantize activations to INT8: a_int8 = a * 127 / a_max_abs
// 6. Execute dpas: int32_acc = sum_k(w_int8 * a_int8)
// 7. Dequantize result: fp_result = int32_acc * w_scale * a_scale / (127 * 127)
//
// Key differences from FP16:
// - K-tile = 32 (not 16)
// - dpas outputs INT32 accumulator
// - Both weights and activations dynamically quantized
//
// IMPORTANT: INT8 is LOSSY - not bit-exact with FP16/FP32 path!

/**
 * ESIMD INT8 matmul kernel using xmx::dpas instruction.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
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

    // Initialize FP32 accumulator for final result: [8 x 16]
    // We accumulate in FP32 because each K-tile has different scales
    esimd::simd<float, ESIMD_ACC_SIZE> fp_acc = 0.0f;

    // Number of K tiles (each dpas processes 32 K elements for INT8)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS_INT8 - 1) / ESIMD_K_PER_DPAS_INT8;

    // ========================================================================
    // Step 1: Compute per-row activation scales (max-abs for each M row)
    // This needs to scan all K elements for each of the 8 rows in this tile
    // ========================================================================
    esimd::simd<float, TILE_M> act_max_abs = 0.0f;

    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) {
            act_max_abs[m] = 1.0f;  // Dummy scale for out-of-bounds rows
            continue;
        }

        float max_abs = 0.0f;
        for (int64_t k = 0; k < args.K; k++) {
            float val = args.activations[m_global * args.K + k];
            float abs_val = (val >= 0.0f) ? val : -val;
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        act_max_abs[m] = (max_abs > 1e-10f) ? max_abs : 1.0f;
    }

    // ========================================================================
    // K-loop: iterate over K dimension in tiles of 32
    // ========================================================================
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start_local = kt * ESIMD_K_PER_DPAS_INT8;
        const int64_t k_remaining = args.K - k_start_local;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS_INT8 ? k_remaining : ESIMD_K_PER_DPAS_INT8);

        // ====================================================================
        // Step 2a: Dequantize Q4_0 weights to FP and find max-abs per N column
        // ====================================================================
        // Temporary storage for dequantized weights [TILE_N x K_TILE]
        float w_fp[TILE_N][ESIMD_K_PER_DPAS_INT8];
        esimd::simd<float, TILE_N> w_max_abs = 0.0f;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (n_global >= args.N || k >= k_len) {
                    w_fp[n][k] = 0.0f;
                    continue;
                }

                const int64_t k_global = k_start_local + k;
                const int k_block_idx = static_cast<int>(k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx);
                const block_q4_0_unified * blk = &weights[block_idx];
                const float d = static_cast<float>(blk->d);

                int qs_val;
                if (idx_in_block < 16) {
                    qs_val = blk->qs[idx_in_block] & 0x0F;
                } else {
                    qs_val = blk->qs[idx_in_block - 16] >> 4;
                }

                // Dequantize to FP: w = (nibble - 8) * d
                const float w_val = static_cast<float>(qs_val - 8) * d;
                w_fp[n][k] = w_val;

                // Track max-abs for this column
                float abs_w = (w_val >= 0.0f) ? w_val : -w_val;
                if (abs_w > w_max_abs[n]) {
                    w_max_abs[n] = abs_w;
                }
            }
        }

        // Ensure no divide by zero
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            if (w_max_abs[n] < 1e-10f) {
                w_max_abs[n] = 1.0f;
            }
        }

        // ====================================================================
        // Step 2b: Quantize weights to INT8 with VNNI packing
        // ====================================================================
        esimd::simd<int8_t, ESIMD_B_SIZE_INT8> b_vec = 0;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                // Quantize: w_int8 = w_fp * 127 / max_abs
                float qval = w_fp[n][k] * 127.0f / w_max_abs[n];
                if (qval > 127.0f) qval = 127.0f;
                if (qval < -127.0f) qval = -127.0f;
                const int8_t w_int8 = static_cast<int8_t>(qval);

                // VNNI layout for INT8: b[(k/4) * N * 4 + n * 4 + (k%4)]
                const int vnni_idx = (k / 4) * (TILE_N * 4) + n * 4 + (k % 4);
                b_vec[vnni_idx] = w_int8;
            }
        }

        // ====================================================================
        // Step 3: Quantize activations to INT8
        // ====================================================================
        esimd::simd<int8_t, ESIMD_A_SIZE_INT8> a_vec = 0;

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const int64_t m_global = m_start + m;
            if (m_global >= args.M) continue;

            const float scale_inv = 127.0f / act_max_abs[m];

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (k >= k_len) break;

                const int64_t k_global = k_start_local + k;
                const float act_f32 = args.activations[m_global * args.K + k_global];

                float qval = act_f32 * scale_inv;
                if (qval > 127.0f) qval = 127.0f;
                if (qval < -127.0f) qval = -127.0f;

                const int8_t a_int8 = static_cast<int8_t>(qval);
                a_vec[m * ESIMD_K_PER_DPAS_INT8 + k] = a_int8;
            }
        }

        // ====================================================================
        // Step 4: Execute dpas: int32_acc = sum_k(w_int8 * a_int8)
        // ====================================================================
        esimd::simd<int32_t, ESIMD_ACC_SIZE_INT8> int32_acc = 0;
        int32_acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                              int32_t, int32_t, int8_t, int8_t>(int32_acc, b_vec, a_vec);

        // ====================================================================
        // Step 5: Dequantize and accumulate to FP32
        // fp_result += int32_acc * (w_max_abs / 127) * (a_max_abs / 127)
        //            = int32_acc * w_max_abs * a_max_abs / (127 * 127)
        // ====================================================================
        constexpr float scale_denom = 127.0f * 127.0f;

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const float a_scale = act_max_abs[m] / scale_denom;

            #pragma unroll
            for (int n = 0; n < TILE_N; n++) {
                const int idx = m * TILE_N + n;
                const float w_scale = w_max_abs[n];
                const float combined_scale = w_scale * a_scale;

                fp_acc[idx] += static_cast<float>(int32_acc[idx]) * combined_scale;
            }
        }
    }

    // ========================================================================
    // Write output with boundary checking
    // ========================================================================
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = fp_acc[m * TILE_N + n];
        }
    }
}

// =============================================================================
// Cooperative ESIMD FP16 Kernel Implementation
// =============================================================================
// Uses multiple work-items with named barriers for work-group level loading.
// Each sub-group (16 work-items) owns one 8x16 output tile.
// All work-items cooperate on loading data to SLM using strided pattern.
//
// Key differences from single work-item kernel:
// 1. Work-group size: 32 (2 sub-groups) vs 1
// 2. Cooperative loading: All work-items load together, then barrier
// 3. Each sub-group computes its own output tile after loading
// 4. Uses esimd::named_barrier for work-group synchronization

/**
 * Cooperative ESIMD FP16 matmul kernel using xmx::dpas with work-group level loading.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Work distribution:
 * - 2D grid of work-groups, each handles COOP_WG_M x COOP_WG_N output region
 * - Each work-group has 32 work-items (2 sub-groups of 16)
 * - Sub-group 0: handles M-rows [0..7], sub-group 1: handles M-rows [8..15]
 * - All work-items cooperate on loading weights [16 x K_TILE] and activations [16 x K_TILE]
 *
 * Named barrier usage:
 * - barrier 0: Synchronize after cooperative loading
 * - esimd::fence ensures memory visibility before barrier
 *
 * Block operations strategy (for performance):
 * - Loading phase: Each work-item loads one row (16 half) and uses slm_block_store
 * - Compute phase: Use slm_block_load to fetch entire 8x16 or 16x16 tiles
 *
 * @param item        ND-item for work distribution
 * @param args        Kernel arguments
 * @param local_id    Linear work-item ID within work-group [0..31]
 * @param sg_id       Sub-group ID within work-group [0..1]
 * @param lane        Lane ID within sub-group [0..15]
 */
template <int WG_SIZE>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_cooperative_impl(
    sycl::nd_item<2>            item,
    const UnifiedKernelArgs     args,
    int                         local_id,
    int                         sg_id,
    int                         lane) {

    // Compile-time validation of WG_SIZE constraints
    static_assert(WG_SIZE % 16 == 0, "WG_SIZE must be multiple of 16 for sub-group size");
    static_assert(WG_SIZE >= 32, "WG_SIZE must be at least 32 (2 sub-groups)");

    // Work-group coordinates (each work-group handles COOP_WG_M x COOP_WG_N output)
    const int wg_row = item.get_group(0);  // M dimension
    const int wg_col = item.get_group(1);  // N dimension

    // Global output coordinates for this work-group
    const int64_t m_wg_start = wg_row * COOP_WG_M;  // Starting output row for work-group
    const int64_t n_wg_start = wg_col * COOP_WG_N;  // Starting output column for work-group

    // Boundary check: skip if entire work-group is out of bounds
    if (m_wg_start >= args.M || n_wg_start >= args.N) {
        return;
    }

    // This sub-group's output tile coordinates
    // Sub-group 0: M-rows [0..7], Sub-group 1: M-rows [8..15]
    const int64_t m_sg_start = m_wg_start + sg_id * ESIMD_REPEAT_COUNT;
    const int64_t n_sg_start = n_wg_start;  // All sub-groups handle same N range

    // Check if this sub-group has work (boundary check)
    const bool sg_has_work = (m_sg_start < args.M && n_sg_start < args.N);

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize SLM for this work-group
    // Layout: [weights: 16 x 16 half][activations: 16 x 16 half]
    esimd::slm_init<COOP_SLM_TOTAL_BYTES>();

    // Initialize accumulator for this sub-group's output tile: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Initialize named barrier 0 for work-group synchronization
    // All WG_SIZE work-items participate
    esimd::named_barrier_init<1>();  // 1 barrier with ID 0

    // Number of K tiles
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // Get prefetch distance from kernel args (set by host-side launch_unified_matmul)
    // args.prefetch_depth is set from get_prefetch_distance() on the host side
    const int prefetch_distance = args.prefetch_depth;

    // K-loop: iterate over K dimension in tiles of 16
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * ESIMD_K_PER_DPAS;
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // ================================================================
        // Phase 0: LSC Prefetch for Future K-Tiles (Memory/Compute Overlap)
        // ================================================================
        // Prefetch future tile data while loading current tile.
        // Uses cache hints based on data reuse patterns:
        // - Weights: streaming (used once per output element)
        // - Activations: cached (reused across N columns)
        //
        // Bounds checking: Only prefetch if future tile exists
        // Guard: if (kt + prefetch_distance < k_tiles) prefetch(...)
        if (prefetch_distance > 0 && kt + prefetch_distance < k_tiles) {
            const int64_t prefetch_k_start = (kt + prefetch_distance) * ESIMD_K_PER_DPAS;

            // Prefetch weights for future K-tile (work-items 0..15)
            prefetch_weights_cooperative<ESIMD_K_PER_DPAS, COOP_WG_N>(
                weights, n_wg_start, prefetch_k_start, args,
                k_blocks_per_row, local_id, COOP_WG_N);

            // Prefetch activations for future K-tile (work-items 16..31)
            prefetch_activations_cooperative<ESIMD_K_PER_DPAS, COOP_WG_M>(
                args.activations, m_wg_start, prefetch_k_start, args,
                local_id, COOP_WG_N, COOP_WG_M);
        }

        // ================================================================
        // Phase 1: Cooperative Loading with Block Operations
        // ================================================================
        // SLM layout: [weights: 16 rows x 16 cols][activations: 16 rows x 16 cols]
        // 32 work-items load 32 rows total (16 weight rows + 16 activation rows)
        // Each work-item loads one row (16 half values) using block store
        //
        // Work-item assignment:
        // - local_id 0-15: Load weight rows 0-15
        // - local_id 16-31: Load activation rows 0-15

        // Each work-item loads exactly one row of 16 half values
        constexpr int ELEMS_PER_ROW = ESIMD_K_PER_DPAS;  // 16

        if (local_id < COOP_WG_N) {
            // Work-items 0-15: Load weight row local_id using vectorized dequantization
            const int n_off = local_id;
            const int64_t n_global = n_wg_start + n_off;

            // Use vectorized tile dequantization for this row
            esimd::simd<sycl::half, ELEMS_PER_ROW> w_row = sycl::half(0.0f);

            if (n_global < args.N) {
                // Vectorized dequantization: load and unpack entire tile at once
                w_row = dequant_q4_0_tile_vectorized<ELEMS_PER_ROW>(
                    weights, n_global, k_start, args.K, k_blocks_per_row, k_len);

                // Zero out elements beyond k_len for boundary handling
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) {
                        w_row[k] = sycl::half(0.0f);
                    }
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = COOP_SLM_WEIGHTS_OFFSET + n_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, w_row);

        } else {
            // Work-items 16-31: Load activation row (local_id - 16)
            const int m_off = local_id - COOP_WG_N;  // 0-15
            const int64_t m_global = m_wg_start + m_off;

            // Build row vector in registers, then block store
            esimd::simd<sycl::half, ELEMS_PER_ROW> a_row = sycl::half(0.0f);

            if (m_global < args.M) {
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) break;
                    const int64_t k_global = k_start + k;
                    a_row[k] = static_cast<sycl::half>(args.activations[m_global * args.K + k_global]);
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = COOP_SLM_ACTS_OFFSET + m_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, a_row);
        }

        // Fence and barrier to ensure all loading is complete
        // named_barrier_signal(barrier_id, producer_consumer_mode, num_producers, num_consumers)
        // Note: Mode=0 acts as simple barrier on Arc/Xe2 (all work-items sync without role tracking).
        // PVC docs say 0x1=producer, 0x2=consumer, 0x3=both, but mode=3 hangs on Arc.
        // Keep mode=0 for Arc compatibility - all WG_SIZE work-items participate as peers.
        esimd::fence<esimd::fence_mask::local_barrier>();
        esimd::named_barrier_signal</*Fence=*/false>(0, 0, WG_SIZE, WG_SIZE);
        esimd::named_barrier_wait(0);

        // ================================================================
        // Phase 2: Compute with Block Loads
        // ================================================================
        if (sg_has_work) {
            // Block load activations for this sub-group's M-rows [sg_id*8 .. sg_id*8+7]
            // SLM layout is row-major: row m at offset m * 16 * sizeof(half)
            // A layout for dpas: a[m * K_per + k] for m in [0..7], k in [0..15]
            const int m_base = sg_id * ESIMD_REPEAT_COUNT;  // 0 or 8
            const uint32_t a_slm_base = COOP_SLM_ACTS_OFFSET + m_base * ELEMS_PER_ROW * sizeof(sycl::half);

            // Block load all 8 rows (8 x 16 = 128 half values)
            esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec =
                esimd::slm_block_load<sycl::half, ESIMD_A_SIZE>(a_slm_base);

            // Handle boundary: zero out rows beyond M
            #pragma unroll
            for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                const int64_t m_global = m_wg_start + m_base + m;
                if (m_global >= args.M) {
                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        a_vec[m * ESIMD_K_PER_DPAS + k] = sycl::half(0.0f);
                    }
                }
            }

            // Block load weights (all 16 rows = 256 half values)
            // Then repack to VNNI format for dpas
            esimd::simd<sycl::half, COOP_SLM_WEIGHTS_SIZE> w_raw =
                esimd::slm_block_load<sycl::half, COOP_SLM_WEIGHTS_SIZE>(COOP_SLM_WEIGHTS_OFFSET);

            // Repack weights from row-major to VNNI format
            // Row-major: w_raw[n * 16 + k] for n in [0..15], k in [0..15]
            // VNNI: b_vec[(k/2) * N * 2 + n * 2 + (k%2)]
            esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec;
            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                #pragma unroll
                for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                    const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                    b_vec[vnni_idx] = w_raw[n * ESIMD_K_PER_DPAS + k];
                }
            }

            // Handle boundary: zero out weights beyond N
            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_wg_start + n;
                if (n_global >= args.N) {
                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                        b_vec[vnni_idx] = sycl::half(0.0f);
                    }
                }
            }

            // Execute dpas: acc += A @ B
            acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                            float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
        }

        // Barrier before next K-tile (ensure all sub-groups done before overwriting SLM)
        // Mode=0 for Arc/Xe2 compatibility (see note above)
        if (kt + 1 < k_tiles) {
            esimd::fence<esimd::fence_mask::local_barrier>();
            esimd::named_barrier_signal</*Fence=*/false>(0, 0, WG_SIZE, WG_SIZE);
            esimd::named_barrier_wait(0);
        }
    }

    // ================================================================
    // Phase 3: Write output (each sub-group writes its tile)
    // ================================================================
    if (sg_has_work) {
        #pragma unroll
        for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
            const int64_t m_global = m_sg_start + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_sg_start + n;
                if (n_global >= args.N) continue;

                args.output[m_global * args.N + n_global] = acc[m * ESIMD_EXEC_SIZE + n];
            }
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
 * - ESIMD enabled (default on; disable with GGML_SYCL_XMX_ESIMD=0)
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
    // ESIMD path enabled by default; disable with GGML_SYCL_XMX_ESIMD=0
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

void launch_unified_matmul(sycl::queue & q, const UnifiedKernelArgs & args_in) {
    // Make a mutable copy of args to set prefetch_depth from host configuration
    UnifiedKernelArgs args = args_in;

    // Set prefetch distance from host-side configuration if not already set
    // This ensures the kernel gets the correct value without calling getenv from device
    if (args.prefetch_depth <= 0) {
        args.prefetch_depth = get_prefetch_distance();
    }

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

    // ==========================================================================
    // Batch-Size Gating: Select optimal kernel path
    // ==========================================================================
    // Query XMX config for dispatch decision (cached in XMXConfig::from_device)
    // Use device 0 by default (single-GPU typical case)
    XMXConfig cfg = XMXConfig::from_device(0);

    // Batch size is the M dimension (number of tokens being processed)
    const int batch_size = static_cast<int>(args.M);

    // Select kernel path based on batch size and hardware capabilities
    KernelPath selected_path = select_kernel_path(
        batch_size, args.M, args.N, args.K, args.quant_type, cfg);

    // Debug: Print dispatch decision (controlled by GGML_SYCL_DEBUG=1)
    if (ggml_sycl_unified_debug_enabled()) {
        fprintf(stderr, "[unified-kernel] DISPATCH: batch=%d path=%s M=%lld N=%lld K=%lld "
                "min_batch=%d xmx_supported=%d force_mmvq=%d force_esimd=%d\n",
                batch_size, kernel_path_name(selected_path),
                static_cast<long long>(args.M), static_cast<long long>(args.N),
                static_cast<long long>(args.K),
                get_esimd_min_batch(), cfg.supported ? 1 : 0,
                env_force_mmvq() ? 1 : 0, env_force_esimd() ? 1 : 0);
        fflush(stderr);
    }

    // Early return for DMMV and MMVQ paths - these are handled by existing kernels
    // in ggml-sycl.cpp. The unified kernel only handles the ESIMD_DPAS path here.
    // The caller (ggml_sycl_mul_mat) should check the path and dispatch accordingly.
    //
    // NOTE: This function is the unified kernel launcher. For now, we continue
    // with ESIMD path selection below if ESIMD is available. The DMMV/MMVQ
    // fallback is implicit: if ESIMD paths don't match, scalar path is used.

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
    const bool prefer_esimd_small = ggml_sycl_unified::prefer_esimd_small();
    const int  esimd_min_batch    = ggml_sycl_unified::get_esimd_min_batch();
    const int  prefer_esimd_max_m = ggml_sycl_unified::prefer_esimd_max_m();
    const bool allow_joint_matrix = !(prefer_esimd_small &&
                                     (batch_size < esimd_min_batch || args.M <= prefer_esimd_max_m));
    bool use_xmx_path = false;
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (allow_joint_matrix && args.use_xmx && can_use_xmx(args.M, args.N, args.K)) {
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
        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] XMX path: M=%lld N=%lld K=%lld grid=(%d,%d)\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), xmx_grid_m, xmx_grid_n);
            fflush(stderr);
        }

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
    // Enabled by default while optimizing; set GGML_SYCL_XMX_ESIMD=0 to disable.
    // Each work-item processes one 8x16 output tile using ESIMD dpas instruction.
    //
    // Two variants:
    // 1. INT8 (GGML_SYCL_XMX_INT8=1): Dynamic activation quantization, K=32 per dpas
    // 2. FP16 (default when ESIMD enabled): K=16 per dpas
    //
    // NOTE: INT8 is LOSSY - not bit-exact with FP16 path!
    //
    // Batch-size gating: Only use ESIMD path if selected_path == ESIMD_DPAS
    // Small batches should use scalar DMMV/MMVQ kernels instead.

#if GGML_SYCL_ESIMD_AVAILABLE
    // Only proceed with ESIMD paths if batch-size gating selected ESIMD_DPAS
    const bool esimd_enabled_by_gating = (selected_path == KernelPath::ESIMD_DPAS);

    // Try INT8 path first (requires both ESIMD and INT8 flags AND batch-size gating)
    if (esimd_enabled_by_gating && can_use_esimd_int8_dpas(args.M, args.N, args.K)) {
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

    // Cooperative ESIMD FP16 path (multi-work-item with named barriers)
    // Enabled by default while optimizing; set GGML_SYCL_XMX_COOPERATIVE=0 to disable
    // Work-group size configurable via GGML_SYCL_ESIMD_WG_SIZE (valid: 32, 64)
    if (esimd_enabled_by_gating && can_use_cooperative_esimd(args.M, args.N, args.K)) {
        // Get configured work-group size (default: 32)
        const int wg_size = get_cooperative_wg_size();

        // Grid dimensions: each work-group handles COOP_WG_M x COOP_WG_N output
        const int coop_grid_m = (static_cast<int>(args.M) + COOP_WG_M - 1) / COOP_WG_M;
        const int coop_grid_n = (static_cast<int>(args.N) + COOP_WG_N - 1) / COOP_WG_N;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] Cooperative ESIMD FP16 path: M=%lld N=%lld K=%lld "
                    "grid=(%d,%d) wg_size=%d\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), coop_grid_m, coop_grid_n, wg_size);
            fflush(stderr);
        }

        // Currently only WG_SIZE=32 is fully implemented
        // WG_SIZE=64 requires larger SLM tiles (TODO: implement in future)
        // The wg_size variable is checked but always returns 32 for now
        (void)wg_size;  // Suppress unused variable warning

        q.submit([&](sycl::handler & cgh) {
            constexpr int WG_SIZE = 32;
            sycl::range<2> global(coop_grid_m * WG_SIZE, coop_grid_n);
            sycl::range<2> local(WG_SIZE, 1);

            cgh.parallel_for<esimd_fp16_cooperative_kernel<WG_SIZE>>(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    const int local_id = item.get_local_id(0);
                    const int sg_id = local_id / COOP_SUBGROUP_SIZE;
                    const int lane = local_id % COOP_SUBGROUP_SIZE;
                    esimd_matmul_fp16_cooperative_impl<WG_SIZE>(
                        item, args, local_id, sg_id, lane);
                }
            );
        });
        return;
    }

    // FP16 path (ESIMD enabled but INT8 not enabled)
    if (esimd_enabled_by_gating && can_use_esimd_dpas(args.M, args.N, args.K)) {
        // ESIMD FP16 dpas tile sizes (fixed by hardware)
        constexpr int ESIMD_TM = 8;   // RepeatCount = 8
        constexpr int ESIMD_TN = 16;  // ExecutionSize = 16

        const int esimd_grid_m = (static_cast<int>(args.M) + ESIMD_TM - 1) / ESIMD_TM;
        const int esimd_grid_n = (static_cast<int>(args.N) + ESIMD_TN - 1) / ESIMD_TN;

        // Query XMX config for double-buffer capability
        // Use device 0 by default (single-GPU typical case)
        XMXConfig cfg = XMXConfig::from_device(0);

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD FP16 path: M=%lld N=%lld K=%lld grid=(%d,%d) double_buf=%d\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), esimd_grid_m, esimd_grid_n,
                    cfg.use_double_buffer ? 1 : 0);
            fflush(stderr);
        }

        // Use double-buffered kernel if enabled and SLM permits
        if (cfg.use_double_buffer) {
            q.submit([&](sycl::handler & cgh) {
                // ESIMD kernel: one work-item per output tile (no work-group cooperation)
                sycl::range<2> global(esimd_grid_m, esimd_grid_n);
                sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

                // Capture cfg by value for device execution
                XMXConfig cfg_copy = cfg;

                cgh.parallel_for<esimd_fp16_double_buffered_kernel<ESIMD_TM, ESIMD_TN>>(
                    sycl::nd_range<2>(global, local),
                    [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                        // Calculate tile coordinates
                        const int tile_row = item.get_global_id(0);  // M tile index
                        const int tile_col = item.get_global_id(1);  // N tile index

                        const int64_t m_start = tile_row * ESIMD_TM;
                        const int64_t n_start = tile_col * ESIMD_TN;

                        // Call double-buffered ESIMD FP16 kernel implementation
                        esimd_matmul_fp16_double_buffered_impl<ESIMD_TM, ESIMD_TN>(
                            args, m_start, n_start, cfg_copy);
                    }
                );
            });
        } else {
            // Fall back to non-buffered path
            q.submit([&](sycl::handler & cgh) {
                // ESIMD kernel: one work-item per output tile (no work-group cooperation)
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
        }
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
