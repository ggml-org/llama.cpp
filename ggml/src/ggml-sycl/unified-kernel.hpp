//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Architecture for SYCL Matmul
//
// This header defines the unified kernel interface that supports:
// - Q4_0 quantization (initially, other types to follow)
// - Scalar and XMX (dpas) compute paths
// - Configurable tile sizes for auto-tuning
// - Boundary handling for non-aligned dimensions
//
// Design principles:
// - Single kernel entry point with runtime dispatch
// - Template parameters for compile-time optimization
// - SLM-based weight staging for memory efficiency
// - Extensible to multiple quantization formats
//
// XMX Path:
// - Uses Intel joint_matrix extensions for dpas acceleration
// - Tile dimensions: 8x16x8 (M x N x K step) for half precision
// - Requires sub-group size 16 for joint_matrix operations
//

#ifndef GGML_SYCL_UNIFIED_KERNEL_HPP
#define GGML_SYCL_UNIFIED_KERNEL_HPP

#include <cstdint>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>

// Check for joint_matrix support
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE 0
#endif

// =============================================================================
// ESIMD dpas Support
// =============================================================================
// Check for ESIMD extension availability for low-level dpas access.
// ESIMD provides explicit SIMD control for XMX instructions.
//
// ESIMD dpas Layout Requirements (FP16):
// - A operand: Row-major [M x K] where M=Repeat=8, K=16
//   Layout: a[m * K + k] for m=0..7, k=0..15
// - B operand: VNNI-packed [K x N] where K=16, N=ExecSize=16
//   Layout: b[(k/2) * N * 2 + n * 2 + (k%2)] for k=0..15, n=0..15
//   This groups consecutive K values together for efficient systolic array processing.
// - Accumulator: Row-major [M x N] where M=8, N=16
//   Layout: acc[m * N + n] for m=0..7, n=0..15
//
// NOTE: These includes MUST be before any namespace declaration to avoid
// namespace collision issues with SYCL internal headers.

#if __has_include(<sycl/ext/intel/esimd.hpp>) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    define GGML_SYCL_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
#    include <sycl/ext/intel/esimd/xmx/dpas.hpp>
// Namespace aliases for cleaner ESIMD dpas code (in global namespace)
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
#else
#    define GGML_SYCL_ESIMD_AVAILABLE 0
#endif

namespace ggml_sycl_unified {

// =============================================================================
// XMX Tile Dimension Constants
// =============================================================================
// Intel XMX (Xe Matrix eXtensions) tile dimensions for dpas instructions.
// These are hardware-defined constraints for joint_matrix operations.
//
// For half precision (fp16) on Intel Arc GPUs:
// - A matrix: 8x16 (M x K) row-major
// - B matrix: 16x16 (K x N) column-major
// - C matrix: 8x16 (M x N) accumulator
//
// Note: INT8 uses different dimensions (8x32 for K).

constexpr int XMX_TILE_M = 8;   // M dimension of XMX output tile
constexpr int XMX_TILE_N = 16;  // N dimension of XMX output tile
constexpr int XMX_TILE_K = 16;  // K step per dpas instruction (for half precision)

// Sub-group size required for joint_matrix operations
constexpr int XMX_SUBGROUP_SIZE = 16;

// =============================================================================
// Environment Variable Checks for XMX Configuration
// =============================================================================
// These functions use static caching to avoid repeated getenv() calls.
// GGML_SYCL_XMX_ESIMD=1: Enable ESIMD dpas path (disabled by default)
// GGML_SYCL_XMX_INT8=1: Enable INT8 quantization in dpas (disabled by default)

/**
 * Check if ESIMD dpas path is enabled via environment.
 *
 * The ESIMD dpas path uses low-level ESIMD instructions instead of joint_matrix.
 * This is disabled by default. Set GGML_SYCL_XMX_ESIMD=1 to enable.
 *
 * @return true if ESIMD dpas path is enabled
 */
inline bool use_esimd_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_ESIMD");
        enabled          = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

/**
 * Check if INT8 dpas quantization is enabled via environment.
 *
 * When enabled, uses INT8 quantization in dpas instructions (K=32 per step).
 * Default is FP16 (K=16 per step). Set GGML_SYCL_XMX_INT8=1 to enable.
 *
 * @return true if INT8 dpas quantization is enabled
 */
inline bool use_int8_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_INT8");
        enabled          = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// XMXConfig: Hardware-queried Configuration for ESIMD dpas
// =============================================================================
// This struct captures hardware-specific XMX dimensions and capabilities.
// Use XMXConfig::from_device(device_id) to query actual hardware values.
// Default constructor provides safe fallback values for Intel Arc GPUs.

// Note: XMXConfig::from_device() is implemented in unified-kernel.cpp
// and requires ggml_sycl_device_info which is defined in common.hpp (global namespace).

/**
 * XMX configuration for ESIMD dpas kernels.
 *
 * Captures hardware-queried tile dimensions and capabilities.
 * Intel ESIMD dpas parameters:
 * - SystolicDepth: Always 8 (fixed in hardware)
 * - RepeatCount: 1-8, determines M dimension (we use 8 for full utilization)
 * - ExecutionSize: 8 for DG2, 16 for PVC/Arc (determines N dimension)
 */
struct XMXConfig {
    // =========================================================================
    // XMX Tile Dimensions
    // =========================================================================
    // These are hardware-defined constraints for dpas instructions.
    // Default values are for Intel Arc B580.

    size_t xmx_m = 8;  // RepeatCount determines M (1-8, we use 8)
    size_t xmx_n = 16; // ExecutionSize: 8 for DG2, 16 for PVC/Arc
    size_t xmx_k_fp16 = 16;  // K for FP16: SystolicDepth(8) x OpsPerChannel(2) = 16
    size_t xmx_k_int8 = 32;  // K for INT8: SystolicDepth(8) x OpsPerChannel(4) = 32

    // =========================================================================
    // Hardware Resources
    // =========================================================================

    size_t slm_size = 65536;  // SLM bytes per work-group (default 64KB)
    int    nsm      = 20;     // Compute units (streaming multiprocessors)

    // =========================================================================
    // Capability Flags (from hardware query)
    // =========================================================================

    bool supported     = false;  // Hardware has XMX support
    bool supports_int8 = false;  // INT8 dpas available
    bool supports_fp16 = false;  // FP16 dpas available

    // =========================================================================
    // Derived Configuration
    // =========================================================================

    bool use_double_buffer  = false;  // SLM can hold 2x tile buffers
    int  tiles_per_workitem = 1;      // Tiles processed per work-item

    // =========================================================================
    // Factory Method
    // =========================================================================

    /**
     * Query hardware and create configuration for a specific device.
     *
     * @param device_id  Device index (0-based), or -1 for default config
     * @return XMXConfig populated with hardware values, or defaults if unavailable
     *
     * Edge cases handled:
     * - device_id < 0: Returns default config
     * - device_id >= device_count: Returns default config
     * - xmx.M/N/K = 0: Uses fallback defaults
     * - xmx.slm_size = 0: Uses default 65536
     * - xmx.supported = false: Returns config with supported=false
     */
    static XMXConfig from_device(int device_id);
};

// Note: XMXConfig::from_device() is implemented in unified-kernel.cpp
// because it requires access to ggml_sycl_device_info which is defined in common.hpp.

// =============================================================================
// Batch Strategy for XMX Path
// =============================================================================
// Different batch sizes benefit from different tiling strategies.

enum class BatchStrategy {
    WIDE_N,      // Batch 1-7: Wide N-tiles (single row, process multiple N columns per sub-group)
    STANDARD,    // Batch 8-63: Standard tiling with multiple M and N tiles
    PERSISTENT   // Batch 64+: Multiple tiles per workgroup with persistent threads
};

/**
 * Determine the optimal batch strategy for XMX path.
 *
 * @param batch_size  Number of tokens (M dimension)
 * @return Recommended batch strategy
 */
inline BatchStrategy get_batch_strategy(int batch_size) {
    if (batch_size <= 7) {
        return BatchStrategy::WIDE_N;
    } else if (batch_size <= 63) {
        return BatchStrategy::STANDARD;
    } else {
        return BatchStrategy::PERSISTENT;
    }
}

// =============================================================================
// Layout Mode Constants
// =============================================================================
// These mirror the reorder_mode enum from common.hpp
// 0 = NONE (AoS), 1 = SOA, 2 = COALESCED, 3 = XMX_COALESCED, 4 = XMX_GEMM_TILED

constexpr int LAYOUT_NONE          = 0;
constexpr int LAYOUT_SOA           = 1;
constexpr int LAYOUT_COALESCED     = 2;
constexpr int LAYOUT_XMX_COALESCED = 3;
constexpr int LAYOUT_XMX_GEMM_TILED = 4;

// =============================================================================
// LayoutMode Enum
// =============================================================================
// Strongly-typed enum for layout selection in unified kernel.
// Values match LAYOUT_* constants above for interoperability.

enum class LayoutMode : int {
    AOS          = LAYOUT_NONE,           // Array of Structures (original contiguous blocks)
    SOA          = LAYOUT_SOA,            // Structure of Arrays (qs bytes first, then scales)
    COALESCED    = LAYOUT_COALESCED,      // Word-major interleaved for sub-group reads
    XMX_COALESCED = LAYOUT_XMX_COALESCED  // K_TILE=32 aligned for dpas instructions
};

// =============================================================================
// Quantization Type Constants
// =============================================================================
// These mirror GGML_TYPE_* enum values
// Used for runtime dispatch to appropriate dequantization code

constexpr int QUANT_TYPE_Q4_0 = 2;   // GGML_TYPE_Q4_0
constexpr int QUANT_TYPE_Q4_1 = 3;   // GGML_TYPE_Q4_1
constexpr int QUANT_TYPE_Q8_0 = 8;   // GGML_TYPE_Q8_0
constexpr int QUANT_TYPE_Q6_K = 14;  // GGML_TYPE_Q6_K

// =============================================================================
// Q4_0 Block Structure
// =============================================================================
// Q4_0: 32 weights per block, 4 bits per weight
// Block layout: [d: fp16] [qs: 16 bytes (32 nibbles)]
// Total size: 18 bytes per block

// Note: UNIFIED_QK4_0 may already be defined by ggml-common.h as a macro.
// Use namespaced constant to avoid conflicts.
constexpr int UNIFIED_QK4_0 = 32;  // Weights per Q4_0 block

struct block_q4_0_unified {
    sycl::half d;                       // Scale factor
    uint8_t    qs[UNIFIED_QK4_0 / 2];  // Quantized values: 16 bytes = 32 nibbles
};

static_assert(sizeof(block_q4_0_unified) == sizeof(sycl::half) + UNIFIED_QK4_0 / 2, "wrong q4_0 block size");

// =============================================================================
// UnifiedKernelArgs: Kernel launch parameters
// =============================================================================
// Contains all information needed to launch the unified matmul kernel.
// Designed to be POD for efficient device-side access.
//
// GGML Tensor Layout Convention:
// ==============================
// In GGML, mul_mat computes: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
//
// This is equivalent to: C = B @ A^T where:
// - A = src0 (weights) with shape [N, K]
// - B = src1 (activations) with shape [M, K]
// - C = dst (output) with shape [M, N]
//
// Key insight: weights are indexed by output column (n), NOT output row (m)!
// Each weight row corresponds to an output column.

struct UnifiedKernelArgs {
    // Matrix dimensions (GGML convention)
    int64_t M;  // Output rows (batch * tokens) - from src1->ne[1]
    int64_t N;  // Output columns (hidden dim / output features) - from src0->ne[1]
    int64_t K;  // Reduction dimension (must be multiple of block size) - from src0->ne[0]

    // Tile configuration (from auto-tuning or heuristics)
    int tile_m;  // M dimension tile size (output rows per tile)
    int tile_n;  // N dimension tile size (output columns per tile)
    int tile_k;  // K dimension tile size (typically 32 for Q4_0)

    // Compute path selection
    bool use_xmx;  // true = XMX/dpas path, false = scalar path

    // Memory layout (legacy int field for compatibility)
    int layout_mode;  // 0=NONE(AoS), 1=SOA, 2=COALESCED, etc.

    // Memory layout (strongly-typed enum)
    LayoutMode layout = LayoutMode::AOS;  // Default: Array of Structures

    // Quantization format
    int quant_type;  // GGML_TYPE_* enum value

    // Prefetch configuration
    int prefetch_depth;  // 0 = none, 1-4 typical

    // Data pointers (device memory)
    // GGML Convention: weights indexed by (n, k), activations indexed by (m, k)
    const void *  weights;      // Quantized weight matrix [N, K/block_size blocks] - src0
    const float * activations;  // Activation matrix [M, K] (row-major F32) - src1
    float *       output;       // Output matrix [M, N] (row-major F32) - dst
};

// =============================================================================
// Kernel Launch Function
// =============================================================================

/**
 * Launch the unified matmul kernel.
 *
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) has shape [N, K] - indexed by output column n
 * - activations (src1) has shape [M, K] - indexed by output row m
 * - output (dst) has shape [M, N]
 *
 * The kernel automatically handles:
 * - Q4_0 dequantization during computation
 * - Boundary conditions for non-aligned dimensions
 * - SLM staging for weight reuse
 *
 * @param q     SYCL queue for submission
 * @param args  Kernel arguments (dimensions, tiles, data pointers)
 */
void launch_unified_matmul(sycl::queue & q, const UnifiedKernelArgs & args);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate required SLM size for unified kernel.
 *
 * @param tile_m  M tile size (output rows)
 * @param tile_n  N tile size (output columns)
 * @param tile_k  K tile size (reduction dimension)
 * @return Size in bytes needed for SLM
 */
inline size_t calculate_slm_size(int tile_m, int tile_n, int tile_k) {
    // SLM usage (GGML convention):
    // - Weight tile: tile_n * tile_k weights (dequantized to float)
    //   Weights are indexed by output column (n), so we load tile_n rows of weights
    // - Activation tile: tile_m * tile_k floats
    //   Activations are indexed by output row (m), so we load tile_m rows of activations
    // For Q4_0 with scalar path, we dequantize to float in SLM
    size_t weight_slm     = static_cast<size_t>(tile_n) * tile_k * sizeof(float);
    size_t activation_slm = static_cast<size_t>(tile_m) * tile_k * sizeof(float);
    return weight_slm + activation_slm;
}

/**
 * Check if dimensions are valid for unified kernel.
 *
 * @param args Kernel arguments
 * @return true if dimensions are valid
 */
inline bool validate_args(const UnifiedKernelArgs & args) {
    // K must be multiple of block size for Q4_0
    if (args.quant_type == QUANT_TYPE_Q4_0 && (args.K % UNIFIED_QK4_0) != 0) {
        return false;
    }

    // Dimensions must be positive
    if (args.M <= 0 || args.N <= 0 || args.K <= 0) {
        return false;
    }

    // Tile sizes must be positive
    if (args.tile_m <= 0 || args.tile_n <= 0 || args.tile_k <= 0) {
        return false;
    }

    // Pointers must be valid
    if (args.weights == nullptr || args.activations == nullptr || args.output == nullptr) {
        return false;
    }

    return true;
}

// =============================================================================
// Scalar Fallback Path Functions
// =============================================================================
// Used for non-XMX devices and partial tiles with explicit boundary checking

/**
 * Determine if scalar fallback should be used instead of XMX.
 *
 * Use scalar for:
 * - Very small M (< 8, too small for XMX)
 * - K not aligned to 32 (dpas requirement)
 * - Device without XMX support
 * - Partial tiles at boundaries
 *
 * @param args Kernel arguments
 * @return true if scalar fallback should be used
 */
inline bool should_use_scalar_fallback(const UnifiedKernelArgs & args) {
    // Use scalar for very small M (too small for XMX)
    if (args.M < 8) {
        return true;
    }
    // Use scalar for K not aligned to 32 (dpas requirement)
    if (args.K % 32 != 0) {
        return true;
    }
    // Use scalar when XMX is explicitly disabled
    if (!args.use_xmx) {
        return true;
    }
    return false;
}

/**
 * Check if a specific tile requires scalar fallback due to boundary conditions.
 *
 * @param m_start   Starting M index for this tile
 * @param n_start   Starting N index for this tile
 * @param k_start   Starting K index for this tile
 * @param tile_m    Tile size in M dimension
 * @param tile_n    Tile size in N dimension
 * @param tile_k    Tile size in K dimension
 * @param M         Total M dimension
 * @param N         Total N dimension
 * @param K         Total K dimension
 * @return true if this tile is a partial tile requiring scalar fallback
 */
inline bool is_partial_tile(int64_t m_start, int64_t n_start, int64_t k_start,
                            int tile_m, int tile_n, int tile_k,
                            int64_t M, int64_t N, int64_t K) {
    // Check if tile extends beyond matrix boundaries
    if (m_start + tile_m > M) return true;
    if (n_start + tile_n > N) return true;
    if (k_start + tile_k > K) return true;
    return false;
}

/**
 * Compute scalar tile with explicit boundary checking.
 *
 * This function handles partial tiles at matrix boundaries where
 * dimensions may not align with tile sizes. Each work-item processes
 * one or more output elements using a simple nested loop.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns)
 * @tparam TILE_K   K tile size (reduction)
 * @param activations   Activation matrix (not used - data loaded to SLM)
 * @param slm_weights   Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile (may be < TILE_M)
 * @param N_actual      Actual N elements in this tile (may be < TILE_N)
 * @param K_actual      Actual K elements in this tile (may be < TILE_K)
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension (not used - tile size TILE_K used for indexing)
 * @param N             Full N dimension (for output indexing)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_bounded(
    const float *                         /* activations */,  // Not used - data loaded to SLM
    sycl::local_accessor<float, 1> &      slm_weights,
    sycl::local_accessor<float, 1> &      slm_activations,
    float *                               output,
    int                                   M_actual,
    int                                   N_actual,
    int                                   K_actual,
    int64_t                               m_offset,
    int64_t                               n_offset,
    int64_t                               /* K */,  // Not used - tile size TILE_K used for indexing
    int64_t                               N,
    const sycl::nd_item<2> &              item) {

    const int local_row  = item.get_local_id(0);
    const int local_col  = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Each work-item handles a subset of output elements
    // Iterate over output elements assigned to this thread
    for (int m = local_row; m < M_actual; m += local_size_row) {
        for (int n = local_col; n < N_actual; n += local_size_col) {
            float sum = 0.0f;

            // Dot product over K dimension using SLM data
            // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
            for (int k = 0; k < K_actual; k++) {
                // slm_weights layout: [TILE_N x TILE_K] indexed as [n * TILE_K + k]
                // slm_activations layout: [TILE_M x TILE_K] indexed as [m * TILE_K + k]
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                sum += w * a;
            }

            // Accumulate to output (atomically if needed, but with tile-based
            // partitioning each output element is owned by exactly one work-group)
            output[(m_offset + m) * N + (n_offset + n)] += sum;
        }
    }
}

/**
 * Compute scalar tile with sub-group optimization.
 *
 * Uses sub-group collective operations for efficient horizontal reduction.
 * Each sub-group processes rows cooperatively, with lanes distributing
 * the K-dimension work and reducing results.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns)
 * @tparam TILE_K   K tile size (reduction)
 * @param activations   Activation matrix (not used, data in SLM)
 * @param slm_weights   Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile
 * @param N_actual      Actual N elements in this tile
 * @param K_actual      Actual K elements in this tile
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension (not used)
 * @param N             Full N dimension (for output indexing)
 * @param sg            Sub-group handle
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_subgroup(
    const float *                         /* activations */,  // Not used - data loaded to SLM
    sycl::local_accessor<float, 1> &      slm_weights,
    sycl::local_accessor<float, 1> &      slm_activations,
    float *                               output,
    int                                   M_actual,
    int                                   N_actual,
    int                                   K_actual,
    int64_t                               m_offset,
    int64_t                               n_offset,
    int64_t                               /* K */,  // Not used - tile size TILE_K used for indexing
    int64_t                               N,
    sycl::sub_group                       sg,
    const sycl::nd_item<2> &              item) {

    const int sg_id   = sg.get_local_id()[0];
    const int sg_size = sg.get_local_range()[0];

    // Work-group coordinates
    const int wg_row = item.get_local_id(0);
    const int wg_size_row = item.get_local_range(0);

    // Each row of work-items handles different M values
    // Within a row, the sub-group cooperates on the K-reduction
    for (int m = wg_row; m < M_actual; m += wg_size_row) {
        for (int n = 0; n < N_actual; n++) {
            float partial = 0.0f;

            // Distribute K across sub-group lanes
            // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
            for (int k = sg_id; k < K_actual; k += sg_size) {
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                partial += w * a;
            }

            // Reduce within sub-group using collective operation
            float sum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());

            // Lane 0 writes result
            if (sg_id == 0) {
                output[(m_offset + m) * N + (n_offset + n)] += sum;
            }
        }
    }
}

/**
 * Vectorized scalar compute for aligned tiles.
 *
 * Uses SYCL vec<float, 4> for better memory throughput when
 * dimensions are aligned to vector width.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns, must be multiple of 4 for vectorization)
 * @tparam TILE_K   K tile size (reduction)
 * @param slm_weights      Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations  Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output           Output matrix [M x N]
 * @param M_actual         Actual M elements
 * @param N_actual         Actual N elements (should be multiple of 4)
 * @param K_actual         Actual K elements
 * @param m_offset         Starting M index
 * @param n_offset         Starting N index
 * @param N                Full N dimension (for output indexing)
 * @param item             ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_vectorized(
    sycl::local_accessor<float, 1> &      slm_weights,
    sycl::local_accessor<float, 1> &      slm_activations,
    float *                               output,
    int                                   M_actual,
    int                                   N_actual,
    int                                   K_actual,
    int64_t                               m_offset,
    int64_t                               n_offset,
    int64_t                               N,
    const sycl::nd_item<2> &              item) {

    const int local_row  = item.get_local_id(0);
    const int local_col  = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Process 4 K elements at a time when aligned (vectorize over K for better weight reuse)
    // Note: We iterate over output (m, n) and reduce over k
    for (int m = local_row; m < M_actual; m += local_size_row) {
        for (int n = local_col; n < N_actual; n += local_size_col) {
            float sum = 0.0f;

            // Vectorized K reduction when K_actual is multiple of 4
            const int K_vec = (K_actual / 4) * 4;

            for (int k = 0; k < K_vec; k += 4) {
                // Load 4 weight values for this n
                // GGML: weights[n,k] for consecutive k values
                sycl::vec<float, 4> w;
                w[0] = slm_weights[n * TILE_K + k + 0];
                w[1] = slm_weights[n * TILE_K + k + 1];
                w[2] = slm_weights[n * TILE_K + k + 2];
                w[3] = slm_weights[n * TILE_K + k + 3];

                // Load 4 activation values for this m
                // GGML: activations[m,k] for consecutive k values
                sycl::vec<float, 4> a;
                a[0] = slm_activations[m * TILE_K + k + 0];
                a[1] = slm_activations[m * TILE_K + k + 1];
                a[2] = slm_activations[m * TILE_K + k + 2];
                a[3] = slm_activations[m * TILE_K + k + 3];

                // Dot product contribution
                sum += w[0] * a[0] + w[1] * a[1] + w[2] * a[2] + w[3] * a[3];
            }

            // Scalar cleanup for remaining K elements
            for (int k = K_vec; k < K_actual; k++) {
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                sum += w * a;
            }

            output[(m_offset + m) * N + (n_offset + n)] += sum;
        }
    }
}

// =============================================================================
// Layout-Aware Weight Loading Functions
// =============================================================================
// These functions load Q4_0 quantized weights from global memory into SLM,
// handling different memory layouts (AOS, SOA, COALESCED, XMX_COALESCED).
// All functions dequantize to sycl::half in SLM for XMX compatibility.

// COALESCED layout constants (matches dmmv.cpp)
constexpr int MMVQ_COALESCED_TILE_BLOCKS = 32;  // Blocks per tile
constexpr int MMVQ_COALESCED_TILE_BYTES  = MMVQ_COALESCED_TILE_BLOCKS * (UNIFIED_QK4_0 / 2);  // 512 bytes

// XMX constants for weight loading
constexpr int XMX_K_TILE_LOADING = 32;  // K dimension alignment for dpas

/**
 * Dequantize a Q4_0 block to half precision.
 *
 * @param block  Pointer to Q4_0 block
 * @param output Output array of UNIFIED_QK4_0 half values
 */
SYCL_EXTERNAL inline void dequant_q4_0_to_half(const block_q4_0_unified * block, sycl::half * output) {
    const sycl::half d = block->d;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const uint8_t qs = block->qs[i];
        const int     lo = (qs & 0x0F) - 8;
        const int     hi = (qs >> 4) - 8;

        output[i]      = static_cast<sycl::half>(lo) * d;
        output[i + 16] = static_cast<sycl::half>(hi) * d;
    }
}

/**
 * Load weights from AOS (Array of Structures) layout.
 *
 * AOS layout: Contiguous blocks, each 18 bytes [d: fp16][qs: 16 bytes]
 * Indexed as: blocks[row * blocks_per_row + col]
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be multiple of UNIFIED_QK4_0)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to Q4_0 blocks
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_aos(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               n_start,
                             int64_t                               k_start,
                             int64_t                               N,
                             int64_t                               K,
                             const sycl::nd_item<3> &              item) {
    const block_q4_0_unified * blocks       = static_cast<const block_q4_0_unified *>(weights);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Total elements to load: TILE_N rows x TILE_K weights
    // Each thread handles a subset of blocks
    const int tile_k_blocks = TILE_K / UNIFIED_QK4_0;
    const int total_blocks  = TILE_N * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int n_off     = idx / tile_k_blocks;
        const int k_block   = idx % tile_k_blocks;
        const int64_t n_global = n_start + n_off;
        const int64_t k_block_global = k_start / UNIFIED_QK4_0 + k_block;

        // Bounds check
        if (n_global >= N || k_block_global >= blocks_per_row) {
            // Zero-fill for out-of-bounds
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Load and dequantize block
        // GGML: weights[n_global, k] - row n_global
        const block_q4_0_unified * block = &blocks[n_global * blocks_per_row + k_block_global];
        sycl::half        temp[UNIFIED_QK4_0];
        dequant_q4_0_to_half(block, temp);

        // Store to SLM: [n_off * TILE_K + k]
        for (int i = 0; i < UNIFIED_QK4_0; i++) {
            slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = temp[i];
        }
    }
}

/**
 * Load weights from SOA (Structure of Arrays) layout.
 *
 * SOA layout: All qs bytes contiguous, then all scales contiguous.
 * qs: [nrows * K/2 bytes]
 * d:  [nrows * K/32 half values]
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be multiple of UNIFIED_QK4_0)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (SOA layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor (for scale offset calculation)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_soa(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               n_start,
                             int64_t                               k_start,
                             int64_t                               N,
                             int64_t                               K,
                             int64_t                               nrows_full,
                             const sycl::nd_item<3> &              item) {
    const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes = static_cast<int>(K / 2);
    const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Each thread loads a subset of blocks
    const int tile_k_blocks = TILE_K / UNIFIED_QK4_0;
    const int total_blocks  = TILE_N * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int     n_off       = idx / tile_k_blocks;
        const int     k_block     = idx % tile_k_blocks;
        const int64_t n_global    = n_start + n_off;
        const int64_t k_start_idx = k_start + k_block * UNIFIED_QK4_0;

        // Bounds check
        if (n_global >= N || k_start_idx >= K) {
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Get scale for this block
        // GGML: weights[n_global, k] - row n_global
        const int64_t    block_idx = n_global * blocks_per_row + k_start_idx / UNIFIED_QK4_0;
        const sycl::half d         = d_base[block_idx];

        // Get qs pointer for this block
        const uint8_t * qs = qs_base + n_global * row_qs_bytes + k_start_idx / 2;

        // Dequantize and store to SLM: [n_off * TILE_K + k]
        for (int i = 0; i < 16; i++) {
            const uint8_t qs_byte = qs[i];
            const int     lo      = (qs_byte & 0x0F) - 8;
            const int     hi      = (qs_byte >> 4) - 8;

            slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i]      = static_cast<sycl::half>(lo) * d;
            slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i + 16] = static_cast<sycl::half>(hi) * d;
        }
    }
}

/**
 * Load weights from COALESCED layout.
 *
 * COALESCED layout: Word-major interleaved for efficient sub-group reads.
 * Tiles of MMVQ_COALESCED_TILE_BLOCKS (32) blocks, with 4-byte word interleaving.
 *
 * Layout within a tile:
 * - Word 0: bytes 0-3 from all 32 blocks (128 bytes)
 * - Word 1: bytes 4-7 from all 32 blocks (128 bytes)
 * - ...
 * - Word 3: bytes 12-15 from all 32 blocks (128 bytes)
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (COALESCED layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                   const void *                          weights,
                                   int64_t                               n_start,
                                   int64_t                               k_start,
                                   int64_t                               N,
                                   int64_t                               K,
                                   int64_t                               nrows_full,
                                   const sycl::nd_item<3> &              item) {
    const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes = static_cast<int>(K / 2);
    const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;  // 128 bytes

    // Each thread loads a subset of weights
    const int total_weights = TILE_N * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     n_off   = idx / TILE_K;
        const int     k_off   = idx % TILE_K;
        const int64_t n_global = n_start + n_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (n_global >= N || k_global >= K) {
            slm[idx] = sycl::half(0.0f);
            continue;
        }

        // Compute block and position within block
        const int block_idx      = static_cast<int>(k_global / UNIFIED_QK4_0);
        const int pos_in_block   = static_cast<int>(k_global % UNIFIED_QK4_0);

        // Compute tile and position within tile
        const int tile_idx        = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile   = block_idx % MMVQ_COALESCED_TILE_BLOCKS;

        // Compute byte position within the 16-byte qs region
        const int qs_byte_idx = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
        const int word_idx    = qs_byte_idx / 4;
        const int byte_in_word = qs_byte_idx % 4;

        // Compute coalesced offset
        // GGML: weights[n_global, k] - row n_global
        const int64_t tile_base = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[n_global * blocks_per_row + block_idx];

        // Dequantize and store to SLM: [n_off * TILE_K + k_off]
        const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
        slm[idx] = static_cast<sycl::half>(nibble - 8) * d;
    }
}

/**
 * Load weights from XMX_COALESCED layout.
 *
 * XMX_COALESCED layout: Optimized for dpas with K_TILE=32 alignment.
 * Similar to COALESCED but with additional padding/alignment for XMX.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be 32 for dpas)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (XMX_COALESCED layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_xmx_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                       const void *                          weights,
                                       int64_t                               n_start,
                                       int64_t                               k_start,
                                       int64_t                               N,
                                       int64_t                               K,
                                       int64_t                               nrows_full,
                                       const sycl::nd_item<3> &              item) {
    // XMX_COALESCED uses the same basic structure as COALESCED
    // but is aligned to XMX_K_TILE_LOADING boundaries
    static_assert(TILE_K % XMX_K_TILE_LOADING == 0 || TILE_K < XMX_K_TILE_LOADING,
                  "TILE_K must be multiple of XMX_K_TILE_LOADING for XMX_COALESCED");

    const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes = static_cast<int>(K / 2);
    const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Each thread handles a subset of N x TILE_K weights
    const int total_weights = TILE_N * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     n_off    = idx / TILE_K;
        const int     k_off    = idx % TILE_K;
        const int64_t n_global = n_start + n_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (n_global >= N || k_global >= K) {
            slm[idx] = sycl::half(0.0f);
            continue;
        }

        // Compute block and position
        const int block_idx    = static_cast<int>(k_global / UNIFIED_QK4_0);
        const int pos_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

        // Compute tile and position within tile (XMX tiles are K_TILE aligned)
        const int tile_idx       = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile  = block_idx % MMVQ_COALESCED_TILE_BLOCKS;

        // Compute byte position
        const int qs_byte_idx  = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
        const int word_idx     = qs_byte_idx / 4;
        const int byte_in_word = qs_byte_idx % 4;

        constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

        // Compute coalesced offset
        // GGML: weights[n_global, k] - row n_global
        const int64_t tile_base = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[n_global * blocks_per_row + block_idx];

        // Dequantize and store to SLM: [n_off * TILE_K + k_off]
        const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
        slm[idx] = static_cast<sycl::half>(nibble - 8) * d;
    }
}

/**
 * Layout dispatcher for weight loading (3D nd_item version).
 *
 * Dispatches to the appropriate weight loading function based on LayoutMode.
 * All functions dequantize Q4_0 weights to sycl::half in SLM.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               n_start,
                                int64_t                               k_start,
                                int64_t                               N,
                                int64_t                               K,
                                int64_t                               nrows_full,
                                const sycl::nd_item<3> &              item) {
    switch (layout) {
        case LayoutMode::AOS:
            load_weights_aos<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, item);
            break;

        case LayoutMode::SOA:
            load_weights_soa<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;

        case LayoutMode::COALESCED:
            load_weights_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;

        case LayoutMode::XMX_COALESCED:
            load_weights_xmx_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;
    }
}

/**
 * Layout dispatcher for weight loading (2D nd_item version).
 *
 * Uses flat linear indexing to distribute work across work-items.
 * This overload is used by the existing kernel implementation.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param item2d        2D ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               n_start,
                                int64_t                               k_start,
                                int64_t                               N,
                                int64_t                               K,
                                int64_t                               nrows_full,
                                const sycl::nd_item<2> &              item2d) {
    // Use flat linear indexing within the work-group
    const int local_id   = static_cast<int>(item2d.get_local_linear_id());
    const int local_size = static_cast<int>(item2d.get_local_range().size());

    const block_q4_0_unified * blocks = static_cast<const block_q4_0_unified *>(weights);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    // Handle based on layout
    if (layout == LayoutMode::AOS) {
        // AOS: Direct block access
        const int tile_k_blocks = TILE_K / UNIFIED_QK4_0;
        const int total_blocks  = TILE_N * tile_k_blocks;

        for (int idx = local_id; idx < total_blocks; idx += local_size) {
            const int     n_off       = idx / tile_k_blocks;
            const int     k_block     = idx % tile_k_blocks;
            const int64_t n_global    = n_start + n_off;
            const int64_t k_block_global = k_start / UNIFIED_QK4_0 + k_block;

            if (n_global >= N || k_block_global >= blocks_per_row) {
                for (int i = 0; i < UNIFIED_QK4_0; i++) {
                    slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
                }
                continue;
            }

            // GGML: weights[n_global, k] - row n_global
            const block_q4_0_unified * block = &blocks[n_global * blocks_per_row + k_block_global];
            sycl::half         temp[UNIFIED_QK4_0];
            dequant_q4_0_to_half(block, temp);

            // Store to SLM: [n_off * TILE_K + k]
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = temp[i];
            }
        }
    } else {
        // SOA/COALESCED/XMX_COALESCED: Per-element loading
        const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
        const int          row_qs_bytes = static_cast<int>(K / 2);
        const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);

        const int total_weights = TILE_N * TILE_K;

        for (int idx = local_id; idx < total_weights; idx += local_size) {
            const int     n_off    = idx / TILE_K;
            const int     k_off    = idx % TILE_K;
            const int64_t n_global = n_start + n_off;
            const int64_t k_global = k_start + k_off;

            if (n_global >= N || k_global >= K) {
                slm[idx] = sycl::half(0.0f);
                continue;
            }

            const int block_idx    = static_cast<int>(k_global / UNIFIED_QK4_0);
            const int pos_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

            if (layout == LayoutMode::SOA) {
                // SOA: qs bytes contiguous, then scales
                // GGML: weights[n_global, k] - row n_global
                const uint8_t qs_byte = qs_base[n_global * row_qs_bytes + k_global / 2];
                const sycl::half d = d_base[n_global * blocks_per_row + block_idx];
                const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm[idx] = static_cast<sycl::half>(nibble - 8) * d;
            } else {
                // COALESCED / XMX_COALESCED: Word-interleaved
                // GGML: weights[n_global, k] - row n_global
                const int tile_idx       = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
                const int block_in_tile  = block_idx % MMVQ_COALESCED_TILE_BLOCKS;
                const int qs_byte_idx    = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
                const int word_idx       = qs_byte_idx / 4;
                const int byte_in_word   = qs_byte_idx % 4;
                constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

                const int64_t tile_base = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
                const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

                const uint8_t qs_byte = qs_base[word_offset];
                const sycl::half d = d_base[n_global * blocks_per_row + block_idx];
                const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm[idx] = static_cast<sycl::half>(nibble - 8) * d;
            }
        }
    }
}

// =============================================================================
// XMX Compute Path Functions
// =============================================================================
// Uses Intel joint_matrix API for dpas (Dot Product Accumulate Systolic)
// acceleration on XMX hardware.

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// Namespace alias for brevity
namespace sycl_matrix = sycl::ext::oneapi::experimental::matrix;

/**
 * XMX tile compute using joint_matrix.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 *
 * For XMX, we compute: C[M,N] = B[M,K] @ A[N,K]^T
 * Where:
 * - A (weights) in SLM: [TILE_N x TILE_K] row-major, indexed as [n * TILE_K + k]
 * - B (activations) in SLM: [TILE_M x TILE_K] row-major, indexed as [m * TILE_K + k]
 * - C (output): [TILE_M x TILE_N]
 *
 * For each output tile (m, n), we need weights[n, :] and activations[m, :]
 *
 * @tparam TILE_M   M tile size (must be multiple of XMX_TILE_M=8)
 * @tparam TILE_N   N tile size (must be multiple of XMX_TILE_N=16)
 * @tparam TILE_K   K tile size (must be multiple of XMX_TILE_K=8)
 * @param sg            Sub-group handle
 * @param weights_slm   Pointer to weights in SLM [TILE_N x TILE_K] half, indexed as [n * TILE_K + k]
 * @param act_slm       Pointer to activations in SLM [TILE_M x TILE_K] half, indexed as [m * TILE_K + k]
 * @param c_regs        Pointer to C accumulator in registers [TILE_M x TILE_N] float
 * @param slm_acc       SLM for accumulator extraction
 * @param item          ND-item for work distribution
 */
template<int TILE_M, int TILE_N, int TILE_K>
[[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]]
inline void compute_tile_xmx(
    sycl::sub_group sg,
    const sycl::half* weights_slm,  // Weights in SLM [TILE_N x TILE_K] row-major
    const sycl::half* act_slm,      // Activations in SLM [TILE_M x TILE_K] row-major
    float* c_regs,                  // Accumulator in registers [TILE_M x TILE_N]
    sycl::local_accessor<float, 1>& slm_acc,  // SLM for accumulator extraction
    const sycl::nd_item<2>& /* item */)       // Unused, kept for API consistency
{
    // Number of XMX tiles needed to cover the full tile
    constexpr int NUM_TILES_M = TILE_M / XMX_TILE_M;
    constexpr int NUM_TILES_N = TILE_N / XMX_TILE_N;
    constexpr int NUM_K_STEPS = TILE_K / XMX_TILE_K;

    // Per-tile output size
    constexpr int XMX_OUTPUT_SIZE = XMX_TILE_M * XMX_TILE_N;

    const int lane = sg.get_local_linear_id();

    // Declare joint matrices
    // For C = B @ A^T, we load:
    // - mat_a from activations B[m, k] (row-major)
    // - mat_b from weights A[n, k]^T (col-major view of row-major data)
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half,
                              sycl_matrix::use::a, XMX_TILE_M, XMX_TILE_K,
                              sycl_matrix::layout::row_major> mat_a;  // Activations
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half,
                              sycl_matrix::use::b, XMX_TILE_K, XMX_TILE_N,
                              sycl_matrix::layout::col_major> mat_b;  // Weights transposed
    sycl_matrix::joint_matrix<sycl::sub_group, float,
                              sycl_matrix::use::accumulator, XMX_TILE_M, XMX_TILE_N> acc;

    // Get raw pointer to SLM accumulator region for this sub-group
    float* acc_slm_ptr = const_cast<float*>(&slm_acc[0]);
    auto acc_slm_cast = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(acc_slm_ptr);

    // Process each XMX tile
    for (int tm = 0; tm < NUM_TILES_M; tm++) {
        const int m_base = tm * XMX_TILE_M;

        for (int tn = 0; tn < NUM_TILES_N; tn++) {
            const int n_base = tn * XMX_TILE_N;

            // Initialize accumulator
            sycl_matrix::joint_matrix_fill(sg, acc, 0.0f);

            // K-dimension loop
            for (int tk = 0; tk < NUM_K_STEPS; tk++) {
                const int k_base = tk * XMX_TILE_K;

                // Load activations tile (row-major: activations[m, k])
                // Index: act_slm[m_base * TILE_K + k_base]
                auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(act_slm + m_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_a, a_ptr, TILE_K);

                // Load weights tile (transposed view: weights[n, k] loaded as col-major)
                // Weights are stored row-major as [n * TILE_K + k]
                // For col-major load of transposed matrix, we read from weights[n_base, k_base]
                // The col-major load will read columns of weights (which are rows after transpose)
                auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(weights_slm + n_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_b, b_ptr, TILE_K);

                // Compute: acc += A * B (where B is transposed weights)
                sycl_matrix::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            }

            // Store accumulator to SLM for extraction
            sycl_matrix::joint_matrix_store(sg, acc, acc_slm_cast, XMX_TILE_N,
                                            sycl_matrix::layout::row_major);

            // Sub-group barrier to ensure store is complete
            sycl::group_barrier(sg);

            // Extract and accumulate to output registers
            for (int i = lane; i < XMX_OUTPUT_SIZE; i += XMX_SUBGROUP_SIZE) {
                int row = i / XMX_TILE_N;
                int col = i % XMX_TILE_N;
                int out_idx = (m_base + row) * TILE_N + (n_base + col);
                c_regs[out_idx] += acc_slm_ptr[i];
            }

            sycl::group_barrier(sg);
        }
    }
}

/**
 * Check if XMX unified kernel path is enabled via environment.
 *
 * XMX unified path is DISABLED by default due to 27% performance regression
 * (PP512: 25.73 -> 18.78 t/s). Set GGML_SYCL_XMX_UNIFIED=1 to enable for testing.
 *
 * The XMX path correctness issues have been resolved.
 * TODO: Enable by default once kernel is optimized (see llama.cpp-gkvk).
 *
 * @return true if XMX unified path is enabled
 */
inline bool is_xmx_unified_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_XMX_UNIFIED");
        // Disabled by default (opt-in): XMX path shows 27% regression vs scalar path
        // Enable with GGML_SYCL_XMX_UNIFIED=1 for testing/development only
        // TODO: Enable by default once XMX kernel is optimized (see llama.cpp-gkvk benchmark results)
        enabled = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

/**
 * Check if XMX path can be used for given dimensions.
 *
 * XMX requires:
 * - XMX enabled (GGML_SYCL_XMX_UNIFIED=1, disabled by default)
 * - M >= XMX_TILE_M (8)
 * - N >= XMX_TILE_N (16)
 * - K aligned to XMX_TILE_K (16) for dpas
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if XMX can be used
 */
inline bool can_use_xmx(int64_t M, int64_t N, int64_t K) {
    // XMX path disabled by default, enable with GGML_SYCL_XMX_UNIFIED=1
    if (!is_xmx_unified_enabled()) {
        return false;
    }
    return M >= XMX_TILE_M && N >= XMX_TILE_N && (K % XMX_TILE_K) == 0;
}

#else  // !GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

/**
 * Fallback when joint_matrix is not available.
 *
 * Note: For production use without joint_matrix, consider implementing
 * an ESIMD-based dpas path using sycl::ext::intel::esimd::xmx::dpas.
 */
inline bool can_use_xmx(int64_t /* M */, int64_t /* N */, int64_t /* K */) {
    return false;  // XMX not available
}

#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

}  // namespace ggml_sycl_unified

#endif  // GGML_SYCL_UNIFIED_KERNEL_HPP
