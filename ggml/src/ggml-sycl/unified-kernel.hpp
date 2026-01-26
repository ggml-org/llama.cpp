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

struct UnifiedKernelArgs {
    // Matrix dimensions
    int64_t M;  // Output rows (batch * tokens)
    int64_t N;  // Output columns (hidden dim / output features)
    int64_t K;  // Inner dimension (reduction dim, must be multiple of block size)

    // Tile configuration (from auto-tuning or heuristics)
    int tile_m;  // M dimension tile size
    int tile_n;  // N dimension tile size
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
    const void *  weights;      // Quantized weight matrix [M, K/block_size blocks]
    const float * activations;  // Activation matrix [K, N] (row-major F32)
    float *       output;       // Output matrix [M, N] (row-major F32)
};

// =============================================================================
// Kernel Launch Function
// =============================================================================

/**
 * Launch the unified matmul kernel.
 *
 * Computes: output[M,N] = dequant(weights[M,K]) @ activations[K,N]
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
 * @param tile_m  M tile size
 * @param tile_n  N tile size
 * @param tile_k  K tile size
 * @return Size in bytes needed for SLM
 */
inline size_t calculate_slm_size(int tile_m, int tile_n, int tile_k) {
    // SLM usage:
    // - Weight tile: tile_m * tile_k weights (dequantized to float)
    // - Activation tile: tile_k * tile_n floats
    // For Q4_0 with scalar path, we dequantize to float in SLM
    size_t weight_slm     = static_cast<size_t>(tile_m) * tile_k * sizeof(float);
    size_t activation_slm = static_cast<size_t>(tile_k) * tile_n * sizeof(float);
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
 * @tparam TILE_M   M tile size
 * @tparam TILE_N   N tile size
 * @tparam TILE_K   K tile size
 * @param activations   Activation matrix [M x K] (row-major)
 * @param weights_slm   Dequantized weights in SLM [TILE_M x TILE_K]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile (may be < TILE_M)
 * @param N_actual      Actual N elements in this tile (may be < TILE_N)
 * @param K_actual      Actual K elements in this tile (may be < TILE_K)
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension
 * @param N             Full N dimension
 * @param slm_activations SLM accessor for activations [TILE_K x TILE_N]
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
            for (int k = 0; k < K_actual; k++) {
                // slm_weights layout: [TILE_M * TILE_K] indexed as [m * TILE_K + k]
                // slm_activations layout: [TILE_K * TILE_N] indexed as [k * TILE_N + n]
                float w = slm_weights[m * TILE_K + k];
                float a = slm_activations[k * TILE_N + n];
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
 * @tparam TILE_M   M tile size
 * @tparam TILE_N   N tile size
 * @tparam TILE_K   K tile size
 * @param activations   Activation matrix (not used, data in SLM)
 * @param slm_weights   Dequantized weights in SLM [TILE_M x TILE_K]
 * @param slm_activations SLM accessor for activations [TILE_K x TILE_N]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile
 * @param N_actual      Actual N elements in this tile
 * @param K_actual      Actual K elements in this tile
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension
 * @param N             Full N dimension
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
            for (int k = sg_id; k < K_actual; k += sg_size) {
                float w = slm_weights[m * TILE_K + k];
                float a = slm_activations[k * TILE_N + n];
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
 * @tparam TILE_M   M tile size
 * @tparam TILE_N   N tile size (must be multiple of 4)
 * @tparam TILE_K   K tile size
 * @param slm_weights      Dequantized weights in SLM [TILE_M x TILE_K]
 * @param slm_activations  Activations in SLM [TILE_K x TILE_N]
 * @param output           Output matrix [M x N]
 * @param M_actual         Actual M elements
 * @param N_actual         Actual N elements (should be multiple of 4)
 * @param K_actual         Actual K elements
 * @param m_offset         Starting M index
 * @param n_offset         Starting N index
 * @param N                Full N dimension
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

    // Process 4 N elements at a time when aligned
    const int N_vec = (N_actual / 4) * 4;

    for (int m = local_row; m < M_actual; m += local_size_row) {
        // Vectorized path for aligned portion
        for (int n = local_col * 4; n < N_vec; n += local_size_col * 4) {
            sycl::vec<float, 4> sum(0.0f);

            for (int k = 0; k < K_actual; k++) {
                float w = slm_weights[m * TILE_K + k];

                // Load 4 activation values
                sycl::vec<float, 4> a;
                a[0] = slm_activations[k * TILE_N + n + 0];
                a[1] = slm_activations[k * TILE_N + n + 1];
                a[2] = slm_activations[k * TILE_N + n + 2];
                a[3] = slm_activations[k * TILE_N + n + 3];

                sum += w * a;
            }

            // Write 4 output values
            int64_t out_base = (m_offset + m) * N + (n_offset + n);
            output[out_base + 0] += sum[0];
            output[out_base + 1] += sum[1];
            output[out_base + 2] += sum[2];
            output[out_base + 3] += sum[3];
        }

        // Scalar cleanup for remaining elements
        for (int n = N_vec + local_col; n < N_actual; n += local_size_col) {
            float sum = 0.0f;
            for (int k = 0; k < K_actual; k++) {
                float w = slm_weights[m * TILE_K + k];
                float a = slm_activations[k * TILE_N + n];
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
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size (not used for weights)
 * @tparam TILE_K       K dimension tile size (should be multiple of UNIFIED_QK4_0)
 * @param slm           SLM accessor for dequantized weights [TILE_M * TILE_K]
 * @param weights       Global memory pointer to Q4_0 blocks
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_aos(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               m_start,
                             int64_t                               k_start,
                             int64_t                               M,
                             int64_t                               K,
                             const sycl::nd_item<3> &              item) {
    const block_q4_0_unified * blocks       = static_cast<const block_q4_0_unified *>(weights);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Total elements to load: TILE_M rows x TILE_K weights
    // Each thread handles a subset of blocks
    const int tile_k_blocks = TILE_K / UNIFIED_QK4_0;
    const int total_blocks  = TILE_M * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int m_off     = idx / tile_k_blocks;
        const int k_block   = idx % tile_k_blocks;
        const int64_t m_global = m_start + m_off;
        const int64_t k_block_global = k_start / UNIFIED_QK4_0 + k_block;

        // Bounds check
        if (m_global >= M || k_block_global >= blocks_per_row) {
            // Zero-fill for out-of-bounds
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Load and dequantize block
        const block_q4_0_unified * block = &blocks[m_global * blocks_per_row + k_block_global];
        sycl::half        temp[UNIFIED_QK4_0];
        dequant_q4_0_to_half(block, temp);

        // Store to SLM
        for (int i = 0; i < UNIFIED_QK4_0; i++) {
            slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = temp[i];
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
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size (not used for weights)
 * @tparam TILE_K       K dimension tile size (should be multiple of UNIFIED_QK4_0)
 * @param slm           SLM accessor for dequantized weights [TILE_M * TILE_K]
 * @param weights       Global memory pointer (SOA layout)
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor (for scale offset calculation)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_soa(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               m_start,
                             int64_t                               k_start,
                             int64_t                               M,
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
    const int total_blocks  = TILE_M * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int     m_off       = idx / tile_k_blocks;
        const int     k_block     = idx % tile_k_blocks;
        const int64_t m_global    = m_start + m_off;
        const int64_t k_start_idx = k_start + k_block * UNIFIED_QK4_0;

        // Bounds check
        if (m_global >= M || k_start_idx >= K) {
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Get scale for this block
        const int64_t    block_idx = m_global * blocks_per_row + k_start_idx / UNIFIED_QK4_0;
        const sycl::half d         = d_base[block_idx];

        // Get qs pointer for this block
        const uint8_t * qs = qs_base + m_global * row_qs_bytes + k_start_idx / 2;

        // Dequantize and store
        for (int i = 0; i < 16; i++) {
            const uint8_t qs_byte = qs[i];
            const int     lo      = (qs_byte & 0x0F) - 8;
            const int     hi      = (qs_byte >> 4) - 8;

            slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i]      = static_cast<sycl::half>(lo) * d;
            slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i + 16] = static_cast<sycl::half>(hi) * d;
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
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size (not used)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights
 * @param weights       Global memory pointer (COALESCED layout)
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                   const void *                          weights,
                                   int64_t                               m_start,
                                   int64_t                               k_start,
                                   int64_t                               M,
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
    const int total_weights = TILE_M * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     m_off   = idx / TILE_K;
        const int     k_off   = idx % TILE_K;
        const int64_t m_global = m_start + m_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (m_global >= M || k_global >= K) {
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
        const int64_t tile_base = m_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[m_global * blocks_per_row + block_idx];

        // Dequantize
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
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size (not used)
 * @tparam TILE_K       K dimension tile size (should be 32 for dpas)
 * @param slm           SLM accessor for dequantized weights
 * @param weights       Global memory pointer (XMX_COALESCED layout)
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_xmx_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                       const void *                          weights,
                                       int64_t                               m_start,
                                       int64_t                               k_start,
                                       int64_t                               M,
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

    // Each thread handles a subset of M x TILE_K weights
    const int total_weights = TILE_M * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     m_off    = idx / TILE_K;
        const int     k_off    = idx % TILE_K;
        const int64_t m_global = m_start + m_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (m_global >= M || k_global >= K) {
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
        const int64_t tile_base = m_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[m_global * blocks_per_row + block_idx];

        // Dequantize
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
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               m_start,
                                int64_t                               k_start,
                                int64_t                               M,
                                int64_t                               K,
                                int64_t                               nrows_full,
                                const sycl::nd_item<3> &              item) {
    switch (layout) {
        case LayoutMode::AOS:
            load_weights_aos<TILE_M, TILE_N, TILE_K>(slm, weights, m_start, k_start, M, K, item);
            break;

        case LayoutMode::SOA:
            load_weights_soa<TILE_M, TILE_N, TILE_K>(slm, weights, m_start, k_start, M, K, nrows_full, item);
            break;

        case LayoutMode::COALESCED:
            load_weights_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, m_start, k_start, M, K, nrows_full, item);
            break;

        case LayoutMode::XMX_COALESCED:
            load_weights_xmx_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, m_start, k_start, M, K, nrows_full, item);
            break;
    }
}

/**
 * Layout dispatcher for weight loading (2D nd_item version).
 *
 * Uses flat linear indexing to distribute work across work-items.
 * This overload is used by the existing kernel implementation.
 *
 * @tparam TILE_M       M dimension tile size
 * @tparam TILE_N       N dimension tile size
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param m_start       Starting M (row) index
 * @param k_start       Starting K index
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param item2d        2D ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               m_start,
                                int64_t                               k_start,
                                int64_t                               M,
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
        const int total_blocks  = TILE_M * tile_k_blocks;

        for (int idx = local_id; idx < total_blocks; idx += local_size) {
            const int     m_off       = idx / tile_k_blocks;
            const int     k_block     = idx % tile_k_blocks;
            const int64_t m_global    = m_start + m_off;
            const int64_t k_block_global = k_start / UNIFIED_QK4_0 + k_block;

            if (m_global >= M || k_block_global >= blocks_per_row) {
                for (int i = 0; i < UNIFIED_QK4_0; i++) {
                    slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
                }
                continue;
            }

            const block_q4_0_unified * block = &blocks[m_global * blocks_per_row + k_block_global];
            sycl::half         temp[UNIFIED_QK4_0];
            dequant_q4_0_to_half(block, temp);

            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[m_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = temp[i];
            }
        }
    } else {
        // SOA/COALESCED/XMX_COALESCED: Per-element loading
        const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
        const int          row_qs_bytes = static_cast<int>(K / 2);
        const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);

        const int total_weights = TILE_M * TILE_K;

        for (int idx = local_id; idx < total_weights; idx += local_size) {
            const int     m_off    = idx / TILE_K;
            const int     k_off    = idx % TILE_K;
            const int64_t m_global = m_start + m_off;
            const int64_t k_global = k_start + k_off;

            if (m_global >= M || k_global >= K) {
                slm[idx] = sycl::half(0.0f);
                continue;
            }

            const int block_idx    = static_cast<int>(k_global / UNIFIED_QK4_0);
            const int pos_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

            if (layout == LayoutMode::SOA) {
                // SOA: qs bytes contiguous, then scales
                const uint8_t qs_byte = qs_base[m_global * row_qs_bytes + k_global / 2];
                const sycl::half d = d_base[m_global * blocks_per_row + block_idx];
                const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm[idx] = static_cast<sycl::half>(nibble - 8) * d;
            } else {
                // COALESCED / XMX_COALESCED: Word-interleaved
                const int tile_idx       = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
                const int block_in_tile  = block_idx % MMVQ_COALESCED_TILE_BLOCKS;
                const int qs_byte_idx    = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
                const int word_idx       = qs_byte_idx / 4;
                const int byte_in_word   = qs_byte_idx % 4;
                constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

                const int64_t tile_base = m_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
                const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

                const uint8_t qs_byte = qs_base[word_offset];
                const sycl::half d = d_base[m_global * blocks_per_row + block_idx];
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
 * Computes: C[M,N] += A[M,K] @ B[K,N] using dpas instructions.
 *
 * Data layout expectations:
 * - A (weights): row-major in SLM, dequantized to half [TILE_M x TILE_K]
 * - B (activations): col-major in SLM, converted to half [TILE_K x TILE_N]
 * - C (accumulator): float registers [TILE_M x TILE_N]
 *
 * @tparam TILE_M   M tile size (must be multiple of XMX_TILE_M=8)
 * @tparam TILE_N   N tile size (must be multiple of XMX_TILE_N=16)
 * @tparam TILE_K   K tile size (must be multiple of XMX_TILE_K=8)
 * @param sg            Sub-group handle
 * @param a_slm         Pointer to A matrix in SLM [TILE_M x TILE_K] half
 * @param b_slm         Pointer to B matrix in SLM [TILE_K x TILE_N] half
 * @param c_regs        Pointer to C accumulator in registers [TILE_M x TILE_N] float
 * @param slm_acc       SLM for accumulator extraction
 * @param item          ND-item for work distribution
 */
template<int TILE_M, int TILE_N, int TILE_K>
[[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]]
inline void compute_tile_xmx(
    sycl::sub_group sg,
    const sycl::half* a_slm,    // Weights in SLM [TILE_M x TILE_K] row-major
    const sycl::half* b_slm,    // Activations in SLM [TILE_K x TILE_N] col-major
    float* c_regs,              // Accumulator in registers [TILE_M x TILE_N]
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
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half,
                              sycl_matrix::use::a, XMX_TILE_M, XMX_TILE_K,
                              sycl_matrix::layout::row_major> mat_a;
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half,
                              sycl_matrix::use::b, XMX_TILE_K, XMX_TILE_N,
                              sycl_matrix::layout::col_major> mat_b;
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

                // Load A tile from SLM (row-major: [m_base + row, k_base + col])
                auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(a_slm + m_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_a, a_ptr, TILE_K);

                // Load B tile from SLM (col-major for dpas: [k_base + row, n_base + col])
                // Note: B is stored in column-major for optimal dpas access pattern
                auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(b_slm + n_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_b, b_ptr, TILE_K);

                // Compute: acc += A * B
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
 * Set GGML_SYCL_XMX_UNIFIED=1 to enable the experimental XMX path.
 * Default is disabled until implementation is fully validated.
 *
 * @return true if XMX unified path is enabled
 */
inline bool is_xmx_unified_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_XMX_UNIFIED");
        enabled = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

/**
 * Check if XMX path can be used for given dimensions.
 *
 * XMX requires:
 * - Environment gate enabled (GGML_SYCL_XMX_UNIFIED=1)
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
    // XMX path is gated by environment variable until fully validated
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
