//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// XMX Kernel Configuration Types
//
// Template-based configuration for XMX-optimized matrix multiplication kernels.
// Provides compile-time configuration for:
//   - Tile dimensions (M, N, K)
//   - Tiling factors (TILES_M, TILES_N)
//   - Buffer strategy (single vs double-buffered)
//   - Reduction strategy (warp reduce vs column-fused)
//
// Pre-defined configurations match the 7 existing Q4_0 kernel variants:
//   - BasicConfig: Single-tile 8x16x32
//   - DoubleBufConfig: Single-tile with double buffering
//   - MultiTileConfig: 4 vertical tiles (32 rows)
//   - LargeTileConfig: 8x4 tiles (64x64 output)
//   - MultiTileDbConfig: Multi-tile with double buffering
//   - ColFusedConfig: Column-fused reduction
//   - ColFused4TileConfig: 4-tile column-fused
//

#ifndef GGML_SYCL_XMX_KERNEL_CONFIG_HPP
#define GGML_SYCL_XMX_KERNEL_CONFIG_HPP

namespace ggml_sycl_xmx {

// =============================================================================
// Enum Definitions
// =============================================================================

/**
 * @brief Buffer strategy for K-dimension iteration
 */
enum class BufferStrategy {
    SINGLE,         ///< Single buffer - load, barrier, compute, barrier
    DOUBLE_BUFFER   ///< Double buffer - overlap load and compute
};

/**
 * @brief Reduction strategy for accumulating partial results
 */
enum class ReductionStrategy {
    WARP_REDUCE,    ///< Standard warp-level reduction
    COLUMN_FUSED    ///< Column-fused: single sub-group processes multiple column tiles
};

// =============================================================================
// XMXKernelConfig Template
// =============================================================================

/**
 * @brief Compile-time configuration for XMX matrix multiplication kernels
 *
 * Template parameters:
 *   - TileM: XMX tile M dimension (typically 8 for Intel Arc)
 *   - TileN: XMX tile N dimension (typically 16 for Intel Arc)
 *   - TileK: XMX tile K dimension (typically 32 for INT8)
 *   - TilesM: Number of M tiles per work-group (default 1)
 *   - TilesN: Number of N tiles per work-group (default 1)
 *   - Buffer: Buffering strategy (default SINGLE)
 *   - Reduce: Reduction strategy (default WARP_REDUCE)
 *
 * Derived constants:
 *   - OUTPUT_M: Total output rows per work-group = TileM * TilesM
 *   - OUTPUT_N: Total output columns per work-group = TileN * TilesN
 */
template <
    int TileM,
    int TileN,
    int TileK,
    int TilesM = 1,
    int TilesN = 1,
    BufferStrategy Buffer = BufferStrategy::SINGLE,
    ReductionStrategy Reduce = ReductionStrategy::WARP_REDUCE
>
struct XMXKernelConfig {
    // Core tile dimensions (XMX hardware constants)
    static constexpr int TILE_M = TileM;
    static constexpr int TILE_N = TileN;
    static constexpr int TILE_K = TileK;

    // Tiling factors (work-group configuration)
    static constexpr int TILES_M = TilesM;
    static constexpr int TILES_N = TilesN;

    // Strategy flags
    static constexpr BufferStrategy BUFFER = Buffer;
    static constexpr ReductionStrategy REDUCE = Reduce;

    // Derived output dimensions
    static constexpr int OUTPUT_M = TileM * TilesM;
    static constexpr int OUTPUT_N = TileN * TilesN;

    // Sub-group configuration (Intel XMX uses 16-wide sub-groups)
    static constexpr int SG_SIZE = 16;

    // Work-group thread count based on tiling
    static constexpr int WG_THREADS = TilesM * TilesN * SG_SIZE;

    // SLM size calculations
    static constexpr int SLM_A_ELEMENTS = TILE_M * TILE_K * TilesM;
    static constexpr int SLM_B_ELEMENTS = TILE_K * TILE_N * TilesN;
    static constexpr int SLM_SCALES_A = TILE_M * TilesM;
    static constexpr int SLM_SCALES_B = TILE_N * TilesN;

    // Double-buffer multiplier
    static constexpr int BUFFER_MULT = (Buffer == BufferStrategy::DOUBLE_BUFFER) ? 2 : 1;
};

// =============================================================================
// Pre-defined Configurations
// =============================================================================

/**
 * @brief Basic single-tile configuration
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_kernel
 * - One sub-group per work-group
 * - 8x16 output tile (128 elements)
 * - Simple load-compute-store pattern
 */
using BasicConfig = XMXKernelConfig<8, 16, 32, 1, 1,
                                     BufferStrategy::SINGLE,
                                     ReductionStrategy::WARP_REDUCE>;

/**
 * @brief Double-buffered single-tile configuration
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_doublebuf_kernel
 * - Overlaps memory loads with computation
 * - Uses ping-pong buffers
 * - Better latency hiding
 */
using DoubleBufConfig = XMXKernelConfig<8, 16, 32, 1, 1,
                                         BufferStrategy::DOUBLE_BUFFER,
                                         ReductionStrategy::WARP_REDUCE>;

/**
 * @brief Multi-tile configuration (4 vertical tiles)
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_multitile_kernel
 * - 4 sub-groups per work-group
 * - 32x16 output (4 row tiles x 1 col tile)
 * - A tile loaded once, shared across sub-groups
 */
using MultiTileConfig = XMXKernelConfig<8, 16, 32, 4, 1,
                                         BufferStrategy::SINGLE,
                                         ReductionStrategy::WARP_REDUCE>;

/**
 * @brief Large-tile configuration (8x4 tiles)
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_largetile_kernel
 * - 32 sub-groups per work-group (8 row tiles x 4 col tiles)
 * - 64x64 output (4096 elements)
 * - Amortizes work-group launch overhead
 */
using LargeTileConfig = XMXKernelConfig<8, 16, 32, 8, 4,
                                         BufferStrategy::SINGLE,
                                         ReductionStrategy::WARP_REDUCE>;

/**
 * @brief Multi-tile with double buffering
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_multitile_db_kernel
 * - 4 sub-groups with ping-pong buffers
 * - Combines multi-tile with latency hiding
 */
using MultiTileDbConfig = XMXKernelConfig<8, 16, 32, 4, 1,
                                           BufferStrategy::DOUBLE_BUFFER,
                                           ReductionStrategy::WARP_REDUCE>;

/**
 * @brief Column-fused configuration
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_colfused_kernel
 * - 2 sub-groups processing 2 column tiles in parallel
 * - Shared A tile between sub-groups
 * - Better B tile locality
 */
using ColFusedConfig = XMXKernelConfig<8, 16, 32, 1, 2,
                                        BufferStrategy::SINGLE,
                                        ReductionStrategy::COLUMN_FUSED>;

/**
 * @brief 4-tile column-fused sequential configuration
 *
 * Maps to: mul_mat_q4_0_q8_1_xmx_colfused_4tile_kernel
 * - Single sub-group processes 4 column tiles sequentially
 * - Maximum A tile reuse
 * - Good for medium batch sizes
 */
using ColFused4TileConfig = XMXKernelConfig<8, 16, 32, 1, 4,
                                             BufferStrategy::SINGLE,
                                             ReductionStrategy::COLUMN_FUSED>;

}  // namespace ggml_sycl_xmx

#endif  // GGML_SYCL_XMX_KERNEL_CONFIG_HPP
