//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMQ_ESIMD_HPP
#define GGML_SYCL_MMQ_ESIMD_HPP

#include "common.hpp"
#include "vecdotq.hpp"

// Check for ESIMD availability
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#define SYCL_ESIMD_MMQ_AVAILABLE 1
#include <sycl/ext/intel/esimd.hpp>
#include <vector>  // For trace mode CPU-side debugging
namespace esimd = sycl::ext::intel::esimd;

// ============================================================================
// MMQ ESIMD Kernel for Intel Arc (Xe2/Battlemage)
//
// Key optimizations:
// 1. Unified block loading - eliminates L3 thrashing from two-phase tile load
// 2. K-partitioned reduction - 32 partitions with tree-based merge
// 3. ESIMD vectorized nibble extraction for Q4_0
// 4. Double-buffered prefetching to hide memory latency
//
// Target metrics (VTune):
// - L3 miss rate: <10% (from 20.6%)
// - XVE stalls: <45% (from 60%)
// - pp2048 throughput: +15-25%
// ============================================================================

// Kernel configuration - V1 (legacy, single work-item per output)
constexpr int MMQ_ESIMD_PARTITIONS = 1;   // Single work-item (for debugging)
constexpr int MMQ_ESIMD_TILE_M = 64;      // Output rows per work-group
constexpr int MMQ_ESIMD_TILE_N = 32;      // Output cols per work-group

// Kernel configuration - V2 (optimized, tiled with SLM)
// Following reference MMQ pattern from mmq.cpp
// ADAPTED FOR WARP_SIZE=32 on Intel Arc Battlemage
// X=32 is optimal - larger tiles increase SLM pressure and hurt performance
constexpr int MMQ_ESIMD_Y = 128;          // Rows per tile (matches reference)
constexpr int MMQ_ESIMD_X = 32;           // Cols per tile (optimal for small batches)
constexpr int MMQ_ESIMD_NWARPS = 8;       // Warps per work-group
constexpr int MMQ_ESIMD_WARP_SIZE_V2 = WARP_SIZE;  // 32 on Intel Arc Battlemage
constexpr int MMQ_ESIMD_BLOCK_SIZE = MMQ_ESIMD_NWARPS * MMQ_ESIMD_WARP_SIZE_V2;  // 8*32=256 threads

// Number of K blocks processed per iteration
// With WARP_SIZE=32: 32/4 = 8 blocks (QI4_0 = QK4_0/(4*QR4_0) = 32/8 = 4)
// With WARP_SIZE=16: 16/4 = 4 blocks
constexpr int MMQ_ESIMD_BLOCKS_PER_ITER = MMQ_ESIMD_WARP_SIZE_V2 / QI4_0;

// Tile stride for Y in K dimension
// Must match X loading rate: blocks_per_iter blocks * QI8_1 ints per block
// With WARP_SIZE=32: 8 * 8 = 64 ints
// With WARP_SIZE=16: 4 * 8 = 32 ints
constexpr int MMQ_ESIMD_TILE_K = MMQ_ESIMD_BLOCKS_PER_ITER * QI8_1;

// Number of Y-loading phases needed per K iteration
// With WARP_SIZE=32, TILE_K=64: need 2 phases (32 threads * 2 = 64 values)
// With WARP_SIZE=16, TILE_K=32: need 2 phases (16 threads * 2 = 32 values)
constexpr int MMQ_ESIMD_Y_PHASES = (MMQ_ESIMD_TILE_K + MMQ_ESIMD_WARP_SIZE_V2 - 1) / MMQ_ESIMD_WARP_SIZE_V2;

// SLM tile sizes for Q4_0 (following mmq.cpp pattern)
// tile_x_qs: store quantized int values, padded for bank conflicts
// With WARP_SIZE=32: 128 * 33 = 4224 ints = 16.5 KB
constexpr int SLM_X_QS_SIZE = MMQ_ESIMD_Y * (MMQ_ESIMD_WARP_SIZE_V2 + 1);
// tile_x_d: store scales as floats, padded for bank conflicts
// With WARP_SIZE=32: 128 * 9 = 1152 floats = 4.5 KB
constexpr int SLM_X_D_SIZE = MMQ_ESIMD_Y * (MMQ_ESIMD_BLOCKS_PER_ITER + 1);
// tile_y_qs: store Y quantized values
// With WARP_SIZE=32, X=32: 32 * 64 = 2048 ints = 8 KB
// With WARP_SIZE=16, X=64: 64 * 32 = 2048 ints = 8 KB
constexpr int SLM_Y_QS_SIZE = MMQ_ESIMD_X * MMQ_ESIMD_TILE_K;
// tile_y_ds: store Y scale+sum pairs
// With WARP_SIZE=32, X=32: 32 * 8 = 256 half2s = 1 KB
// With WARP_SIZE=16, X=64: 64 * 4 = 256 half2s = 1 KB
constexpr int SLM_Y_DS_SIZE = MMQ_ESIMD_X * (MMQ_ESIMD_TILE_K / QI8_1);

// Kernel name classes for VTune profiling (use distinct struct types)
class mmq_esimd_q4_0_kernel;           // V1 legacy kernel
class mmq_esimd_q4_0_v2_kernel;        // V2 optimized kernel (no bounds check)
class mmq_esimd_q4_0_v2_check_kernel;  // V2 optimized kernel (with bounds check)
class mmq_esimd_q4_0_v3_kernel;        // V3 ESIMD-native kernel with K-partitioning

// ============================================================================
// V3 Configuration - Standard SYCL kernel with different tile sizes for comparison
// Uses same pattern as V2 but with smaller tiles for more work-groups
// ============================================================================
constexpr int MMQ_V3_TILE_M = 64;         // Rows per work-group
constexpr int MMQ_V3_TILE_N = 32;         // Cols per work-group
constexpr int MMQ_V3_NWARPS = 8;          // Warps per work-group
constexpr int MMQ_V3_THREADS = 256;       // Threads per work-group (8 warps × 32)

// ============================================================================
// ESIMD Nibble Extraction for Q4_0
// Input: 16 bytes = 32 nibbles packed (low nibble first, then high nibble)
// Output: 32 int8 values in [-8, 7]
// ============================================================================

// Returns RAW Q4_0 values in [0, 15] - NOT centered
// The centering offset (-8) is applied in the dot product formula
SYCL_ESIMD_FUNCTION esimd::simd<int8_t, 32> dequant_q4_0_nibbles(esimd::simd<uint8_t, 16> packed)
{
    esimd::simd<int8_t, 32> result;

    // Q4_0 layout: byte[i] contains quant[i] in bits[0:3], quant[i+16] in bits[4:7]
    // Extract low and high nibbles
    esimd::simd<uint8_t, 16> lo = packed & 0x0F;    // Low nibbles = quant[0..15]
    esimd::simd<uint8_t, 16> hi = packed >> 4;       // High nibbles = quant[16..31]

    // Sequential layout: [quant[0..15], quant[16..31]]
    // Keep raw values [0,15] - centering is done in the formula
    result.template select<16, 1>(0) = esimd::simd<int8_t, 16>(lo);   // indices 0-15
    result.template select<16, 1>(16) = esimd::simd<int8_t, 16>(hi);  // indices 16-31

    return result;
}

// ============================================================================
// ESIMD dp4a equivalent - 4-way int8 dot product
// Takes two packed int32 values (each containing 4 int8 values) and accumulates
// ============================================================================

SYCL_ESIMD_FUNCTION int32_t esimd_dp4a(int32_t a, int32_t b, int32_t c)
{
    // Extract 4 signed bytes from each int32
    // a and b each contain 4 packed int8 values in little-endian order
    int8_t a0 = static_cast<int8_t>(a & 0xFF);
    int8_t a1 = static_cast<int8_t>((a >> 8) & 0xFF);
    int8_t a2 = static_cast<int8_t>((a >> 16) & 0xFF);
    int8_t a3 = static_cast<int8_t>((a >> 24) & 0xFF);

    int8_t b0 = static_cast<int8_t>(b & 0xFF);
    int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFF);
    int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFF);
    int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFF);

    // Compute dot product and accumulate
    return c + (int32_t)a0 * (int32_t)b0
             + (int32_t)a1 * (int32_t)b1
             + (int32_t)a2 * (int32_t)b2
             + (int32_t)a3 * (int32_t)b3;
}

// ESIMD-safe int32 load from byte array (avoids alignment issues)
SYCL_ESIMD_FUNCTION int32_t load_int32_from_bytes(const uint8_t* ptr)
{
    return static_cast<int32_t>(ptr[0])
         | (static_cast<int32_t>(ptr[1]) << 8)
         | (static_cast<int32_t>(ptr[2]) << 16)
         | (static_cast<int32_t>(ptr[3]) << 24);
}

// ESIMD-safe int32 load from signed byte array
SYCL_ESIMD_FUNCTION int32_t load_int32_from_int8(const int8_t* ptr)
{
    // Pack as unsigned then reinterpret - maintain sign bits properly
    return static_cast<int32_t>(static_cast<uint8_t>(ptr[0]))
         | (static_cast<int32_t>(static_cast<uint8_t>(ptr[1])) << 8)
         | (static_cast<int32_t>(static_cast<uint8_t>(ptr[2])) << 16)
         | (static_cast<int32_t>(static_cast<uint8_t>(ptr[3])) << 24);
}

// ESIMD-safe float load from half
SYCL_ESIMD_FUNCTION float load_half_as_float(const ggml_half* ptr)
{
    ggml_half h = *ptr;
    return static_cast<float>(h);
}

// ============================================================================
// SLM Tree Reduction
// Reduce partial sums from K partitions using binary tree reduction
// ============================================================================

SYCL_ESIMD_FUNCTION float reduce_k_partitions(
    float partial_sum,
    int partition_id,
    size_t slm_offset)
{
    // Store partial sum to SLM
    esimd::slm_scalar_store<float>(slm_offset + partition_id * sizeof(float), partial_sum);
    esimd::barrier();

    // Tree reduction: log2(32) = 5 rounds
    #pragma unroll
    for (int stride = MMQ_ESIMD_PARTITIONS / 2; stride > 0; stride /= 2) {
        if (partition_id < stride) {
            float my_val = esimd::slm_scalar_load<float>(slm_offset + partition_id * sizeof(float));
            float other_val = esimd::slm_scalar_load<float>(slm_offset + (partition_id + stride) * sizeof(float));
            esimd::slm_scalar_store<float>(slm_offset + partition_id * sizeof(float), my_val + other_val);
        }
        esimd::barrier();
    }

    // Return final reduced value (only partition 0 has correct result)
    return esimd::slm_scalar_load<float>(slm_offset);
}

// ============================================================================
// MMQ ESIMD Kernel for Q4_0
//
// Grid: (nrows, ncols, 1)
// Block: MMQ_ESIMD_PARTITIONS work-items
//
// Each work-group handles one (row, col) output element
// Each work-item processes 1/PARTITIONS of the K dimension
// ============================================================================

// Debug buffer for comparing ESIMD vs reference computations
// Set GGML_SYCL_MMQ_TRACE=1 to enable tracing to /tmp/mmq_esimd_trace.txt
struct mmq_debug_info {
    int row;
    int col;
    int blk;
    float x_d;
    float y_d;
    float y_s;
    int32_t sumi;
    float block_result;
    float partial_sum;
};

// Returns true if ESIMD kernel was launched, false if caller should use fallback
inline bool launch_mmq_q4_0_esimd(
    const block_q4_0 * __restrict__ x,     // Quantized weights [nrows, k/32]
    const block_q8_1 * __restrict__ y,     // Quantized activations [ncols, k/32]
    float * __restrict__ dst,              // Output [nrows, ncols]
    const int64_t nrows,                   // Number of output rows
    const int64_t ncols,                   // Number of output columns
    const int64_t k,                       // Inner dimension (must be multiple of 32)
    const int64_t nrows_dst,               // Stride for destination
    sycl::queue & stream)
{
    const int64_t blocks_per_row = k / QK4_0;  // QK4_0 = 32

    // Debug: print kernel launch info
    static int launch_count = 0;
    bool debug_mode = std::getenv("GGML_SYCL_MMQ_DEBUG") != nullptr;

    launch_count++;

    if (debug_mode) {
        fprintf(stderr, "[MMQ-ESIMD #%d] nrows=%ld ncols=%ld k=%ld blocks_per_row=%ld\n",
                launch_count, (long)nrows, (long)ncols, (long)k, (long)blocks_per_row);
    }

    // Safety: fall back to standard MMQ for edge cases
    // Note: Intel Arc can handle large grids (20 Xe cores × 256 threads = 5120 HW threads)
    // Raising limit to 500000 to cover FFN layers (14336 × 16 = 229376)
    const int64_t grid_size = nrows * ncols;
    if (blocks_per_row < 4 || grid_size > 500000) {
        if (debug_mode) {
            fprintf(stderr, "[MMQ-ESIMD] Skipping - blocks_per_row=%ld, grid_size=%ld\n",
                    (long)blocks_per_row, (long)grid_size);
        }
        return false;  // Signal caller to use fallback
    }

    // Grid: one work-group per output element
    // Block: single work-item (simple version, will optimize later)
    sycl::range<2> grid(ncols, nrows);
    sycl::range<2> block(1, 1);

    // Use a simpler kernel structure that processes blocks using ESIMD vectors
    // Each work-item computes one output element by processing all K blocks
    stream.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmq_esimd_q4_0_kernel>(
            sycl::nd_range<2>(grid, block),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                using namespace esimd;

                const int row = item.get_global_id(1);
                const int col = item.get_global_id(0);

                if (row >= nrows || col >= ncols) {
                    return;
                }

                // Base pointers for this row/col
                const block_q4_0* x_row = x + row * blocks_per_row;
                const block_q8_1* y_col = y + col * blocks_per_row;

                float sum = 0.0f;

                // Process 4 blocks at a time for better vectorization
                // Each block: 32 quants, so 4 blocks = 128 quants
                const int blocks_per_iter = 4;
                const int full_iters = blocks_per_row / blocks_per_iter;
                const int remainder = blocks_per_row % blocks_per_iter;

                for (int iter = 0; iter < full_iters; iter++) {
                    const int blk_base = iter * blocks_per_iter;

                    // Process 4 blocks
                    #pragma unroll
                    for (int b = 0; b < blocks_per_iter; b++) {
                        const block_q4_0* x_blk = x_row + blk_base + b;
                        const block_q8_1* y_blk = y_col + blk_base + b;

                        // Load scales using ESIMD-safe helpers
                        float x_d = load_half_as_float(&x_blk->d);
                        const ggml_half* y_ds_ptr = reinterpret_cast<const ggml_half*>(&y_blk->ds);
                        float y_d = load_half_as_float(y_ds_ptr);
                        float y_s = load_half_as_float(y_ds_ptr + 1);

                        // Compute dot product using the same pattern as vec_dot_q4_0_q8_1_impl
                        // Load x_qs as 4 ints, y_qs as 8 ints, and use dp4a
                        int32_t sumi = 0;

                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            // Use ESIMD-safe byte-by-byte loading
                            int32_t v = load_int32_from_bytes(&x_blk->qs[i * 4]);
                            int32_t vi0 = v & 0x0F0F0F0F;         // Low nibbles (x indices i*4 to i*4+3)
                            int32_t vi1 = (v >> 4) & 0x0F0F0F0F;  // High nibbles (x indices i*4+16 to i*4+19)

                            // Y indexing must match X nibble layout:
                            // vi0 pairs with y[i*4 : i*4+3], vi1 pairs with y[i*4+16 : i*4+19]
                            int32_t y0 = load_int32_from_int8(&y_blk->qs[i * 4]);
                            int32_t y1 = load_int32_from_int8(&y_blk->qs[i * 4 + 16]);

                            // dp4a: dot product of 4 int8 values
                            sumi = esimd_dp4a(vi0, y0, sumi);
                            sumi = esimd_dp4a(vi1, y1, sumi);
                        }

                        // Apply Q4_0 @ Q8_1 formula
                        // result = d4 * (sumi * d8 - 8 * s8)
                        sum += x_d * (static_cast<float>(sumi) * y_d - 8.0f * y_s);
                    }
                }

                // Handle remainder blocks
                for (int b = 0; b < remainder; b++) {
                    const int blk = full_iters * blocks_per_iter + b;
                    const block_q4_0* x_blk = x_row + blk;
                    const block_q8_1* y_blk = y_col + blk;

                    float x_d = load_half_as_float(&x_blk->d);
                    const ggml_half* y_ds_ptr = reinterpret_cast<const ggml_half*>(&y_blk->ds);
                    float y_d = load_half_as_float(y_ds_ptr);
                    float y_s = load_half_as_float(y_ds_ptr + 1);

                    int32_t sumi = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int32_t v = load_int32_from_bytes(&x_blk->qs[i * 4]);
                        int32_t vi0 = v & 0x0F0F0F0F;         // Low nibbles (x indices i*4 to i*4+3)
                        int32_t vi1 = (v >> 4) & 0x0F0F0F0F;  // High nibbles (x indices i*4+16 to i*4+19)
                        // Y indexing must match X nibble layout
                        int32_t y0 = load_int32_from_int8(&y_blk->qs[i * 4]);
                        int32_t y1 = load_int32_from_int8(&y_blk->qs[i * 4 + 16]);
                        sumi = esimd_dp4a(vi0, y0, sumi);
                        sumi = esimd_dp4a(vi1, y1, sumi);
                    }

                    sum += x_d * (static_cast<float>(sumi) * y_d - 8.0f * y_s);
                }

                // Write result - column-major order to match reference MMQ kernel
                // dst[col*nrows_dst + row] is the correct indexing for MMQ output
                dst[col * nrows_dst + row] = sum;
            });
    });

    return true;  // Kernel was launched
}

// ============================================================================
// V2 Optimized Kernel - Tiled with SLM caching
// Following the reference MMQ kernel pattern from mmq.cpp
// ============================================================================

// V2 vec_dot function - computes Q4_0 @ Q8_1 dot product from SLM tiles
// ADAPTED FOR WARP_SIZE=32 (TILE_K=64) and WARP_SIZE=16 (TILE_K=32)
template <int vdr>
static __dpct_inline__ float vec_dot_q4_0_q8_1_v2(
    const int* __restrict__ tile_x_qs,
    const float* __restrict__ tile_x_d,
    const int* __restrict__ tile_y_qs,
    const sycl::half2* __restrict__ tile_y_ds,
    const int i, const int j, const int k)
{
    // SLM strides - must match loading code
    constexpr int y_tile_stride = MMQ_ESIMD_TILE_K;  // 64 for WARP=32, 32 for WARP=16
    constexpr int x_qs_stride = MMQ_ESIMD_WARP_SIZE_V2 + 1;  // 33 for WARP=32
    constexpr int x_d_stride = MMQ_ESIMD_BLOCKS_PER_ITER + 1;  // 9 for WARP=32
    constexpr int y_ds_stride = MMQ_ESIMD_TILE_K / QI8_1;  // 8 for WARP=32, 4 for WARP=16

    // Get y indices for dot product (following reference mmq.cpp pattern)
    // k is in range [0, WARP_SIZE/QR4_0) for the inner k loop
    // Q4_0 layout: low nibbles pair with y[0..15], high nibbles pair with y[16..31]
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2 * vdr];
    #pragma unroll
    for (int l = 0; l < vdr; ++l) {
        // Access Y qs values - wrap within tile
        u[2*l+0] = tile_y_qs[j * y_tile_stride + (kyqs + l) % y_tile_stride];
        u[2*l+1] = tile_y_qs[j * y_tile_stride + (kyqs + l + QI4_0) % y_tile_stride];
    }

    // Get x values from SLM
    const int* x_ql = &tile_x_qs[i * x_qs_stride + k];

    // Compute dot product using native dp4a
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < vdr; ++l) {
        const int vi0 = (x_ql[l] >> 0) & 0x0F0F0F0F;
        const int vi1 = (x_ql[l] >> 4) & 0x0F0F0F0F;

        sumi = dpct::dp4a(vi0, u[2*l+0], sumi);
        sumi = dpct::dp4a(vi1, u[2*l+1], sumi);
    }

    // Get scales - X scale per QI4_0 values, Y ds per QI8_1 values
    const float x_d = tile_x_d[i * x_d_stride + k / QI4_0];
    const sycl::half2 y_ds = tile_y_ds[j * y_ds_stride + (2*k/QI8_1) % y_ds_stride];

    const sycl::float2 ds8f = y_ds.convert<float, sycl::rounding_mode::automatic>();

    // Q4_0 @ Q8_1 formula: d4 * (sumi * d8 - 8 * s8)
    return x_d * (static_cast<float>(sumi) * ds8f.x() - (8.0f * vdr / QI4_0) * ds8f.y());
}

// V2 kernel launcher - optimized with tiling and SLM
template <bool need_check>
inline bool launch_mmq_q4_0_esimd_v2_impl(
    const block_q4_0 * __restrict__ x,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int64_t nrows,
    const int64_t ncols,
    const int64_t k,
    const int64_t nrows_dst,
    sycl::queue & stream)
{
    const int blocks_per_row_x = k / QK4_0;
    const int blocks_per_col_y = k / QK8_1;

    // Calculate grid dimensions
    const int grid_y = (nrows + MMQ_ESIMD_Y - 1) / MMQ_ESIMD_Y;
    const int grid_x = (ncols + MMQ_ESIMD_X - 1) / MMQ_ESIMD_X;

    // Grid: (grid_x * NWARPS, grid_y * WARP_SIZE) - flattened for nd_range
    // Each work-group is (NWARPS, WARP_SIZE) = (8, 16) = 128 threads
    sycl::range<2> global(grid_x * MMQ_ESIMD_NWARPS, grid_y * MMQ_ESIMD_WARP_SIZE_V2);
    sycl::range<2> local(MMQ_ESIMD_NWARPS, MMQ_ESIMD_WARP_SIZE_V2);

    stream.submit([&](sycl::handler& cgh) {
        // SLM allocations
        sycl::local_accessor<int, 1> tile_x_qs(sycl::range<1>(SLM_X_QS_SIZE), cgh);
        sycl::local_accessor<float, 1> tile_x_d(sycl::range<1>(SLM_X_D_SIZE), cgh);
        sycl::local_accessor<int, 1> tile_y_qs(sycl::range<1>(SLM_Y_QS_SIZE), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds(sycl::range<1>(SLM_Y_DS_SIZE), cgh);

        using kernel_class = std::conditional_t<need_check, mmq_esimd_q4_0_v2_check_kernel, mmq_esimd_q4_0_v2_kernel>;

        cgh.parallel_for<kernel_class>(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                // Thread identification
                const int warp_id = item.get_local_id(0);   // 0..7 (which warp in work-group)
                const int lane_id = item.get_local_id(1);   // 0..15 (which thread in warp)

                // Work-group identification
                const int group_x = item.get_group(0);      // Which column tile
                const int group_y = item.get_group(1);      // Which row tile

                // Output tile position
                const int row_dst_0 = group_y * MMQ_ESIMD_Y;
                const int col_dst_0 = group_x * MMQ_ESIMD_X;

                // Get SLM pointers using get_multi_ptr().get() (modern SYCL 2020)
                int* slm_x_qs = tile_x_qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* slm_x_d = tile_x_d.template get_multi_ptr<sycl::access::decorated::no>().get();
                int* slm_y_qs = tile_y_qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                sycl::half2* slm_y_ds = tile_y_ds.template get_multi_ptr<sycl::access::decorated::no>().get();

                // Accumulators - each thread computes (MMQ_Y/WARP_SIZE) x (MMQ_X/NWARPS) outputs
                // WARP_SIZE=32: 128/32 = 4 rows, X/8 cols per thread
                constexpr int acc_rows = MMQ_ESIMD_Y / MMQ_ESIMD_WARP_SIZE_V2;
                constexpr int acc_cols = MMQ_ESIMD_X / MMQ_ESIMD_NWARPS;  // X/8
                float sum[acc_rows][acc_cols] = {{0.0f}};

                // Blocks per iteration (how many K blocks fit in one SLM tile load)
                // WARP_SIZE=32: 32/4 = 8 blocks
                // WARP_SIZE=16: 16/4 = 4 blocks
                constexpr int blocks_per_iter = MMQ_ESIMD_WARP_SIZE_V2 / QI4_0;

                // X scale SLM stride (with +1 padding for bank conflicts)
                constexpr int x_d_slm_stride = blocks_per_iter + 1;

                // Process K dimension in blocks
                for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_iter) {

                    // === COOPERATIVE X-TILE LOADING ===
                    // Each thread loads different elements into SLM
                    // With WARP_SIZE=32, QI4_0=4: 8 blocks per iteration, 4 ints per block
                    const int kbx = lane_id / QI4_0;      // Which block (0..7 for WARP=32)
                    const int kqsx = lane_id % QI4_0;     // Which int within block (0..3)

                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_ESIMD_Y; i0 += MMQ_ESIMD_NWARPS) {
                        int i = i0 + warp_id;
                        if (need_check) {
                            i = sycl::min(i, static_cast<int>(nrows - row_dst_0 - 1));
                        }

                        const block_q4_0* bxi = x + (row_dst_0 + i) * blocks_per_row_x + ib0 + kbx;
                        slm_x_qs[i * (MMQ_ESIMD_WARP_SIZE_V2 + 1) + lane_id] =
                            get_int_from_uint8(bxi->qs, kqsx);
                    }

                    // Load X scales into SLM
                    // Each thread loads one scale from a different block
                    // With WARP_SIZE=32: 32 threads, blocks_per_iter=8
                    // So 32/8=4 threads share each block index kbxd
                    const int kbxd = lane_id % blocks_per_iter;
                    #pragma unroll
                    for (int i0 = 0; i0 < MMQ_ESIMD_Y; i0 += MMQ_ESIMD_NWARPS * (MMQ_ESIMD_WARP_SIZE_V2 / blocks_per_iter)) {
                        int i = i0 + warp_id * (MMQ_ESIMD_WARP_SIZE_V2 / blocks_per_iter) + lane_id / blocks_per_iter;
                        if (need_check) {
                            i = sycl::min(i, static_cast<int>(nrows - row_dst_0 - 1));
                        }

                        const block_q4_0* bxi = x + (row_dst_0 + i) * blocks_per_row_x + ib0 + kbxd;
                        // Use consistent stride = blocks_per_iter + 1 (matches vec_dot)
                        slm_x_d[i * x_d_slm_stride + kbxd] = static_cast<float>(bxi->d);
                    }

                    // === COOPERATIVE Y-TILE LOADING ===
                    // Y tile stores TILE_K ints per column (64 for WARP=32, 32 for WARP=16)
                    constexpr int y_tile_stride = MMQ_ESIMD_TILE_K;
                    constexpr int y_ds_stride = y_tile_stride / QI8_1;
                    // Number of phases needed: ceil(TILE_K / WARP_SIZE)
                    // WARP=32, TILE_K=64: 2 phases
                    // WARP=16, TILE_K=32: 2 phases
                    constexpr int y_phases = MMQ_ESIMD_Y_PHASES;

                    #pragma unroll
                    for (int ir = 0; ir < y_phases; ++ir) {
                        // Each phase loads WARP_SIZE values
                        const int kqs = ir * MMQ_ESIMD_WARP_SIZE_V2 + lane_id;

                        // Only load if within TILE_K bounds
                        if (kqs < y_tile_stride) {
                            const int kbxd_y = kqs / QI8_1;

                            #pragma unroll
                            for (int j0 = 0; j0 < MMQ_ESIMD_X; j0 += MMQ_ESIMD_NWARPS) {
                                const int col_y_eff = need_check ?
                                    sycl::min(col_dst_0 + warp_id + j0, static_cast<int>(ncols - 1)) :
                                    col_dst_0 + warp_id + j0;

                                // Y block index: ib0 + kbxd_y (blocks_per_iter blocks loaded)
                                const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + ib0 + kbxd_y];

                                const int index_y = (warp_id + j0) * y_tile_stride + kqs;
                                slm_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, lane_id % QI8_1);
                            }
                        }

                        // Load Y scales and sums
                        // Each thread loads one half2 (d, s) from a different block
                        #pragma unroll
                        for (int ids0 = 0; ids0 < MMQ_ESIMD_X; ids0 += MMQ_ESIMD_NWARPS * (MMQ_ESIMD_WARP_SIZE_V2 / y_ds_stride)) {
                            const int ids = (ids0 + warp_id * (MMQ_ESIMD_WARP_SIZE_V2 / y_ds_stride) + lane_id / y_ds_stride) % MMQ_ESIMD_X;
                            const int kby = lane_id % y_ds_stride;

                            if (ir * (MMQ_ESIMD_WARP_SIZE_V2 / QI8_1) + kby < y_ds_stride) {
                                const int col_y_eff = need_check ?
                                    sycl::min(col_dst_0 + ids, static_cast<int>(ncols - 1)) :
                                    col_dst_0 + ids;

                                const int y_block_idx = ir * (MMQ_ESIMD_WARP_SIZE_V2 / QI8_1) + kby;
                                const sycl::half2* dsi_src = &y[col_y_eff * blocks_per_col_y + ib0 + y_block_idx].ds;
                                slm_y_ds[ids * y_ds_stride + y_block_idx] = *dsi_src;
                            }
                        }
                    }

                    // === BARRIER - ensure all loads complete ===
                    item.barrier(sycl::access::fence_space::local_space);

                    // === COMPUTE - dot products from SLM ===
                    // Process all WARP_SIZE ints loaded in X SLM
                    // VDR_Q4_0_Q8_1_MMQ = 4 ints per vec_dot call
                    // Total iterations: WARP_SIZE / VDR = 32/4 = 8 for WARP=32
                    #pragma unroll
                    for (int k_inner = 0; k_inner < MMQ_ESIMD_WARP_SIZE_V2; k_inner += VDR_Q4_0_Q8_1_MMQ) {
                        #pragma unroll
                        for (int j = 0; j < MMQ_ESIMD_X; j += MMQ_ESIMD_NWARPS) {
                            #pragma unroll
                            for (int i = 0; i < MMQ_ESIMD_Y; i += MMQ_ESIMD_WARP_SIZE_V2) {
                                sum[i / MMQ_ESIMD_WARP_SIZE_V2][j / MMQ_ESIMD_NWARPS] +=
                                    vec_dot_q4_0_q8_1_v2<VDR_Q4_0_Q8_1_MMQ>(
                                        slm_x_qs, slm_x_d, slm_y_qs, slm_y_ds,
                                        lane_id + i, warp_id + j, k_inner);
                            }
                        }
                    }

                    // === BARRIER - before next iteration ===
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // === WRITE OUTPUT - column-major order ===
                #pragma unroll
                for (int j = 0; j < MMQ_ESIMD_X; j += MMQ_ESIMD_NWARPS) {
                    const int col_dst = col_dst_0 + j + warp_id;
                    if (col_dst >= ncols) continue;

                    #pragma unroll
                    for (int i = 0; i < MMQ_ESIMD_Y; i += MMQ_ESIMD_WARP_SIZE_V2) {
                        const int row_dst = row_dst_0 + lane_id + i;
                        if (row_dst >= nrows_dst) continue;

                        dst[col_dst * nrows_dst + row_dst] = sum[i / MMQ_ESIMD_WARP_SIZE_V2][j / MMQ_ESIMD_NWARPS];
                    }
                }
            });
    });

    return true;
}

// V2 kernel entry point
inline bool launch_mmq_q4_0_esimd_v2(
    const block_q4_0 * __restrict__ x,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int64_t nrows,
    const int64_t ncols,
    const int64_t k,
    const int64_t nrows_dst,
    sycl::queue & stream)
{
    const int64_t blocks_per_row = k / QK4_0;

    // Debug output
    static int launch_count = 0;
    bool debug_mode = std::getenv("GGML_SYCL_MMQ_DEBUG") != nullptr;

    launch_count++;
    if (debug_mode) {
        if (launch_count == 1) {
            // Print config once
            fprintf(stderr, "[MMQ-ESIMD-V2] WARP_SIZE=%d MMQ_ESIMD_WARP_SIZE_V2=%d NWARPS=%d BLOCK_SIZE=%d\n",
                    WARP_SIZE, MMQ_ESIMD_WARP_SIZE_V2, MMQ_ESIMD_NWARPS, MMQ_ESIMD_BLOCK_SIZE);
            fprintf(stderr, "[MMQ-ESIMD-V2] Y=%d X=%d TILE_K=%d blocks_per_iter=%d\n",
                    MMQ_ESIMD_Y, MMQ_ESIMD_X, MMQ_ESIMD_TILE_K, MMQ_ESIMD_WARP_SIZE_V2 / QI4_0);
        }
        fprintf(stderr, "[MMQ-ESIMD-V2 #%d] nrows=%ld ncols=%ld k=%ld blocks_per_row=%ld\n",
                launch_count, (long)nrows, (long)ncols, (long)k, (long)blocks_per_row);
    }

    // Fall back to standard MMQ only for very small K dimension
    // With MMQ_ESIMD_X=32 matching MMQ_MAX_BATCH_SIZE=32, ncols will always be valid
    if (blocks_per_row < 2) {
        if (debug_mode) {
            fprintf(stderr, "[MMQ-ESIMD-V2] Falling back - blocks_per_row=%ld too small\n",
                    (long)blocks_per_row);
        }
        return false;
    }

    // Choose kernel based on whether bounds checking is needed
    const bool need_check = (nrows % MMQ_ESIMD_Y != 0) || (ncols % MMQ_ESIMD_X != 0);

    if (need_check) {
        return launch_mmq_q4_0_esimd_v2_impl<true>(x, y, dst, nrows, ncols, k, nrows_dst, stream);
    } else {
        return launch_mmq_q4_0_esimd_v2_impl<false>(x, y, dst, nrows, ncols, k, nrows_dst, stream);
    }
}

// ============================================================================
// V3 Kernel - ESIMD-native with better memory access and compute
// Key optimizations:
// 1. Uses only ESIMD-compatible operations (no sycl::vec in ESIMD context)
// 2. Processes multiple output elements per thread
// 3. Coalesced memory access patterns
// 4. Reduced barrier overhead with smaller tiles
// ============================================================================

// ESIMD-compatible dp4a for V3 kernel
SYCL_ESIMD_FUNCTION int32_t v3_dp4a(int32_t a, int32_t b, int32_t c)
{
    // Extract 4 signed bytes from each int32
    int8_t a0 = static_cast<int8_t>(a & 0xFF);
    int8_t a1 = static_cast<int8_t>((a >> 8) & 0xFF);
    int8_t a2 = static_cast<int8_t>((a >> 16) & 0xFF);
    int8_t a3 = static_cast<int8_t>((a >> 24) & 0xFF);

    int8_t b0 = static_cast<int8_t>(b & 0xFF);
    int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFF);
    int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFF);
    int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFF);

    return c + (int32_t)a0 * (int32_t)b0
             + (int32_t)a1 * (int32_t)b1
             + (int32_t)a2 * (int32_t)b2
             + (int32_t)a3 * (int32_t)b3;
}

inline bool launch_mmq_q4_0_esimd_v3(
    const block_q4_0 * __restrict__ x,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int64_t nrows,
    const int64_t ncols,
    const int64_t k,
    const int64_t nrows_dst,
    sycl::queue & stream)
{
    using namespace esimd;

    const int blocks_per_row = k / QK4_0;

    // Debug output
    static int launch_count = 0;
    bool debug_mode = std::getenv("GGML_SYCL_MMQ_DEBUG") != nullptr;

    launch_count++;
    if (debug_mode && launch_count == 1) {
        fprintf(stderr, "[MMQ-ESIMD-V3] TILE_M=%d TILE_N=%d THREADS=%d\n",
                MMQ_V3_TILE_M, MMQ_V3_TILE_N, MMQ_V3_THREADS);
    }

    // Grid dimensions
    const int grid_m = (nrows + MMQ_V3_TILE_M - 1) / MMQ_V3_TILE_M;
    const int grid_n = (ncols + MMQ_V3_TILE_N - 1) / MMQ_V3_TILE_N;

    // Each work-group handles TILE_M x TILE_N = 32 x 32 outputs
    // 64 threads = 2 warps x 32 threads (ESIMD limit)
    // Each thread computes (TILE_M/32) x (TILE_N/2) = 1 x 16 = 16 outputs
    sycl::range<2> global(grid_n * 2, grid_m * 32);
    sycl::range<2> local(2, 32);

    // SLM sizes for V3 - store scales as separate d and s arrays instead of half2
    constexpr int V3_SLM_X_SIZE = MMQ_V3_TILE_M * 33;     // 32 * 33 = 1056 ints = 4.1 KB
    constexpr int V3_SLM_X_D_SIZE = MMQ_V3_TILE_M * 9;    // 32 * 9 = 288 floats = 1.1 KB
    constexpr int V3_SLM_Y_SIZE = MMQ_V3_TILE_N * 64;     // 32 * 64 = 2048 ints = 8 KB
    constexpr int V3_SLM_Y_D_SIZE = MMQ_V3_TILE_N * 8;    // 32 * 8 = 256 floats = 1 KB (d values)
    constexpr int V3_SLM_Y_S_SIZE = MMQ_V3_TILE_N * 8;    // 32 * 8 = 256 floats = 1 KB (s values)

    stream.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> slm_x_qs(sycl::range<1>(V3_SLM_X_SIZE), cgh);
        sycl::local_accessor<float, 1> slm_x_d(sycl::range<1>(V3_SLM_X_D_SIZE), cgh);
        sycl::local_accessor<int, 1> slm_y_qs(sycl::range<1>(V3_SLM_Y_SIZE), cgh);
        sycl::local_accessor<float, 1> slm_y_d(sycl::range<1>(V3_SLM_Y_D_SIZE), cgh);
        sycl::local_accessor<float, 1> slm_y_s(sycl::range<1>(V3_SLM_Y_S_SIZE), cgh);

        cgh.parallel_for<mmq_esimd_q4_0_v3_kernel>(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                const int warp_id = item.get_local_id(0);   // 0..1 (2 warps)
                const int lane_id = item.get_local_id(1);   // 0..31
                const int tid = warp_id * 32 + lane_id;     // 0..63

                const int group_n = item.get_group(0);
                const int group_m = item.get_group(1);

                const int row_base = group_m * MMQ_V3_TILE_M;
                const int col_base = group_n * MMQ_V3_TILE_N;

                // Get SLM pointers
                int* tile_x_qs = slm_x_qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* tile_x_d = slm_x_d.template get_multi_ptr<sycl::access::decorated::no>().get();
                int* tile_y_qs = slm_y_qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* tile_y_d = slm_y_d.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* tile_y_s = slm_y_s.template get_multi_ptr<sycl::access::decorated::no>().get();

                // Each thread computes 1 row x 16 cols = 16 outputs
                // With 64 threads (2 warps × 32): 32 threads per warp
                // Each warp handles 16 cols (warp 0: cols 0-15, warp 1: cols 16-31)
                constexpr int ROWS_PER_THREAD = 1;   // 32/32 = 1
                constexpr int COLS_PER_THREAD = 16;  // 32/2 = 16

                // Use scalar accumulators
                float acc[COLS_PER_THREAD] = {0.0f};

                // Process K dimension in chunks of 8 blocks (256 quants)
                constexpr int K_BLOCKS_PER_ITER = 8;

                for (int kb = 0; kb < blocks_per_row; kb += K_BLOCKS_PER_ITER) {
                    // === LOAD X TILE ===
                    // 64 threads load 32 rows × 8 blocks × 4 ints = 1024 values
                    // Each thread loads 16 values
                    #pragma unroll
                    for (int load_idx = tid; load_idx < MMQ_V3_TILE_M * 32; load_idx += 64) {
                        const int row = load_idx / 32;
                        const int k_idx = load_idx % 32;
                        const int blk = k_idx / 4;
                        const int qs_idx = k_idx % 4;

                        const int global_row = row_base + row;
                        if (global_row < nrows && kb + blk < blocks_per_row) {
                            const block_q4_0* bx = x + global_row * blocks_per_row + kb + blk;
                            tile_x_qs[row * 33 + k_idx] = get_int_from_uint8(bx->qs, qs_idx);
                        }
                    }

                    // Load X scales
                    #pragma unroll
                    for (int load_idx = tid; load_idx < MMQ_V3_TILE_M * 8; load_idx += 64) {
                        const int row = load_idx / 8;
                        const int blk = load_idx % 8;

                        const int global_row = row_base + row;
                        if (global_row < nrows && kb + blk < blocks_per_row) {
                            const block_q4_0* bx = x + global_row * blocks_per_row + kb + blk;
                            tile_x_d[row * 9 + blk] = static_cast<float>(bx->d);
                        }
                    }

                    // === LOAD Y TILE ===
                    // 32 cols × 64 ints = 2048 values
                    #pragma unroll
                    for (int load_idx = tid; load_idx < MMQ_V3_TILE_N * 64; load_idx += 64) {
                        const int col = load_idx / 64;
                        const int k_idx = load_idx % 64;
                        const int blk = k_idx / 8;
                        const int qs_idx = k_idx % 8;

                        const int global_col = col_base + col;
                        if (global_col < ncols && kb + blk < blocks_per_row) {
                            const block_q8_1* by = y + global_col * blocks_per_row + kb + blk;
                            tile_y_qs[col * 64 + k_idx] = get_int_from_int8_aligned(by->qs, qs_idx);
                        }
                    }

                    // Load Y scales - extract d and s from half2
                    #pragma unroll
                    for (int load_idx = tid; load_idx < MMQ_V3_TILE_N * 8; load_idx += 64) {
                        const int col = load_idx / 8;
                        const int blk = load_idx % 8;

                        const int global_col = col_base + col;
                        if (global_col < ncols && kb + blk < blocks_per_row) {
                            const block_q8_1* by = y + global_col * blocks_per_row + kb + blk;
                            // Load half2 as raw uint32 and extract halves
                            const uint32_t ds_raw = *reinterpret_cast<const uint32_t*>(&by->ds);
                            const sycl::half d_half = *reinterpret_cast<const sycl::half*>(&ds_raw);
                            const sycl::half s_half = *reinterpret_cast<const sycl::half*>(reinterpret_cast<const char*>(&ds_raw) + 2);
                            tile_y_d[col * 8 + blk] = static_cast<float>(d_half);
                            tile_y_s[col * 8 + blk] = static_cast<float>(s_half);
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    // === COMPUTE ===
                    // Each thread computes 1 row × 16 cols
                    // lane_id selects the row (0..31), warp_id selects col range (0-15 or 16-31)
                    const int row = lane_id;
                    const int col_start = warp_id * COLS_PER_THREAD;

                    if (row < MMQ_V3_TILE_M) {
                        #pragma unroll
                        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                            const int col = col_start + ci;
                            float partial = 0.0f;

                            // Process 8 K blocks
                            #pragma unroll
                            for (int blk = 0; blk < K_BLOCKS_PER_ITER; blk++) {
                                if (kb + blk >= blocks_per_row) break;

                                const float x_d_val = tile_x_d[row * 9 + blk];
                                const float y_d_val = tile_y_d[col * 8 + blk];
                                const float y_s_val = tile_y_s[col * 8 + blk];

                                int32_t sumi = 0;
                                #pragma unroll
                                for (int i = 0; i < 4; i++) {
                                    const int x_qs = tile_x_qs[row * 33 + blk * 4 + i];
                                    const int vi0 = x_qs & 0x0F0F0F0F;
                                    const int vi1 = (x_qs >> 4) & 0x0F0F0F0F;

                                    const int y0 = tile_y_qs[col * 64 + blk * 8 + i];
                                    const int y1 = tile_y_qs[col * 64 + blk * 8 + i + 4];

                                    sumi = v3_dp4a(vi0, y0, sumi);
                                    sumi = v3_dp4a(vi1, y1, sumi);
                                }

                                partial += x_d_val * (static_cast<float>(sumi) * y_d_val - 8.0f * y_s_val);
                            }

                            acc[ci] += partial;
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                // === WRITE OUTPUT ===
                const int row = lane_id;
                const int global_row = row_base + row;
                const int col_start = warp_id * COLS_PER_THREAD;

                if (global_row < nrows) {
                    #pragma unroll
                    for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                        const int col = col_base + col_start + ci;
                        if (col < ncols) {
                            dst[col * nrows_dst + global_row] = acc[ci];
                        }
                    }
                }
            });
    });

    return true;
}

// ============================================================================
// Dispatch function - called from mmq.cpp
// ============================================================================

inline int mmq_esimd_version() {
    static int version = []() {
        const char* env = std::getenv("GGML_SYCL_MMQ_ESIMD");
        if (env == nullptr) return 0;
        return std::atoi(env);
    }();
    return version;
}

inline bool mmq_esimd_enabled() {
    return mmq_esimd_version() > 0;
}

inline bool mmq_esimd_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        // TODO: Add Q8_0, Q4_K, etc.
        default:
            return false;
    }
}

inline bool mmq_esimd_available() {
    return true;
}

#else // !SYCL_ESIMD_MMQ_AVAILABLE

// Stub implementations when ESIMD is not available

inline bool mmq_esimd_enabled() {
    return false;
}

inline bool mmq_esimd_supported(ggml_type type) {
    GGML_UNUSED(type);
    return false;
}

inline bool mmq_esimd_available() {
    return false;
}

inline bool launch_mmq_q4_0_esimd(
    const block_q4_0 * __restrict__ x,
    const block_q8_1 * __restrict__ y,
    float * __restrict__ dst,
    const int64_t nrows,
    const int64_t ncols,
    const int64_t k,
    const int64_t nrows_dst,
    sycl::queue & stream)
{
    GGML_UNUSED(x);
    GGML_UNUSED(y);
    GGML_UNUSED(dst);
    GGML_UNUSED(nrows);
    GGML_UNUSED(ncols);
    GGML_UNUSED(k);
    GGML_UNUSED(nrows_dst);
    GGML_UNUSED(stream);
    return false;  // ESIMD not available, use fallback
}

#endif // SYCL_ESIMD_MMQ_AVAILABLE

#endif // GGML_SYCL_MMQ_ESIMD_HPP
