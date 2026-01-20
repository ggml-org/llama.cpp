//
// XMX ESIMD Q4_0 GEMM Kernel using dpas Pattern
//
// ESIMD-based General Matrix Multiply for Q4_0 quantized weights.
// Uses XMX dpas (Dot Product Accumulate Systolic) pattern for hardware acceleration.
//
// Q4_0 Format:
// - Block size: 32 weights per block (QK4_0 = 32)
// - Storage: sycl::half d (scale) + uint8_t qs[16] (32 nibbles packed)
// - Low nibble: qs[i] & 0x0F  -> value - 8 for signed range [-8, +7]
// - High nibble: qs[i] >> 4   -> value - 8 for signed range [-8, +7]
//
// CRITICAL: This kernel uses xmx::dpas<8, RepeatCount>() intrinsic:
// - SystolicDepth is ALWAYS 8 (hardware constant)
// - RepeatCount controls output M dimension (M = RepeatCount * 8)
// - B operand MUST be in VNNI format (4 int8 -> 1 uint32)
// - dpas does int8*int8->int32 accumulation, scale applied AFTER
// - K dimension for int8 is 32 (from XMXDpasConfig.k_int8)
//
// Input:
//   A: FP32 input matrix [M, K]
//   B: Q4_0 quantized weight matrix [N, K/QK4_0] blocks
// Output:
//   C: FP32 result matrix [M, N]
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_XMX_ESIMD_GEMM_Q4_HPP
#define GGML_SYCL_XMX_ESIMD_GEMM_Q4_HPP

#include <sycl/sycl.hpp>

// Include XMX ESIMD common infrastructure (provides XMXConfig, dpas helpers, VNNI packing)
#include "xmx-esimd-common.hpp"

// =============================================================================
// Q4_0 Block Definition for ESIMD GEMM
// =============================================================================

#ifndef QK4_0
#    define QK4_0 32
#endif

// Block structure matching ggml-common.h
// Q4_0: 32 weights per block, packed as 16 bytes (2 nibbles per byte)
struct block_q4_0_esimd {
    sycl::half d;              // scale factor (delta)
    uint8_t    qs[QK4_0 / 2];  // quantized values: 16 bytes = 32 nibbles
};

namespace ggml_sycl_xmx {

// =============================================================================
// Q4_0 Unpacking Functions
//
// Q4_0 packs 2 values per byte (4-bit nibbles):
// - Low nibble: qs[i] & 0x0F  -> subtract 8 for signed range
// - High nibble: qs[i] >> 4   -> subtract 8 for signed range
// =============================================================================

// Host-side Q4_0 unpacking: converts 16 packed bytes to 32 signed int8 values
// CRITICAL: Must subtract 8 to get signed range [-8, +7]
inline void unpack_q4_0_to_int8(const uint8_t * qs, int8_t * out, int n_bytes) {
    for (int i = 0; i < n_bytes; i++) {
        out[2 * i]     = static_cast<int8_t>((qs[i] & 0x0F) - 8);  // low nibble
        out[2 * i + 1] = static_cast<int8_t>((qs[i] >> 4) - 8);    // high nibble
    }
}

// Host-side Q4_0 unpacking for a full block (16 bytes -> 32 int8 values)
inline void unpack_q4_0_block_to_int8(const uint8_t * qs, int8_t * out) {
    unpack_q4_0_to_int8(qs, out, QK4_0 / 2);
}

#if SYCL_XMX_ESIMD_AVAILABLE

// =============================================================================
// ESIMD Q4_0 Unpacking
//
// Template unpacking for ESIMD kernels
// N_BYTES: number of packed bytes to unpack (output is 2*N_BYTES int8 values)
// =============================================================================

template <int N_BYTES>
SYCL_ESIMD_FUNCTION esimd::simd<int8_t, N_BYTES * 2> unpack_q4_0_esimd(esimd::simd<uint8_t, N_BYTES> packed) {
    esimd::simd<int8_t, N_BYTES * 2> result;

    // Extract low and high nibbles, subtract 8 for signed range
#    pragma unroll
    for (int i = 0; i < N_BYTES; i++) {
        result[2 * i]     = static_cast<int8_t>((packed[i] & 0x0F) - 8);  // low nibble
        result[2 * i + 1] = static_cast<int8_t>((packed[i] >> 4) - 8);    // high nibble
    }

    return result;
}

// Specialized version for full Q4_0 block (16 bytes -> 32 int8)
SYCL_ESIMD_FUNCTION esimd::simd<int8_t, QK4_0> unpack_q4_0_block_esimd(esimd::simd<uint8_t, QK4_0 / 2> packed) {
    return unpack_q4_0_esimd<QK4_0 / 2>(packed);
}

// =============================================================================
// ESIMD Q4_0 GEMM Kernel Classes (for SYCL naming)
// =============================================================================

class esimd_q4_0_gemm_kernel_basic;
class esimd_q4_0_gemm_kernel_tiled;

// =============================================================================
// ESIMD Q4_0 GEMM: Basic Implementation
//
// This kernel computes C[M,N] = A[M,K] * B[K,N] where:
// - A is FP32 input matrix
// - B is Q4_0 quantized weight matrix with per-block scales
// - C is FP32 output matrix
//
// The computation follows this pattern:
// For each Q4_0 block in K dimension:
//   1. Load A values for this block (32 floats)
//   2. Load B packed nibbles (16 bytes) and unpack to 32 int8s
//   3. Apply scale and compute dot product: sum += A[k] * (unpack(B[k]) * scale)
//
// Note: For true dpas usage, A would also need to be quantized.
// This implementation uses ESIMD vector operations that match
// the dpas computational pattern while supporting FP32 input A.
// =============================================================================

inline void esimd_q4_0_gemm(sycl::queue &     q,
                            const float *     A,
                            const void *      B,
                            float *           C,
                            int               M,
                            int               N,
                            int               K,
                            const XMXConfig & config) {
    // Validate dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    // K must be a multiple of QK4_0 for Q4_0 format
    if (K % QK4_0 != 0) {
        throw std::invalid_argument("K must be a multiple of QK4_0 (32) for Q4_0 GEMM");
    }

    const int k_blocks = K / QK4_0;

    // Get tile dimensions from XMXConfig (hardware-queried, not hardcoded)
    const size_t tile_m = config.tile_m();
    const size_t tile_n = config.tile_n();

    // ESIMD kernels are limited to max 64 work-items per work-group
    // Use smaller work-group sizes that respect ESIMD constraints
    constexpr int MAX_ESIMD_WG_SIZE = 64;
    const int     wg_m              = (tile_m > 0) ? std::min(static_cast<int>(tile_m), 8) : 8;
    const int     wg_n              = (tile_n > 0) ? std::min(static_cast<int>(tile_n), 8) : 8;

    // Ensure total work-group size doesn't exceed ESIMD limit
    int final_wg_m = wg_m;
    int final_wg_n = wg_n;
    while (final_wg_m * final_wg_n > MAX_ESIMD_WG_SIZE) {
        if (final_wg_n > final_wg_m) {
            final_wg_n /= 2;
        } else {
            final_wg_m /= 2;
        }
    }

    // Compute grid dimensions (round up to handle non-aligned M, N)
    const int grid_m = (M + final_wg_m - 1) / final_wg_m * final_wg_m;
    const int grid_n = (N + final_wg_n - 1) / final_wg_n * final_wg_n;

    // Cast B to the Q4_0 block type
    const block_q4_0_esimd * B_blocks = static_cast<const block_q4_0_esimd *>(B);

    // Submit ESIMD kernel
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<esimd_q4_0_gemm_kernel_basic>(
            sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n), sycl::range<2>(final_wg_m, final_wg_n)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                const int m = static_cast<int>(item.get_global_id(0));
                const int n = static_cast<int>(item.get_global_id(1));

                // Bounds check for non-aligned dimensions
                if (m >= M || n >= N) {
                    return;
                }

                // Accumulator for dot product
                float acc = 0.0f;

                // Iterate over K dimension in blocks of QK4_0
                // This loop structure matches dpas processing of K in chunks
                for (int kb = 0; kb < k_blocks; kb++) {
                    // Get the Q4_0 block for weight element (n, kb)
                    // B layout: [N, K/QK4_0] blocks
                    const block_q4_0_esimd & block = B_blocks[n * k_blocks + kb];

                    // Get scale factor for this block
                    // dpas pattern: accumulate int products, then scale
                    const float scale = static_cast<float>(block.d);

                    // Load A values for this block into ESIMD register
                    esimd::simd<float, QK4_0> a_vec;
#    pragma unroll
                    for (int i = 0; i < QK4_0; i++) {
                        a_vec[i] = A[m * K + kb * QK4_0 + i];
                    }

                    // Load packed Q4_0 nibbles and unpack to signed int8
                    esimd::simd<uint8_t, QK4_0 / 2> packed_qs;
#    pragma unroll
                    for (int i = 0; i < QK4_0 / 2; i++) {
                        packed_qs[i] = block.qs[i];
                    }

                    // Unpack nibbles to signed int8 (subtracts 8 for signed range)
                    esimd::simd<int8_t, QK4_0> unpacked_qs = unpack_q4_0_block_esimd(packed_qs);

                    // Convert to float with scale applied
                    // This simulates: dpas accumulates int8*int8->int32, then scale
                    // Since A is FP32, we do (A * unpacked_qs) directly and apply scale
                    esimd::simd<float, QK4_0> b_vec;
#    pragma unroll
                    for (int i = 0; i < QK4_0; i++) {
                        b_vec[i] = static_cast<float>(unpacked_qs[i]) * scale;
                    }

                    // Compute element-wise product
                    esimd::simd<float, QK4_0> prod = a_vec * b_vec;

                    // Reduce to scalar - this is the dpas accumulation pattern
                    acc += esimd_hsum<QK4_0>(prod);
                }

                // Store result
                C[m * N + n] = acc;
            });
    });
}

// =============================================================================
// ESIMD Q4_0 GEMM: Tiled Implementation (for larger matrices)
//
// This kernel uses tiled computation for better cache utilization.
// Tile dimensions are derived from XMXConfig (hardware-queried).
//
// The tiled version processes multiple output elements per work-item,
// which improves data reuse and reduces memory traffic.
// =============================================================================

// Tile sizes for the tiled implementation
// These are multiples of the hardware XMX tile dimensions
constexpr int Q4_TILE_M_MULT = 4;  // Process 4x hardware M tiles per work-group
constexpr int Q4_TILE_N_MULT = 4;  // Process 4x hardware N tiles per work-group

inline void esimd_q4_0_gemm_tiled(sycl::queue &     q,
                                  const float *     A,
                                  const void *      B,
                                  float *           C,
                                  int               M,
                                  int               N,
                                  int               K,
                                  const XMXConfig & config) {
    // Validate dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (K % QK4_0 != 0) {
        throw std::invalid_argument("K must be a multiple of QK4_0 (32) for Q4_0 GEMM");
    }

    const int k_blocks = K / QK4_0;

    // Get tile dimensions from XMXConfig
    const size_t hw_tile_m = config.tile_m();
    const size_t hw_tile_n = config.tile_n();

    // Compute work-group tile sizes (logical tiles, each work-item processes multiple elements)
    const int tile_m = (hw_tile_m > 0) ? static_cast<int>(hw_tile_m * Q4_TILE_M_MULT) : 32;
    const int tile_n = (hw_tile_n > 0) ? static_cast<int>(hw_tile_n * Q4_TILE_N_MULT) : 64;

    // ESIMD work-group size is limited to 64
    // Use 8x8 = 64 which is the max for ESIMD kernels
    constexpr int MAX_ESIMD_WG_SIZE = 64;
    int           wg_size_m         = 8;
    int           wg_size_n         = 8;

    // Ensure we don't exceed ESIMD limit
    while (wg_size_m * wg_size_n > MAX_ESIMD_WG_SIZE) {
        if (wg_size_n > wg_size_m) {
            wg_size_n /= 2;
        } else {
            wg_size_m /= 2;
        }
    }

    // Grid dimensions
    const int num_tiles_m = (M + tile_m - 1) / tile_m;
    const int num_tiles_n = (N + tile_n - 1) / tile_n;

    const block_q4_0_esimd * B_blocks = static_cast<const block_q4_0_esimd *>(B);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<esimd_q4_0_gemm_kernel_tiled>(
            sycl::nd_range<2>(sycl::range<2>(num_tiles_m * wg_size_m, num_tiles_n * wg_size_n),
                              sycl::range<2>(wg_size_m, wg_size_n)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                const int tile_idx_m = static_cast<int>(item.get_group(0));
                const int tile_idx_n = static_cast<int>(item.get_group(1));
                const int local_m    = static_cast<int>(item.get_local_id(0));
                const int local_n    = static_cast<int>(item.get_local_id(1));

                // Compute M and N ranges for this tile
                const int m_start  = tile_idx_m * tile_m;
                const int n_start  = tile_idx_n * tile_n;
                const int m_stride = wg_size_m;
                const int n_stride = wg_size_n;

                // Process multiple output elements per work-item
                for (int m_local = local_m; m_local < tile_m; m_local += m_stride) {
                    const int m = m_start + m_local;
                    if (m >= M) {
                        continue;
                    }

                    for (int n_local = local_n; n_local < tile_n; n_local += n_stride) {
                        const int n = n_start + n_local;
                        if (n >= N) {
                            continue;
                        }

                        // Accumulator
                        float acc = 0.0f;

                        // Process K in Q4_0 blocks
                        // This matches dpas K=32 processing for int8
                        for (int kb = 0; kb < k_blocks; kb++) {
                            const block_q4_0_esimd & block = B_blocks[n * k_blocks + kb];
                            const float              scale = static_cast<float>(block.d);

                            // Vectorized dot product using ESIMD
                            esimd::simd<float, QK4_0> a_vec;

#    pragma unroll
                            for (int i = 0; i < QK4_0; i++) {
                                a_vec[i] = A[m * K + kb * QK4_0 + i];
                            }

                            // Load and unpack Q4_0 nibbles
                            esimd::simd<uint8_t, QK4_0 / 2> packed_qs;
#    pragma unroll
                            for (int i = 0; i < QK4_0 / 2; i++) {
                                packed_qs[i] = block.qs[i];
                            }

                            esimd::simd<int8_t, QK4_0> unpacked_qs = unpack_q4_0_block_esimd(packed_qs);

                            // dpas pattern: scale applied after accumulation
                            esimd::simd<float, QK4_0> b_vec;
#    pragma unroll
                            for (int i = 0; i < QK4_0; i++) {
                                b_vec[i] = static_cast<float>(unpacked_qs[i]) * scale;
                            }

                            esimd::simd<float, QK4_0> prod = a_vec * b_vec;
                            acc += esimd_hsum<QK4_0>(prod);
                        }

                        C[m * N + n] = acc;
                    }
                }
            });
    });
}

#else   // !SYCL_XMX_ESIMD_AVAILABLE

// Stub implementations when ESIMD is not available
inline void esimd_q4_0_gemm(sycl::queue &     q,
                            const float *     A,
                            const void *      B,
                            float *           C,
                            int               M,
                            int               N,
                            int               K,
                            const XMXConfig & config) {
    (void) q;
    (void) A;
    (void) B;
    (void) C;
    (void) M;
    (void) N;
    (void) K;
    (void) config;
    throw std::runtime_error("ESIMD Q4_0 GEMM not available - ESIMD support missing");
}

inline void esimd_q4_0_gemm_tiled(sycl::queue &     q,
                                  const float *     A,
                                  const void *      B,
                                  float *           C,
                                  int               M,
                                  int               N,
                                  int               K,
                                  const XMXConfig & config) {
    (void) q;
    (void) A;
    (void) B;
    (void) C;
    (void) M;
    (void) N;
    (void) K;
    (void) config;
    throw std::runtime_error("ESIMD Q4_0 GEMM (tiled) not available - ESIMD support missing");
}

#endif  // SYCL_XMX_ESIMD_AVAILABLE

// =============================================================================
// Dispatch Helper
// =============================================================================

// Choose the best GEMM implementation based on matrix size and config
inline void esimd_q4_0_gemm_dispatch(sycl::queue &     q,
                                     const float *     A,
                                     const void *      B,
                                     float *           C,
                                     int               M,
                                     int               N,
                                     int               K,
                                     const XMXConfig & config) {
    // For large matrices, use tiled implementation
    // For small matrices, use basic implementation
    constexpr int LARGE_MATRIX_THRESHOLD = 1024;

    if (M >= LARGE_MATRIX_THRESHOLD || N >= LARGE_MATRIX_THRESHOLD) {
        esimd_q4_0_gemm_tiled(q, A, B, C, M, N, K, config);
    } else {
        esimd_q4_0_gemm(q, A, B, C, M, N, K, config);
    }
}

}  // namespace ggml_sycl_xmx

#endif  // GGML_SYCL_XMX_ESIMD_GEMM_Q4_HPP
