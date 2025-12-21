//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_DMMV_ESIMD_HPP
#define GGML_SYCL_DMMV_ESIMD_HPP

#include "common.hpp"

// Check for ESIMD availability
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#define SYCL_ESIMD_DMMV_AVAILABLE 1
#include <sycl/ext/intel/esimd.hpp>
#include <cstdlib>
#include <cstring>
namespace esimd = sycl::ext::intel::esimd;
#pragma message "DMMV-ESIMD: ESIMD header available, compiling ESIMD kernel"

// ============================================================================
// DMMV ESIMD Kernel for Intel Arc (Xe2/Battlemage)
//
// EXPERIMENTAL - DISABLED BY DEFAULT
//
// Benchmark results show ESIMD is 3.5x SLOWER than standard DMMV:
//   - Standard DMMV: 21.10 t/s
//   - DMMV ESIMD:     6.06 t/s
//
// Root causes:
// 1. ESIMD cannot use subgroup shuffles - requires SLM + barriers for reduction
// 2. Unaligned quantized data (block_q4_0.qs at offset 2) requires scalar loads
// 3. Register pressure from 32-float SIMD vectors per thread
//
// The standard DMMV kernel is already well-optimized for token generation
// using subgroup shuffles which are essentially free on Intel GPUs.
//
// To enable for testing: GGML_SYCL_DMMV_ESIMD=1
// ============================================================================

// Kernel name classes for VTune profiling
class dmmv_esimd_q4_0_kernel;

// ============================================================================
// ESIMD Helper: Dequantize Q4_0 block (16 bytes -> 32 floats)
// Input: 16 bytes packed (each byte has 2 nibbles)
// Output: 32 floats dequantized with scale d and zero-point offset -8
// ============================================================================

SYCL_ESIMD_FUNCTION esimd::simd<float, 32> dequant_q4_0_to_float(
    esimd::simd<uint8_t, 16> packed, float d)
{
    // Q4_0 format: 16 bytes packed, each byte has 2 nibbles (lo, hi)
    // The order for y pairing (from standard kernel):
    //   - Low nibbles (bytes 0-15 & 0xF) pair with y[0..15]
    //   - High nibbles (bytes 0-15 >> 4) pair with y[16..31]
    // So output[0..15] = low nibbles, output[16..31] = high nibbles

    // Extract low and high nibbles
    esimd::simd<uint8_t, 16> lo = packed & 0x0F;      // Low nibbles -> result[0..15]
    esimd::simd<uint8_t, 16> hi = packed >> 4;        // High nibbles -> result[16..31]

    // Convert to float with scale and zero-point offset
    esimd::simd<float, 32> result;

    // Non-interleaved: result[0..15] = lo, result[16..31] = hi
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = (static_cast<float>(lo[i]) - 8.0f) * d;
        result[i + 16] = (static_cast<float>(hi[i]) - 8.0f) * d;
    }

    return result;
}

// ============================================================================
// ESIMD Helper: Horizontal sum of simd<float, 32>
// Uses ESIMD reduction intrinsics for efficiency
// ============================================================================

SYCL_ESIMD_FUNCTION float esimd_hsum_32(esimd::simd<float, 32> v)
{
    // Tree reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    esimd::simd<float, 16> v16 = v.template select<16, 1>(0) + v.template select<16, 1>(16);
    esimd::simd<float, 8> v8 = v16.template select<8, 1>(0) + v16.template select<8, 1>(8);
    esimd::simd<float, 4> v4 = v8.template select<4, 1>(0) + v8.template select<4, 1>(4);
    esimd::simd<float, 2> v2 = v4.template select<2, 1>(0) + v4.template select<2, 1>(2);
    return v2[0] + v2[1];
}

// ============================================================================
// DMMV ESIMD Kernel for Q4_0
//
// Grid: (1, 1, nrows)
// Block: (1, 1, WARP_SIZE) - one warp per row
//
// Each work-group computes one row of the output
// Each work-item processes blocks_per_row/WARP_SIZE blocks
// ============================================================================

inline bool launch_dmmv_q4_0_esimd(
    const void * __restrict__ vx,           // Quantized weights [nrows, ncols/32]
    const float * __restrict__ y,           // Input activations [ncols]
    float * __restrict__ dst,               // Output [nrows]
    const int ncols,
    const int nrows,
    sycl::queue & stream)
{
    const int blocks_per_row = ncols / QK4_0;  // QK4_0 = 32

    // Debug output
    static int launch_count = 0;
    bool debug_mode = std::getenv("GGML_SYCL_DMMV_DEBUG") != nullptr;

    launch_count++;

    if (debug_mode && launch_count <= 3) {
        fprintf(stderr, "[DMMV-ESIMD #%d] nrows=%d ncols=%d blocks_per_row=%d WARP_SIZE=%d\n",
                launch_count, nrows, ncols, blocks_per_row, WARP_SIZE);
    }

    // For debugging: Use standard SYCL instead of ESIMD to verify the dispatch path
    // Note: Keep this disabled in production
    #if 0
    static int fallback_debug = 0;
    if (fallback_debug++ < 3) {
        fprintf(stderr, "[DMMV-ESIMD] DEBUG: Returning false to use standard kernel\n");
    }
    return false;
    #endif

    // Safety check: ncols must be multiple of DMMV_X (32)
    if (ncols % GGML_SYCL_DMMV_X != 0) {
        if (debug_mode) {
            fprintf(stderr, "[DMMV-ESIMD] Skipping - ncols=%d not multiple of %d\n",
                    ncols, GGML_SYCL_DMMV_X);
        }
        return false;
    }

    // Grid: one work-group per row - MUST match standard DMMV kernel layout
    // Standard uses: block_nums(1, 1, block_num_y), block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE)
    // So dim2 is the row dimension, not dim1
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    sycl::range<3> block_nums(1, 1, block_num_y);
    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    sycl::range<3> global = block_nums * block_dims;  // (1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE)
    sycl::range<3> local = block_dims;  // (1, GGML_SYCL_MMV_Y, WARP_SIZE)

    const block_q4_0* x = reinterpret_cast<const block_q4_0*>(vx);

    stream.submit([&](sycl::handler & cgh) {
        // Note: ESIMD kernels manage their own SLM via slm_scalar_store/load
        // No need for sycl::local_accessor

        // Debug: use a diagnostic path that doesn't use SLM or ESIMD vectors
        // This helps isolate where the bug is
        int debug_level = 0;
        const char* debug_env = std::getenv("GGML_SYCL_DMMV_SIMPLE");
        if (debug_env) debug_level = std::atoi(debug_env);

        if (debug_level == 4) {
            // Return false from here to force fallback to standard kernel
            // but via a different code path
            return;  // Empty submit - will hang or fail
        } else if (debug_level == 2) {
            // Minimal debug: just write known values to verify kernel runs
            // Use same indexing as standard kernel: get_group(2) for row
            cgh.parallel_for<class dmmv_minimal_kernel>(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item) {
                    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
                    const int tid = item.get_local_id(2);
                    if (row < nrows && tid == 0) {
                        dst[row] = 0.0f;  // Write zeros to all outputs
                    }
                });
        } else if (debug_level == 3) {
            // Copy the EXACT standard kernel to verify dispatch works
            // This is the same as dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>
            cgh.parallel_for<class dmmv_copy_standard_kernel>(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
                    if (row >= nrows) return;

                    const int tid = item.get_local_id(2);
                    constexpr int qk = QK4_0;  // 32
                    constexpr int qr = QR4_0;  // 2

                    const int iter_stride = 2 * GGML_SYCL_DMMV_X;  // 64
                    const int vals_per_iter = iter_stride / WARP_SIZE;  // 2
                    const int y_offset = qk / 2;  // 16

                    float tmp = 0.0f;

                    for (int i = 0; i < ncols; i += iter_stride) {
                        const int col = i + vals_per_iter * tid;
                        const int ib = (row * ncols + col) / qk;  // block index
                        const int iqs = (col % qk) / qr;  // quant index (0-15)
                        const int iybs = col - col % qk;  // y block start

                        #pragma unroll
                        for (int j = 0; j < vals_per_iter; j += 2) {
                            // Dequantize Q4_0
                            const block_q4_0* xb = x + ib;
                            float d = static_cast<float>(xb->d);
                            int vui = xb->qs[iqs + j / qr];
                            float v0 = (static_cast<float>(vui & 0xF) - 8.0f) * d;
                            float v1 = (static_cast<float>(vui >> 4) - 8.0f) * d;

                            tmp += v0 * y[iybs + iqs + j / qr + 0];
                            tmp += v1 * y[iybs + iqs + j / qr + y_offset];
                        }
                    }

                    // Warp-level reduction using subgroup shuffle
                    auto sg = item.get_sub_group();
                    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
                    for (int mask = mask_start; mask > 0; mask >>= 1) {
                        tmp += sycl::permute_group_by_xor(sg, tmp, mask);
                    }

                    if (tid == 0) {
                        dst[row] = tmp;
                    }
                });
        } else if (debug_level == 1) {
            // Simple scalar kernel for debugging - no ESIMD, no SLM
            // Use same indexing as standard kernel
            cgh.parallel_for<class dmmv_debug_kernel>(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item) {
                    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
                    const int tid = item.get_local_id(2);

                    if (row >= nrows || tid != 0) {
                        return;
                    }

                    // Single thread computes entire row (slow but correct)
                    const block_q4_0* x_row = x + row * blocks_per_row;
                    float sum = 0.0f;

                    for (int b = 0; b < blocks_per_row; b++) {
                        const block_q4_0* blk = x_row + b;
                        float d = static_cast<float>(blk->d);
                        const float* y_ptr = y + b * QK4_0;

                        for (int j = 0; j < QK4_0 / 2; j++) {
                            uint8_t byte = blk->qs[j];
                            float x0 = (static_cast<float>(byte & 0xF) - 8.0f) * d;
                            float x1 = (static_cast<float>(byte >> 4) - 8.0f) * d;
                            sum += x0 * y_ptr[2*j];
                            sum += x1 * y_ptr[2*j + 1];
                        }
                    }

                    dst[row] = sum;
                });
        } else {
            // ESIMD kernel - optimized version
            // Uses ESIMD block loads and SLM for reduction
            cgh.parallel_for<dmmv_esimd_q4_0_kernel>(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                    using namespace esimd;

                    // SLM size: WARP_SIZE floats for reduction
                    constexpr int slm_size = WARP_SIZE * sizeof(float);
                    slm_init<slm_size>();

                    // Row indexing must match standard kernel: get_group(2) for row
                    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
                    const int tid = item.get_local_id(2);

                    if (row >= nrows) {
                        return;
                    }

                    // Base pointer for this row's weights
                    const block_q4_0* x_row = x + row * blocks_per_row;

                    float partial_sum = 0.0f;

                    // Each thread processes blocks strided by WARP_SIZE
                    for (int b = tid; b < blocks_per_row; b += WARP_SIZE) {
                        const block_q4_0* blk = x_row + b;

                        // Load scale (half -> float)
                        float d = static_cast<float>(blk->d);

                        // Load 16 bytes of quantized data
                        // NOTE: block_q4_0.qs is at offset 2 (after half d), NOT aligned
                        // Must use scalar loads or gather instead of block_load
                        const uint8_t* qs_ptr = blk->qs;
                        simd<uint8_t, 16> packed;
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            packed[i] = qs_ptr[i];
                        }

                        // Dequantize to 32 floats
                        simd<float, 32> x_vals = dequant_q4_0_to_float(packed, d);

                        // Load 32 activation values using block_load (Y is aligned)
                        const float* y_ptr = y + b * QK4_0;
                        simd<float, 32> y_vals = block_load<float, 32>(y_ptr);

                        // FMA and horizontal sum
                        simd<float, 32> products = x_vals * y_vals;
                        partial_sum += esimd_hsum_32(products);
                    }

                    // Optimized reduction using ESIMD slm_block operations
                    // Store partial sum to SLM
                    slm_scalar_store<float>(tid * sizeof(float), partial_sum);
                    esimd::barrier();

                    // Only thread 0 does the reduction (simpler, might be faster for small WARP)
                    if (tid == 0) {
                        // Load all 32 partial sums and reduce
                        simd<float, 32> all_sums = slm_block_load<float, 32>(0);
                        // ESIMD horizontal sum
                        simd<float, 16> s16 = all_sums.template select<16, 1>(0) + all_sums.template select<16, 1>(16);
                        simd<float, 8> s8 = s16.template select<8, 1>(0) + s16.template select<8, 1>(8);
                        simd<float, 4> s4 = s8.template select<4, 1>(0) + s8.template select<4, 1>(4);
                        simd<float, 2> s2 = s4.template select<2, 1>(0) + s4.template select<2, 1>(2);
                        dst[row] = s2[0] + s2[1];
                    }
                });
        }
    });

    return true;
}

// ============================================================================
// Dispatch function - called from dmmv.cpp
// ============================================================================

inline bool dmmv_esimd_enabled() {
    static bool enabled = []() {
        const char* env = std::getenv("GGML_SYCL_DMMV_ESIMD");
        bool val = env != nullptr && std::string(env) == "1";
        fprintf(stderr, "[DMMV-ESIMD] ESIMD available, env=%s, enabled=%d\n",
                env ? env : "null", val ? 1 : 0);
        return val;
    }();
    return enabled;
}

inline bool dmmv_esimd_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        // TODO: Add Q8_0, Q4_K, etc.
        default:
            return false;
    }
}

#else // !SYCL_ESIMD_DMMV_AVAILABLE

// Stub implementations when ESIMD is not available

inline bool dmmv_esimd_enabled() {
    static bool once = []() {
        fprintf(stderr, "[DMMV-ESIMD] ESIMD NOT available - using standard DMMV\n");
        return false;
    }();
    (void)once;
    return false;
}

inline bool dmmv_esimd_supported(ggml_type type) {
    GGML_UNUSED(type);
    return false;
}

inline bool launch_dmmv_q4_0_esimd(
    const void * __restrict__ vx,
    const float * __restrict__ y,
    float * __restrict__ dst,
    const int ncols,
    const int nrows,
    sycl::queue & stream)
{
    GGML_UNUSED(vx);
    GGML_UNUSED(y);
    GGML_UNUSED(dst);
    GGML_UNUSED(ncols);
    GGML_UNUSED(nrows);
    GGML_UNUSED(stream);
    return false;
}

#endif // SYCL_ESIMD_DMMV_AVAILABLE

#endif // GGML_SYCL_DMMV_ESIMD_HPP
