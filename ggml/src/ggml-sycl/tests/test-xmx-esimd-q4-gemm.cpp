//
// Test: ESIMD Q4_0 GEMM Kernel with XMX dpas Pattern
//
// Tests the dpas-based Q4_0 GEMM kernel for:
// 1. NibbleUnpackToInt8 - Verify unpacking correctness
// 2. ScaleFactorCorrect - Verify scale application
// 3. DpasCorrectnessWithKnownInputs - Verify GEMM with known values
// 4. MatchesCPUReference - Verify against CPU reference
// 5. FallbackOnUnsupportedHardware - Graceful handling
// 6. NonAlignedDimensionsWork - M=23, N=29
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Enable standalone mode for XMX ESIMD common header
#define XMX_TEST_STANDALONE 1

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
#else
#    define SYCL_ESIMD_AVAILABLE 0
#endif

// Include XMX ESIMD common header (provides XMXConfig, XMXCapabilities, dpas helpers)
#include "../xmx-esimd-common.hpp"

// Include the Q4_0 ESIMD GEMM kernel (uses dpas pattern)
#include "../xmx-esimd-gemm-q4.hpp"

// =============================================================================
// Q4_0 Block Definition (matches ggml-common.h)
// =============================================================================
#define QK4_0 32

struct block_q4_0_test {
    sycl::half d;              // scale factor (delta)
    uint8_t    qs[QK4_0 / 2];  // quantized values: 16 bytes = 32 nibbles
};

static_assert(sizeof(block_q4_0_test) == sizeof(sycl::half) + QK4_0 / 2, "wrong q4_0 block size");

// =============================================================================
// Test Helpers
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_SKIP(reason)                          \
    do {                                           \
        g_tests_skipped++;                         \
        fprintf(stderr, "SKIPPED (%s)\n", reason); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tol, msg)                                                      \
    do {                                                                                                  \
        float diff = std::abs((float) (actual) - (float) (expected));                                     \
        if (diff > (tol)) {                                                                               \
            fprintf(stderr, "FAILED: %s (expected %.6f, got %.6f, diff=%.6f)\n", msg, (float) (expected), \
                    (float) (actual), diff);                                                              \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

// =============================================================================
// Reference Implementation: Q4_0 GEMM (CPU)
//
// Computes C[M,N] = A[M,K] * B[K,N] where:
// - A is FP32 input matrix
// - B is Q4_0 quantized weight matrix (stored as blocks)
// - C is FP32 output matrix
//
// Q4_0 format:
// - 32 weights per block packed as 16 bytes (2 nibbles per byte)
// - Low nibble: qs[i] & 0x0F, value = nibble - 8 (signed range [-8, +7])
// - High nibble: qs[i] >> 4, value = nibble - 8 (signed range [-8, +7])
// - Dequantized: value * scale
// =============================================================================

void reference_q4_0_gemm(const float * A, const block_q4_0_test * B, float * C, int M, int N, int K) {
    // Number of Q4_0 blocks in K dimension
    const int k_blocks = K / QK4_0;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;

            for (int kb = 0; kb < k_blocks; kb++) {
                // Get the block for this (n, kb) position
                // B layout: [N, K/QK4_0] blocks
                const block_q4_0_test & block = B[n * k_blocks + kb];
                float                   scale = static_cast<float>(block.d);

                // Unpack and compute dot product for this block
                // Q4_0: 16 packed bytes -> 32 signed int8 values
                for (int i = 0; i < QK4_0 / 2; i++) {
                    uint8_t packed = block.qs[i];

                    // Low nibble: (packed & 0x0F) - 8
                    int8_t val_lo = static_cast<int8_t>((packed & 0x0F) - 8);
                    float  b_lo   = static_cast<float>(val_lo) * scale;
                    int    k_lo   = kb * QK4_0 + 2 * i;
                    sum += A[m * K + k_lo] * b_lo;

                    // High nibble: (packed >> 4) - 8
                    int8_t val_hi = static_cast<int8_t>((packed >> 4) - 8);
                    float  b_hi   = static_cast<float>(val_hi) * scale;
                    int    k_hi   = kb * QK4_0 + 2 * i + 1;
                    sum += A[m * K + k_hi] * b_hi;
                }
            }

            C[m * N + n] = sum;
        }
    }
}

// =============================================================================
// Helper: Quantize FP32 to Q4_0
// =============================================================================

void quantize_fp32_to_q4_0(const float * src, block_q4_0_test * dst, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        // Find max absolute value in block
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            float val = std::abs(src[i * QK4_0 + j]);
            if (val > amax) {
                amax = val;
            }
        }

        // Compute scale (Q4_0 range is -8 to +7, so max magnitude is 7)
        float d  = amax / 7.0f;
        dst[i].d = sycl::half(d);

        // Quantize and pack nibbles
        if (d > 0) {
            float id = 1.0f / d;
            for (int j = 0; j < QK4_0 / 2; j++) {
                // Get two values to pack
                float v0 = src[i * QK4_0 + 2 * j];
                float v1 = src[i * QK4_0 + 2 * j + 1];

                // Quantize to [-8, +7] range, then add 8 to get [0, 15]
                int8_t q0 = static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, v0 * id))));
                int8_t q1 = static_cast<int8_t>(std::round(std::max(-8.0f, std::min(7.0f, v1 * id))));

                // Pack: low nibble = q0 + 8, high nibble = q1 + 8
                dst[i].qs[j] = static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
            }
        } else {
            // Zero scale: pack zeros (8 for each nibble since 0 + 8 = 8)
            for (int j = 0; j < QK4_0 / 2; j++) {
                dst[i].qs[j] = 0x88;  // 8 | (8 << 4)
            }
        }
    }
}

// =============================================================================
// Test 1: NibbleUnpackToInt8
//
// Verify unpacking correctness: each nibble should be correctly extracted
// and converted to signed int8 by subtracting 8.
// =============================================================================
#if SYCL_ESIMD_AVAILABLE

bool test_nibble_unpack_to_int8(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;     // Not used for this test
    (void) caps;  // Not used for this test
    TEST_BEGIN("NibbleUnpackToInt8");

    // Test host-side unpacking function
    uint8_t packed[16];
    int8_t  unpacked[32];

    // Create test pattern: low nibble = i % 16, high nibble = (15 - i % 16)
    for (int i = 0; i < 16; i++) {
        uint8_t lo = static_cast<uint8_t>(i % 16);
        uint8_t hi = static_cast<uint8_t>(15 - (i % 16));
        packed[i]  = lo | (hi << 4);
    }

    // Unpack using the host function
    ggml_sycl_xmx::unpack_q4_0_block_to_int8(packed, unpacked);

    // Verify unpacking
    for (int i = 0; i < 16; i++) {
        int8_t expected_lo = static_cast<int8_t>((i % 16) - 8);
        int8_t expected_hi = static_cast<int8_t>((15 - (i % 16)) - 8);

        if (unpacked[2 * i] != expected_lo) {
            fprintf(stderr, "Low nibble mismatch at %d: expected %d, got %d\n", i, expected_lo, unpacked[2 * i]);
            TEST_FAIL("Low nibble unpacking incorrect");
        }
        if (unpacked[2 * i + 1] != expected_hi) {
            fprintf(stderr, "High nibble mismatch at %d: expected %d, got %d\n", i, expected_hi, unpacked[2 * i + 1]);
            TEST_FAIL("High nibble unpacking incorrect");
        }
    }

    // Test edge cases: 0x00 should unpack to (-8, -8), 0xFF should unpack to (7, 7)
    uint8_t edge_packed[2] = { 0x00, 0xFF };
    int8_t  edge_unpacked[4];
    ggml_sycl_xmx::unpack_q4_0_to_int8(edge_packed, edge_unpacked, 2);

    TEST_ASSERT(edge_unpacked[0] == -8, "0x00 low nibble should be -8");
    TEST_ASSERT(edge_unpacked[1] == -8, "0x00 high nibble should be -8");
    TEST_ASSERT(edge_unpacked[2] == 7, "0xFF low nibble should be 7");
    TEST_ASSERT(edge_unpacked[3] == 7, "0xFF high nibble should be 7");

    fprintf(stderr, "(verified 32 values + edge cases) ");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: ScaleFactorCorrect
//
// Verify scale is applied correctly in dequantization.
// Uses different scales for each B column.
// =============================================================================

bool test_scale_factor_correct(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("ScaleFactorCorrect");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    constexpr int M = 2;
    constexpr int N = 2;
    constexpr int K = 32;  // One Q4_0 block

    const int k_blocks = K / QK4_0;

    std::vector<float>           h_A(M * K);
    std::vector<block_q4_0_test> h_B(N * k_blocks);
    std::vector<float>           h_C_ref(M * N);
    std::vector<float>           h_C_gpu(M * N, 0.0f);

    // A = all 1.0f
    std::fill(h_A.begin(), h_A.end(), 1.0f);

    // B column 0: all nibbles = 9 (value = 9 - 8 = 1), scale = 0.5
    // B column 1: all nibbles = 10 (value = 10 - 8 = 2), scale = 2.0
    h_B[0].d = sycl::half(0.5f);
    h_B[1].d = sycl::half(2.0f);
    for (int i = 0; i < QK4_0 / 2; i++) {
        h_B[0].qs[i] = 0x99;  // 9 | (9 << 4) -> values are 1, 1
        h_B[1].qs[i] = 0xAA;  // 10 | (10 << 4) -> values are 2, 2
    }

    // Expected:
    // C[*,0] = sum(1.0 * 1 * 0.5) = 32 * 0.5 = 16.0
    // C[*,1] = sum(1.0 * 2 * 2.0) = 32 * 4.0 = 128.0
    reference_q4_0_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    float expected_col0 = 16.0f;
    float expected_col1 = 128.0f;

    TEST_ASSERT_NEAR(h_C_ref[0], expected_col0, 1e-3f, "Reference C[0,0] incorrect");
    TEST_ASSERT_NEAR(h_C_ref[1], expected_col1, 1e-3f, "Reference C[0,1] incorrect");

    // Run on GPU
    float *           d_A = sycl::malloc_device<float>(M * K, q);
    block_q4_0_test * d_B = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    float *           d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        if (d_A) {
            sycl::free(d_A, q);
        }
        if (d_B) {
            sycl::free(d_B, q);
        }
        if (d_C) {
            sycl::free(d_C, q);
        }
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
    q.memcpy(d_B, h_B.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memset(d_C, 0, M * N * sizeof(float)).wait();

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    try {
        ggml_sycl_xmx::esimd_q4_0_gemm(q, d_A, d_B, d_C, M, N, K, config);
        q.wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        fprintf(stderr, "Kernel failed: %s\n", e.what());
        TEST_FAIL("Kernel execution failed");
    }

    q.memcpy(h_C_gpu.data(), d_C, M * N * sizeof(float)).wait();

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    // Verify scale was applied correctly
    const float tolerance = 1e-1f;  // Q4_0 has quantization error

    fprintf(stderr, "\n  C[0,0]: expected=%.1f, got=%.1f\n", expected_col0, h_C_gpu[0]);
    fprintf(stderr, "  C[0,1]: expected=%.1f, got=%.1f\n", expected_col1, h_C_gpu[1]);

    TEST_ASSERT_NEAR(h_C_gpu[0], expected_col0, tolerance, "Scale factor not applied correctly for column 0");
    TEST_ASSERT_NEAR(h_C_gpu[1], expected_col1, tolerance, "Scale factor not applied correctly for column 1");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: DpasCorrectnessWithKnownInputs
//
// Setup known A and B values, verify C matches expected computation.
// Uses simple all-1s pattern for A and known nibble values for B.
// =============================================================================

bool test_dpas_correctness_with_known_inputs(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("DpasCorrectnessWithKnownInputs");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    constexpr int M = 8;   // One dpas tile (RepeatCount=1)
    constexpr int N = 16;  // Two N tiles (each 8 wide for Arc)
    constexpr int K = 32;  // One Q4_0 block

    const int k_blocks = K / QK4_0;

    // Allocate host memory
    std::vector<float>           h_A(M * K);
    std::vector<block_q4_0_test> h_B(N * k_blocks);
    std::vector<float>           h_C_ref(M * N);
    std::vector<float>           h_C_gpu(M * N, 0.0f);

    // Initialize A with all 1.0f
    std::fill(h_A.begin(), h_A.end(), 1.0f);

    // Initialize B: all nibbles = 9 (value = 1), scale = 1.0
    // So dequantized value = 1 * 1.0 = 1.0 for each element
    for (int n = 0; n < N; n++) {
        h_B[n * k_blocks].d = sycl::half(1.0f);
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_B[n * k_blocks].qs[i] = 0x99;  // 9 | (9 << 4) -> values are 1, 1
        }
    }

    // Expected result: C[m,n] = sum_k(1.0 * 1 * 1.0) = 32 * 1.0 = 32.0
    float expected_value = 32.0f;
    reference_q4_0_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // Verify reference matches expected
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C_ref[i] - expected_value) > 1e-3f) {
            fprintf(stderr, "Reference verification failed at %d: expected %.2f, got %.2f\n", i, expected_value,
                    h_C_ref[i]);
            TEST_FAIL("Reference implementation incorrect");
        }
    }

    // Allocate device memory
    float *           d_A = sycl::malloc_device<float>(M * K, q);
    block_q4_0_test * d_B = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    float *           d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        if (d_A) {
            sycl::free(d_A, q);
        }
        if (d_B) {
            sycl::free(d_B, q);
        }
        if (d_C) {
            sycl::free(d_C, q);
        }
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
    q.memcpy(d_B, h_B.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memset(d_C, 0, M * N * sizeof(float)).wait();

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    try {
        ggml_sycl_xmx::esimd_q4_0_gemm(q, d_A, d_B, d_C, M, N, K, config);
        q.wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        fprintf(stderr, "Kernel failed: %s\n", e.what());
        TEST_FAIL("dpas kernel execution failed");
    }

    q.memcpy(h_C_gpu.data(), d_C, M * N * sizeof(float)).wait();

    // Compare results
    const float tolerance   = 1e-2f;
    bool        all_correct = true;
    int         error_count = 0;

    for (int i = 0; i < M * N && error_count < 5; i++) {
        float diff = std::abs(h_C_gpu[i] - expected_value);
        if (diff > tolerance) {
            fprintf(stderr, "\n  C[%d]: expected %.4f, got %.4f (diff=%.4f)", i, expected_value, h_C_gpu[i], diff);
            all_correct = false;
            error_count++;
        }
    }

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    TEST_ASSERT(all_correct, "dpas output does not match expected values");
    fprintf(stderr, "(expected=%.1f) ", expected_value);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: MatchesCPUReference
//
// Random input matrices, verify GPU result matches CPU reference within tolerance.
// =============================================================================

bool test_matches_cpu_reference(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("MatchesCPUReference");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 128;  // Multiple Q4_0 blocks

    const int k_blocks = K / QK4_0;

    std::vector<float>           h_A(M * K);
    std::vector<float>           h_B_fp32(N * K);
    std::vector<block_q4_0_test> h_B(N * k_blocks);
    std::vector<float>           h_C_ref(M * N);
    std::vector<float>           h_C_gpu(M * N, 0.0f);

    // Initialize with random values
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = dist(rng);
    }
    for (size_t i = 0; i < h_B_fp32.size(); i++) {
        h_B_fp32[i] = dist(rng);
    }

    // Quantize B
    for (int n = 0; n < N; n++) {
        quantize_fp32_to_q4_0(&h_B_fp32[n * K], &h_B[n * k_blocks], k_blocks);
    }

    // CPU reference
    reference_q4_0_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // GPU computation
    float *           d_A = sycl::malloc_device<float>(M * K, q);
    block_q4_0_test * d_B = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    float *           d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        if (d_A) {
            sycl::free(d_A, q);
        }
        if (d_B) {
            sycl::free(d_B, q);
        }
        if (d_C) {
            sycl::free(d_C, q);
        }
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
    q.memcpy(d_B, h_B.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memset(d_C, 0, M * N * sizeof(float)).wait();

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    try {
        ggml_sycl_xmx::esimd_q4_0_gemm(q, d_A, d_B, d_C, M, N, K, config);
        q.wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        fprintf(stderr, "Kernel failed: %s\n", e.what());
        TEST_FAIL("Kernel execution failed");
    }

    q.memcpy(h_C_gpu.data(), d_C, M * N * sizeof(float)).wait();

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    // Compare with tolerance (Q4_0 has higher quantization error than Q8_0)
    const float tolerance   = 1e-1f;
    bool        all_correct = true;
    int         errors      = 0;
    float       max_diff    = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_C_gpu[i] - h_C_ref[i]);
        max_diff   = std::max(max_diff, diff);
        if (diff > tolerance) {
            if (errors < 5) {
                fprintf(stderr, "\n  C[%d]: ref=%.4f, gpu=%.4f, diff=%.4f", i, h_C_ref[i], h_C_gpu[i], diff);
            }
            all_correct = false;
            errors++;
        }
    }

    TEST_ASSERT(all_correct, "GPU result does not match CPU reference");
    fprintf(stderr, "(max_diff=%.4f) ", max_diff);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: FallbackOnUnsupportedHardware
//
// Verify graceful handling when XMX is not supported.
// =============================================================================

bool test_fallback_on_unsupported_hardware(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("FallbackOnUnsupportedHardware");

    // Create mock unsupported capabilities
    XMXCapabilities unsupported;
    unsupported.supported     = false;
    unsupported.M             = 0;
    unsupported.N             = 0;
    unsupported.K             = 0;
    unsupported.supports_int8 = false;
    unsupported.supports_fp16 = false;

    // Verify XMXConfig reports as unsupported
    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(unsupported);
    TEST_ASSERT(!config.is_supported(), "Config should report as unsupported");
    TEST_ASSERT(!config.supports_int8(), "Int8 should not be supported");

    // Verify XMXDpasConfig is invalid
    ggml_sycl_xmx::XMXDpasConfig dpas_cfg = ggml_sycl_xmx::XMXDpasConfig::from_capabilities(unsupported);
    TEST_ASSERT(!dpas_cfg.is_valid(), "DpasConfig should be invalid");
    TEST_ASSERT(!dpas_cfg.supports_int8, "Int8 dpas should not be supported");

    // Verify supports_dpas returns false
    TEST_ASSERT(!ggml_sycl_xmx::supports_dpas(unsupported), "supports_dpas should return false");

    // Verify Q4_0 type is reported as unsupported
    TEST_ASSERT(!config.supports_qtype(GGML_TYPE_Q4_0), "Q4_0 should not be supported on unsupported hardware");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: NonAlignedDimensionsWork
//
// Verify kernel handles non-aligned M/N dimensions correctly with bounds checking.
// Uses M=23 and N=29 which are prime numbers, not multiples of typical tile sizes.
// =============================================================================

bool test_non_aligned_dimensions_work(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("NonAlignedDimensionsWork");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

    constexpr int M = 23;  // Prime, not aligned to any power of 2
    constexpr int N = 29;  // Prime, definitely not aligned
    constexpr int K = 64;  // Two Q4_0 blocks

    const int k_blocks = K / QK4_0;

    std::vector<float>           h_A(M * K);
    std::vector<float>           h_B_fp32(N * K);
    std::vector<block_q4_0_test> h_B(N * k_blocks);
    std::vector<float>           h_C_ref(M * N);
    std::vector<float>           h_C_gpu(M * N, 0.0f);

    // Initialize with random values
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = dist(rng);
    }
    for (size_t i = 0; i < h_B_fp32.size(); i++) {
        h_B_fp32[i] = dist(rng);
    }

    // Quantize B
    for (int n = 0; n < N; n++) {
        quantize_fp32_to_q4_0(&h_B_fp32[n * K], &h_B[n * k_blocks], k_blocks);
    }

    // CPU reference
    reference_q4_0_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // GPU computation
    float *           d_A = sycl::malloc_device<float>(M * K, q);
    block_q4_0_test * d_B = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    float *           d_C = sycl::malloc_device<float>(M * N, q);

    if (!d_A || !d_B || !d_C) {
        if (d_A) {
            sycl::free(d_A, q);
        }
        if (d_B) {
            sycl::free(d_B, q);
        }
        if (d_C) {
            sycl::free(d_C, q);
        }
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
    q.memcpy(d_B, h_B.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memset(d_C, 0, M * N * sizeof(float)).wait();

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    try {
        ggml_sycl_xmx::esimd_q4_0_gemm(q, d_A, d_B, d_C, M, N, K, config);
        q.wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        fprintf(stderr, "Kernel failed: %s\n", e.what());
        TEST_FAIL("Kernel failed with non-aligned dimensions");
    }

    q.memcpy(h_C_gpu.data(), d_C, M * N * sizeof(float)).wait();

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    // Compare results (Q4_0 has higher quantization error)
    const float tolerance   = 1e-1f;
    bool        all_correct = true;
    int         errors      = 0;

    for (int i = 0; i < M * N && errors < 10; i++) {
        float diff = std::abs(h_C_gpu[i] - h_C_ref[i]);
        if (diff > tolerance) {
            if (errors < 5) {
                int m = i / N;
                int n = i % N;
                fprintf(stderr, "\n  C[%d,%d]: ref=%.4f, gpu=%.4f, diff=%.4f", m, n, h_C_ref[i], h_C_gpu[i], diff);
            }
            all_correct = false;
            errors++;
        }
    }

    TEST_ASSERT(all_correct, "Non-aligned dimensions produced incorrect results");
    fprintf(stderr, "(M=%d, N=%d) ", M, N);
    TEST_PASS();
    return true;
}

#else   // !SYCL_ESIMD_AVAILABLE

// Stub implementations when ESIMD is not available
bool test_nibble_unpack_to_int8(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("NibbleUnpackToInt8");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_scale_factor_correct(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("ScaleFactorCorrect");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_dpas_correctness_with_known_inputs(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("DpasCorrectnessWithKnownInputs");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_matches_cpu_reference(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("MatchesCPUReference");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_fallback_on_unsupported_hardware(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("FallbackOnUnsupportedHardware");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_non_aligned_dimensions_work(sycl::queue & q, const XMXCapabilities & caps) {
    (void) q;
    (void) caps;
    TEST_BEGIN("NonAlignedDimensionsWork");
    TEST_SKIP("ESIMD not available");
    return true;
}

#endif  // SYCL_ESIMD_AVAILABLE

// =============================================================================
// Main
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== ESIMD Q4_0 GEMM dpas Pattern Kernel Tests ===\n");

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "WARNING: ESIMD header not available, tests will be skipped.\n\n");
#endif

    // Create SYCL queue
    sycl::queue  q;
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
        q   = sycl::queue(dev);
        fprintf(stderr, "Using GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU found, using default device\n");
        dev = sycl::device(sycl::default_selector_v);
        q   = sycl::queue(dev);
    }

    // Query XMX capabilities
    XMXCapabilities caps = query_xmx_capabilities(dev);
    fprintf(stderr, "XMX support: %s\n", caps.supported ? "YES" : "NO");
    if (caps.supported) {
        fprintf(stderr, "XMX dimensions: M=%zu, N=%zu, K=%zu\n", caps.M, caps.N, caps.K);
        fprintf(stderr, "XMX int8: %s, fp16: %s\n", caps.supports_int8 ? "YES" : "NO",
                caps.supports_fp16 ? "YES" : "NO");
    }
    fprintf(stderr, "\n");

    // Run the 6 required tests
    bool all_passed = true;

    // Test 1: NibbleUnpackToInt8
    all_passed &= test_nibble_unpack_to_int8(q, caps);

    // Test 2: ScaleFactorCorrect
    all_passed &= test_scale_factor_correct(q, caps);

    // Test 3: DpasCorrectnessWithKnownInputs
    all_passed &= test_dpas_correctness_with_known_inputs(q, caps);

    // Test 4: MatchesCPUReference
    all_passed &= test_matches_cpu_reference(q, caps);

    // Test 5: FallbackOnUnsupportedHardware
    all_passed &= test_fallback_on_unsupported_hardware(q, caps);

    // Test 6: NonAlignedDimensionsWork
    all_passed &= test_non_aligned_dimensions_work(q, caps);

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "\nNote: All tests were skipped because ESIMD is not available.\n");
    return 0;
#endif

    if (!caps.supported) {
        fprintf(stderr, "\nNote: XMX tests were skipped because hardware does not support XMX.\n");
        return 0;
    }

    return all_passed ? 0 : 1;
}
