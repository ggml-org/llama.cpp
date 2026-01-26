//
// Test: XMX Compute Path for Unified Kernel
//
// TDD tests for the XMX dpas compute path using Intel joint_matrix extensions.
// Tests verify:
// 1. UnifiedKernelArgs has use_xmx field
// 2. Reference scalar matmul for verification
// 3. XMX tile dimensions are correct (8x16x32)
// 4. Batch strategy selection (wide_n, standard, persistent)
// 5. Q4_0 to half conversion for XMX
// 6. XMX compute path produces correct results (when hardware supports it)
// 7. Graceful fallback to scalar when XMX unavailable
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

// Include the unified kernel header
#include "../unified-kernel.hpp"

// =============================================================================
// Q4_0 Block Definition (match ggml-common.h)
// =============================================================================
#define QK4_0 32

struct block_q4_0_test {
    sycl::half d;              // scale factor
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
// Test 1: UnifiedKernelArgs has use_xmx field
// =============================================================================
static bool test_args_has_use_xmx() {
    TEST_BEGIN("test_args_has_use_xmx");

    ggml_sycl_unified::UnifiedKernelArgs args{};
    args.use_xmx = true;
    TEST_ASSERT(args.use_xmx == true, "use_xmx should be true");
    args.use_xmx = false;
    TEST_ASSERT(args.use_xmx == false, "use_xmx should be false");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Reference scalar matmul for verification
// =============================================================================
static bool test_scalar_reference() {
    TEST_BEGIN("test_scalar_reference");

    // Small 4x4 matmul for verification
    constexpr int M = 4, N = 4, K = 4;
    float         A[M * K] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };  // Identity
    float         B[K * N] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    float         C[M * N] = { 0 };

    // Reference matmul: C = A * B
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }

    // With identity A, C should equal B
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT(std::abs(C[i] - B[i]) < 1e-5, "Scalar reference should match");
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: XMX tile dimensions are correct
// =============================================================================
static bool test_xmx_tile_dimensions() {
    TEST_BEGIN("test_xmx_tile_dimensions");

    // Intel XMX for half precision: 8x16 output, K=16 per step
    // These are compile-time constants from unified-kernel.hpp
    constexpr int XMX_M = ggml_sycl_unified::XMX_TILE_M;
    constexpr int XMX_N = ggml_sycl_unified::XMX_TILE_N;
    constexpr int XMX_K = ggml_sycl_unified::XMX_TILE_K;

    static_assert(XMX_M == 8, "XMX M dimension must be 8");
    static_assert(XMX_N == 16, "XMX N dimension must be 16");
    static_assert(XMX_K == 16, "XMX K step must be 16 for half precision");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Batch strategy selection
// =============================================================================
static bool test_batch_strategy_selection() {
    TEST_BEGIN("test_batch_strategy_selection");

    // Test batch strategy helper
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(1) == ggml_sycl_unified::BatchStrategy::WIDE_N,
                "Batch 1 should use WIDE_N");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(4) == ggml_sycl_unified::BatchStrategy::WIDE_N,
                "Batch 4 should use WIDE_N");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(7) == ggml_sycl_unified::BatchStrategy::WIDE_N,
                "Batch 7 should use WIDE_N");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(8) == ggml_sycl_unified::BatchStrategy::STANDARD,
                "Batch 8 should use STANDARD");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(32) == ggml_sycl_unified::BatchStrategy::STANDARD,
                "Batch 32 should use STANDARD");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(63) == ggml_sycl_unified::BatchStrategy::STANDARD,
                "Batch 63 should use STANDARD");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(64) == ggml_sycl_unified::BatchStrategy::PERSISTENT,
                "Batch 64 should use PERSISTENT");
    TEST_ASSERT(ggml_sycl_unified::get_batch_strategy(128) == ggml_sycl_unified::BatchStrategy::PERSISTENT,
                "Batch 128 should use PERSISTENT");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: Q4_0 to half conversion for XMX
// =============================================================================
static bool test_q4_to_half_conversion() {
    TEST_BEGIN("test_q4_to_half_conversion");

    // Q4_0 dequant: (nibble - 8) * scale
    // Then convert to half for XMX
    uint8_t qs    = 0x5A;  // nibbles: 10, 5
    float   scale = 0.5f;

    int low_nibble  = qs & 0xF;   // 10
    int high_nibble = qs >> 4;    // 5

    float val_low  = (low_nibble - 8) * scale;   // (10-8) * 0.5 = 1.0
    float val_high = (high_nibble - 8) * scale;  // (5-8) * 0.5 = -1.5

    TEST_ASSERT(std::abs(val_low - 1.0f) < 1e-5, "Low nibble conversion should be 1.0");
    TEST_ASSERT(std::abs(val_high - (-1.5f)) < 1e-5, "High nibble conversion should be -1.5");

    // Verify half conversion preserves values
    sycl::half h_low  = static_cast<sycl::half>(val_low);
    sycl::half h_high = static_cast<sycl::half>(val_high);
    TEST_ASSERT(std::abs(static_cast<float>(h_low) - val_low) < 1e-3, "Half conversion should preserve low value");
    TEST_ASSERT(std::abs(static_cast<float>(h_high) - val_high) < 1e-3, "Half conversion should preserve high value");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: XMX compute path produces correct results (GPU test)
// =============================================================================

// CPU reference matmul: C[M,N] = A_q4_0[M,K] * B_f32[K,N]
static void dequantize_q4_0_block_test(const block_q4_0_test & block, float * out) {
    const float d = static_cast<float>(block.d);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = block.qs[i];
        out[i]       = (static_cast<float>(byte & 0x0F) - 8.0f) * d;
        out[i + 16]  = (static_cast<float>(byte >> 4) - 8.0f) * d;
    }
}

static void matmul_q4_0_f32_cpu_reference(const block_q4_0_test * A,
                                          const float *           B,
                                          float *                 C,
                                          int                     M,
                                          int                     N,
                                          int                     K) {
    const int k_blocks = K / QK4_0;
    std::vector<float> a_row(K);

    for (int m = 0; m < M; m++) {
        for (int kb = 0; kb < k_blocks; kb++) {
            dequantize_q4_0_block_test(A[m * k_blocks + kb], a_row.data() + kb * QK4_0);
        }
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a_row[k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

static bool test_xmx_compute_correctness(sycl::queue & q) {
    TEST_BEGIN("test_xmx_compute_correctness");

    // Check for XMX hardware support
    sycl::device dev = q.get_device();
    bool has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
    if (!has_xmx) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    // Test with dimensions suitable for XMX (multiples of XMX tile sizes)
    // M=8, N=16, K=32 (one full XMX tile)
    const int M        = 8;
    const int N        = 16;
    const int K        = 32;
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::mt19937                 rng(789);
    for (auto & block : A) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations
    std::vector<float> B(K * N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < K * N; i++) {
        B[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), B.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(M * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(K * N, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), M * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, B.data(), K * N * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args - request XMX path
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;  // Request XMX path
    args.layout_mode    = 0;
    args.quant_type     = 2;  // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Launch kernel
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 1e-2f;
    float       max_diff  = 0.0f;
    int         max_idx   = 0;

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C_gpu[i] - C_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx  = i;
        }
    }

    if (max_diff > tolerance) {
        fprintf(stderr, "FAILED: Max diff %.6f at index %d (expected %.6f, got %.6f)\n",
                max_diff, max_idx, C_ref[max_idx], C_gpu[max_idx]);
        sycl::free(A_dev, q);
        sycl::free(B_dev, q);
        sycl::free(C_dev, q);
        return false;
    }

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: Graceful fallback to scalar when XMX unavailable
// =============================================================================
static bool test_xmx_fallback(sycl::queue & q) {
    TEST_BEGIN("test_xmx_fallback");

    // This test verifies that even when use_xmx=true, the kernel still works
    // on hardware without XMX support (falls back to scalar path)

    const int M        = 4;
    const int N        = 8;
    const int K        = 32;
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::mt19937                 rng(999);
    for (auto & block : A) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = static_cast<uint8_t>((i % 16) | ((i % 16) << 4));
        }
    }

    // Allocate F32 activations
    std::vector<float> B(K * N);
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B[k * N + n] = 0.1f * (k + n);
        }
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), B.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(M * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(K * N, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), M * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, B.data(), K * N * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args - request XMX but kernel should fallback if unavailable
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 4;
    args.tile_n         = 8;
    args.tile_k         = 32;
    args.use_xmx        = true;  // Request XMX, but should fallback on non-XMX hardware
    args.layout_mode    = 0;
    args.quant_type     = 2;  // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Launch kernel - should not crash regardless of XMX support
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 1e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(C_gpu[i], C_ref[i], tolerance,
                         (std::string("Mismatch at index ") + std::to_string(i)).c_str());
    }

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 8: XMX path with larger matrices
// =============================================================================
static bool test_xmx_larger_matrix(sycl::queue & q) {
    TEST_BEGIN("test_xmx_larger_matrix");

    // Check for XMX hardware support
    sycl::device dev = q.get_device();
    bool has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
    if (!has_xmx) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    // Test with larger dimensions (multiple XMX tiles)
    const int M        = 32;   // 4 tiles in M
    const int N        = 64;   // 4 tiles in N
    const int K        = 128;  // 4 Q4_0 blocks
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights
    std::vector<block_q4_0_test> A(M * k_blocks);
    std::mt19937                 rng(12345);
    for (auto & block : A) {
        block.d = sycl::half(0.05f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations
    std::vector<float> B(K * N);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < K * N; i++) {
        B[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), B.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(M * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(K * N, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), M * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, B.data(), K * N * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args - request XMX path
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 16;
    args.tile_n         = 32;
    args.tile_k         = 32;
    args.use_xmx        = true;  // Request XMX path
    args.layout_mode    = 0;
    args.quant_type     = 2;  // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Launch kernel
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 5e-2f;  // Larger tolerance for accumulated errors
    float       max_diff  = 0.0f;
    int         max_idx   = 0;

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C_gpu[i] - C_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx  = i;
        }
    }

    if (max_diff > tolerance) {
        fprintf(stderr, "FAILED: Max diff %.6f at index %d (expected %.6f, got %.6f)\n",
                max_diff, max_idx, C_ref[max_idx], C_gpu[max_idx]);
        sycl::free(A_dev, q);
        sycl::free(B_dev, q);
        sycl::free(C_dev, q);
        return false;
    }

    fprintf(stderr, "(max_diff=%.6f) ", max_diff);

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "XMX Compute Path Tests\n");
    fprintf(stderr, "===========================================\n");

    // Run pure unit tests (no GPU required)
    bool all_passed = true;

    all_passed &= test_args_has_use_xmx();
    all_passed &= test_scalar_reference();
    all_passed &= test_xmx_tile_dimensions();
    all_passed &= test_batch_strategy_selection();
    all_passed &= test_q4_to_half_conversion();

    // GPU tests
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU device found: %s\n", e.what());
        fprintf(stderr, "-------------------------------------------\n");
        fprintf(stderr, "Tests: %d run, %d passed, %d skipped (GPU tests skipped)\n",
                g_tests_run, g_tests_passed, g_tests_skipped);
        return all_passed ? 0 : 1;
    }

    fprintf(stderr, "Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "XMX support: %s\n", dev.has(sycl::aspect::ext_intel_matrix) ? "YES" : "NO");
    fprintf(stderr, "-------------------------------------------\n");

    sycl::queue q(dev, sycl::property::queue::in_order{});

    all_passed &= test_xmx_compute_correctness(q);
    all_passed &= test_xmx_fallback(q);
    all_passed &= test_xmx_larger_matrix(q);

    // Summary
    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n",
            g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }

    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
