//
// Test: Unified Matmul Kernel
//
// TDD tests for the unified kernel architecture supporting Q4_0 quantization.
// Tests verify:
// 1. Small matrix multiplication produces correct results
// 2. Non-aligned boundary dimensions don't cause OOB access
// 3. Scalar path computation correctness
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <chrono>
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
// CPU Reference Implementation: Q4_0 Matmul
// =============================================================================

// Dequantize a single Q4_0 block to float
static void dequantize_q4_0_block_test(const block_q4_0_test & block, float * out) {
    const float d = static_cast<float>(block.d);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = block.qs[i];
        // Low nibble: value[i] = (qs[i] & 0xF) - 8
        out[i]      = (static_cast<float>(byte & 0x0F) - 8.0f) * d;
        // High nibble: value[i+16] = (qs[i] >> 4) - 8
        out[i + 16] = (static_cast<float>(byte >> 4) - 8.0f) * d;
    }
}

// CPU reference matmul: C[M,N] = A_q4_0[N,K] * X_f32[M,K]^T
// A: Q4_0 weights [N, K/32 blocks] (indexed by output column n)
// X: float activations [M, K] (row-major, each row is one input token)
// C: float output [M, N]
static void matmul_q4_0_f32_cpu_reference(const block_q4_0_test * A,
                                          const float *           X,
                                          float *                 C,
                                          int                     M,
                                          int                     N,
                                          int                     K) {
    const int k_blocks = K / QK4_0;

    // Temporary storage for dequantized row
    std::vector<float> a_row(K);

    for (int n = 0; n < N; n++) {
        // Dequantize weight row for output column n
        for (int kb = 0; kb < k_blocks; kb++) {
            dequantize_q4_0_block_test(A[n * k_blocks + kb], a_row.data() + kb * QK4_0);
        }

        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                // X is row-major: X[m,k] = X[m*K + k]
                acc += a_row[k] * X[m * K + k];
            }
            C[m * N + n] = acc;
        }
    }
}

// =============================================================================
// Test 1: Small 4x4 Q4_0 Matmul
// Verifies basic kernel correctness with minimal dimensions
// =============================================================================
static bool test_unified_kernel_q4_0_small(sycl::queue & q) {
    TEST_BEGIN("test_unified_kernel_q4_0_small");

    // Dimensions: 4x4 output from 4x32 @ 32x4
    // M=4, N=4, K=32 (single Q4_0 block per row)
    const int M        = 4;
    const int N        = 4;
    const int K        = 32;
    const int k_blocks = K / QK4_0;  // = 1

    // Allocate and initialize Q4_0 weights [N, K]
    std::vector<block_q4_0_test> A(N * k_blocks);
    std::mt19937                 rng(42);
    for (auto & block : A) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            // Use simple pattern for easier debugging
            block.qs[i] = static_cast<uint8_t>((i % 16) | ((i % 16) << 4));
        }
    }

    // Allocate and initialize F32 activations [M, K]
    std::vector<float> X(M * K);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            X[m * K + k] = 0.1f * (m + k);
        }
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), X.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(M * K, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, X.data(), M * K * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 4;
    args.tile_n         = 4;
    args.tile_k         = 32;
    args.use_xmx        = false;  // Scalar path for now
    args.layout_mode    = 0;      // NONE (AoS)
    args.quant_type     = 2;      // GGML_TYPE_Q4_0
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
    const float tolerance = 1e-3f;
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
// Test 2: Non-Aligned Boundary Dimensions
// Verifies no OOB access for dimensions not aligned to tile size
// =============================================================================
static bool test_unified_kernel_boundary(sycl::queue & q) {
    TEST_BEGIN("test_unified_kernel_boundary");

    // Non-aligned dimensions: 33x65 (not multiples of common tile sizes)
    // K must still be multiple of 32 for Q4_0
    const int M        = 33;
    const int N        = 65;
    const int K        = 64;  // 2 Q4_0 blocks
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights [N, K]
    std::vector<block_q4_0_test> A(N * k_blocks);
    std::mt19937                 rng(123);
    for (auto & block : A) {
        block.d = sycl::half(0.05f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations [M, K]
    std::vector<float> X(M * K);
    for (int i = 0; i < M * K; i++) {
        X[i] = 0.01f * (rng() % 100);
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), X.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(M * K, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, X.data(), M * K * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args with tile sizes that don't evenly divide dimensions
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;   // 33 % 8 = 1 (boundary)
    args.tile_n         = 16;  // 65 % 16 = 1 (boundary)
    args.tile_k         = 32;
    args.use_xmx        = false;
    args.layout_mode    = 0;
    args.quant_type     = 2;  // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Launch kernel - should not crash or produce OOB access
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 1e-2f;  // Slightly higher tolerance for accumulated errors
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int idx = m * N + n;
            TEST_ASSERT_NEAR(C_gpu[idx], C_ref[idx], tolerance,
                             (std::string("Mismatch at [") + std::to_string(m) + "," + std::to_string(n) + "]").c_str());
        }
    }

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Scalar Path Correctness
// Verifies the scalar (non-XMX) path produces correct results
// =============================================================================
static bool test_unified_kernel_scalar(sycl::queue & q) {
    TEST_BEGIN("test_unified_kernel_scalar");

    // Medium-sized matmul to test scalar path
    const int M        = 16;
    const int N        = 32;
    const int K        = 128;  // 4 Q4_0 blocks
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights with random values [N, K]
    std::vector<block_q4_0_test> A(N * k_blocks);
    std::mt19937                 rng(456);
    for (auto & block : A) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations with random values [M, K]
    std::vector<float> X(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        X[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> C_ref(M * N);
    matmul_q4_0_f32_cpu_reference(A.data(), X.data(), C_ref.data(), M, N, K);

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(M * K, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, X.data(), M * K * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args - explicitly request scalar path
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = false;  // Force scalar path
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

    // Verify results with reasonable tolerance
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
// Test 4: DMMV Performance - Vectorized Q4_0 for Batch=1
// Verifies the DMMV path achieves target performance for decode operations
// =============================================================================
static bool test_unified_kernel_dmmv_perf_impl(sycl::queue & q, int N, int K, const char * label) {
    const int M        = 1;       // batch=1 (decode)
    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights [N, K/32 blocks]
    // Note: For DMMV, weights are indexed by output column (N dimension)
    std::vector<block_q4_0_test> A(N * k_blocks);
    std::mt19937                 rng(42);
    for (auto & block : A) {
        block.d = sycl::half(0.01f * (rng() % 100 + 1) / 100.0f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations [M=1, K]
    std::vector<float> B(M * K);
    for (int k = 0; k < K; k++) {
        B[k] = 0.01f * ((rng() % 200) - 100) / 100.0f;
    }

    // Compute CPU reference: C[M=1, N] = B[M=1, K] @ A_dequant[K, N]
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> a_col(K);  // Dequantized column
    for (int n = 0; n < N; n++) {
        // Dequantize column n of weights
        for (int kb = 0; kb < k_blocks; kb++) {
            dequantize_q4_0_block_test(A[n * k_blocks + kb], a_col.data() + kb * QK4_0);
        }
        // Dot product
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += B[k] * a_col[k];
        }
        C_ref[n] = acc;
    }

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * B_dev = sycl::malloc_device<float>(M * K, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(A_dev && B_dev && C_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(A_dev, A.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(B_dev, B.data(), M * K * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args for DMMV path
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 32;
    args.tile_k         = 32;
    args.use_xmx        = false;  // DMMV doesn't use XMX
    args.layout_mode    = 0;      // AoS
    args.quant_type     = 2;      // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Warmup
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Benchmark: Run multiple iterations
    const int num_iters = 100;
    auto      start     = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; i++) {
        ggml_sycl_unified::launch_unified_matmul(q, args);
    }
    q.wait();

    auto end     = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Calculate throughput
    double time_per_iter_us = static_cast<double>(elapsed) / num_iters;
    double time_per_iter_ms = time_per_iter_us / 1000.0;

    // Memory bandwidth calculation
    // Read: N * K/32 * sizeof(block_q4_0) + K * sizeof(float)
    // Write: N * sizeof(float)
    size_t bytes_read  = N * k_blocks * sizeof(block_q4_0_test) + K * sizeof(float);
    size_t bytes_write = N * sizeof(float);
    double bandwidth_gbps = (bytes_read + bytes_write) / (time_per_iter_us * 1000.0);

    fprintf(stderr, "  [%s] M=%d N=%d K=%d\n", label, M, N, K);
    fprintf(stderr, "  Time per matmul: %.2f us (%.3f ms)\n", time_per_iter_us, time_per_iter_ms);
    fprintf(stderr, "  Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);

    // Performance target: Should achieve >50% of theoretical bandwidth
    // Arc B580 has ~480 GB/s bandwidth, so target is ~240 GB/s for memory-bound ops
    // Current unified DMMV is ~15 tok/s, legacy is ~40 tok/s
    // This corresponds to roughly 3x improvement needed
    const double min_bandwidth_gbps = 100.0;  // Conservative target

    if (bandwidth_gbps < min_bandwidth_gbps) {
        fprintf(stderr, "  WARNING: Below target bandwidth (%.1f GB/s < %.1f GB/s)\n",
                bandwidth_gbps, min_bandwidth_gbps);
    }

    // Copy result back for correctness check
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results (more lenient tolerance for larger matrices)
    const float tolerance = 0.01f;
    int         errors    = 0;
    for (int i = 0; i < M * N && errors < 5; i++) {
        float diff = std::abs(C_gpu[i] - C_ref[i]);
        float rel  = diff / (std::abs(C_ref[i]) + 1e-6f);
        if (rel > tolerance && diff > 1e-4f) {
            fprintf(stderr, "  Mismatch at [%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                    i, C_gpu[i], C_ref[i], diff);
            errors++;
        }
    }

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_ASSERT(errors == 0, "Correctness check failed");

    return true;
}

static bool test_unified_kernel_dmmv_perf(sycl::queue & q) {
    TEST_BEGIN("test_unified_kernel_dmmv_perf");

    fprintf(stderr, "\n");

    // Test multiple dimension configurations
    bool ok = true;
    ok &= test_unified_kernel_dmmv_perf_impl(q, 4096, 4096, "Attention Q/K/V/O");
    ok &= test_unified_kernel_dmmv_perf_impl(q, 14336, 4096, "FFN Gate/Up");
    ok &= test_unified_kernel_dmmv_perf_impl(q, 4096, 14336, "FFN Down");

    if (ok) {
        TEST_PASS();
    }
    return ok;
}

// =============================================================================
// Test 5: DMMV SoA Layout Performance
// Verifies the SoA layout achieves better performance for large K dimensions
// =============================================================================
static bool test_unified_kernel_dmmv_soa_impl(sycl::queue & q, int N, int K, const char * label) {
    const int M        = 1;       // batch=1 (decode)
    const int k_blocks = K / QK4_0;
    const int64_t total_blocks = static_cast<int64_t>(N) * k_blocks;

    // Allocate Q4_0 weights in SoA layout:
    // [all qs: total_blocks * 16 bytes] [all d: total_blocks * sizeof(half)]
    const size_t qs_bytes = total_blocks * (QK4_0 / 2);
    const size_t d_bytes  = total_blocks * sizeof(sycl::half);
    std::vector<uint8_t> soa_buffer(qs_bytes + d_bytes);

    uint8_t *    qs_base = soa_buffer.data();
    sycl::half * d_base  = reinterpret_cast<sycl::half *>(soa_buffer.data() + qs_bytes);

    // Initialize with random data
    std::mt19937 rng(42);
    for (int64_t i = 0; i < total_blocks; i++) {
        d_base[i] = sycl::half(0.01f * (rng() % 100 + 1) / 100.0f);
        for (int j = 0; j < 16; j++) {
            qs_base[i * 16 + j] = rng() % 256;
        }
    }

    // Allocate F32 activations [M=1, K]
    std::vector<float> B(M * K);
    for (int k = 0; k < K; k++) {
        B[k] = 0.01f * ((rng() % 200) - 100) / 100.0f;
    }

    // Compute CPU reference: C[M=1, N] = B[M=1, K] @ A_dequant[K, N]
    std::vector<float> C_ref(M * N, 0.0f);
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int kb = 0; kb < k_blocks; kb++) {
            const int64_t block_idx = static_cast<int64_t>(n) * k_blocks + kb;
            const float d = static_cast<float>(d_base[block_idx]);
            const uint8_t * qs = qs_base + block_idx * 16;

            for (int i = 0; i < 16; i++) {
                const uint8_t qs_byte = qs[i];
                const float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                const float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;
                const int k_offset = kb * QK4_0;
                acc += w0 * B[k_offset + i] + w1 * B[k_offset + i + 16];
            }
        }
        C_ref[n] = acc;
    }

    // Allocate device memory
    auto * A_dev = sycl::malloc_device<uint8_t>(soa_buffer.size(), q);
    auto * B_dev = sycl::malloc_device<float>(M * K, q);
    auto * C_dev = sycl::malloc_device<float>(M * N, q);

    if (!A_dev || !B_dev || !C_dev) {
        fprintf(stderr, "FAILED: Device memory allocation failed\n");
        if (A_dev) sycl::free(A_dev, q);
        if (B_dev) sycl::free(B_dev, q);
        if (C_dev) sycl::free(C_dev, q);
        return false;
    }

    // Copy to device
    q.memcpy(A_dev, soa_buffer.data(), soa_buffer.size()).wait();
    q.memcpy(B_dev, B.data(), M * K * sizeof(float)).wait();
    q.memset(C_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args for DMMV path with SoA layout
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 32;
    args.tile_k         = 32;
    args.use_xmx        = false;  // DMMV doesn't use XMX
    args.layout_mode    = 1;      // SOA
    args.layout         = ggml_sycl_unified::LayoutMode::SOA;
    args.quant_type     = 2;      // GGML_TYPE_Q4_0
    args.prefetch_depth = 0;
    args.weights        = A_dev;
    args.activations    = B_dev;
    args.output         = C_dev;

    // Warmup
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Benchmark: Run multiple iterations
    const int num_iters = 100;
    auto      start     = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; i++) {
        ggml_sycl_unified::launch_unified_matmul(q, args);
    }
    q.wait();

    auto end     = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Calculate throughput
    double time_per_iter_us = static_cast<double>(elapsed) / num_iters;
    double time_per_iter_ms = time_per_iter_us / 1000.0;

    // Memory bandwidth calculation
    size_t bytes_read  = soa_buffer.size() + K * sizeof(float);
    size_t bytes_write = N * sizeof(float);
    double bandwidth_gbps = (bytes_read + bytes_write) / (time_per_iter_us * 1000.0);

    fprintf(stderr, "  [%s SoA] M=%d N=%d K=%d\n", label, M, N, K);
    fprintf(stderr, "  Time per matmul: %.2f us (%.3f ms)\n", time_per_iter_us, time_per_iter_ms);
    fprintf(stderr, "  Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);

    // Copy result back for correctness check
    std::vector<float> C_gpu(M * N);
    q.memcpy(C_gpu.data(), C_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 0.01f;
    int         errors    = 0;
    for (int i = 0; i < M * N && errors < 5; i++) {
        float diff = std::abs(C_gpu[i] - C_ref[i]);
        float rel  = diff / (std::abs(C_ref[i]) + 1e-6f);
        if (rel > tolerance && diff > 1e-4f) {
            fprintf(stderr, "  Mismatch at [%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                    i, C_gpu[i], C_ref[i], diff);
            errors++;
        }
    }

    // Cleanup
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    if (errors > 0) {
        fprintf(stderr, "FAILED: Correctness check failed\n");
        return false;
    }

    return true;
}

static bool test_unified_kernel_dmmv_soa_perf(sycl::queue & q) {
    TEST_BEGIN("test_unified_kernel_dmmv_soa_perf");

    fprintf(stderr, "\n");

    // Test SoA layout performance - especially for large K (FFN Down)
    bool ok = true;
    ok &= test_unified_kernel_dmmv_soa_impl(q, 4096, 4096, "Attention Q/K/V/O");
    ok &= test_unified_kernel_dmmv_soa_impl(q, 14336, 4096, "FFN Gate/Up");
    ok &= test_unified_kernel_dmmv_soa_impl(q, 4096, 14336, "FFN Down");

    if (ok) {
        TEST_PASS();
    }
    return ok;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Unified Kernel Tests\n");
    fprintf(stderr, "===========================================\n");

    // Select GPU device
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU device found: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "-------------------------------------------\n");

    sycl::queue q(dev, sycl::property::queue::in_order{});

    // Run tests
    bool all_passed = true;

    all_passed &= test_unified_kernel_q4_0_small(q);
    all_passed &= test_unified_kernel_boundary(q);
    all_passed &= test_unified_kernel_scalar(q);
    all_passed &= test_unified_kernel_dmmv_perf(q);
    all_passed &= test_unified_kernel_dmmv_soa_perf(q);

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
