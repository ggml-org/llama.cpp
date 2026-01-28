//
// Test: ESIMD SLM Prefetch Optimization (Task llama.cpp-attk)
//
// TDD tests for LSC prefetch optimization with cache hints in ESIMD dpas kernels.
// Tests verify:
// 1. Prefetch distance configuration works correctly
// 2. Bounds checking prevents OOB prefetch
// 3. Cache hints are correctly applied (streaming for weights, cached for activations)
// 4. Named barriers synchronize correctly with prefetch
// 5. SLM capacity is not exceeded
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
// Note: UNIFIED_KERNEL_TEST_STANDALONE is defined via CMakeLists.txt to provide
// stub implementations for common.cpp symbols needed by unified-kernel.cpp
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

// CPU reference matmul following GGML convention:
// dst[m,n] = sum_k(weights[n,k] * activations[m,k])
// Where:
// - weights [N, K/32 blocks] - Q4_0 quantized
// - activations [M, K] - float row-major
// - output [M, N] - float row-major
static void matmul_q4_0_f32_cpu_reference(const block_q4_0_test * weights,
                                          const float *           activations,
                                          float *                 output,
                                          int                     M,
                                          int                     N,
                                          int                     K) {
    const int k_blocks = K / QK4_0;

    // Temporary storage for dequantized weight row
    std::vector<float> w_row(K);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            // Dequantize weight row n
            for (int kb = 0; kb < k_blocks; kb++) {
                dequantize_q4_0_block_test(weights[n * k_blocks + kb], w_row.data() + kb * QK4_0);
            }

            // Dot product: output[m,n] = sum_k(weights[n,k] * activations[m,k])
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += w_row[k] * activations[m * K + k];
            }
            output[m * N + n] = acc;
        }
    }
}

// =============================================================================
// Test 1: Prefetch Distance 1 - Verify Correct Output with Minimal Prefetch
// =============================================================================
static bool test_prefetch_distance_1(sycl::queue & q) {
    TEST_BEGIN("test_prefetch_distance_1");

    // Check if prefetch configuration API exists
#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // Check if get_prefetch_distance function exists in the namespace
    // This tests that the prefetch distance configuration was implemented
    int prefetch_dist = ggml_sycl_unified::get_prefetch_distance();
    (void)prefetch_dist;  // Use it to avoid unused warning

    // Test with small matrix to verify basic functionality with prefetch=1
    const int M = 16;
    const int N = 32;
    const int K = 128;  // 4 K-tiles with K_PER_DPAS=16, or 4 Q4_0 blocks

    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights [N, K]
    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(42);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations [M, K]
    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    // Allocate device memory
    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args with prefetch_distance = 1
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;  // Use ESIMD path
    args.layout_mode    = 0;
    args.quant_type     = 2;     // GGML_TYPE_Q4_0
    args.prefetch_depth = 1;     // Prefetch 1 tile ahead
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    // Launch kernel
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 5e-2f;  // Higher tolerance for FP16 intermediate values
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance,
                         (std::string("Mismatch at index ") + std::to_string(i)).c_str());
    }

    // Cleanup
    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Prefetch Distance 4 - Verify Correct Output with Aggressive Prefetch
// =============================================================================
static bool test_prefetch_distance_4(sycl::queue & q) {
    TEST_BEGIN("test_prefetch_distance_4");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // Test with larger K to have multiple prefetch opportunities
    const int M = 16;
    const int N = 32;
    const int K = 512;  // 16 K-tiles with K_PER_DPAS=32, plenty for prefetch distance 4

    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights [N, K]
    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(123);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f * (rng() % 10 + 1));
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations [M, K]
    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    // Allocate device memory
    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args with prefetch_distance = 4
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;     // GGML_TYPE_Q4_0
    args.prefetch_depth = 4;     // Prefetch 4 tiles ahead
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    // Launch kernel
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    // Verify results
    const float tolerance = 5e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance,
                         (std::string("Mismatch at index ") + std::to_string(i)).c_str());
    }

    // Cleanup
    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Prefetch Boundary Check - Verify Last K-Tile Doesn't Prefetch OOB
// =============================================================================
static bool test_prefetch_boundary_check(sycl::queue & q) {
    TEST_BEGIN("test_prefetch_boundary_check");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // Use K dimension that creates exactly N K-tiles where N-1 + prefetch_distance >= N
    // This tests the boundary check: if (k_tile + PREFETCH_DISTANCE < K_tiles) prefetch(...)
    // With K=64 (2 K-tiles of 32 each) and prefetch_distance=2, we should NOT prefetch
    // on the first K-tile because 0+2 >= 2

    const int M = 8;
    const int N = 16;
    const int K = 64;  // Only 2 K-tiles when TILE_K=32

    const int k_blocks = K / QK4_0;

    // Allocate and initialize Q4_0 weights [N, K]
    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(456);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    // Allocate F32 activations [M, K]
    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    // Compute CPU reference
    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    // Allocate device memory
    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    // Copy to device
    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    // Setup kernel args - prefetch_distance >= K_tiles should be handled gracefully
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;
    args.prefetch_depth = 4;  // Larger than number of K-tiles (2)
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    // Launch kernel - should NOT crash or access out of bounds
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    // Copy result back
    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    // Verify results - correctness proves no OOB occurred
    const float tolerance = 5e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance,
                         (std::string("Mismatch at index ") + std::to_string(i)).c_str());
    }

    // Cleanup
    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Cache Hint for Weights - Verify Streaming Hint is Used
// =============================================================================
static bool test_cache_hint_weights_streaming(sycl::queue & q) {
    TEST_BEGIN("test_cache_hint_weights_streaming");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // This test verifies that weights use streaming/uncached hints
    // Weights are used once per output element and then evicted.
    //
    // Design-level verification: The implementation should use:
    // esimd::lsc_prefetch<..., cache_hint::streaming, cache_hint::uncached>(weight_ptr)
    //
    // We verify this indirectly through correctness - if wrong hints cause
    // data corruption, the test will fail.

    // Test that the cache hint configuration API exists
    ggml_sycl_unified::CacheHintPolicy weights_policy = ggml_sycl_unified::get_weights_cache_hint();
    TEST_ASSERT(weights_policy == ggml_sycl_unified::CacheHintPolicy::STREAMING,
                "Weights should use STREAMING cache hint policy");

    // Run a basic correctness test to ensure hints don't break functionality
    const int M = 16;
    const int N = 32;
    const int K = 128;

    const int k_blocks = K / QK4_0;

    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(789);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;
    args.prefetch_depth = 2;
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    const float tolerance = 5e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance, "Mismatch with streaming hints");
    }

    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: Cache Hint for Activations - Verify Cached Hint is Used
// =============================================================================
static bool test_cache_hint_activations_cached(sycl::queue & q) {
    TEST_BEGIN("test_cache_hint_activations_cached");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // This test verifies that activations use cached/cached hints
    // Activations are reused across N columns, so they benefit from caching.
    //
    // Design-level verification: The implementation should use:
    // esimd::lsc_prefetch<..., cache_hint::cached, cache_hint::cached>(act_ptr)

    // Test that the cache hint configuration API exists
    ggml_sycl_unified::CacheHintPolicy acts_policy = ggml_sycl_unified::get_activations_cache_hint();
    TEST_ASSERT(acts_policy == ggml_sycl_unified::CacheHintPolicy::CACHED,
                "Activations should use CACHED cache hint policy");

    // Run a basic correctness test to ensure hints don't break functionality
    const int M = 16;
    const int N = 32;
    const int K = 128;

    const int k_blocks = K / QK4_0;

    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(321);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 8;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;
    args.prefetch_depth = 2;
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    const float tolerance = 5e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance, "Mismatch with cached hints");
    }

    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: SLM Capacity Max Tile - Verify SLM Usage < 64KB
// =============================================================================
static bool test_slm_capacity_max_tile(sycl::queue & q) {
    TEST_BEGIN("test_slm_capacity_max_tile");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // Verify that maximum prefetch depth doesn't exceed SLM capacity (64KB)
    //
    // SLM usage calculation for cooperative kernel with prefetch:
    // - Weights buffer: TILE_N * K_TILE * sizeof(half) = 16 * 16 * 2 = 512 bytes per buffer
    // - Activations buffer: TILE_M * K_TILE * sizeof(half) = 16 * 16 * 2 = 512 bytes per buffer
    // - With double-buffering: 2 * (512 + 512) = 2048 bytes
    // - With max prefetch depth 4: Still only uses 2 buffers (current + next)
    //
    // This should never exceed 64KB (65536 bytes)

    size_t max_slm_usage = ggml_sycl_unified::get_max_slm_usage_with_prefetch();
    const size_t MAX_SLM = 65536;  // 64KB

    TEST_ASSERT(max_slm_usage <= MAX_SLM,
                (std::string("SLM usage ") + std::to_string(max_slm_usage) +
                 " exceeds 64KB limit").c_str());

    // Also verify via a kernel launch with large problem
    const int M = 64;
    const int N = 64;
    const int K = 256;

    const int k_blocks = K / QK4_0;

    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(555);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 16;  // Larger tiles
    args.tile_n         = 32;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;
    args.prefetch_depth = 4;  // Maximum prefetch
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    // Launch kernel - should not fail due to SLM overflow
    ggml_sycl_unified::launch_unified_matmul(q, args);
    q.wait();

    std::vector<float> output_gpu(M * N);
    q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

    const float tolerance = 5e-2f;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(output_gpu[i] - output_ref[i]) > tolerance) {
            correct = false;
            break;
        }
    }
    TEST_ASSERT(correct, "Kernel with max prefetch produced incorrect results");

    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: Named Barrier All Work-Items - Verify Barrier Sequence
// =============================================================================
static bool test_named_barrier_all_wis(sycl::queue & q) {
    TEST_BEGIN("test_named_barrier_all_wis");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // This test verifies that all work-items execute the barrier sequence correctly
    // with prefetch enabled. The prefetch-load-barrier pattern must synchronize
    // all work-items before compute.
    //
    // Pattern:
    // 1. Prefetch future tile (non-blocking)
    // 2. Load current tile to SLM
    // 3. Named barrier - all work-items sync
    // 4. Compute on loaded data
    //
    // If barriers are incorrect, race conditions will cause data corruption.

    // Use larger work-group to stress-test barrier
    const int M = 32;  // COOP_WG_M = 16, so 2 work-groups in M
    const int N = 32;  // COOP_WG_N = 16, so 2 work-groups in N
    const int K = 256;

    const int k_blocks = K / QK4_0;

    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(777);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();
    q.memset(output_dev, 0, M * N * sizeof(float)).wait();

    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.tile_m         = 16;
    args.tile_n         = 16;
    args.tile_k         = 32;
    args.use_xmx        = true;
    args.layout_mode    = 0;
    args.quant_type     = 2;
    args.prefetch_depth = 2;  // Enable prefetch with barriers
    args.weights        = weights_dev;
    args.activations    = activations_dev;
    args.output         = output_dev;

    // Run multiple times to catch race conditions
    for (int iter = 0; iter < 5; iter++) {
        q.memset(output_dev, 0, M * N * sizeof(float)).wait();
        ggml_sycl_unified::launch_unified_matmul(q, args);
        q.wait();

        std::vector<float> output_gpu(M * N);
        q.memcpy(output_gpu.data(), output_dev, M * N * sizeof(float)).wait();

        const float tolerance = 5e-2f;
        for (int i = 0; i < M * N; i++) {
            TEST_ASSERT_NEAR(output_gpu[i], output_ref[i], tolerance,
                             (std::string("Mismatch at iteration ") + std::to_string(iter) +
                              " index " + std::to_string(i)).c_str());
        }
    }

    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 8: Prefetch Improves Stalls - Design-Level Verification
// =============================================================================
static bool test_prefetch_improves_stalls(sycl::queue & q) {
    TEST_BEGIN("test_prefetch_improves_stalls");

#ifndef GGML_SYCL_ESIMD_AVAILABLE
    TEST_SKIP("ESIMD not available");
    return true;
#endif

    // This is a design-level test. We verify:
    // 1. Prefetch distance > 0 doesn't break correctness
    // 2. The prefetch mechanism is properly integrated
    //
    // Actual performance improvement (reduced memory stalls) would need
    // VTune profiling which is outside the scope of unit tests.

    const int M = 32;
    const int N = 64;
    const int K = 512;

    const int k_blocks = K / QK4_0;

    std::vector<block_q4_0_test> weights(N * k_blocks);
    std::mt19937 rng(999);
    for (auto & block : weights) {
        block.d = sycl::half(0.1f);
        for (int i = 0; i < 16; i++) {
            block.qs[i] = rng() % 256;
        }
    }

    std::vector<float> activations(M * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        activations[i] = dist(rng);
    }

    std::vector<float> output_ref(M * N);
    matmul_q4_0_f32_cpu_reference(weights.data(), activations.data(), output_ref.data(), M, N, K);

    auto * weights_dev = sycl::malloc_device<block_q4_0_test>(N * k_blocks, q);
    auto * activations_dev = sycl::malloc_device<float>(M * K, q);
    auto * output_dev = sycl::malloc_device<float>(M * N, q);

    TEST_ASSERT(weights_dev && activations_dev && output_dev, "Device memory allocation failed");

    q.memcpy(weights_dev, weights.data(), N * k_blocks * sizeof(block_q4_0_test)).wait();
    q.memcpy(activations_dev, activations.data(), M * K * sizeof(float)).wait();

    // Compare outputs with different prefetch distances
    std::vector<float> output_no_prefetch(M * N);
    std::vector<float> output_with_prefetch(M * N);

    // Run without prefetch
    {
        q.memset(output_dev, 0, M * N * sizeof(float)).wait();

        ggml_sycl_unified::UnifiedKernelArgs args;
        args.M              = M;
        args.N              = N;
        args.K              = K;
        args.tile_m         = 8;
        args.tile_n         = 16;
        args.tile_k         = 32;
        args.use_xmx        = true;
        args.layout_mode    = 0;
        args.quant_type     = 2;
        args.prefetch_depth = 0;  // No prefetch
        args.weights        = weights_dev;
        args.activations    = activations_dev;
        args.output         = output_dev;

        ggml_sycl_unified::launch_unified_matmul(q, args);
        q.wait();

        q.memcpy(output_no_prefetch.data(), output_dev, M * N * sizeof(float)).wait();
    }

    // Run with prefetch
    {
        q.memset(output_dev, 0, M * N * sizeof(float)).wait();

        ggml_sycl_unified::UnifiedKernelArgs args;
        args.M              = M;
        args.N              = N;
        args.K              = K;
        args.tile_m         = 8;
        args.tile_n         = 16;
        args.tile_k         = 32;
        args.use_xmx        = true;
        args.layout_mode    = 0;
        args.quant_type     = 2;
        args.prefetch_depth = 2;  // With prefetch
        args.weights        = weights_dev;
        args.activations    = activations_dev;
        args.output         = output_dev;

        ggml_sycl_unified::launch_unified_matmul(q, args);
        q.wait();

        q.memcpy(output_with_prefetch.data(), output_dev, M * N * sizeof(float)).wait();
    }

    // Both versions should produce the same correct result
    const float tolerance = 5e-2f;
    for (int i = 0; i < M * N; i++) {
        TEST_ASSERT_NEAR(output_no_prefetch[i], output_ref[i], tolerance,
                         "No-prefetch version incorrect");
        TEST_ASSERT_NEAR(output_with_prefetch[i], output_ref[i], tolerance,
                         "With-prefetch version incorrect");
        // Also verify both versions match each other
        TEST_ASSERT_NEAR(output_no_prefetch[i], output_with_prefetch[i], 1e-4f,
                         "Prefetch vs non-prefetch mismatch");
    }

    sycl::free(weights_dev, q);
    sycl::free(activations_dev, q);
    sycl::free(output_dev, q);

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
    fprintf(stderr, "ESIMD SLM Prefetch Optimization Tests\n");
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

    all_passed &= test_prefetch_distance_1(q);
    all_passed &= test_prefetch_distance_4(q);
    all_passed &= test_prefetch_boundary_check(q);
    all_passed &= test_cache_hint_weights_streaming(q);
    all_passed &= test_cache_hint_activations_cached(q);
    all_passed &= test_slm_capacity_max_tile(q);
    all_passed &= test_named_barrier_all_wis(q);
    all_passed &= test_prefetch_improves_stalls(q);

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
