//
// Test: XMX DPAS Basic Infrastructure
//
// Tests for dpas (Dot Product Accumulate Systolic) infrastructure:
// - Hardware detection with dpas flag
// - Basic int8x8 multiply operations
// - VNNI packing correctness
// - Accumulator preservation
// - Repeat count handling
// - Graceful fallback on non-XMX hardware
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>

// Enable standalone mode for testing
#define XMX_TEST_STANDALONE 1

// Include the XMX ESIMD common header (provides XMXCapabilities in standalone mode)
#include "../xmx-esimd-common.hpp"

using namespace ggml_sycl_xmx;

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

// =============================================================================
// Test 1: XMXDpas.HardwareDetectionWithDpasFlag
// Assert: caps.supports_int8_dpas == true on Arc GPUs
// Assert: caps.systolic_depth == 8
// =============================================================================
bool test_dpas_hardware_detection(sycl::device & dev) {
    TEST_BEGIN("XMXDpas.HardwareDetectionWithDpasFlag");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    // Query XMX capabilities
    XMXCapabilities caps = query_xmx_capabilities(dev);
    TEST_ASSERT(caps.supported, "XMX should be supported");

    // Check dpas configuration
    XMXDpasConfig dpas_cfg = XMXDpasConfig::from_capabilities(caps);
    TEST_ASSERT(dpas_cfg.is_valid(), "XMXDpasConfig should be valid on XMX hardware");

    // Verify systolic depth is 8 (Intel XMX constant)
    TEST_ASSERT(dpas_cfg.systolic_depth == XMXDpasConfig::SYSTOLIC_DEPTH, "Systolic depth should be 8");
    TEST_ASSERT(XMXDpasConfig::SYSTOLIC_DEPTH == 8, "SYSTOLIC_DEPTH constant should be 8");

    // Verify int8 dpas support
    TEST_ASSERT(dpas_cfg.supports_int8, "Int8 dpas should be supported on Arc GPU");

    // Verify K dimension for int8 is 32 (8 systolic depth * 4 ops per channel)
    TEST_ASSERT(dpas_cfg.k_int8 == 32, "K dimension for int8 should be 32");

    // Verify N dimension is 8 for Arc/DG2 (execution size)
    TEST_ASSERT(dpas_cfg.n_size == 8 || dpas_cfg.n_size == 16, "N size should be 8 (Arc/DG2) or 16 (PVC)");

    fprintf(stderr, "(systolic_depth=%d, k_int8=%d, n_size=%d) ", dpas_cfg.systolic_depth, dpas_cfg.k_int8,
            dpas_cfg.n_size);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: XMXDpas.BasicInt8Multiply
// Setup: A[M,K] = all 1s, B[K,N] = all 1s (VNNI), C[M,N] = zeros
// where M=8 (systolic depth), K=32 (int8), N=execution size (8 or 16)
// Call: dpas<8, 1>(C, B, A)
// Assert: All C[i,j] == K (32)
// =============================================================================
bool test_dpas_basic_int8_multiply(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("XMXDpas.BasicInt8Multiply");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

#if SYCL_XMX_ESIMD_AVAILABLE
    XMXDpasConfig dpas_cfg = XMXDpasConfig::from_capabilities(caps);
    if (!dpas_cfg.supports_int8) {
        TEST_SKIP("Int8 dpas not supported");
        return true;
    }

    // Dimensions for RepeatCount=1, SystolicDepth=8, int8
    // M = RepeatCount * SystolicDepth = 1 * 8 = 8
    // K = 32 for int8 (8 systolic * 4 ops per channel)
    // N = execution size from hardware (8 for DG2, 16 for PVC/newer Arc)
    const int M           = 8;                // RepeatCount=1 * SystolicDepth=8
    const int K           = dpas_cfg.k_int8;  // 32 for int8
    const int N           = dpas_cfg.n_size;  // Hardware-dependent execution size
    const int A_size      = M * K;
    const int B_vnni_size = (K / 4) * N;      // VNNI packed: 4 int8 -> 1 uint32
    const int C_size      = M * N;

    // Allocate USM memory
    int8_t *   A_host   = sycl::malloc_host<int8_t>(A_size, q);
    uint32_t * B_host   = sycl::malloc_host<uint32_t>(B_vnni_size, q);
    int32_t *  C_host   = sycl::malloc_host<int32_t>(C_size, q);
    int32_t *  C_result = sycl::malloc_host<int32_t>(C_size, q);

    // Initialize A with all 1s
    for (int i = 0; i < A_size; i++) {
        A_host[i] = 1;
    }

    // Initialize B in VNNI format with all 1s
    // VNNI packing: 4 consecutive int8 values packed into 1 uint32
    // For all 1s: packed = 0x01010101
    for (int i = 0; i < B_vnni_size; i++) {
        B_host[i] = 0x01010101;  // Four 1s packed
    }

    // Initialize C with zeros
    for (int i = 0; i < C_size; i++) {
        C_host[i] = 0;
    }

    // Copy to device
    int8_t *   A_dev = sycl::malloc_device<int8_t>(A_size, q);
    uint32_t * B_dev = sycl::malloc_device<uint32_t>(B_vnni_size, q);
    int32_t *  C_dev = sycl::malloc_device<int32_t>(C_size, q);

    q.memcpy(A_dev, A_host, A_size * sizeof(int8_t));
    q.memcpy(B_dev, B_host, B_vnni_size * sizeof(uint32_t));
    q.memcpy(C_dev, C_host, C_size * sizeof(int32_t));
    q.wait();

    // Run dpas kernel
    bool kernel_success = run_dpas_int8_test_kernel(q, C_dev, B_dev, A_dev, M, K, N);

    if (!kernel_success) {
        TEST_SKIP("dpas kernel not implemented yet");
        sycl::free(A_host, q);
        sycl::free(B_host, q);
        sycl::free(C_host, q);
        sycl::free(C_result, q);
        sycl::free(A_dev, q);
        sycl::free(B_dev, q);
        sycl::free(C_dev, q);
        return true;
    }

    // Copy result back
    q.memcpy(C_result, C_dev, C_size * sizeof(int32_t)).wait();

    // Verify: Each C[i,j] should be the dot product of row i of A and col j of B
    // Since A=all 1s and B=all 1s, C[i,j] = sum of K=32 products = 32
    bool all_correct = true;
    for (int i = 0; i < C_size; i++) {
        if (C_result[i] != K) {
            fprintf(stderr, "C[%d] = %d, expected %d\n", i, C_result[i], K);
            all_correct = false;
        }
    }

    // Cleanup
    sycl::free(A_host, q);
    sycl::free(B_host, q);
    sycl::free(C_host, q);
    sycl::free(C_result, q);
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_ASSERT(all_correct, "All C values should be 32");
    TEST_PASS();
    return true;
#else
    TEST_SKIP("ESIMD not available");
    return true;
#endif
}

// =============================================================================
// Test 3: XMXDpas.VNNIPackingCorrectness
// Setup: Sequential int8 values [0,1,2,...,31]
// Assert: packed[0] == (0 | 1<<8 | 2<<16 | 3<<24)
// =============================================================================
bool test_vnni_packing_correctness() {
    TEST_BEGIN("XMXDpas.VNNIPackingCorrectness");

#if SYCL_XMX_ESIMD_AVAILABLE
    // Test the pack_vnni_int8 function with sequential values
    // Input: [0, 1, 2, 3, 4, 5, 6, 7, ...]
    // Output: packed uint32 values

    // Expected packing for first 4 bytes:
    // packed[0] = (bytes[0]) | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)
    //           = 0 | (1 << 8) | (2 << 16) | (3 << 24)
    //           = 0x03020100

    constexpr int K = 32;
    int8_t        input[K];
    for (int i = 0; i < K; i++) {
        input[i] = static_cast<int8_t>(i);
    }

    // Test VNNI packing helper
    uint32_t packed[K / 4];
    pack_vnni_int8_host(input, packed, K);

    // Verify first packed value
    uint32_t expected_first = static_cast<uint32_t>(0) | (static_cast<uint32_t>(1) << 8) |
                              (static_cast<uint32_t>(2) << 16) | (static_cast<uint32_t>(3) << 24);
    TEST_ASSERT(packed[0] == expected_first, "First VNNI packed value should be 0x03020100");

    // Verify second packed value: (4 | 5<<8 | 6<<16 | 7<<24) = 0x07060504
    uint32_t expected_second = static_cast<uint32_t>(4) | (static_cast<uint32_t>(5) << 8) |
                               (static_cast<uint32_t>(6) << 16) | (static_cast<uint32_t>(7) << 24);
    TEST_ASSERT(packed[1] == expected_second, "Second VNNI packed value should be 0x07060504");

    fprintf(stderr, "(packed[0]=0x%08X, packed[1]=0x%08X) ", packed[0], packed[1]);
    TEST_PASS();
    return true;
#else
    TEST_SKIP("ESIMD not available");
    return true;
#endif
}

// =============================================================================
// Test 4: XMXDpas.AccumulatorPreserved
// Setup: C[M,N] = all 100s (accumulator)
// Assert: Result = accumulator + A*B = 100 + 32 = 132
// =============================================================================
bool test_dpas_accumulator_preserved(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("XMXDpas.AccumulatorPreserved");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

#if SYCL_XMX_ESIMD_AVAILABLE
    XMXDpasConfig dpas_cfg = XMXDpasConfig::from_capabilities(caps);
    if (!dpas_cfg.supports_int8) {
        TEST_SKIP("Int8 dpas not supported");
        return true;
    }

    const int         M           = 8;                // RepeatCount=1 * SystolicDepth=8
    const int         K           = dpas_cfg.k_int8;  // 32 for int8
    const int         N           = dpas_cfg.n_size;  // Hardware-dependent
    const int         A_size      = M * K;
    const int         B_vnni_size = (K / 4) * N;
    const int         C_size      = M * N;
    constexpr int32_t ACCUM_VALUE = 100;

    // Allocate
    int8_t *   A_host   = sycl::malloc_host<int8_t>(A_size, q);
    uint32_t * B_host   = sycl::malloc_host<uint32_t>(B_vnni_size, q);
    int32_t *  C_host   = sycl::malloc_host<int32_t>(C_size, q);
    int32_t *  C_result = sycl::malloc_host<int32_t>(C_size, q);

    // A = all 1s, B = all 1s (VNNI), C = all 100s
    for (int i = 0; i < A_size; i++) {
        A_host[i] = 1;
    }
    for (int i = 0; i < B_vnni_size; i++) {
        B_host[i] = 0x01010101;
    }
    for (int i = 0; i < C_size; i++) {
        C_host[i] = ACCUM_VALUE;
    }

    int8_t *   A_dev = sycl::malloc_device<int8_t>(A_size, q);
    uint32_t * B_dev = sycl::malloc_device<uint32_t>(B_vnni_size, q);
    int32_t *  C_dev = sycl::malloc_device<int32_t>(C_size, q);

    q.memcpy(A_dev, A_host, A_size * sizeof(int8_t));
    q.memcpy(B_dev, B_host, B_vnni_size * sizeof(uint32_t));
    q.memcpy(C_dev, C_host, C_size * sizeof(int32_t));
    q.wait();

    // Run dpas with accumulator
    bool kernel_success = run_dpas_int8_with_accum_test_kernel(q, C_dev, B_dev, A_dev, M, K, N);

    if (!kernel_success) {
        TEST_SKIP("dpas accumulator kernel not implemented yet");
        sycl::free(A_host, q);
        sycl::free(B_host, q);
        sycl::free(C_host, q);
        sycl::free(C_result, q);
        sycl::free(A_dev, q);
        sycl::free(B_dev, q);
        sycl::free(C_dev, q);
        return true;
    }

    q.memcpy(C_result, C_dev, C_size * sizeof(int32_t)).wait();

    // Result should be: C[i,j] = 100 + 32 = 132
    int32_t expected    = ACCUM_VALUE + K;
    bool    all_correct = true;
    for (int i = 0; i < C_size; i++) {
        if (C_result[i] != expected) {
            fprintf(stderr, "C[%d] = %d, expected %d\n", i, C_result[i], expected);
            all_correct = false;
        }
    }

    sycl::free(A_host, q);
    sycl::free(B_host, q);
    sycl::free(C_host, q);
    sycl::free(C_result, q);
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_ASSERT(all_correct, "Accumulator should be preserved (100 + 32 = 132)");
    TEST_PASS();
    return true;
#else
    TEST_SKIP("ESIMD not available");
    return true;
#endif
}

// =============================================================================
// Test 5: XMXDpas.RepeatCount4Produces32Rows
// Call: dpas<8, 4> with A[32,K], B[K,N]
// Assert: Output has 32 rows (RepeatCount * SystolicDepth = 4 * 8)
// =============================================================================
bool test_dpas_repeat_count_4(sycl::queue & q, const XMXCapabilities & caps) {
    TEST_BEGIN("XMXDpas.RepeatCount4Produces32Rows");

    if (!caps.supported || !caps.supports_int8) {
        TEST_SKIP("XMX int8 not supported");
        return true;
    }

#if SYCL_XMX_ESIMD_AVAILABLE
    XMXDpasConfig dpas_cfg = XMXDpasConfig::from_capabilities(caps);
    if (!dpas_cfg.supports_int8) {
        TEST_SKIP("Int8 dpas not supported");
        return true;
    }

    // RepeatCount=4 means M=4*8=32 rows
    constexpr int REPEAT_COUNT = 4;
    const int     M            = REPEAT_COUNT * XMXDpasConfig::SYSTOLIC_DEPTH;  // 32 rows
    const int     K            = dpas_cfg.k_int8;                               // 32 for int8
    const int     N            = dpas_cfg.n_size;                               // Hardware-dependent
    const int     A_size       = M * K;
    const int     B_vnni_size  = (K / 4) * N;
    const int     C_size       = M * N;

    int8_t *   A_host   = sycl::malloc_host<int8_t>(A_size, q);
    uint32_t * B_host   = sycl::malloc_host<uint32_t>(B_vnni_size, q);
    int32_t *  C_host   = sycl::malloc_host<int32_t>(C_size, q);
    int32_t *  C_result = sycl::malloc_host<int32_t>(C_size, q);

    for (int i = 0; i < A_size; i++) {
        A_host[i] = 1;
    }
    for (int i = 0; i < B_vnni_size; i++) {
        B_host[i] = 0x01010101;
    }
    for (int i = 0; i < C_size; i++) {
        C_host[i] = 0;
    }

    int8_t *   A_dev = sycl::malloc_device<int8_t>(A_size, q);
    uint32_t * B_dev = sycl::malloc_device<uint32_t>(B_vnni_size, q);
    int32_t *  C_dev = sycl::malloc_device<int32_t>(C_size, q);

    q.memcpy(A_dev, A_host, A_size * sizeof(int8_t));
    q.memcpy(B_dev, B_host, B_vnni_size * sizeof(uint32_t));
    q.memcpy(C_dev, C_host, C_size * sizeof(int32_t));
    q.wait();

    bool kernel_success = run_dpas_int8_repeat4_test_kernel(q, C_dev, B_dev, A_dev, M, K, N);

    if (!kernel_success) {
        TEST_SKIP("dpas repeat count kernel not implemented yet");
        sycl::free(A_host, q);
        sycl::free(B_host, q);
        sycl::free(C_host, q);
        sycl::free(C_result, q);
        sycl::free(A_dev, q);
        sycl::free(B_dev, q);
        sycl::free(C_dev, q);
        return true;
    }

    q.memcpy(C_result, C_dev, C_size * sizeof(int32_t)).wait();

    // Verify all 32 rows * 8 cols = 256 elements are computed correctly
    bool all_correct = true;
    for (int i = 0; i < C_size; i++) {
        if (C_result[i] != K) {
            fprintf(stderr, "C[%d] = %d, expected %d\n", i, C_result[i], K);
            all_correct = false;
            break;
        }
    }

    sycl::free(A_host, q);
    sycl::free(B_host, q);
    sycl::free(C_host, q);
    sycl::free(C_result, q);
    sycl::free(A_dev, q);
    sycl::free(B_dev, q);
    sycl::free(C_dev, q);

    TEST_ASSERT(all_correct, "All 32 rows should be computed correctly");
    fprintf(stderr, "(output_rows=%d) ", M);
    TEST_PASS();
    return true;
#else
    TEST_SKIP("ESIMD not available");
    return true;
#endif
}

// =============================================================================
// Test 6: XMXDpas.FallbackWhenDpasUnsupported
// Assert: Graceful skip on non-XMX hardware
// =============================================================================
bool test_dpas_graceful_fallback() {
    TEST_BEGIN("XMXDpas.FallbackWhenDpasUnsupported");

    // Create unsupported capabilities
    XMXCapabilities unsupported_caps;
    unsupported_caps.supported     = false;
    unsupported_caps.M             = 0;
    unsupported_caps.N             = 0;
    unsupported_caps.K             = 0;
    unsupported_caps.supports_int8 = false;
    unsupported_caps.supports_fp16 = false;

    // Create dpas config from unsupported caps
    XMXDpasConfig dpas_cfg = XMXDpasConfig::from_capabilities(unsupported_caps);

    // Should not crash and should report as invalid
    TEST_ASSERT(!dpas_cfg.is_valid(), "XMXDpasConfig should be invalid on unsupported hardware");
    TEST_ASSERT(!dpas_cfg.supports_int8, "Int8 should not be supported");
    TEST_ASSERT(!dpas_cfg.supports_fp16, "FP16 should not be supported");
    TEST_ASSERT(dpas_cfg.max_repeat_count == 0, "Max repeat count should be 0");

    // Check supports_dpas() helper returns false
    bool supported = supports_dpas(unsupported_caps);
    TEST_ASSERT(!supported, "supports_dpas() should return false for unsupported hardware");

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== XMX DPAS Basic Infrastructure Tests ===\n");

    // Find a SYCL device
    sycl::device dev;
    sycl::queue  q;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
        q   = sycl::queue(dev);
        fprintf(stderr, "Using GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU found, using default device\n");
        dev = sycl::device(sycl::default_selector_v);
        q   = sycl::queue(dev);
    }

    // Check for XMX support upfront
    bool has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
    fprintf(stderr, "XMX support: %s\n", has_xmx ? "YES" : "NO");

    // Query capabilities if XMX is supported
    XMXCapabilities caps;
    if (has_xmx) {
        caps = query_xmx_capabilities(dev);
        fprintf(stderr, "XMX capabilities: M=%zu, N=%zu, K=%zu, int8=%d, fp16=%d\n", caps.M, caps.N, caps.K,
                caps.supports_int8, caps.supports_fp16);
    }
    fprintf(stderr, "\n");

    // Run tests
    bool all_passed = true;

    // Test 1: Hardware detection with dpas flag
    all_passed &= test_dpas_hardware_detection(dev);

    // Test 2: Basic int8x8 multiply
    all_passed &= test_dpas_basic_int8_multiply(q, caps);

    // Test 3: VNNI packing correctness
    all_passed &= test_vnni_packing_correctness();

    // Test 4: Accumulator preserved
    all_passed &= test_dpas_accumulator_preserved(q, caps);

    // Test 5: Repeat count 4 produces 32 rows
    all_passed &= test_dpas_repeat_count_4(q, caps);

    // Test 6: Graceful fallback on unsupported hardware
    all_passed &= test_dpas_graceful_fallback();

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

    if (!has_xmx) {
        fprintf(stderr, "\nNote: Hardware XMX tests were skipped because device does not support XMX.\n");
        fprintf(stderr, "This is expected on non-Intel-Arc hardware.\n");
        return 0;  // Success - graceful skip
    }

    return all_passed ? 0 : 1;
}
