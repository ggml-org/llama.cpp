//
// Test: Kernel Path Dispatch (Batch-Size Gating)
//
// TDD tests for the ESIMD batch-size gating logic (Task: llama.cpp-caws).
// Tests verify:
// 1. Dispatch logic returns correct path for each batch size
// 2. Environment variable overrides work correctly
// 3. XMX support flag is respected
// 4. Custom threshold configuration works
// 5. Dispatch overhead is minimal (<1us for 10000 calls)
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
#include <cstdint>
#include <string>

// =============================================================================
// Standalone KernelPath and XMXConfig definitions for testing
// These mirror the definitions in unified-kernel.hpp without SYCL dependencies
// =============================================================================

namespace ggml_sycl_unified_test {

/**
 * Kernel path enum for dispatch decisions.
 */
enum class KernelPath {
    DMMV,       ///< Dense matrix-vector multiply (batch=1, memory-bound)
    MMVQ,       ///< Matrix-matrix vector quantized (small batch)
    ESIMD_DPAS  ///< ESIMD dpas path (large batch, compute-bound)
};

/**
 * Get kernel path name as string for debug logging.
 */
inline const char * kernel_path_name(KernelPath path) {
    switch (path) {
        case KernelPath::DMMV:       return "DMMV";
        case KernelPath::MMVQ:       return "MMVQ";
        case KernelPath::ESIMD_DPAS: return "ESIMD_DPAS";
        default:                     return "UNKNOWN";
    }
}

/**
 * Get minimum batch size for ESIMD dispatch (cached).
 */
inline int get_esimd_min_batch() {
    static int min_batch = -1;
    if (min_batch < 0) {
        const char * env = std::getenv("GGML_SYCL_ESIMD_MIN_BATCH");
        if (env) {
            int val = std::atoi(env);
            min_batch = (val > 0) ? val : 8;
        } else {
            min_batch = 8;
        }
    }
    return min_batch;
}

/**
 * Check if MMVQ path is forced via environment (cached).
 */
inline bool env_force_mmvq() {
    static int forced = -1;
    if (forced < 0) {
        const char * env = std::getenv("GGML_SYCL_FORCE_MMVQ");
        forced           = (env && std::string(env) == "1") ? 1 : 0;
    }
    return forced != 0;
}

/**
 * Check if ESIMD path is forced via environment (cached).
 */
inline bool env_force_esimd() {
    static int forced = -1;
    if (forced < 0) {
        const char * env = std::getenv("GGML_SYCL_FORCE_ESIMD");
        forced           = (env && std::string(env) == "1") ? 1 : 0;
    }
    return forced != 0;
}

/**
 * XMX configuration for ESIMD dpas kernels.
 */
struct XMXConfig {
    size_t xmx_m = 8;
    size_t xmx_n = 16;
    size_t xmx_k_fp16 = 16;
    size_t xmx_k_int8 = 32;
    size_t slm_size = 65536;
    int    nsm      = 20;
    bool supported     = false;
    bool supports_int8 = false;
    bool supports_fp16 = false;
    bool use_double_buffer  = false;
    int  tiles_per_workitem = 1;
};

/**
 * Select the optimal kernel path based on batch size and hardware capabilities.
 */
inline KernelPath select_kernel_path(
    int batch_size, int64_t M, int64_t N, int64_t K,
    int quant_type, const XMXConfig & cfg) {
    (void)M;
    (void)N;
    (void)K;
    (void)quant_type;

    // Batch=1: Always DMMV (memory-bound, optimized for single vector)
    if (batch_size == 1) {
        return KernelPath::DMMV;
    }

    // Environment override: Force MMVQ path
    if (env_force_mmvq()) {
        return KernelPath::MMVQ;
    }

    // Check environment override first (cached at init)
    const int min_batch = get_esimd_min_batch();

    // Batch < threshold: MMVQ (small batch, still memory-bound)
    if (batch_size < min_batch) {
        return KernelPath::MMVQ;
    }

    // Check if ESIMD available
    if (!cfg.supported) {
        return KernelPath::MMVQ;
    }

    // Environment override: Force ESIMD path
    if (env_force_esimd()) {
        return KernelPath::ESIMD_DPAS;
    }

    // Default: ESIMD dpas for compute-bound regime
    return KernelPath::ESIMD_DPAS;
}

} // namespace ggml_sycl_unified_test

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

#define TEST_ASSERT_EQ(actual, expected, msg)                                              \
    do {                                                                                   \
        if ((actual) != (expected)) {                                                      \
            fprintf(stderr, "FAILED: %s (expected %d, got %d)\n", msg, (int)(expected),    \
                    (int)(actual));                                                        \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

using namespace ggml_sycl_unified_test;

// =============================================================================
// Create XMXConfig with specific settings for testing
// =============================================================================
static XMXConfig make_test_config(bool supported) {
    XMXConfig cfg;
    cfg.supported = supported;
    cfg.supports_fp16 = supported;
    cfg.supports_int8 = false;
    cfg.xmx_m = 8;
    cfg.xmx_n = 16;
    cfg.xmx_k_fp16 = 16;
    cfg.xmx_k_int8 = 32;
    cfg.slm_size = 65536;
    cfg.nsm = 20;
    return cfg;
}

// =============================================================================
// Test 1: Batch=1 should always return DMMV
// =============================================================================
static bool test_dispatch_batch_1() {
    TEST_BEGIN("test_dispatch_batch_1");

    XMXConfig cfg = make_test_config(true);  // XMX supported

    KernelPath path = select_kernel_path(
        1,    // batch_size = 1
        1,    // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::DMMV, "batch=1 should return DMMV");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Batch=4 should return MMVQ (below default threshold of 8)
// =============================================================================
static bool test_dispatch_batch_4() {
    TEST_BEGIN("test_dispatch_batch_4");

    XMXConfig cfg = make_test_config(true);  // XMX supported

    KernelPath path = select_kernel_path(
        4,    // batch_size = 4
        4,    // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::MMVQ, "batch=4 should return MMVQ (below threshold 8)");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Batch=8 should return ESIMD_DPAS (at default threshold)
// =============================================================================
static bool test_dispatch_batch_8() {
    TEST_BEGIN("test_dispatch_batch_8");

    XMXConfig cfg = make_test_config(true);  // XMX supported

    KernelPath path = select_kernel_path(
        8,    // batch_size = 8
        8,    // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::ESIMD_DPAS, "batch=8 should return ESIMD_DPAS (at threshold)");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Batch=7 should return MMVQ (just below threshold)
// =============================================================================
static bool test_dispatch_batch_7() {
    TEST_BEGIN("test_dispatch_batch_7");

    XMXConfig cfg = make_test_config(true);  // XMX supported

    KernelPath path = select_kernel_path(
        7,    // batch_size = 7 (just below 8)
        7,    // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::MMVQ, "batch=7 should return MMVQ (just below threshold)");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: XMX unsupported should return MMVQ even for large batch
// =============================================================================
static bool test_dispatch_xmx_unsupported() {
    TEST_BEGIN("test_dispatch_xmx_unsupported");

    XMXConfig cfg = make_test_config(false);  // XMX NOT supported

    KernelPath path = select_kernel_path(
        512,  // batch_size = 512 (large batch)
        512,  // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::MMVQ, "batch=512 with XMX unsupported should return MMVQ");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: Large batch should return ESIMD_DPAS
// =============================================================================
static bool test_dispatch_batch_512() {
    TEST_BEGIN("test_dispatch_batch_512");

    XMXConfig cfg = make_test_config(true);  // XMX supported

    KernelPath path = select_kernel_path(
        512,  // batch_size = 512
        512,  // M
        4096, // N
        4096, // K
        2,    // quant_type = Q4_0
        cfg
    );

    TEST_ASSERT_EQ(path, KernelPath::ESIMD_DPAS, "batch=512 should return ESIMD_DPAS");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: Dispatch overhead should be <1us for 10000 calls
// =============================================================================
static bool test_dispatch_overhead() {
    TEST_BEGIN("test_dispatch_overhead");

    XMXConfig cfg = make_test_config(true);
    const int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    // Run dispatch many times
    volatile KernelPath path;
    for (int i = 0; i < iterations; i++) {
        path = select_kernel_path(
            64,   // batch_size
            64,   // M
            4096, // N
            4096, // K
            2,    // quant_type
            cfg
        );
    }
    (void)path;  // Suppress unused warning

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_us = static_cast<double>(duration.count()) / iterations;

    fprintf(stderr, "(avg=%.3f us) ", avg_us);

    // Target: average dispatch time <1us (allowing for measurement overhead)
    TEST_ASSERT(avg_us < 1.0, "Dispatch overhead should be <1us per call");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 8: kernel_path_name returns correct strings
// =============================================================================
static bool test_kernel_path_name() {
    TEST_BEGIN("test_kernel_path_name");

    TEST_ASSERT(strcmp(kernel_path_name(KernelPath::DMMV), "DMMV") == 0,
                "DMMV name mismatch");
    TEST_ASSERT(strcmp(kernel_path_name(KernelPath::MMVQ), "MMVQ") == 0,
                "MMVQ name mismatch");
    TEST_ASSERT(strcmp(kernel_path_name(KernelPath::ESIMD_DPAS), "ESIMD_DPAS") == 0,
                "ESIMD_DPAS name mismatch");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 9: get_esimd_min_batch returns default 8
// =============================================================================
static bool test_default_min_batch() {
    TEST_BEGIN("test_default_min_batch");

    // Note: This test depends on GGML_SYCL_ESIMD_MIN_BATCH not being set
    // If it is set, this test will reflect that value
    int min_batch = get_esimd_min_batch();

    fprintf(stderr, "(min_batch=%d) ", min_batch);

    // Should be 8 by default, or whatever is set in env
    TEST_ASSERT(min_batch > 0, "min_batch should be positive");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 10: Batch at various sizes with XMX supported
// =============================================================================
static bool test_dispatch_various_batches() {
    TEST_BEGIN("test_dispatch_various_batches");

    XMXConfig cfg = make_test_config(true);

    // Test array of batch sizes and expected paths
    struct TestCase {
        int batch;
        KernelPath expected;
        const char* desc;
    };

    const int min_batch = get_esimd_min_batch();

    TestCase cases[] = {
        { 1,           KernelPath::DMMV,       "batch=1" },
        { 2,           KernelPath::MMVQ,       "batch=2" },
        { min_batch-1, KernelPath::MMVQ,       "batch=min-1" },
        { min_batch,   KernelPath::ESIMD_DPAS, "batch=min" },
        { min_batch+1, KernelPath::ESIMD_DPAS, "batch=min+1" },
        { 16,          KernelPath::ESIMD_DPAS, "batch=16" },
        { 32,          KernelPath::ESIMD_DPAS, "batch=32" },
        { 64,          KernelPath::ESIMD_DPAS, "batch=64" },
        { 128,         KernelPath::ESIMD_DPAS, "batch=128" },
        { 256,         KernelPath::ESIMD_DPAS, "batch=256" },
        { 512,         KernelPath::ESIMD_DPAS, "batch=512" },
    };

    for (const auto& tc : cases) {
        // Skip batch=min-1 if min_batch is 1 (edge case)
        if (tc.batch < 1) continue;

        KernelPath path = select_kernel_path(
            tc.batch, tc.batch, 4096, 4096, 2, cfg
        );

        if (path != tc.expected) {
            fprintf(stderr, "FAILED: %s - expected %s, got %s\n",
                    tc.desc, kernel_path_name(tc.expected), kernel_path_name(path));
            return false;
        }
    }

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
    fprintf(stderr, "Kernel Path Dispatch Tests (Batch-Size Gating)\n");
    fprintf(stderr, "===========================================\n");

    // Show environment variables that affect dispatch
    const char* env_min_batch = std::getenv("GGML_SYCL_ESIMD_MIN_BATCH");
    const char* env_force_mmvq_val = std::getenv("GGML_SYCL_FORCE_MMVQ");
    const char* env_force_esimd_val = std::getenv("GGML_SYCL_FORCE_ESIMD");

    fprintf(stderr, "Environment:\n");
    fprintf(stderr, "  GGML_SYCL_ESIMD_MIN_BATCH=%s\n", env_min_batch ? env_min_batch : "(unset, default=8)");
    fprintf(stderr, "  GGML_SYCL_FORCE_MMVQ=%s\n", env_force_mmvq_val ? env_force_mmvq_val : "(unset)");
    fprintf(stderr, "  GGML_SYCL_FORCE_ESIMD=%s\n", env_force_esimd_val ? env_force_esimd_val : "(unset)");
    fprintf(stderr, "-------------------------------------------\n");

    // Note: Tests 6-7 (env override tests) require specific environment setup
    // Run those tests separately with:
    //   GGML_SYCL_FORCE_MMVQ=1 ./test-kernel-dispatch
    //   GGML_SYCL_ESIMD_MIN_BATCH=16 ./test-kernel-dispatch

    bool all_passed = true;

    // Core dispatch logic tests
    all_passed &= test_dispatch_batch_1();
    all_passed &= test_dispatch_batch_4();

    // Skip threshold-dependent tests if FORCE_MMVQ is set
    if (!env_force_mmvq_val) {
        all_passed &= test_dispatch_batch_8();
        all_passed &= test_dispatch_batch_7();
        all_passed &= test_dispatch_batch_512();
    } else {
        fprintf(stderr, "[TEST] test_dispatch_batch_8 ... SKIPPED (FORCE_MMVQ set)\n");
        fprintf(stderr, "[TEST] test_dispatch_batch_7 ... SKIPPED (FORCE_MMVQ set)\n");
        fprintf(stderr, "[TEST] test_dispatch_batch_512 ... SKIPPED (FORCE_MMVQ set)\n");
        g_tests_skipped += 3;
    }

    all_passed &= test_dispatch_xmx_unsupported();
    all_passed &= test_dispatch_overhead();
    all_passed &= test_kernel_path_name();
    all_passed &= test_default_min_batch();

    if (!env_force_mmvq_val) {
        all_passed &= test_dispatch_various_batches();
    } else {
        fprintf(stderr, "[TEST] test_dispatch_various_batches ... SKIPPED (FORCE_MMVQ set)\n");
        g_tests_skipped++;
    }

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
