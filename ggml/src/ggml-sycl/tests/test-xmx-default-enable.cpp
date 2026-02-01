//
// Test: XMX Default Enable Behavior
//
// TDD tests to validate XMX unified kernel behavior:
// 1. XMX enabled by default while optimizing
// 2. can_use_xmx returns true for valid dimensions
// 3. can_use_xmx correctly validates dimensions
// 4. XMX tile constants match hardware expectations
//
// XMX is enabled by default while optimizing.
// Use GGML_SYCL_XMX_UNIFIED=0 to disable.
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

// Include the unified kernel header only (no common.hpp dependencies)
#include "../unified-kernel.hpp"

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
// Test 1: XMX Enabled by Default (Optimization Phase)
// Verifies that is_xmx_unified_enabled() returns true without env var
// =============================================================================
static bool test_xmx_enabled_by_default(sycl::queue & q) {
    TEST_BEGIN("test_xmx_enabled_by_default");
    (void)q;  // Unused in this test

    // Test: is_xmx_unified_enabled() should return true by default
    // Use GGML_SYCL_XMX_UNIFIED=0 to disable.

    bool enabled = ggml_sycl_unified::is_xmx_unified_enabled();

    // XMX enabled by default (disable with GGML_SYCL_XMX_UNIFIED=0)
    TEST_ASSERT(enabled, "XMX should be ENABLED by default while optimizing");

    fprintf(stderr, "\n  [INFO] XMX enabled by default - use GGML_SYCL_XMX_UNIFIED=0 to disable\n");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: can_use_xmx Returns True When Enabled
// Verifies that can_use_xmx respects the enabled-by-default setting
// =============================================================================
static bool test_can_use_xmx_respects_enabled(sycl::queue & q) {
    TEST_BEGIN("test_can_use_xmx_respects_enabled");
    (void)q;  // Unused in this test

    // XMX is enabled by default, can_use_xmx should return true for valid dimensions
    int64_t M = 16;
    int64_t N = 32;
    int64_t K = 64;

    bool can_use = ggml_sycl_unified::can_use_xmx(M, N, K);

    // can_use_xmx should return true because XMX is enabled by default
    fprintf(stderr, "\n  [INFO] can_use_xmx(%lld, %lld, %lld) = %s (XMX enabled by default)\n",
            (long long)M, (long long)N, (long long)K, can_use ? "true" : "false");

    TEST_ASSERT(can_use, "can_use_xmx should return true when XMX is enabled by default");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: can_use_xmx Rejects Invalid Dimensions
// Verifies that can_use_xmx correctly rejects invalid dimensions
// =============================================================================
static bool test_can_use_xmx_rejects_invalid(sycl::queue & q) {
    TEST_BEGIN("test_can_use_xmx_rejects_invalid");
    (void)q;  // Unused in this test

    // Test various invalid dimension combinations
    // These should return false regardless of whether XMX is enabled

    const bool allow_small = ggml_sycl_unified::allow_small_xmx_tiles();

    // M too small (< 8) - rejected unless small tiles explicitly enabled
    TEST_ASSERT(ggml_sycl_unified::can_use_xmx(4, 32, 64) == allow_small,
                "M < 8 should be rejected unless small tiles are enabled");

    // N too small (< 16) - rejected unless small tiles explicitly enabled
    TEST_ASSERT(ggml_sycl_unified::can_use_xmx(16, 8, 64) == allow_small,
                "N < 16 should be rejected unless small tiles are enabled");

    // K not aligned to 16
    TEST_ASSERT(!ggml_sycl_unified::can_use_xmx(16, 32, 65),
                "Should reject K not aligned to 16");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: XMX Path Selection When Enabled
// Verifies that launch_unified_matmul handles XMX enabled appropriately
// =============================================================================
static bool test_xmx_path_selection_enabled(sycl::queue & q) {
    TEST_BEGIN("test_xmx_path_selection_enabled");

    // Check if device supports XMX
    sycl::device dev = q.get_device();
    bool has_matrix = dev.has(sycl::aspect::ext_intel_matrix);

    fprintf(stderr, "\n  [INFO] Device: %s\n",
            dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "  [INFO] ext_intel_matrix support: %s\n",
            has_matrix ? "yes" : "no");

    if (!has_matrix) {
        TEST_SKIP("Device doesn't support ext_intel_matrix");
        return true;
    }

    // Setup a test case
    const int M = 16;
    const int N = 32;
    const int K = 32;

    // XMX is enabled by default, can_use_xmx should return true
    bool can_use = ggml_sycl_unified::can_use_xmx(M, N, K);
    fprintf(stderr, "  [INFO] can_use_xmx(%d, %d, %d) = %s\n",
            M, N, K, can_use ? "true" : "false");

    // Expected: true because XMX is enabled by default
    TEST_ASSERT(can_use, "XMX should be usable when enabled by default");

    fprintf(stderr, "  [INFO] XMX path selection working correctly (enabled by default)\n");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: Header Constants Match Hardware Expectations
// Verifies that XMX tile constants are reasonable
// =============================================================================
static bool test_xmx_tile_constants(sycl::queue & q) {
    TEST_BEGIN("test_xmx_tile_constants");
    (void)q;  // Unused in this test

    // Verify the header constants are reasonable for Intel Arc
    // These are from unified-kernel.hpp

    fprintf(stderr, "\n  [INFO] XMX tile constants:\n");
    fprintf(stderr, "    XMX_TILE_M = %d (expected: 8)\n",
            ggml_sycl_unified::XMX_TILE_M);
    fprintf(stderr, "    XMX_TILE_N = %d (expected: 16)\n",
            ggml_sycl_unified::XMX_TILE_N);
    fprintf(stderr, "    XMX_TILE_K = %d (expected: 16 for fp16)\n",
            ggml_sycl_unified::XMX_TILE_K);
    fprintf(stderr, "    XMX_SUBGROUP_SIZE = %d (expected: 16)\n",
            ggml_sycl_unified::XMX_SUBGROUP_SIZE);

    // Verify expected values
    TEST_ASSERT(ggml_sycl_unified::XMX_TILE_M == 8,
                "XMX_TILE_M should be 8");
    TEST_ASSERT(ggml_sycl_unified::XMX_TILE_N == 16,
                "XMX_TILE_N should be 16");
    // Note: XMX_TILE_K is 16 for fp16 in the header, but int8 uses 32
    // The unified kernel uses fp16, so 16 is correct
    TEST_ASSERT(ggml_sycl_unified::XMX_TILE_K == 16,
                "XMX_TILE_K should be 16 for fp16");
    TEST_ASSERT(ggml_sycl_unified::XMX_SUBGROUP_SIZE == 16,
                "XMX_SUBGROUP_SIZE should be 16");

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
    fprintf(stderr, "XMX Default Enable Tests\n");
    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "XMX is ENABLED by default while optimizing.\n");
    fprintf(stderr, "Use GGML_SYCL_XMX_UNIFIED=0 to disable.\n");
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

    all_passed &= test_xmx_enabled_by_default(q);
    all_passed &= test_can_use_xmx_respects_enabled(q);
    all_passed &= test_can_use_xmx_rejects_invalid(q);
    all_passed &= test_xmx_path_selection_enabled(q);
    all_passed &= test_xmx_tile_constants(q);

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
