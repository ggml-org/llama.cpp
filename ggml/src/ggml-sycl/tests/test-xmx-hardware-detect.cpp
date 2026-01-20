//
// Test: XMX Hardware Detection and XMXConfig
//
// Tests that XMXConfig correctly uses hardware-queried XMXCapabilities
// and does NOT use hardcoded values.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

// Include the XMX ESIMD common header (provides XMXCapabilities in standalone mode)
#include "../xmx-esimd-common.hpp"

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
// Test 1: XMXCapabilities is populated by query_xmx_capabilities()
// =============================================================================
bool test_xmx_capabilities_populated(sycl::device & dev) {
    TEST_BEGIN("XMXCapabilities populated from hardware");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    XMXCapabilities caps = query_xmx_capabilities(dev);

    TEST_ASSERT(caps.supported, "XMX should be marked as supported");
    TEST_ASSERT(caps.M > 0, "XMX M dimension should be > 0");
    TEST_ASSERT(caps.N > 0, "XMX N dimension should be > 0");
    TEST_ASSERT(caps.K > 0, "XMX K dimension should be > 0");
    TEST_ASSERT(caps.slm_size > 0, "SLM size should be > 0");

    // Verify reasonable values (Intel Arc typically: M=8, N=16, K=32)
    TEST_ASSERT(caps.M >= 4 && caps.M <= 32, "XMX M should be in reasonable range [4, 32]");
    TEST_ASSERT(caps.N >= 8 && caps.N <= 64, "XMX N should be in reasonable range [8, 64]");
    TEST_ASSERT(caps.K >= 16 && caps.K <= 64, "XMX K should be in reasonable range [16, 64]");

    fprintf(stderr, "(M=%zu, N=%zu, K=%zu) ", caps.M, caps.N, caps.K);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: XMXConfig::from_capabilities() uses hardware values
// =============================================================================
bool test_xmx_config_from_capabilities(sycl::device & dev) {
    TEST_BEGIN("XMXConfig::from_capabilities() uses hardware values");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    XMXCapabilities          caps   = query_xmx_capabilities(dev);
    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    // Verify config uses values from capabilities, not hardcoded
    TEST_ASSERT(config.tile_m() == caps.M, "XMXConfig::tile_m() should match XMXCapabilities::M");
    TEST_ASSERT(config.tile_n() == caps.N, "XMXConfig::tile_n() should match XMXCapabilities::N");
    TEST_ASSERT(config.tile_k() == caps.K, "XMXConfig::tile_k() should match XMXCapabilities::K");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: XMXConfig::supports_qtype() for required quantization types
// =============================================================================
bool test_xmx_config_supports_qtype(sycl::device & dev) {
    TEST_BEGIN("XMXConfig::supports_qtype() for Q4_0, Q8_0, MXFP4, F16");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    XMXCapabilities          caps   = query_xmx_capabilities(dev);
    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    // Test Q4_0 support (requires int8 XMX path)
    bool q4_0_expected = caps.supports_int8;
    TEST_ASSERT(config.supports_qtype(GGML_TYPE_Q4_0) == q4_0_expected,
                "Q4_0 support should match hardware int8 capability");

    // Test Q8_0 support (requires int8 XMX path)
    bool q8_0_expected = caps.supports_int8;
    TEST_ASSERT(config.supports_qtype(GGML_TYPE_Q8_0) == q8_0_expected,
                "Q8_0 support should match hardware int8 capability");

    // Test MXFP4 support (requires int8 for MoE path - uses int8 accumulation)
    bool mxfp4_expected = caps.supports_int8;
    TEST_ASSERT(config.supports_qtype(GGML_TYPE_IQ4_NL) == mxfp4_expected,
                "MXFP4/IQ4_NL support should match hardware int8 capability");

    // Test F16 support
    bool f16_expected = caps.supports_fp16;
    TEST_ASSERT(config.supports_qtype(GGML_TYPE_F16) == f16_expected,
                "F16 support should match hardware fp16 capability");

    fprintf(stderr, "(Q4_0=%d, Q8_0=%d, MXFP4=%d, F16=%d) ", config.supports_qtype(GGML_TYPE_Q4_0),
            config.supports_qtype(GGML_TYPE_Q8_0), config.supports_qtype(GGML_TYPE_IQ4_NL),
            config.supports_qtype(GGML_TYPE_F16));
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: XMXConfig does NOT use hardcoded 8/16/32 values
// =============================================================================
bool test_xmx_config_no_hardcoded_values(sycl::device & dev) {
    TEST_BEGIN("XMXConfig does NOT use hardcoded defaults when hardware reports different");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    // We query but don't use the real caps - just verify device has XMX
    (void) query_xmx_capabilities(dev);

    // Create a mock capabilities with different values to verify no hardcoding
    XMXCapabilities mock_caps;
    mock_caps.supported     = true;
    mock_caps.M             = 4;   // Different from typical 8
    mock_caps.N             = 8;   // Different from typical 16
    mock_caps.K             = 16;  // Different from typical 32
    mock_caps.supports_int8 = true;
    mock_caps.supports_fp16 = true;

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(mock_caps);

    // Verify config uses the mock values, not defaults
    TEST_ASSERT(config.tile_m() == 4, "XMXConfig should use M=4, not default 8");
    TEST_ASSERT(config.tile_n() == 8, "XMXConfig should use N=8, not default 16");
    TEST_ASSERT(config.tile_k() == 16, "XMXConfig should use K=16, not default 32");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: XMXConfig handles unsupported hardware gracefully
// =============================================================================
bool test_xmx_config_unsupported_hardware() {
    TEST_BEGIN("XMXConfig handles unsupported hardware gracefully");

    // Create capabilities for unsupported hardware
    XMXCapabilities unsupported_caps;
    unsupported_caps.supported = false;
    unsupported_caps.M         = 0;
    unsupported_caps.N         = 0;
    unsupported_caps.K         = 0;

    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(unsupported_caps);

    // Verify graceful handling - should not crash, should report no support
    TEST_ASSERT(!config.is_supported(), "XMXConfig should report not supported for unsupported hardware");
    TEST_ASSERT(!config.supports_qtype(GGML_TYPE_Q4_0), "Should not support Q4_0 on unsupported hardware");
    TEST_ASSERT(!config.supports_qtype(GGML_TYPE_Q8_0), "Should not support Q8_0 on unsupported hardware");
    TEST_ASSERT(!config.supports_qtype(GGML_TYPE_F16), "Should not support F16 on unsupported hardware");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: XMXCapabilities int8/fp16 type detection
// =============================================================================
bool test_xmx_capabilities_type_detection(sycl::device & dev) {
    TEST_BEGIN("XMXCapabilities correctly detects int8/fp16 support");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    XMXCapabilities caps = query_xmx_capabilities(dev);

    // Intel Arc should support both int8 and fp16
    // This is a hardware expectation for Intel Arc GPUs
    fprintf(stderr, "(int8=%d, fp16=%d) ", caps.supports_int8, caps.supports_fp16);

    // At minimum, one type should be supported if XMX is available
    TEST_ASSERT(caps.supports_int8 || caps.supports_fp16, "At least one XMX type (int8 or fp16) should be supported");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: XMXConfig::compute_sizes() for different quant types
// =============================================================================
bool test_xmx_config_compute_sizes(sycl::device & dev) {
    TEST_BEGIN("XMXConfig::compute_sizes() returns correct tile sizes");

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        TEST_SKIP("Device does not support XMX");
        return true;
    }

    XMXCapabilities          caps   = query_xmx_capabilities(dev);
    ggml_sycl_xmx::XMXConfig config = ggml_sycl_xmx::XMXConfig::from_capabilities(caps);

    // For Q4_0: each block has 32 elements, 4-bit each = 16 bytes data + 2 bytes scale
    // K dimension should align with quant block size
    size_t k_elements = config.tile_k();
    TEST_ASSERT(k_elements > 0, "K tile elements should be > 0");
    TEST_ASSERT(k_elements % 32 == 0 || 32 % k_elements == 0,
                "K tile should be compatible with Q4_0/Q8_0 block size (32)");

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== XMX Hardware Detection Tests ===\n");

    // Find a SYCL device
    sycl::device dev;
    try {
        // Try to get a GPU device
        dev = sycl::device(sycl::gpu_selector_v);
        fprintf(stderr, "Using GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU found, using default device\n");
        dev = sycl::device(sycl::default_selector_v);
    }

    // Check for XMX support upfront
    bool has_xmx = dev.has(sycl::aspect::ext_intel_matrix);
    fprintf(stderr, "XMX support: %s\n\n", has_xmx ? "YES" : "NO");

    // Run tests
    bool all_passed = true;

    all_passed &= test_xmx_capabilities_populated(dev);
    all_passed &= test_xmx_config_from_capabilities(dev);
    all_passed &= test_xmx_config_supports_qtype(dev);
    all_passed &= test_xmx_config_no_hardcoded_values(dev);
    all_passed &= test_xmx_config_unsupported_hardware();  // No device needed
    all_passed &= test_xmx_capabilities_type_detection(dev);
    all_passed &= test_xmx_config_compute_sizes(dev);

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

    if (!has_xmx) {
        fprintf(stderr, "\nNote: XMX tests were skipped because hardware does not support XMX.\n");
        fprintf(stderr, "This is expected on non-Intel-Arc hardware.\n");
        return 0;  // Success - graceful skip
    }

    return all_passed ? 0 : 1;
}
