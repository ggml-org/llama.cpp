// Unit tests for cold-start hardware heuristics
// Tests GPU capability detection and initial kernel configuration derivation

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Include the cold-start header (to be created)
#include "ggml-sycl/cold-start.hpp"

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                                             \
    do {                                                                                                   \
        if (!(cond)) {                                                                                     \
            fprintf(stderr, "FAIL: %s\n  Condition: %s\n  File: %s:%d\n", msg, #cond, __FILE__, __LINE__); \
            tests_failed++;                                                                                \
            return false;                                                                                  \
        }                                                                                                  \
    } while (0)

#define TEST_ASSERT_EQ(actual, expected, msg)                                                            \
    do {                                                                                                 \
        if ((actual) != (expected)) {                                                                    \
            fprintf(stderr, "FAIL: %s\n  Expected: %d, Got: %d\n  File: %s:%d\n", msg, (int) (expected), \
                    (int) (actual), __FILE__, __LINE__);                                                 \
            tests_failed++;                                                                              \
            return false;                                                                                \
        }                                                                                                \
    } while (0)

// ============================================================================
// Test 1: Arc A770/A750 detection (512+ EUs -> aggressive config)
// ============================================================================
bool test_cold_start_a770() {
    printf("\n=== Test 1: Arc A770/A750 cold-start config (512+ EUs) ===\n");

    // Create mock GPU capabilities for Arc A770/A750 (512 EUs)
    ggml_sycl::GPUCapabilities caps;
    caps.eu_count           = 512;
    caps.slm_size_kb        = 64;
    caps.max_workgroup_size = 1024;
    caps.has_dpas           = true;
    caps.device_name        = "Intel(R) Arc(TM) A770 Graphics";
    caps.device_id          = 0;

    // Derive initial config
    ggml_sycl::KernelConfig config = ggml_sycl::derive_initial_config(caps);

    // Arc A770/A750 should use aggressive XMX config
    TEST_ASSERT_EQ(config.tile_m, 64, "A770 should use tile_m=64");
    TEST_ASSERT_EQ(config.tile_n, 64, "A770 should use tile_n=64");
    TEST_ASSERT_EQ(config.tile_k, 32, "A770 should use tile_k=32");
    TEST_ASSERT(config.use_dpas, "A770 should enable dpas");

    // Verify GPU family detection
    ggml_sycl::GPUFamily family = ggml_sycl::detect_gpu_family(caps);
    TEST_ASSERT(family == ggml_sycl::GPUFamily::ARC_ALCHEMIST, "A770 should be ARC_ALCHEMIST");

    printf("  eu_count: %d\n", caps.eu_count);
    printf("  config: tile_m=%d, tile_n=%d, tile_k=%d, use_dpas=%s\n", config.tile_m, config.tile_n, config.tile_k,
           config.use_dpas ? "true" : "false");
    printf("  PASS\n");
    tests_passed++;
    return true;
}

// ============================================================================
// Test 2: Arc B580 detection (Xe2/Battlemage architecture)
// ============================================================================
bool test_cold_start_b580() {
    printf("\n=== Test 2: Arc B580 cold-start config (Battlemage) ===\n");

    // Create mock GPU capabilities for Arc B580 (160 EUs, Xe2 architecture)
    ggml_sycl::GPUCapabilities caps;
    caps.eu_count           = 160;
    caps.slm_size_kb        = 128;  // Xe2 has larger SLM
    caps.max_workgroup_size = 1024;
    caps.has_dpas           = true;
    caps.device_name        = "Intel(R) Arc(TM) B580 Graphics";
    caps.device_id          = 0;

    // Derive initial config
    ggml_sycl::KernelConfig config = ggml_sycl::derive_initial_config(caps);

    // Arc B580 should use aggressive XMX config (Battlemage is efficient)
    TEST_ASSERT_EQ(config.tile_m, 64, "B580 should use tile_m=64");
    TEST_ASSERT_EQ(config.tile_n, 64, "B580 should use tile_n=64");
    TEST_ASSERT_EQ(config.tile_k, 32, "B580 should use tile_k=32");
    TEST_ASSERT(config.use_dpas, "B580 should enable dpas");

    // Verify GPU family detection
    ggml_sycl::GPUFamily family = ggml_sycl::detect_gpu_family(caps);
    TEST_ASSERT(family == ggml_sycl::GPUFamily::ARC_BATTLEMAGE, "B580 should be ARC_BATTLEMAGE");

    printf("  eu_count: %d\n", caps.eu_count);
    printf("  config: tile_m=%d, tile_n=%d, tile_k=%d, use_dpas=%s\n", config.tile_m, config.tile_n, config.tile_k,
           config.use_dpas ? "true" : "false");
    printf("  PASS\n");
    tests_passed++;
    return true;
}

// ============================================================================
// Test 3: Unknown device detection (fallback conservative config)
// ============================================================================
bool test_cold_start_unknown() {
    printf("\n=== Test 3: Unknown device cold-start config (conservative fallback) ===\n");

    // Create mock GPU capabilities for unknown device
    ggml_sycl::GPUCapabilities caps;
    caps.eu_count           = 96;  // Some unknown device
    caps.slm_size_kb        = 32;
    caps.max_workgroup_size = 512;
    caps.has_dpas           = false;
    caps.device_name        = "Unknown SYCL Device";
    caps.device_id          = 0;

    // Derive initial config
    ggml_sycl::KernelConfig config = ggml_sycl::derive_initial_config(caps);

    // Unknown device should use conservative config
    TEST_ASSERT_EQ(config.tile_m, 16, "Unknown should use tile_m=16");
    TEST_ASSERT_EQ(config.tile_n, 16, "Unknown should use tile_n=16");
    TEST_ASSERT_EQ(config.tile_k, 32, "Unknown should use tile_k=32");
    TEST_ASSERT(!config.use_dpas, "Unknown should not enable dpas");

    // Verify GPU family detection
    ggml_sycl::GPUFamily family = ggml_sycl::detect_gpu_family(caps);
    TEST_ASSERT(family == ggml_sycl::GPUFamily::UNKNOWN, "Unknown device should be UNKNOWN");

    printf("  eu_count: %d\n", caps.eu_count);
    printf("  config: tile_m=%d, tile_n=%d, tile_k=%d, use_dpas=%s\n", config.tile_m, config.tile_n, config.tile_k,
           config.use_dpas ? "true" : "false");
    printf("  PASS\n");
    tests_passed++;
    return true;
}

// ============================================================================
// Test 4: SLM size constraints are respected
// ============================================================================
bool test_slm_size_respected() {
    printf("\n=== Test 4: SLM size constraints respected ===\n");

    // Create mock GPU with limited SLM
    ggml_sycl::GPUCapabilities caps;
    caps.eu_count           = 512;
    caps.slm_size_kb        = 16;  // Very limited SLM
    caps.max_workgroup_size = 1024;
    caps.has_dpas           = true;
    caps.device_name        = "Intel(R) Arc(TM) A770 Graphics";
    caps.device_id          = 0;

    // Derive initial config
    ggml_sycl::KernelConfig config = ggml_sycl::derive_initial_config(caps);

    // With limited SLM, tiles should be smaller to fit
    // tile_m * tile_k + tile_k * tile_n should fit in SLM
    // 16KB = 16384 bytes
    // With int8: 64*32 + 32*64 = 4096 bytes for A + B tiles
    // This should still fit, but config should respect SLM

    // Verify config is valid for SLM
    size_t min_slm_needed = (size_t) config.tile_m * config.tile_k + (size_t) config.tile_k * config.tile_n;
    size_t slm_bytes      = (size_t) caps.slm_size_kb * 1024;

    // We need at least the tile data to fit (with some margin for scales, etc.)
    TEST_ASSERT(min_slm_needed * 2 <= slm_bytes, "Tile config should fit in SLM with margin");

    printf("  SLM: %d KB, min_slm_needed: %zu bytes\n", caps.slm_size_kb, min_slm_needed);
    printf("  config: tile_m=%d, tile_n=%d, tile_k=%d\n", config.tile_m, config.tile_n, config.tile_k);
    printf("  PASS\n");
    tests_passed++;
    return true;
}

// ============================================================================
// Test 5: Arc A580 detection (256 EUs -> medium config)
// ============================================================================
bool test_cold_start_a580() {
    printf("\n=== Test 5: Arc A580 cold-start config (256 EUs) ===\n");

    // Create mock GPU capabilities for Arc A580 (256 EUs)
    ggml_sycl::GPUCapabilities caps;
    caps.eu_count           = 256;
    caps.slm_size_kb        = 64;
    caps.max_workgroup_size = 1024;
    caps.has_dpas           = true;
    caps.device_name        = "Intel(R) Arc(TM) A580 Graphics";
    caps.device_id          = 0;

    // Derive initial config
    ggml_sycl::KernelConfig config = ggml_sycl::derive_initial_config(caps);

    // Arc A580 with 256 EUs should use medium config
    TEST_ASSERT_EQ(config.tile_m, 32, "A580 should use tile_m=32");
    TEST_ASSERT_EQ(config.tile_n, 32, "A580 should use tile_n=32");
    TEST_ASSERT_EQ(config.tile_k, 32, "A580 should use tile_k=32");
    TEST_ASSERT(config.use_dpas, "A580 should enable dpas");

    // Verify GPU family detection
    ggml_sycl::GPUFamily family = ggml_sycl::detect_gpu_family(caps);
    TEST_ASSERT(family == ggml_sycl::GPUFamily::ARC_ALCHEMIST, "A580 should be ARC_ALCHEMIST");

    printf("  eu_count: %d\n", caps.eu_count);
    printf("  config: tile_m=%d, tile_n=%d, tile_k=%d, use_dpas=%s\n", config.tile_m, config.tile_n, config.tile_k,
           config.use_dpas ? "true" : "false");
    printf("  PASS\n");
    tests_passed++;
    return true;
}

// ============================================================================
// Main test runner
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Cold-Start Hardware Heuristics Tests\n");
    printf("========================================\n");

    test_cold_start_a770();
    test_cold_start_b580();
    test_cold_start_unknown();
    test_slm_size_respected();
    test_cold_start_a580();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
