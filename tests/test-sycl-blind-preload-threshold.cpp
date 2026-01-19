// SYCL Blind Preload Threshold unit tests
// Tests for disabling blind graph_preload_moe_experts for large expert counts
// Part of routing-aware MoE expert pre-staging optimization
//
// TDD: These tests written FIRST, before implementation.
// Implementation must make these tests pass.
//
// The blind preload problem:
// - graph_preload_moe_experts() pre-loads ALL experts for every MoE layer
// - For 128 experts x 4.2MB = 537MB per layer, this causes cache thrashing and DEVICE_LOST
// - Solution: Add threshold to skip blind preload when expert count exceeds limit
//
// Environment variable: GGML_SYCL_MAX_BLIND_PRELOAD_EXPERTS
// - Default: 64 experts
// - If n_experts > threshold, skip blind preload (routing-aware handles it)
// - If n_experts <= threshold, perform blind preload as before

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Include the SYCL test header with function declarations
#include "ggml-sycl-test.hpp"

// =============================================================================
// Test 1: Default threshold is 64 when env var not set
// =============================================================================
static bool test_default_threshold_is_64() {
    printf("TEST: test_default_threshold_is_64\n");

    // Clear the env var to ensure we get default
    unsetenv("GGML_SYCL_MAX_BLIND_PRELOAD_EXPERTS");

    // Force re-read by calling the reset function if available
    // For now, we test the expected default behavior
    int threshold = ggml_sycl_get_blind_preload_threshold();

    // Note: Due to static caching, this may not reflect the unsetenv.
    // The test is designed to verify the API contract, not the caching behavior.
    // When run as the first test with no prior env var set, default should be 64.
    if (threshold != 64) {
        printf("  FAIL: expected default threshold=64, got %d\n", threshold);
        printf("  (Note: if env var was set earlier in this process, cached value persists)\n");
        return false;
    }

    printf("  PASS: default threshold is 64\n");
    return true;
}

// =============================================================================
// Test 2: 32-expert model (GPT-OSS 20B) should use blind preload
// =============================================================================
static bool test_32_experts_uses_blind_preload() {
    printf("TEST: test_32_experts_uses_blind_preload\n");

    // 32 experts is well under the default 64 threshold
    bool should_skip = ggml_sycl_should_skip_blind_preload(32);

    if (should_skip) {
        printf("  FAIL: 32 experts should NOT skip blind preload (threshold=64)\n");
        return false;
    }

    printf("  PASS: 32-expert model uses blind preload\n");
    return true;
}

// =============================================================================
// Test 3: 128-expert model (GPT-OSS 120B) should skip blind preload
// =============================================================================
static bool test_128_experts_skips_blind_preload() {
    printf("TEST: test_128_experts_skips_blind_preload\n");

    // 128 experts exceeds the default 64 threshold
    bool should_skip = ggml_sycl_should_skip_blind_preload(128);

    if (!should_skip) {
        printf("  FAIL: 128 experts SHOULD skip blind preload (threshold=64)\n");
        return false;
    }

    printf("  PASS: 128-expert model skips blind preload\n");
    return true;
}

// =============================================================================
// Test 4: Exactly 64 experts should use blind preload (threshold is inclusive)
// =============================================================================
static bool test_exact_threshold_uses_blind_preload() {
    printf("TEST: test_exact_threshold_uses_blind_preload\n");

    // Exactly at threshold should NOT skip (threshold is inclusive upper bound)
    bool should_skip = ggml_sycl_should_skip_blind_preload(64);

    if (should_skip) {
        printf("  FAIL: exactly 64 experts should NOT skip blind preload\n");
        return false;
    }

    printf("  PASS: exactly at threshold (64) uses blind preload\n");
    return true;
}

// =============================================================================
// Test 5: 65 experts should skip blind preload
// =============================================================================
static bool test_65_experts_skips_blind_preload() {
    printf("TEST: test_65_experts_skips_blind_preload\n");

    // 65 experts exceeds the 64 threshold
    bool should_skip = ggml_sycl_should_skip_blind_preload(65);

    if (!should_skip) {
        printf("  FAIL: 65 experts SHOULD skip blind preload (threshold=64)\n");
        return false;
    }

    printf("  PASS: 65 experts (just over threshold) skips blind preload\n");
    return true;
}

// =============================================================================
// Test 6: Edge case - 0 experts should use blind preload (trivially)
// =============================================================================
static bool test_zero_experts_uses_blind_preload() {
    printf("TEST: test_zero_experts_uses_blind_preload\n");

    bool should_skip = ggml_sycl_should_skip_blind_preload(0);

    if (should_skip) {
        printf("  FAIL: 0 experts should NOT skip blind preload\n");
        return false;
    }

    printf("  PASS: 0 experts uses blind preload (trivial case)\n");
    return true;
}

// =============================================================================
// Test 7: Edge case - 1 expert should use blind preload
// =============================================================================
static bool test_one_expert_uses_blind_preload() {
    printf("TEST: test_one_expert_uses_blind_preload\n");

    bool should_skip = ggml_sycl_should_skip_blind_preload(1);

    if (should_skip) {
        printf("  FAIL: 1 expert should NOT skip blind preload\n");
        return false;
    }

    printf("  PASS: 1 expert uses blind preload\n");
    return true;
}

// =============================================================================
// Test 8: 8 experts (Mixtral) should use blind preload
// =============================================================================
static bool test_8_experts_uses_blind_preload() {
    printf("TEST: test_8_experts_uses_blind_preload\n");

    bool should_skip = ggml_sycl_should_skip_blind_preload(8);

    if (should_skip) {
        printf("  FAIL: 8 experts (Mixtral) should NOT skip blind preload\n");
        return false;
    }

    printf("  PASS: 8-expert model (Mixtral) uses blind preload\n");
    return true;
}

// =============================================================================
// Test 9: 256 experts should skip blind preload
// =============================================================================
static bool test_256_experts_skips_blind_preload() {
    printf("TEST: test_256_experts_skips_blind_preload\n");

    bool should_skip = ggml_sycl_should_skip_blind_preload(256);

    if (!should_skip) {
        printf("  FAIL: 256 experts SHOULD skip blind preload\n");
        return false;
    }

    printf("  PASS: 256-expert model skips blind preload\n");
    return true;
}

// =============================================================================
// Test 10: Threshold getter returns consistent value (cached)
// =============================================================================
static bool test_threshold_is_consistent() {
    printf("TEST: test_threshold_is_consistent\n");

    int t1 = ggml_sycl_get_blind_preload_threshold();
    int t2 = ggml_sycl_get_blind_preload_threshold();
    int t3 = ggml_sycl_get_blind_preload_threshold();

    if (t1 != t2 || t2 != t3) {
        printf("  FAIL: threshold should be consistent across calls (got %d, %d, %d)\n", t1, t2, t3);
        return false;
    }

    printf("  PASS: threshold is consistent across calls (cached value: %d)\n", t1);
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== SYCL Blind Preload Threshold Unit Tests ===\n");
    printf("Testing GGML_SYCL_MAX_BLIND_PRELOAD_EXPERTS threshold behavior\n\n");

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_fn)(), const char * name) {
        bool result = test_fn();
        if (result) {
            passed++;
        } else {
            failed++;
            printf("  >>> TEST FAILED: %s\n\n", name);
        }
    };

    // Core threshold tests (default 64)
    run_test(test_default_threshold_is_64, "test_default_threshold_is_64");
    run_test(test_32_experts_uses_blind_preload, "test_32_experts_uses_blind_preload");
    run_test(test_128_experts_skips_blind_preload, "test_128_experts_skips_blind_preload");
    run_test(test_exact_threshold_uses_blind_preload, "test_exact_threshold_uses_blind_preload");
    run_test(test_65_experts_skips_blind_preload, "test_65_experts_skips_blind_preload");

    // Edge cases
    run_test(test_zero_experts_uses_blind_preload, "test_zero_experts_uses_blind_preload");
    run_test(test_one_expert_uses_blind_preload, "test_one_expert_uses_blind_preload");
    run_test(test_8_experts_uses_blind_preload, "test_8_experts_uses_blind_preload");
    run_test(test_256_experts_skips_blind_preload, "test_256_experts_skips_blind_preload");

    // Consistency
    run_test(test_threshold_is_consistent, "test_threshold_is_consistent");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
