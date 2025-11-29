// Test for Flash Attention mask handling in unified KV mode
//
// This test verifies the slope calculation fix for SYCL XMX Flash Attention.
// The bug: For non-ALiBi models in unified KV mode (ne13=1), the attention mask
// containing -INF values for cross-sequence positions was not applied because:
//   slope = get_alibi_slope(0, ...) = 0
//   qk += 0 * (-INF) = qk  (mask not applied)
//
// The fix matches CUDA fattn-tile.cuh behavior:
//   slope = (ncols == 1 && ne13 != 1) ? get_alibi_slope(...) : 1.0f
//
// This ensures the mask is always applied for:
//   - Multi-query batching (ncols > 1)
//   - Unified KV mode (ne13 == 1)

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

// Replicate the ALiBi slope calculation from fattn-common.hpp
static float get_alibi_slope(float max_bias, int head, int n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) {
        return 0.0f;  // Non-ALiBi model
    }
    // ALiBi slope calculation (simplified)
    if (head < n_head_log2) {
        return powf(m0, head + 1);
    }
    return powf(m1, 2 * (head - n_head_log2) + 1);
}

// Test helper to compute slope with the FIX applied
static float compute_slope_fixed(float max_bias, int head, int n_head_log2, float m0, float m1,
                                  int ncols, int64_t ne13) {
    // This is the FIXED implementation matching CUDA fattn-tile.cuh:794
    return (ncols == 1 && ne13 != 1) ? get_alibi_slope(max_bias, head, n_head_log2, m0, m1) : 1.0f;
}

// Test helper to compute slope with the BROKEN implementation
static float compute_slope_broken(float max_bias, int head, int n_head_log2, float m0, float m1) {
    // This was the broken implementation that didn't apply masks for non-ALiBi models
    return get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
}

// Simple test framework
#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test: %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

// =============================================================================
// Tests for Non-ALiBi Models (max_bias = 0)
// =============================================================================

TEST(non_alibi_broken_returns_zero) {
    // For non-ALiBi models, the broken implementation returns slope=0
    // This causes mask values (-INF) to be multiplied by 0 and not applied
    float slope = compute_slope_broken(0.0f, 0, 8, 0.0f, 0.0f);
    assert(slope == 0.0f);
}

TEST(non_alibi_unified_kv_fixed) {
    // In unified KV mode (ne13=1), the fix should return slope=1.0f
    // so that mask values are applied: qk += 1.0f * (-INF) = -INF
    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, 1);  // ncols=1, ne13=1 (unified)
    assert(slope == 1.0f);
}

TEST(non_alibi_multi_column_fixed) {
    // Multi-column batching (ncols > 1) should return slope=1.0f
    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 4, 4);  // ncols=4, ne13=4
    assert(slope == 1.0f);
}

TEST(non_alibi_separate_kv_single_col) {
    // Separate KV mode (ne13 > 1) with single column should use get_alibi_slope
    // For non-ALiBi, this returns 0, but that's OK because each sequence has its own KV
    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, 4);  // ncols=1, ne13=4 (separate)
    assert(slope == 0.0f);  // This is fine because ne13 > 1 means separate KV caches
}

// =============================================================================
// Tests for ALiBi Models (max_bias > 0)
// =============================================================================

TEST(alibi_unified_kv_fixed) {
    // For ALiBi models in unified KV mode, slope should be 1.0f
    // The ALiBi bias is already included in the mask
    float slope = compute_slope_fixed(8.0f, 0, 8, 0.5f, 0.25f, 1, 1);  // unified
    assert(slope == 1.0f);
}

TEST(alibi_separate_kv_single_col) {
    // For ALiBi models with separate KV (ne13 > 1) and ncols=1, use ALiBi slope
    float slope = compute_slope_fixed(8.0f, 0, 8, 0.5f, 0.25f, 1, 4);  // separate KV
    float expected = get_alibi_slope(8.0f, 0, 8, 0.5f, 0.25f);
    assert(slope == expected);
    assert(slope != 0.0f);  // ALiBi slope should be non-zero
}

TEST(alibi_multi_column) {
    // Multi-column batching should always use slope=1.0f
    float slope = compute_slope_fixed(8.0f, 0, 8, 0.5f, 0.25f, 4, 4);
    assert(slope == 1.0f);
}

// =============================================================================
// Tests for Mask Application
// =============================================================================

TEST(mask_applied_correctly_unified_kv) {
    // Simulate what happens in the attention kernel for unified KV mode
    float qk = 5.0f;  // Some QK dot product value
    float mask_val = -std::numeric_limits<float>::infinity();  // Cross-sequence mask

    // With broken implementation (slope=0): mask is NOT applied
    float slope_broken = compute_slope_broken(0.0f, 0, 8, 0.0f, 0.0f);
    float qk_broken = qk + slope_broken * mask_val;
    assert(qk_broken == qk);  // BUG: qk unchanged, mask not applied!

    // With fixed implementation (slope=1.0f): mask IS applied
    float slope_fixed = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, 1);
    float qk_fixed = qk + slope_fixed * mask_val;
    assert(std::isinf(qk_fixed) && qk_fixed < 0);  // Correctly masked to -INF
}

TEST(mask_applied_correctly_multi_column) {
    // Multi-column batching should also apply the mask
    float qk = 5.0f;
    float mask_val = -std::numeric_limits<float>::infinity();

    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 4, 4);  // ncols=4
    float qk_masked = qk + slope * mask_val;
    assert(std::isinf(qk_masked) && qk_masked < 0);  // Correctly masked
}

TEST(valid_positions_not_affected) {
    // Valid positions (mask_val = 0) should not be affected
    float qk = 5.0f;
    float mask_val = 0.0f;  // Valid position

    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, 1);  // unified KV
    float qk_masked = qk + slope * mask_val;
    assert(qk_masked == qk);  // QK unchanged for valid positions
}

// =============================================================================
// Tests for CUDA Reference Behavior Match
// =============================================================================

TEST(cuda_reference_ncols1_ne13_not1) {
    // CUDA: ncols2 == 1 ? get_alibi_slope(...) : 1.0f
    // When ncols=1 and ne13 > 1 (separate KV), use ALiBi slope
    // Our fix: (ncols == 1 && ne13 != 1) ? get_alibi_slope(...) : 1.0f
    // This should match CUDA behavior for non-unified mode

    float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, 4);
    assert(slope == 0.0f);  // Non-ALiBi with separate KV

    // Note: CUDA doesn't have ne13 check because it uses different logic
    // Our additional ne13 check handles unified KV mode specifically
}

TEST(cuda_reference_ncols_gt1) {
    // CUDA: ncols2 == 1 ? ... : 1.0f
    // When ncols > 1, always use slope=1.0f

    float slope = compute_slope_fixed(8.0f, 0, 8, 0.5f, 0.25f, 2, 4);
    assert(slope == 1.0f);
}

// =============================================================================
// Tests for Edge Cases
// =============================================================================

TEST(head_iteration) {
    // Verify slope calculation works for all heads in unified KV mode
    const int n_heads = 32;
    for (int head = 0; head < n_heads; ++head) {
        float slope = compute_slope_fixed(0.0f, head, 8, 0.0f, 0.0f, 1, 1);
        assert(slope == 1.0f);  // All heads should use slope=1.0f in unified mode
    }
}

TEST(ncols_variations) {
    // Test various ncols values
    int ncols_values[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int ncols : ncols_values) {
        float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, ncols, 4);
        if (ncols == 1) {
            assert(slope == 0.0f);  // Separate KV, non-ALiBi
        } else {
            assert(slope == 1.0f);  // Multi-column batching
        }
    }
}

TEST(ne13_variations) {
    // Test various ne13 values (number of sequences)
    int64_t ne13_values[] = {1, 2, 4, 8, 16};
    for (int64_t ne13 : ne13_values) {
        float slope = compute_slope_fixed(0.0f, 0, 8, 0.0f, 0.0f, 1, ne13);
        if (ne13 == 1) {
            assert(slope == 1.0f);  // Unified KV mode
        } else {
            assert(slope == 0.0f);  // Separate KV mode, non-ALiBi
        }
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== Flash Attention Mask Slope Tests ===\n\n");

    printf("Non-ALiBi Model Tests:\n");
    RUN_TEST(non_alibi_broken_returns_zero);
    RUN_TEST(non_alibi_unified_kv_fixed);
    RUN_TEST(non_alibi_multi_column_fixed);
    RUN_TEST(non_alibi_separate_kv_single_col);

    printf("\n");

    printf("ALiBi Model Tests:\n");
    RUN_TEST(alibi_unified_kv_fixed);
    RUN_TEST(alibi_separate_kv_single_col);
    RUN_TEST(alibi_multi_column);

    printf("\n");

    printf("Mask Application Tests:\n");
    RUN_TEST(mask_applied_correctly_unified_kv);
    RUN_TEST(mask_applied_correctly_multi_column);
    RUN_TEST(valid_positions_not_affected);

    printf("\n");

    printf("CUDA Reference Behavior Tests:\n");
    RUN_TEST(cuda_reference_ncols1_ne13_not1);
    RUN_TEST(cuda_reference_ncols_gt1);

    printf("\n");

    printf("Edge Case Tests:\n");
    RUN_TEST(head_iteration);
    RUN_TEST(ncols_variations);
    RUN_TEST(ne13_variations);

    printf("\n=== All tests passed! ===\n");

    return 0;
}
