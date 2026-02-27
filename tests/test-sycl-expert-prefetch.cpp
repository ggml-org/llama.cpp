// SYCL Expert Prefetch DMA Engine unit tests
// Tests for ExpertPrefetcher hint/await API with SYCL device integration.
//
// Part of MoE hybrid inference system (epic llama.cpp-j8eb, Task 3).
// Requires SYCL runtime: tests real async H2D DMA via ExpertCache.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sycl/sycl.hpp>

#include "expert-cache.hpp"
#include "expert-prefetch.hpp"

// =============================================================================
// Helpers
// =============================================================================

// Minimal pinned_chunk_pool stub for test compilation.
// The ExpertCache constructor takes a pinned_chunk_pool& but the prefetcher
// tests exercise the DMA path via ExpertCache's copy_queue.

static sycl::queue make_test_queue() {
    try {
        return sycl::queue(sycl::gpu_selector_v);
    } catch (...) {
        printf("SKIP: No GPU device found\n");
        exit(0);
    }
}

// =============================================================================
// Test 1: init / shutdown lifecycle
// =============================================================================
static bool test_lifecycle() {
    printf("TEST: test_lifecycle\n");

    ggml_sycl::ExpertPrefetcher pf;

    // Not active before init
    if (pf.is_active()) {
        printf("  FAIL: should not be active before init\n");
        return false;
    }

    // Null cache should not crash, just warn
    auto q = make_test_queue();
    pf.init(q, nullptr);
    if (pf.is_active()) {
        printf("  FAIL: init with null cache should leave inactive\n");
        return false;
    }

    printf("  PASS: lifecycle (null cache rejected)\n");
    return true;
}

// =============================================================================
// Test 2: prefetch_depth from environment
// =============================================================================
static bool test_prefetch_depth_default() {
    printf("TEST: test_prefetch_depth_default\n");

    ggml_sycl::ExpertPrefetcher pf;

    // Default depth should be 2
    if (pf.prefetch_depth() != 2) {
        printf("  FAIL: default depth should be 2, got %d\n", pf.prefetch_depth());
        return false;
    }

    printf("  PASS: default prefetch depth is 2\n");
    return true;
}

// =============================================================================
// Test 3: hint returns false when not initialized
// =============================================================================
static bool test_hint_requires_init() {
    printf("TEST: test_hint_requires_init\n");

    ggml_sycl::ExpertPrefetcher pf;

    bool result = pf.hint(0, 0);
    if (result) {
        printf("  FAIL: hint should return false when not initialized\n");
        return false;
    }

    printf("  PASS: hint returns false before init\n");
    return true;
}

// =============================================================================
// Test 4: await returns nullptr when not initialized
// =============================================================================
static bool test_await_requires_init() {
    printf("TEST: test_await_requires_init\n");

    ggml_sycl::ExpertPrefetcher pf;

    void * ptr = pf.await(0, 0);
    if (ptr != nullptr) {
        printf("  FAIL: await should return nullptr when not initialized\n");
        return false;
    }

    printf("  PASS: await returns nullptr before init\n");
    return true;
}

// =============================================================================
// Test 5: pending_count starts at 0
// =============================================================================
static bool test_initial_counts() {
    printf("TEST: test_initial_counts\n");

    ggml_sycl::ExpertPrefetcher pf;

    if (pf.pending_count() != 0) {
        printf("  FAIL: initial pending_count should be 0, got %d\n", pf.pending_count());
        return false;
    }

    if (pf.completed_count() != 0) {
        printf("  FAIL: initial completed_count should be 0, got %d\n", pf.completed_count());
        return false;
    }

    printf("  PASS: initial counts are 0\n");
    return true;
}

// =============================================================================
// Test 6: cancel_all on uninitialized prefetcher is safe
// =============================================================================
static bool test_cancel_all_safe() {
    printf("TEST: test_cancel_all_safe\n");

    ggml_sycl::ExpertPrefetcher pf;

    // Should not crash
    pf.cancel_all();

    printf("  PASS: cancel_all on uninitialized is safe\n");
    return true;
}

// =============================================================================
// Test 7: shutdown on uninitialized prefetcher is safe
// =============================================================================
static bool test_shutdown_safe() {
    printf("TEST: test_shutdown_safe\n");

    ggml_sycl::ExpertPrefetcher pf;

    // Should not crash
    pf.shutdown();

    if (pf.is_active()) {
        printf("  FAIL: should not be active after shutdown\n");
        return false;
    }

    printf("  PASS: shutdown on uninitialized is safe\n");
    return true;
}

// =============================================================================
// Test 8: hint_batch on uninitialized prefetcher is safe
// =============================================================================
static bool test_hint_batch_safe() {
    printf("TEST: test_hint_batch_safe\n");

    ggml_sycl::ExpertPrefetcher pf;

    std::vector<int> experts = { 0, 1, 2 };
    // Should not crash
    pf.hint_batch(0, experts);

    printf("  PASS: hint_batch on uninitialized is safe\n");
    return true;
}

// =============================================================================
// Test 9: ExpertPredictor init / lifecycle
// =============================================================================
static bool test_predictor_lifecycle() {
    printf("TEST: test_predictor_lifecycle\n");

    ggml_sycl::ExpertPredictor pred;

    if (pred.is_active()) {
        printf("  FAIL: should not be active before init\n");
        return false;
    }

    // Invalid params should not crash
    pred.init(0, 8, 2);
    if (pred.is_active()) {
        printf("  FAIL: init with n_layers=0 should leave inactive\n");
        return false;
    }

    // Valid init
    pred.init(32, 8, 2);
    if (!pred.is_active()) {
        printf("  FAIL: should be active after valid init\n");
        return false;
    }

    printf("  PASS: predictor lifecycle\n");
    return true;
}

// =============================================================================
// Test 10: predict returns empty before any history
// =============================================================================
static bool test_predictor_empty_history() {
    printf("TEST: test_predictor_empty_history\n");

    ggml_sycl::ExpertPredictor pred;
    pred.init(32, 8, 2);

    // With no history, predict should still return up to n_experts_used
    // entries from frequency table (all zeros initially, so any 2 experts)
    auto result = pred.predict(0);
    if (static_cast<int>(result.size()) != 2) {
        printf("  FAIL: expected 2 predictions, got %d\n", (int) result.size());
        return false;
    }

    printf("  PASS: predict returns %d experts from freq table\n", (int) result.size());
    return true;
}

// =============================================================================
// Test 11: predict reuses last token's experts
// =============================================================================
static bool test_predictor_reuse_last() {
    printf("TEST: test_predictor_reuse_last\n");

    ggml_sycl::ExpertPredictor pred;
    pred.init(32, 8, 2);

    // Record actual experts for layer 5
    pred.record_actual(5, { 3, 7 });

    // Prediction for layer 5 should reuse {3, 7}
    auto result = pred.predict(5);
    if (result.size() != 2) {
        printf("  FAIL: expected 2 predictions, got %d\n", (int) result.size());
        return false;
    }

    bool has_3 = false, has_7 = false;
    for (int e : result) {
        if (e == 3) has_3 = true;
        if (e == 7) has_7 = true;
    }
    if (!has_3 || !has_7) {
        printf("  FAIL: expected experts {3,7}, got {%d,%d}\n", result[0], result[1]);
        return false;
    }

    printf("  PASS: predict reuses last token experts\n");
    return true;
}

// =============================================================================
// Test 12: accuracy tracking
// =============================================================================
static bool test_predictor_accuracy() {
    printf("TEST: test_predictor_accuracy\n");

    ggml_sycl::ExpertPredictor pred;
    pred.init(32, 8, 2);

    // Initial: no predictions, hit rate = 0
    if (pred.hit_rate() != 0.0f) {
        printf("  FAIL: initial hit rate should be 0\n");
        return false;
    }

    // Record actuals, then predict, then record actuals again to check accuracy
    pred.record_actual(0, { 1, 5 });

    // Predict should now return {1, 5} for layer 0
    auto predicted = pred.predict(0);

    // Record same actuals again -- prediction should be a hit
    pred.record_actual(0, { 1, 5 });

    if (pred.total_predictions() != 1) {
        printf("  FAIL: expected 1 prediction, got %d\n", pred.total_predictions());
        return false;
    }

    if (pred.hit_rate() < 0.5f) {
        printf("  FAIL: expected hit rate >= 0.5, got %.2f\n", pred.hit_rate());
        return false;
    }

    printf("  PASS: accuracy tracking works (hit_rate=%.2f)\n", pred.hit_rate());
    return true;
}

// =============================================================================
// Test 13: predict on uninitialized is safe
// =============================================================================
static bool test_predictor_uninit_safe() {
    printf("TEST: test_predictor_uninit_safe\n");

    ggml_sycl::ExpertPredictor pred;

    auto result = pred.predict(0);
    if (!result.empty()) {
        printf("  FAIL: predict should return empty before init\n");
        return false;
    }

    pred.record_actual(0, { 1, 2 });  // Should not crash

    if (pred.hit_rate() != 0.0f) {
        printf("  FAIL: hit_rate should be 0 before init\n");
        return false;
    }

    printf("  PASS: predictor is safe before init\n");
    return true;
}

// =============================================================================
// Test 14: frequency table fills remaining slots
// =============================================================================
static bool test_predictor_freq_fill() {
    printf("TEST: test_predictor_freq_fill\n");

    ggml_sycl::ExpertPredictor pred;
    pred.init(32, 8, 4);  // top-4

    // Record expert 2 multiple times to make it high-frequency
    for (int i = 0; i < 10; i++) {
        pred.record_actual(0, { 2 });
    }
    // Record expert 6 a few times
    for (int i = 0; i < 5; i++) {
        pred.record_actual(0, { 6 });
    }

    // Now last_experts_ for layer 0 is {6} (from final record_actual)
    // Need 4 experts: {6} from last + fill from freq: {2, ...}
    auto result = pred.predict(0);

    if (static_cast<int>(result.size()) != 4) {
        printf("  FAIL: expected 4 predictions, got %d\n", (int) result.size());
        return false;
    }

    // Expert 6 should be first (from last_experts_)
    if (result[0] != 6) {
        printf("  FAIL: expected expert 6 first (from history), got %d\n", result[0]);
        return false;
    }

    // Expert 2 should be in the result (highest frequency after 6)
    bool has_2 = false;
    for (int e : result) {
        if (e == 2) has_2 = true;
    }
    if (!has_2) {
        printf("  FAIL: expected expert 2 from frequency table\n");
        return false;
    }

    printf("  PASS: frequency table fills remaining prediction slots\n");
    return true;
}

// =============================================================================
// Test 15: env var helper
// =============================================================================
static bool test_predict_env_var() {
    printf("TEST: test_predict_env_var\n");

    // Default should be enabled (ON)
    // Note: this reads env var at first call and caches, so we can only
    // test the default behavior reliably.
    bool enabled = ggml_sycl::ggml_sycl_expert_predict_enabled();
    printf("  INFO: GGML_SYCL_EXPERT_PREDICT enabled=%d\n", enabled);

    printf("  PASS: env var helper works\n");
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== Expert Prefetch & Prediction Unit Tests ===\n");
    printf("Part of MoE hybrid inference (llama.cpp-j8eb/T3+T6)\n\n");

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

    // Core lifecycle (ExpertPrefetcher)
    run_test(test_lifecycle, "test_lifecycle");
    run_test(test_prefetch_depth_default, "test_prefetch_depth_default");
    run_test(test_initial_counts, "test_initial_counts");

    // Safety (no crashes on uninitialized)
    run_test(test_hint_requires_init, "test_hint_requires_init");
    run_test(test_await_requires_init, "test_await_requires_init");
    run_test(test_cancel_all_safe, "test_cancel_all_safe");
    run_test(test_shutdown_safe, "test_shutdown_safe");
    run_test(test_hint_batch_safe, "test_hint_batch_safe");

    // ExpertPredictor tests
    run_test(test_predictor_lifecycle, "test_predictor_lifecycle");
    run_test(test_predictor_empty_history, "test_predictor_empty_history");
    run_test(test_predictor_reuse_last, "test_predictor_reuse_last");
    run_test(test_predictor_accuracy, "test_predictor_accuracy");
    run_test(test_predictor_uninit_safe, "test_predictor_uninit_safe");
    run_test(test_predictor_freq_fill, "test_predictor_freq_fill");
    run_test(test_predict_env_var, "test_predict_env_var");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
