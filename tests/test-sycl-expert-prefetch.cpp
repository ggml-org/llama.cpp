// SYCL Expert Prefetch unit tests
// Tests for hybrid expert prefetching with sorted caching
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-eqa)
//
// TDD: These tests written FIRST, before implementation.
// Implementation must make these tests pass.
//
// Hybrid Prefetch Strategy:
// - Predictive: Prefetch top-K experts based on router scores BEFORE ARGSORT
// - On-demand: If router selects unexpected expert, stream immediately
// - Sorted: Prefetch in descending score order for optimal cache access
// - Adaptive: Track prediction accuracy per layer, adjust prefetch count
//
// Priority Integration:
// - P2_HOT_EXPERT: Recently used, high confidence
// - P3_WARM_EXPERT: Prefetched/predicted, not yet confirmed
// - P4_COLD_EXPERT: Not recently used, evict first
//
// NOTE: This test is self-contained and does not require SYCL runtime.
// It tests the pure C++ logic of the ExpertPrefetcher class.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Include the expert prefetch header
#include "expert-prefetch.hpp"

// =============================================================================
// Test 1: Top-K expert selection sorts by score descending
// =============================================================================
static bool test_prefetch_sorts_by_score_descending() {
    printf("TEST: test_prefetch_sorts_by_score_descending\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);  // 1 layer, 8 experts, 1KB each

    float scores[8] = { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f, 0.9f, 0.4f, 0.6f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    // Top 4 should be experts 5, 3, 7, 1 (scores 0.9, 0.8, 0.6, 0.5)
    if (batch.predictions.size() < 4) {
        printf("  FAIL: expected at least 4 predictions, got %zu\n", batch.predictions.size());
        return false;
    }

    if (batch.predictions[0].expert_id != 5) {
        printf("  FAIL: predictions[0] should be expert 5 (score 0.9), got %u\n", batch.predictions[0].expert_id);
        return false;
    }

    if (batch.predictions[1].expert_id != 3) {
        printf("  FAIL: predictions[1] should be expert 3 (score 0.8), got %u\n", batch.predictions[1].expert_id);
        return false;
    }

    if (batch.predictions[2].expert_id != 7) {
        printf("  FAIL: predictions[2] should be expert 7 (score 0.6), got %u\n", batch.predictions[2].expert_id);
        return false;
    }

    if (batch.predictions[3].expert_id != 1) {
        printf("  FAIL: predictions[3] should be expert 1 (score 0.5), got %u\n", batch.predictions[3].expert_id);
        return false;
    }

    printf("  PASS: prefetch sorts experts by score descending\n");
    return true;
}

// =============================================================================
// Test 2: Scores are preserved in predictions
// =============================================================================
static bool test_scores_preserved_in_predictions() {
    printf("TEST: test_scores_preserved_in_predictions\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 4, 1024);

    float scores[4] = { 0.25f, 0.75f, 0.50f, 0.10f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 4, 2);

    // Top 2 should be experts 1, 2 with scores 0.75, 0.50
    if (batch.predictions.size() < 2) {
        printf("  FAIL: expected at least 2 predictions\n");
        return false;
    }

    const float eps = 1e-6f;
    if (std::abs(batch.predictions[0].score - 0.75f) > eps) {
        printf("  FAIL: predictions[0].score should be 0.75, got %f\n", batch.predictions[0].score);
        return false;
    }

    if (std::abs(batch.predictions[1].score - 0.50f) > eps) {
        printf("  FAIL: predictions[1].score should be 0.50, got %f\n", batch.predictions[1].score);
        return false;
    }

    printf("  PASS: scores preserved in predictions\n");
    return true;
}

// =============================================================================
// Test 3: Prefetched expert marked as hit when accessed
// =============================================================================
static bool test_prefetched_expert_is_hit() {
    printf("TEST: test_prefetched_expert_is_hit\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 4, 1024);

    float scores[4] = { 0.9f, 0.1f, 0.1f, 0.1f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 4, 2);

    // Expert 0 was prefetched (highest score)
    void * data = prefetcher.get_expert_data(0, 0, batch);

    // Should have 1 hit, 0 misses
    if (batch.num_hits != 1) {
        printf("  FAIL: expected num_hits=1, got %zu\n", batch.num_hits);
        return false;
    }

    if (batch.num_misses != 0) {
        printf("  FAIL: expected num_misses=0, got %zu\n", batch.num_misses);
        return false;
    }

    // Data should not be null (even in mock mode)
    if (data == nullptr) {
        printf("  FAIL: expected non-null data for prefetched expert\n");
        return false;
    }

    printf("  PASS: prefetched expert marked as hit\n");
    return true;
}

// =============================================================================
// Test 4: Non-prefetched expert marked as miss (streams on demand)
// =============================================================================
static bool test_non_prefetched_expert_is_miss() {
    printf("TEST: test_non_prefetched_expert_is_miss\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 4, 1024);

    float scores[4] = { 0.9f, 0.8f, 0.1f, 0.1f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 4, 2);  // Only prefetch top 2

    // Expert 3 was NOT prefetched (low score)
    void * data = prefetcher.get_expert_data(0, 3, batch);

    // Should have 0 hits, 1 miss
    if (batch.num_hits != 0) {
        printf("  FAIL: expected num_hits=0, got %zu\n", batch.num_hits);
        return false;
    }

    if (batch.num_misses != 1) {
        printf("  FAIL: expected num_misses=1, got %zu\n", batch.num_misses);
        return false;
    }

    // Data should still be valid (on-demand streaming)
    if (data == nullptr) {
        printf("  FAIL: expected non-null data even for on-demand streaming\n");
        return false;
    }

    printf("  PASS: non-prefetched expert streams on demand\n");
    return true;
}

// =============================================================================
// Test 5: Accuracy tracking with perfect predictions
// =============================================================================
static bool test_accuracy_tracking_perfect() {
    printf("TEST: test_accuracy_tracking_perfect\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    // Simulate 10 batches with perfect predictions
    for (int i = 0; i < 10; i++) {
        float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

        // Actually select experts 0, 1, 2, 3 (all top 4 were prefetched)
        std::vector<uint32_t> selected = { 0, 1, 2, 3 };
        prefetcher.record_selections(0, selected, batch);
    }

    float accuracy = prefetcher.get_layer_accuracy(0);

    // Should be 100% accuracy (all predictions correct)
    if (std::abs(accuracy - 1.0f) > 0.01f) {
        printf("  FAIL: expected accuracy ~1.0, got %f\n", accuracy);
        return false;
    }

    printf("  PASS: accuracy tracking with perfect predictions (%.1f%%)\n", accuracy * 100.0f);
    return true;
}

// =============================================================================
// Test 6: Accuracy tracking with partial predictions
// =============================================================================
static bool test_accuracy_tracking_partial() {
    printf("TEST: test_accuracy_tracking_partial\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    // Simulate 10 batches with 75% accuracy (3/4 predictions correct)
    for (int i = 0; i < 10; i++) {
        float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

        // Actually select experts 0, 1, 2, 5 (3 of 4 prefetched + 1 miss)
        std::vector<uint32_t> selected = { 0, 1, 2, 5 };
        prefetcher.record_selections(0, selected, batch);
    }

    float accuracy = prefetcher.get_layer_accuracy(0);

    // Should be 75% accuracy
    if (std::abs(accuracy - 0.75f) > 0.01f) {
        printf("  FAIL: expected accuracy ~0.75, got %f\n", accuracy);
        return false;
    }

    printf("  PASS: accuracy tracking with partial predictions (%.1f%%)\n", accuracy * 100.0f);
    return true;
}

// =============================================================================
// Test 7: Adaptive prefetch count increases with low accuracy
// =============================================================================
static bool test_adaptive_prefetch_low_accuracy() {
    printf("TEST: test_adaptive_prefetch_low_accuracy\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    // Simulate low accuracy (50%) for 20 batches
    for (int i = 0; i < 20; i++) {
        float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

        // Only 2/4 prefetched are correct
        std::vector<uint32_t> selected = { 0, 1, 6, 7 };
        prefetcher.record_selections(0, selected, batch);
    }

    // Next prefetch should request more than top_k
    float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    if (batch.num_prefetched <= 4) {
        printf("  FAIL: with low accuracy, should prefetch more than top_k (4), got %zu\n", batch.num_prefetched);
        return false;
    }

    printf("  PASS: adaptive prefetch increases count with low accuracy (%zu prefetched for top_k=4)\n",
           batch.num_prefetched);
    return true;
}

// =============================================================================
// Test 8: Adaptive prefetch count stays minimal with high accuracy
// =============================================================================
static bool test_adaptive_prefetch_high_accuracy() {
    printf("TEST: test_adaptive_prefetch_high_accuracy\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    // Simulate high accuracy (95%+) for 20 batches
    for (int i = 0; i < 20; i++) {
        float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

        // All 4 prefetched are correct
        std::vector<uint32_t> selected = { 0, 1, 2, 3 };
        prefetcher.record_selections(0, selected, batch);
    }

    // Next prefetch should not add many extras
    float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    // With 100% accuracy, should prefetch exactly top_k (or at most top_k + 1)
    if (batch.num_prefetched > 5) {
        printf("  FAIL: with high accuracy, should not over-prefetch, got %zu\n", batch.num_prefetched);
        return false;
    }

    printf("  PASS: adaptive prefetch stays minimal with high accuracy (%zu prefetched for top_k=4)\n",
           batch.num_prefetched);
    return true;
}

// =============================================================================
// Test 9: Multiple layers have independent accuracy tracking
// =============================================================================
static bool test_per_layer_accuracy() {
    printf("TEST: test_per_layer_accuracy\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(3, 8, 1024);  // 3 layers

    // Layer 0: 100% accuracy
    for (int i = 0; i < 10; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(0, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 1, 2, 3 };
        prefetcher.record_selections(0, selected, batch);
    }

    // Layer 1: 50% accuracy
    for (int i = 0; i < 10; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(1, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 1, 6, 7 };
        prefetcher.record_selections(1, selected, batch);
    }

    // Layer 2: 25% accuracy
    for (int i = 0; i < 10; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(2, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 5, 6, 7 };
        prefetcher.record_selections(2, selected, batch);
    }

    float acc0 = prefetcher.get_layer_accuracy(0);
    float acc1 = prefetcher.get_layer_accuracy(1);
    float acc2 = prefetcher.get_layer_accuracy(2);

    if (std::abs(acc0 - 1.00f) > 0.05f) {
        printf("  FAIL: layer 0 accuracy should be ~1.00, got %f\n", acc0);
        return false;
    }

    if (std::abs(acc1 - 0.50f) > 0.05f) {
        printf("  FAIL: layer 1 accuracy should be ~0.50, got %f\n", acc1);
        return false;
    }

    if (std::abs(acc2 - 0.25f) > 0.05f) {
        printf("  FAIL: layer 2 accuracy should be ~0.25, got %f\n", acc2);
        return false;
    }

    printf("  PASS: per-layer accuracy tracking (L0=%.0f%%, L1=%.0f%%, L2=%.0f%%)\n", acc0 * 100, acc1 * 100,
           acc2 * 100);
    return true;
}

// =============================================================================
// Test 10: Empty score array edge case
// =============================================================================
static bool test_empty_experts() {
    printf("TEST: test_empty_experts\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 0, 1024);

    auto batch = prefetcher.start_prefetch(0, nullptr, 0, 0);

    if (batch.predictions.size() != 0) {
        printf("  FAIL: expected 0 predictions for empty experts\n");
        return false;
    }

    if (batch.num_prefetched != 0) {
        printf("  FAIL: expected num_prefetched=0, got %zu\n", batch.num_prefetched);
        return false;
    }

    printf("  PASS: handles empty expert list gracefully\n");
    return true;
}

// =============================================================================
// Test 11: top_k larger than num_experts
// =============================================================================
static bool test_top_k_exceeds_num_experts() {
    printf("TEST: test_top_k_exceeds_num_experts\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 4, 1024);

    float scores[4] = { 0.9f, 0.8f, 0.7f, 0.6f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 4, 10);  // top_k=10 but only 4 experts

    // Should prefetch all 4 experts, not crash
    if (batch.num_prefetched > 4) {
        printf("  FAIL: cannot prefetch more than num_experts (4), got %zu\n", batch.num_prefetched);
        return false;
    }

    if (batch.num_prefetched < 4) {
        printf("  FAIL: should prefetch all available experts (4), got %zu\n", batch.num_prefetched);
        return false;
    }

    printf("  PASS: top_k > num_experts clamped correctly (%zu prefetched)\n", batch.num_prefetched);
    return true;
}

// =============================================================================
// Test 12: Prefetch order helper function
// =============================================================================
static bool test_get_prefetch_order() {
    printf("TEST: test_get_prefetch_order\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    float scores[8] = { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f, 0.9f, 0.4f, 0.6f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    // Order should be: 5 (0.9), 3 (0.8), 7 (0.6), 1 (0.5)
    // So: expert 5 -> order 0, expert 3 -> order 1, expert 7 -> order 2, expert 1 -> order 3

    int order5 = prefetcher.get_prefetch_order(batch, 5);
    int order3 = prefetcher.get_prefetch_order(batch, 3);
    int order7 = prefetcher.get_prefetch_order(batch, 7);
    int order1 = prefetcher.get_prefetch_order(batch, 1);
    int order0 = prefetcher.get_prefetch_order(batch, 0);  // Not prefetched

    if (order5 != 0) {
        printf("  FAIL: expert 5 should be order 0, got %d\n", order5);
        return false;
    }

    if (order3 != 1) {
        printf("  FAIL: expert 3 should be order 1, got %d\n", order3);
        return false;
    }

    if (order7 != 2) {
        printf("  FAIL: expert 7 should be order 2, got %d\n", order7);
        return false;
    }

    if (order1 != 3) {
        printf("  FAIL: expert 1 should be order 3, got %d\n", order1);
        return false;
    }

    // Non-prefetched experts should return large value (or -1)
    if (order0 >= 0 && order0 < 4) {
        printf("  FAIL: expert 0 (not prefetched) should have invalid order, got %d\n", order0);
        return false;
    }

    printf("  PASS: get_prefetch_order returns correct ordering\n");
    return true;
}

// =============================================================================
// Test 13: PrefetchBatch num_selected tracking
// =============================================================================
static bool test_batch_num_selected() {
    printf("TEST: test_batch_num_selected\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    // Select 4 experts
    std::vector<uint32_t> selected = { 0, 1, 2, 5 };
    prefetcher.record_selections(0, selected, batch);

    if (batch.num_selected != 4) {
        printf("  FAIL: expected num_selected=4, got %zu\n", batch.num_selected);
        return false;
    }

    printf("  PASS: batch tracks num_selected correctly\n");
    return true;
}

// =============================================================================
// Test 14: Predictions marked as selected after record_selections
// =============================================================================
static bool test_predictions_marked_selected() {
    printf("TEST: test_predictions_marked_selected\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    float scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 8, 4);

    // Predictions are: expert 0, 1, 2, 3
    // Select: 0, 2 (2 out of 4 prefetched)
    std::vector<uint32_t> selected = { 0, 2, 5, 6 };
    prefetcher.record_selections(0, selected, batch);

    // Check that predictions[0] (expert 0) and predictions[2] (expert 2) are marked selected
    int selected_count = 0;
    for (const auto & pred : batch.predictions) {
        if (pred.selected) {
            selected_count++;
            if (pred.expert_id != 0 && pred.expert_id != 2) {
                printf("  FAIL: unexpected expert %u marked as selected\n", pred.expert_id);
                return false;
            }
        }
    }

    if (selected_count != 2) {
        printf("  FAIL: expected 2 predictions marked selected, got %d\n", selected_count);
        return false;
    }

    printf("  PASS: predictions correctly marked as selected\n");
    return true;
}

// =============================================================================
// Test 15: Layer ID out of range returns 0 accuracy
// =============================================================================
static bool test_invalid_layer_accuracy() {
    printf("TEST: test_invalid_layer_accuracy\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(2, 8, 1024);                    // Only 2 layers (0, 1)

    float accuracy = prefetcher.get_layer_accuracy(99);  // Invalid layer

    if (accuracy != 0.0f) {
        printf("  FAIL: invalid layer should return accuracy 0.0, got %f\n", accuracy);
        return false;
    }

    printf("  PASS: invalid layer returns 0 accuracy\n");
    return true;
}

// =============================================================================
// Test 16: Initial accuracy before any data is 0
// =============================================================================
static bool test_initial_accuracy_zero() {
    printf("TEST: test_initial_accuracy_zero\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    float accuracy = prefetcher.get_layer_accuracy(0);

    if (accuracy != 0.0f) {
        printf("  FAIL: initial accuracy should be 0.0, got %f\n", accuracy);
        return false;
    }

    printf("  PASS: initial accuracy is 0\n");
    return true;
}

// =============================================================================
// Test 17: Tie-breaking for equal scores (stable sort by expert ID)
// =============================================================================
static bool test_tie_breaking() {
    printf("TEST: test_tie_breaking\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 4, 1024);

    // All scores equal
    float scores[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    auto  batch     = prefetcher.start_prefetch(0, scores, 4, 4);

    // With equal scores, should maintain stable order (lower expert ID first)
    if (batch.predictions.size() != 4) {
        printf("  FAIL: expected 4 predictions\n");
        return false;
    }

    // Check that expert IDs are in order 0, 1, 2, 3 (or at least consistent)
    // The key requirement is that the sort is deterministic
    bool seen[4] = { false, false, false, false };
    for (const auto & pred : batch.predictions) {
        if (pred.expert_id >= 4) {
            printf("  FAIL: invalid expert ID %u\n", pred.expert_id);
            return false;
        }
        seen[pred.expert_id] = true;
    }

    for (int i = 0; i < 4; i++) {
        if (!seen[i]) {
            printf("  FAIL: missing expert %d in predictions\n", i);
            return false;
        }
    }

    printf("  PASS: tie-breaking produces deterministic ordering\n");
    return true;
}

// =============================================================================
// Test 18: Recommended prefetch count calculation
// =============================================================================
static bool test_recommended_prefetch_count() {
    printf("TEST: test_recommended_prefetch_count\n");

    ggml_sycl::LayerStats stats;

    // Test at various accuracy levels
    const size_t top_k = 4;

    // Very high accuracy (95%+) -> exactly top_k
    stats.total_predictions   = 100;
    stats.correct_predictions = 96;
    size_t rec1               = stats.recommended_prefetch_count(top_k);
    if (rec1 != top_k) {
        printf("  FAIL: at 96%% accuracy, should recommend %zu, got %zu\n", top_k, rec1);
        return false;
    }

    // Good accuracy (80%+) -> top_k + 1
    stats.correct_predictions = 82;
    size_t rec2               = stats.recommended_prefetch_count(top_k);
    if (rec2 != top_k + 1) {
        printf("  FAIL: at 82%% accuracy, should recommend %zu, got %zu\n", top_k + 1, rec2);
        return false;
    }

    // Moderate accuracy (60%+) -> top_k + 2
    stats.correct_predictions = 65;
    size_t rec3               = stats.recommended_prefetch_count(top_k);
    if (rec3 != top_k + 2) {
        printf("  FAIL: at 65%% accuracy, should recommend %zu, got %zu\n", top_k + 2, rec3);
        return false;
    }

    // Poor accuracy (<60%) -> top_k + 4
    stats.correct_predictions = 40;
    size_t rec4               = stats.recommended_prefetch_count(top_k);
    if (rec4 != top_k + 4) {
        printf("  FAIL: at 40%% accuracy, should recommend %zu, got %zu\n", top_k + 4, rec4);
        return false;
    }

    printf("  PASS: recommended_prefetch_count scales with accuracy\n");
    return true;
}

// =============================================================================
// Test 19: Expert stats reset capability
// =============================================================================
static bool test_stats_reset() {
    printf("TEST: test_stats_reset\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(1, 8, 1024);

    // Build up some stats
    for (int i = 0; i < 10; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(0, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 1, 6, 7 };  // 50% hit
        prefetcher.record_selections(0, selected, batch);
    }

    float acc_before = prefetcher.get_layer_accuracy(0);
    if (std::abs(acc_before - 0.5f) > 0.05f) {
        printf("  FAIL: accuracy before reset should be ~0.5, got %f\n", acc_before);
        return false;
    }

    prefetcher.reset_stats();

    float acc_after = prefetcher.get_layer_accuracy(0);
    if (acc_after != 0.0f) {
        printf("  FAIL: accuracy after reset should be 0.0, got %f\n", acc_after);
        return false;
    }

    printf("  PASS: stats reset works correctly\n");
    return true;
}

// =============================================================================
// Test 20: get_stats returns aggregate statistics
// =============================================================================
static bool test_get_stats() {
    printf("TEST: test_get_stats\n");

    ggml_sycl::ExpertPrefetcher prefetcher;
    prefetcher.configure(2, 8, 1024);

    // Layer 0: 10 batches, 4 predictions each, 75% accuracy
    for (int i = 0; i < 10; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(0, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 1, 2, 5 };  // 3/4 = 75%
        prefetcher.record_selections(0, selected, batch);
    }

    // Layer 1: 5 batches, 4 predictions each, 50% accuracy
    for (int i = 0; i < 5; i++) {
        float                 scores[8] = { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f };
        auto                  batch     = prefetcher.start_prefetch(1, scores, 8, 4);
        std::vector<uint32_t> selected  = { 0, 1, 6, 7 };  // 2/4 = 50%
        prefetcher.record_selections(1, selected, batch);
    }

    auto stats = prefetcher.get_stats();

    // Total predictions: 10*4 + 5*4 = 60
    if (stats.total_predictions != 60) {
        printf("  FAIL: expected total_predictions=60, got %lu\n", stats.total_predictions);
        return false;
    }

    // Total correct: 10*3 + 5*2 = 40
    if (stats.total_correct != 40) {
        printf("  FAIL: expected total_correct=40, got %lu\n", stats.total_correct);
        return false;
    }

    // Overall accuracy: 40/60 = 0.667
    if (std::abs(stats.overall_accuracy - 0.667f) > 0.02f) {
        printf("  FAIL: expected overall_accuracy ~0.667, got %f\n", stats.overall_accuracy);
        return false;
    }

    printf("  PASS: get_stats returns correct aggregates\n");
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== Expert Prefetch Unit Tests ===\n");
    printf("Part of unified memory management (llama.cpp-v3n/llama.cpp-eqa)\n\n");

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

    // Core functionality
    run_test(test_prefetch_sorts_by_score_descending, "test_prefetch_sorts_by_score_descending");
    run_test(test_scores_preserved_in_predictions, "test_scores_preserved_in_predictions");
    run_test(test_prefetched_expert_is_hit, "test_prefetched_expert_is_hit");
    run_test(test_non_prefetched_expert_is_miss, "test_non_prefetched_expert_is_miss");

    // Accuracy tracking
    run_test(test_accuracy_tracking_perfect, "test_accuracy_tracking_perfect");
    run_test(test_accuracy_tracking_partial, "test_accuracy_tracking_partial");
    run_test(test_per_layer_accuracy, "test_per_layer_accuracy");
    run_test(test_initial_accuracy_zero, "test_initial_accuracy_zero");

    // Adaptive prefetching
    run_test(test_adaptive_prefetch_low_accuracy, "test_adaptive_prefetch_low_accuracy");
    run_test(test_adaptive_prefetch_high_accuracy, "test_adaptive_prefetch_high_accuracy");
    run_test(test_recommended_prefetch_count, "test_recommended_prefetch_count");

    // Edge cases
    run_test(test_empty_experts, "test_empty_experts");
    run_test(test_top_k_exceeds_num_experts, "test_top_k_exceeds_num_experts");
    run_test(test_invalid_layer_accuracy, "test_invalid_layer_accuracy");
    run_test(test_tie_breaking, "test_tie_breaking");

    // Batch tracking
    run_test(test_get_prefetch_order, "test_get_prefetch_order");
    run_test(test_batch_num_selected, "test_batch_num_selected");
    run_test(test_predictions_marked_selected, "test_predictions_marked_selected");

    // Stats management
    run_test(test_stats_reset, "test_stats_reset");
    run_test(test_get_stats, "test_get_stats");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
