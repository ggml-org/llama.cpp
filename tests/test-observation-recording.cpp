//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// TDD tests for observation recording and statistics
// Tests: TuningObservation, ObservationStats, ObservationQueue, performance tracking

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

// Include the headers under test
#include "tuning-engine-impl.hpp"
#include "tuning-engine.hpp"

using namespace ggml_sycl_tuning;

// =============================================================================
// Test 1: test_observation_struct
// Verify TuningObservation struct has all required fields
// =============================================================================
static bool test_observation_struct() {
    printf("test_observation_struct... ");

    TuningObservation obs;
    obs.time_ms    = 1.5;
    obs.batch_size = 32;
    obs.flops      = 1000000;
    obs.timestamp  = 12345;

    // Also set key and params
    obs.key.quant_type   = 2;
    obs.key.batch_bucket = BatchBucket::SMALL;
    obs.key.K            = 4096;
    obs.key.N            = 4096;

    obs.params.tile_m = 16;
    obs.params.tile_n = 32;

    if (obs.time_ms != 1.5) {
        printf("FAILED: time_ms mismatch\n");
        return false;
    }
    if (obs.batch_size != 32) {
        printf("FAILED: batch_size mismatch\n");
        return false;
    }
    if (obs.flops != 1000000) {
        printf("FAILED: flops mismatch\n");
        return false;
    }
    if (obs.timestamp != 12345) {
        printf("FAILED: timestamp mismatch\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 2: test_stats_single_sample
// Stats update with single sample should have zero variance
// =============================================================================
static bool test_stats_single_sample() {
    printf("test_stats_single_sample... ");

    ObservationStats stats;
    stats.update(10.0);

    if (stats.sample_count != 1) {
        printf("FAILED: sample_count expected 1, got %d\n", stats.sample_count);
        return false;
    }
    if (std::abs(stats.mean_time_ms - 10.0) > 1e-10) {
        printf("FAILED: mean_time_ms expected 10.0, got %f\n", stats.mean_time_ms);
        return false;
    }
    if (stats.variance != 0.0) {
        printf("FAILED: variance expected 0.0, got %f\n", stats.variance);
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 3: test_stats_multiple_samples
// Stats update with multiple samples should compute correct mean
// =============================================================================
static bool test_stats_multiple_samples() {
    printf("test_stats_multiple_samples... ");

    ObservationStats stats;
    stats.update(10.0);
    stats.update(12.0);
    stats.update(11.0);

    if (stats.sample_count != 3) {
        printf("FAILED: sample_count expected 3, got %d\n", stats.sample_count);
        return false;
    }

    // Mean should be 11
    if (std::abs(stats.mean_time_ms - 11.0) > 1e-10) {
        printf("FAILED: mean_time_ms expected 11.0, got %f\n", stats.mean_time_ms);
        return false;
    }

    // Variance should be positive
    if (stats.variance <= 0.0) {
        printf("FAILED: variance should be > 0, got %f\n", stats.variance);
        return false;
    }

    printf("PASSED (variance=%f)\n", stats.variance);
    return true;
}

// =============================================================================
// Test 4: test_confidence_increases
// Confidence should increase with more consistent samples
// =============================================================================
static bool test_confidence_increases() {
    printf("test_confidence_increases... ");

    ObservationStats stats;

    // Add consistent samples (small variance)
    for (int i = 0; i < 100; i++) {
        stats.update(10.0 + (i % 2) * 0.1);  // Alternates between 10.0 and 10.1
    }

    // High sample count + low variance = high confidence
    if (stats.confidence <= 0.5) {
        printf("FAILED: confidence should be > 0.5, got %f\n", stats.confidence);
        return false;
    }
    if (stats.sample_count != 100) {
        printf("FAILED: sample_count expected 100, got %d\n", stats.sample_count);
        return false;
    }

    printf("PASSED (confidence=%f)\n", stats.confidence);
    return true;
}

// =============================================================================
// Test 5: test_low_confidence_high_variance
// High variance should result in low confidence
// =============================================================================
static bool test_low_confidence_high_variance() {
    printf("test_low_confidence_high_variance... ");

    ObservationStats stats;

    // Add wildly varying samples
    stats.update(1.0);
    stats.update(100.0);
    stats.update(50.0);
    stats.update(10.0);
    stats.update(90.0);

    // High variance = low confidence
    if (stats.confidence >= 0.5) {
        printf("FAILED: confidence should be < 0.5, got %f\n", stats.confidence);
        return false;
    }

    printf("PASSED (confidence=%f)\n", stats.confidence);
    return true;
}

// =============================================================================
// Test 6: test_observation_queue
// Verify queue push and pop operations
// =============================================================================
static bool test_observation_queue() {
    printf("test_observation_queue... ");

    ObservationQueue queue;

    TuningObservation obs1;
    obs1.time_ms = 1.0;

    TuningObservation obs2;
    obs2.time_ms = 2.0;

    queue.push(obs1);
    queue.push(obs2);

    TuningObservation result;
    if (!queue.try_pop(result)) {
        printf("FAILED: expected successful pop\n");
        return false;
    }
    if (result.time_ms != 1.0) {
        printf("FAILED: first pop expected 1.0, got %f\n", result.time_ms);
        return false;
    }

    if (!queue.try_pop(result)) {
        printf("FAILED: expected successful second pop\n");
        return false;
    }
    if (result.time_ms != 2.0) {
        printf("FAILED: second pop expected 2.0, got %f\n", result.time_ms);
        return false;
    }

    // Queue should now be empty
    if (queue.try_pop(result)) {
        printf("FAILED: expected empty queue\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 7: test_queue_shutdown
// Verify queue shutdown flag
// =============================================================================
static bool test_queue_shutdown() {
    printf("test_queue_shutdown... ");

    ObservationQueue queue;

    if (queue.is_shutdown()) {
        printf("FAILED: queue should not be shutdown initially\n");
        return false;
    }

    queue.shutdown();

    if (!queue.is_shutdown()) {
        printf("FAILED: queue should be shutdown after shutdown()\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 8: test_welford_accuracy
// Verify Welford's algorithm computes correct variance
// =============================================================================
static bool test_welford_accuracy() {
    printf("test_welford_accuracy... ");

    ObservationStats stats;

    // Known values: 2, 4, 4, 4, 5, 5, 7, 9
    std::vector<double> values = { 2, 4, 4, 4, 5, 5, 7, 9 };
    for (double v : values) {
        stats.update(v);
    }

    // Mean = 5.0
    if (std::abs(stats.mean_time_ms - 5.0) > 1e-10) {
        printf("FAILED: mean expected 5.0, got %f\n", stats.mean_time_ms);
        return false;
    }

    // Sample variance should be approximately 4.57 (n-1 denominator)
    // Population variance would be 4.0
    // Our Welford implementation uses sample variance
    if (stats.variance < 3.5 || stats.variance > 5.5) {
        printf("FAILED: variance expected ~4.0-4.6, got %f\n", stats.variance);
        return false;
    }

    printf("PASSED (variance=%f)\n", stats.variance);
    return true;
}

// =============================================================================
// Test 9: test_engine_record_with_timing
// Verify TuningEngine records timing observations
// =============================================================================
static bool test_engine_record_with_timing() {
    printf("test_engine_record_with_timing... ");

    TuningEngine engine;

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    TunedParams params;
    params.tile_m = 16;
    params.tile_n = 32;

    // Record multiple observations with timing
    for (int i = 0; i < 50; i++) {
        engine.record_observation_async(key, params, 10.0 + (i % 3) * 0.1);
    }

    // Give background thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check that stats were recorded
    ObservationStats stats = engine.get_stats(key);
    if (stats.sample_count != 50) {
        printf("FAILED: sample_count expected 50, got %d\n", stats.sample_count);
        engine.stop();
        return false;
    }

    // Mean should be around 10.1
    if (stats.mean_time_ms < 9.9 || stats.mean_time_ms > 10.3) {
        printf("FAILED: mean_time_ms expected ~10.1, got %f\n", stats.mean_time_ms);
        engine.stop();
        return false;
    }

    engine.stop();
    printf("PASSED (mean=%f, count=%d)\n", stats.mean_time_ms, stats.sample_count);
    return true;
}

// =============================================================================
// Test 10: test_engine_get_confidence
// Verify get_confidence returns correct values
// =============================================================================
static bool test_engine_get_confidence() {
    printf("test_engine_get_confidence... ");

    TuningEngine engine;

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    TunedParams params;

    // Initially, confidence should be 0 (no observations)
    double initial_conf = engine.get_confidence(key);
    if (initial_conf != 0.0) {
        printf("FAILED: initial confidence expected 0.0, got %f\n", initial_conf);
        engine.stop();
        return false;
    }

    // Add consistent observations
    for (int i = 0; i < 100; i++) {
        engine.record_observation_async(key, params, 10.0 + (i % 2) * 0.05);
    }

    // Give time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    double final_conf = engine.get_confidence(key);
    if (final_conf <= 0.5) {
        printf("FAILED: final confidence should be > 0.5, got %f\n", final_conf);
        engine.stop();
        return false;
    }

    engine.stop();
    printf("PASSED (confidence=%f)\n", final_conf);
    return true;
}

// =============================================================================
// Test 11: test_queue_thread_safety
// Verify queue is thread-safe with concurrent push/pop
// =============================================================================
static bool test_queue_thread_safety() {
    printf("test_queue_thread_safety... ");

    ObservationQueue queue;
    constexpr int    NUM_PRODUCERS      = 4;
    constexpr int    NUM_CONSUMERS      = 2;
    constexpr int    ITEMS_PER_PRODUCER = 1000;

    std::atomic<int> produced{ 0 };
    std::atomic<int> consumed{ 0 };

    std::vector<std::thread> threads;

    // Producer threads
    for (int p = 0; p < NUM_PRODUCERS; p++) {
        threads.emplace_back([&queue, &produced, p]() {
            for (int i = 0; i < ITEMS_PER_PRODUCER; i++) {
                TuningObservation obs;
                obs.time_ms = static_cast<double>(p * ITEMS_PER_PRODUCER + i);
                queue.push(obs);
                produced++;
            }
        });
    }

    // Consumer threads
    for (int c = 0; c < NUM_CONSUMERS; c++) {
        threads.emplace_back([&queue, &consumed, &produced]() {
            while (true) {
                TuningObservation obs;
                if (queue.try_pop(obs)) {
                    consumed++;
                } else {
                    // Check if producers are done and queue is empty
                    if (produced.load() >= NUM_PRODUCERS * ITEMS_PER_PRODUCER) {
                        // Try one more time
                        if (!queue.try_pop(obs)) {
                            break;
                        }
                        consumed++;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        });
    }

    // Wait for all threads
    for (auto & t : threads) {
        t.join();
    }

    int total_produced = NUM_PRODUCERS * ITEMS_PER_PRODUCER;
    if (consumed.load() != total_produced) {
        printf("FAILED: consumed %d, expected %d\n", consumed.load(), total_produced);
        return false;
    }

    printf("PASSED (produced=%d, consumed=%d)\n", produced.load(), consumed.load());
    return true;
}

// =============================================================================
// Test 12: test_stats_monotonic_confidence
// Confidence should generally increase with more samples
// =============================================================================
static bool test_stats_monotonic_confidence() {
    printf("test_stats_monotonic_confidence... ");

    ObservationStats stats;
    double           prev_confidence = -1.0;
    int              violations      = 0;

    // Add consistent samples and check confidence trend
    for (int i = 0; i < 100; i++) {
        stats.update(10.0 + (i % 2) * 0.1);

        // After enough samples (to avoid initial instability), check monotonicity
        if (i >= 10) {
            // Allow small decreases due to variance calculation
            if (stats.confidence < prev_confidence - 0.05) {
                violations++;
            }
        }
        prev_confidence = stats.confidence;
    }

    // Allow a few violations due to numerical instability
    if (violations > 5) {
        printf("FAILED: too many confidence decreases (%d)\n", violations);
        return false;
    }

    printf("PASSED (final_confidence=%f, violations=%d)\n", stats.confidence, violations);
    return true;
}

// =============================================================================
// Test 13: test_engine_observation_with_batch_and_flops
// Verify observation records batch_size and flops
// =============================================================================
static bool test_engine_observation_with_batch_and_flops() {
    printf("test_engine_observation_with_batch_and_flops... ");

    TuningObservation obs;
    obs.key.quant_type   = 2;
    obs.key.batch_bucket = BatchBucket::MEDIUM;
    obs.key.K            = 4096;
    obs.key.N            = 4096;
    obs.params.tile_m    = 32;
    obs.params.use_dpas  = true;
    obs.time_ms          = 5.5;
    obs.batch_size       = 64;
    obs.flops            = 2L * 64 * 4096 * 4096;  // 2*M*N*K FLOPS for GEMM
    obs.timestamp        = std::chrono::steady_clock::now().time_since_epoch().count();

    // Verify all fields can be set and retrieved
    if (obs.batch_size != 64) {
        printf("FAILED: batch_size mismatch\n");
        return false;
    }

    int64_t expected_flops = 2L * 64 * 4096 * 4096;
    if (obs.flops != expected_flops) {
        printf("FAILED: flops mismatch (expected %ld, got %ld)\n", expected_flops, obs.flops);
        return false;
    }

    // Verify TFLOPS calculation from recorded data
    // TFLOPS = FLOPS / (time_ms * 1e9) = FLOPS / (time_s * 1e12)
    // For this example: 2*64*4096*4096 / (5.5 * 1e9) = ~0.39 TFLOPS = 390 GFLOPS
    // This is realistic for a small GEMM on GPU
    double tflops = static_cast<double>(obs.flops) / (obs.time_ms * 1e9);
    if (tflops < 0.1 || tflops > 10.0) {  // Sanity check range: 0.1-10 TFLOPS for small GEMM
        printf("FAILED: computed TFLOPS out of range: %f\n", tflops);
        return false;
    }

    printf("PASSED (tflops=%f)\n", tflops);
    return true;
}

int main() {
    printf("=== Observation Recording Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_observation_struct()) {
        passed++;
    } else {
        failed++;
    }
    if (test_stats_single_sample()) {
        passed++;
    } else {
        failed++;
    }
    if (test_stats_multiple_samples()) {
        passed++;
    } else {
        failed++;
    }
    if (test_confidence_increases()) {
        passed++;
    } else {
        failed++;
    }
    if (test_low_confidence_high_variance()) {
        passed++;
    } else {
        failed++;
    }
    if (test_observation_queue()) {
        passed++;
    } else {
        failed++;
    }
    if (test_queue_shutdown()) {
        passed++;
    } else {
        failed++;
    }
    if (test_welford_accuracy()) {
        passed++;
    } else {
        failed++;
    }
    if (test_engine_record_with_timing()) {
        passed++;
    } else {
        failed++;
    }
    if (test_engine_get_confidence()) {
        passed++;
    } else {
        failed++;
    }
    if (test_queue_thread_safety()) {
        passed++;
    } else {
        failed++;
    }
    if (test_stats_monotonic_confidence()) {
        passed++;
    } else {
        failed++;
    }
    if (test_engine_observation_with_batch_and_flops()) {
        passed++;
    } else {
        failed++;
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
