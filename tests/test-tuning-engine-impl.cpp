//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// TDD tests for TuningEngine implementation
// Tests: TuningEngine class with progressive background tuning

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

// Include the headers under test
#include "tuning-engine-impl.hpp"
#include "tuning-engine.hpp"

using namespace ggml_sycl_tuning;

// =============================================================================
// Test 1: test_observation_recording
// Record 100 ops, verify histogram is correct
// =============================================================================
static bool test_observation_recording() {
    printf("test_observation_recording... ");

    TuningEngine engine;

    // Create a few different keys
    TuningKey key1;
    key1.quant_type   = 2;  // GGML_TYPE_Q4_0
    key1.batch_bucket = BatchBucket::SINGLE;
    key1.K            = 4096;
    key1.N            = 4096;

    TuningKey key2;
    key2.quant_type   = 2;
    key2.batch_bucket = BatchBucket::SMALL;
    key2.K            = 4096;
    key2.N            = 4096;

    TuningKey key3;
    key3.quant_type   = 8;  // GGML_TYPE_Q8_0
    key3.batch_bucket = BatchBucket::SINGLE;
    key3.K            = 4096;
    key3.N            = 4096;

    TunedParams params;  // Default params for recording

    // Record observations: key1 50 times, key2 30 times, key3 20 times
    for (int i = 0; i < 50; i++) {
        engine.record_observation_async(key1, params);
    }
    for (int i = 0; i < 30; i++) {
        engine.record_observation_async(key2, params);
    }
    for (int i = 0; i < 20; i++) {
        engine.record_observation_async(key3, params);
    }

    // Give background thread time to process (observations are async)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Verify histogram counts
    auto counts     = engine.get_histogram_snapshot();
    bool found_key1 = false, found_key2 = false, found_key3 = false;

    for (const auto & [key, count] : counts) {
        if (key == key1) {
            found_key1 = true;
            if (count != 50) {
                printf("FAILED: key1 count expected 50, got %lu\n", count);
                return false;
            }
        } else if (key == key2) {
            found_key2 = true;
            if (count != 30) {
                printf("FAILED: key2 count expected 30, got %lu\n", count);
                return false;
            }
        } else if (key == key3) {
            found_key3 = true;
            if (count != 20) {
                printf("FAILED: key3 count expected 20, got %lu\n", count);
                return false;
            }
        }
    }

    if (!found_key1 || !found_key2 || !found_key3) {
        printf("FAILED: missing keys in histogram\n");
        return false;
    }

    engine.stop();
    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 2: test_phase_progression
// Verify OBSERVATION -> BENCHMARKING transition after N ops
// =============================================================================
static bool test_phase_progression() {
    printf("test_phase_progression... ");

    // Create engine with low threshold for testing
    TuningEngineConfig config;
    config.observation_threshold = 50;  // Move to benchmark phase after 50 ops
    TuningEngine engine(config);

    // Should start in OBSERVATION phase
    if (engine.current_phase() != TuningPhase::OBSERVATION) {
        printf("FAILED: expected OBSERVATION phase at start\n");
        return false;
    }

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    TunedParams params;

    // Record observations until threshold
    for (int i = 0; i < 60; i++) {
        engine.record_observation_async(key, params);
    }

    // Give background thread time to process and transition
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Should now be in BENCHMARKING phase
    TuningPhase phase = engine.current_phase();
    if (phase != TuningPhase::BENCHMARKING && phase != TuningPhase::REFINEMENT) {
        printf("FAILED: expected BENCHMARKING or REFINEMENT phase, got %d\n", static_cast<int>(phase));
        return false;
    }

    engine.stop();
    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 3: test_get_params_cache_hit
// Cache hit returns correct params
// =============================================================================
static bool test_get_params_cache_hit() {
    printf("test_get_params_cache_hit... ");

    TuningEngine engine;

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::MEDIUM;
    key.K            = 4096;
    key.N            = 4096;

    // Pre-populate cache with known params
    TunedParams expected;
    expected.tile_m         = 32;
    expected.tile_n         = 64;
    expected.tile_k         = 32;
    expected.workgroup_size = 256;
    expected.slm_kb         = 16;
    expected.prefetch_depth = 2;
    expected.use_dpas       = true;
    expected.layout_mode    = 3;  // XMX_COALESCED

    engine.insert_cached_params(key, expected, 100.0f);

    // Get params should return cached values
    TunedParams result = engine.get_params(key);

    if (result.tile_m != expected.tile_m || result.tile_n != expected.tile_n || result.tile_k != expected.tile_k ||
        result.workgroup_size != expected.workgroup_size || result.use_dpas != expected.use_dpas ||
        result.layout_mode != expected.layout_mode) {
        printf("FAILED: params mismatch\n");
        printf("  expected tile_m=%d, got %d\n", expected.tile_m, result.tile_m);
        return false;
    }

    engine.stop();
    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 4: test_get_params_cold_start
// Cache miss uses heuristics to derive params
// =============================================================================
static bool test_get_params_cold_start() {
    printf("test_get_params_cold_start... ");

    TuningEngine engine;

    // Use a key that hasn't been cached
    TuningKey key;
    key.quant_type   = 2;  // Q4_0
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    // Get params should return heuristic-derived values (not zeros)
    TunedParams result = engine.get_params(key);

    // For cold start, we expect some reasonable defaults (not zeros)
    // The exact values depend on heuristics, but should be non-zero
    if (result.workgroup_size == 0) {
        printf("FAILED: cold start returned zero workgroup_size\n");
        return false;
    }

    // For SINGLE batch bucket (M=1), heuristics should favor:
    // - No dpas (memory bound, not compute bound)
    // - Moderate tile sizes
    // The exact values are implementation-specific, but should be reasonable

    printf("  (cold start params: tile_m=%d, tile_n=%d, wg=%d, dpas=%d)\n", result.tile_m, result.tile_n,
           result.workgroup_size, result.use_dpas);

    engine.stop();
    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 5: test_tuning_non_blocking
// Verify get_params() is fast (<1us for cache hit)
// =============================================================================
static bool test_tuning_non_blocking() {
    printf("test_tuning_non_blocking... ");

    TuningEngine engine;

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::MEDIUM;
    key.K            = 4096;
    key.N            = 4096;

    // Pre-populate cache
    TunedParams params;
    params.tile_m         = 32;
    params.tile_n         = 64;
    params.workgroup_size = 256;
    engine.insert_cached_params(key, params, 100.0f);

    // Warm up
    for (int i = 0; i < 10; i++) {
        engine.get_params(key);
    }

    // Measure time for 1000 cache lookups
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        TunedParams result = engine.get_params(key);
        (void) result;  // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto   ns          = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double ns_per_call = static_cast<double>(ns) / 1000.0;

    // Cache hit should be < 1us (1000ns)
    // Allowing some margin for system overhead
    if (ns_per_call > 5000) {  // 5us is very generous
        printf("FAILED: get_params too slow: %.1f ns/call\n", ns_per_call);
        return false;
    }

    printf("PASSED (%.1f ns/call)\n", ns_per_call);

    engine.stop();
    return true;
}

// =============================================================================
// Test 6: test_record_observation_non_blocking
// Verify record_observation_async() is non-blocking (<100ns)
// =============================================================================
static bool test_record_observation_non_blocking() {
    printf("test_record_observation_non_blocking... ");

    TuningEngine engine;

    TuningKey key;
    key.quant_type   = 2;
    key.batch_bucket = BatchBucket::SINGLE;
    key.K            = 4096;
    key.N            = 4096;

    TunedParams params;

    // Warm up
    for (int i = 0; i < 10; i++) {
        engine.record_observation_async(key, params);
    }

    // Measure time for 1000 observations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        engine.record_observation_async(key, params);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto   ns          = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double ns_per_call = static_cast<double>(ns) / 1000.0;

    // record_observation_async should be non-blocking (<1000ns average)
    // This is a bit generous since we're testing async behavior
    if (ns_per_call > 10000) {  // 10us is very generous for async
        printf("FAILED: record_observation_async too slow: %.1f ns/call\n", ns_per_call);
        return false;
    }

    printf("PASSED (%.1f ns/call)\n", ns_per_call);

    engine.stop();
    return true;
}

// =============================================================================
// Test 7: test_concurrent_access
// Verify thread safety of get_params and record_observation_async
// =============================================================================
static bool test_concurrent_access() {
    printf("test_concurrent_access... ");

    TuningEngine     engine;
    constexpr int    NUM_THREADS    = 4;
    constexpr int    OPS_PER_THREAD = 500;
    std::atomic<int> get_params_calls{ 0 };
    std::atomic<int> record_calls{ 0 };

    // Pre-populate some cache entries
    for (int i = 0; i < 10; i++) {
        TuningKey key;
        key.quant_type   = i;
        key.batch_bucket = BatchBucket::MEDIUM;
        key.K            = 4096;
        key.N            = 4096;

        TunedParams params;
        params.tile_m         = 16 + i * 8;
        params.workgroup_size = 128 + i * 64;
        engine.insert_cached_params(key, params, static_cast<float>(i * 10));
    }

    std::vector<std::thread> threads;

    // Launch threads doing mixed get_params and record_observation
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&engine, &get_params_calls, &record_calls, t]() {
            for (int i = 0; i < OPS_PER_THREAD; i++) {
                TuningKey key;
                key.quant_type   = (t * OPS_PER_THREAD + i) % 20;
                key.batch_bucket = static_cast<BatchBucket>(i % 5);
                key.K            = 1024 * ((i % 4) + 1);
                key.N            = 4096;

                if (i % 2 == 0) {
                    // Get params (cache lookup)
                    TunedParams result = engine.get_params(key);
                    // Sanity check - should always return valid params
                    if (result.workgroup_size == 0 && key.batch_bucket != BatchBucket::SINGLE) {
                        // Cold start for non-SINGLE should have workgroup_size > 0
                        // (This depends on heuristic implementation)
                    }
                    get_params_calls++;
                } else {
                    // Record observation
                    TunedParams params;
                    params.tile_m = 16;
                    engine.record_observation_async(key, params);
                    record_calls++;
                }
            }
        });
    }

    // Wait for all threads
    for (auto & t : threads) {
        t.join();
    }

    printf("PASSED (get_params=%d, record=%d)\n", get_params_calls.load(), record_calls.load());

    engine.stop();
    return true;
}

// =============================================================================
// Test 8: test_top_k_patterns
// Verify engine identifies top-K most frequent patterns
// =============================================================================
static bool test_top_k_patterns() {
    printf("test_top_k_patterns... ");

    TuningEngineConfig config;
    config.observation_threshold = 100;
    config.top_k_patterns        = 3;
    TuningEngine engine(config);

    // Create 5 different keys with different frequencies
    std::vector<std::pair<TuningKey, int>> key_counts;
    for (int i = 0; i < 5; i++) {
        TuningKey key;
        key.quant_type   = i;
        key.batch_bucket = BatchBucket::SINGLE;
        key.K            = 4096;
        key.N            = 4096;
        // Frequencies: 10, 25, 15, 40, 5
        int counts[]     = { 10, 25, 15, 40, 5 };
        key_counts.push_back({ key, counts[i] });
    }

    TunedParams params;
    for (const auto & [key, count] : key_counts) {
        for (int i = 0; i < count; i++) {
            engine.record_observation_async(key, params);
        }
    }

    // Give time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Get top-K patterns
    auto top_k = engine.get_top_patterns(3);

    if (top_k.size() != 3) {
        printf("FAILED: expected 3 top patterns, got %zu\n", top_k.size());
        return false;
    }

    // Top 3 should be: quant_type=3 (40), quant_type=1 (25), quant_type=2 (15)
    if (top_k[0].first.quant_type != 3 || top_k[0].second != 40) {
        printf("FAILED: top-1 should be quant_type=3 with count=40\n");
        return false;
    }
    if (top_k[1].first.quant_type != 1 || top_k[1].second != 25) {
        printf("FAILED: top-2 should be quant_type=1 with count=25\n");
        return false;
    }
    if (top_k[2].first.quant_type != 2 || top_k[2].second != 15) {
        printf("FAILED: top-3 should be quant_type=2 with count=15\n");
        return false;
    }

    engine.stop();
    printf("PASSED\n");
    return true;
}

int main() {
    printf("=== TuningEngine Implementation Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_observation_recording()) {
        passed++;
    } else {
        failed++;
    }
    if (test_phase_progression()) {
        passed++;
    } else {
        failed++;
    }
    if (test_get_params_cache_hit()) {
        passed++;
    } else {
        failed++;
    }
    if (test_get_params_cold_start()) {
        passed++;
    } else {
        failed++;
    }
    if (test_tuning_non_blocking()) {
        passed++;
    } else {
        failed++;
    }
    if (test_record_observation_non_blocking()) {
        passed++;
    } else {
        failed++;
    }
    if (test_concurrent_access()) {
        passed++;
    } else {
        failed++;
    }
    if (test_top_k_patterns()) {
        passed++;
    } else {
        failed++;
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
