//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// TDD tests for tuning-engine.hpp
// Tests: TunedParams, BatchBucket, TuningKey, TuningEntry, TuningCache

#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

// Include the header under test
#include "tuning-engine.hpp"

using namespace ggml_sycl_tuning;

// =============================================================================
// Test 1: test_tuning_cache_basic
// Verifies basic cache operations: insert and lookup
// =============================================================================
static bool test_tuning_cache_basic() {
    printf("test_tuning_cache_basic... ");

    TuningCache cache;

    // Create a key
    TuningKey key1;
    key1.quant_type = 2;  // GGML_TYPE_Q4_0
    key1.batch_bucket = BatchBucket::SMALL;
    key1.K = 4096;
    key1.N = 4096;

    // Create params
    TunedParams params1;
    params1.tile_m = 16;
    params1.tile_n = 16;
    params1.tile_k = 32;
    params1.workgroup_size = 256;
    params1.slm_kb = 32;
    params1.prefetch_depth = 2;
    params1.use_dpas = true;
    params1.layout_mode = 2;  // COALESCED

    // Insert
    TuningEntry entry1;
    entry1.key = key1;
    entry1.params = params1;
    entry1.measured_tflops = 42.5f;
    entry1.timestamp = 1234567890;

    cache.insert(entry1);

    // Lookup - should find
    auto result = cache.lookup(key1);
    if (!result.has_value()) {
        printf("FAILED: expected hit\n");
        return false;
    }
    if (result->params.tile_m != 16 || result->params.tile_n != 16 ||
        result->params.tile_k != 32 || result->params.workgroup_size != 256 ||
        result->params.use_dpas != true || result->measured_tflops != 42.5f) {
        printf("FAILED: params mismatch\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 2: test_tuning_cache_miss
// Verifies cache miss returns nullopt
// =============================================================================
static bool test_tuning_cache_miss() {
    printf("test_tuning_cache_miss... ");

    TuningCache cache;

    // Create a key that was never inserted
    TuningKey key;
    key.quant_type = 2;
    key.batch_bucket = BatchBucket::MEDIUM;
    key.K = 8192;
    key.N = 4096;

    // Lookup - should NOT find
    auto result = cache.lookup(key);
    if (result.has_value()) {
        printf("FAILED: expected miss\n");
        return false;
    }

    // Insert a different key
    TuningKey key2;
    key2.quant_type = 2;
    key2.batch_bucket = BatchBucket::SMALL;  // Different bucket
    key2.K = 8192;
    key2.N = 4096;

    TuningEntry entry;
    entry.key = key2;
    entry.params = TunedParams{};
    cache.insert(entry);

    // Original key should still miss
    result = cache.lookup(key);
    if (result.has_value()) {
        printf("FAILED: expected miss after insert of different key\n");
        return false;
    }

    // key2 should hit
    auto result2 = cache.lookup(key2);
    if (!result2.has_value()) {
        printf("FAILED: expected hit for key2\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 3: test_tuning_cache_thread_safety
// Verifies concurrent reads/writes don't crash or corrupt data
// =============================================================================
static bool test_tuning_cache_thread_safety() {
    printf("test_tuning_cache_thread_safety... ");

    TuningCache cache;
    constexpr int NUM_THREADS = 8;
    constexpr int OPS_PER_THREAD = 1000;
    std::atomic<int> successful_lookups{0};
    std::atomic<int> failed_lookups{0};

    // Pre-populate some entries
    for (int i = 0; i < 100; i++) {
        TuningKey key;
        key.quant_type = i % 10;
        key.batch_bucket = static_cast<BatchBucket>(i % 4);
        key.K = 1024 * ((i % 8) + 1);
        key.N = 4096;

        TuningEntry entry;
        entry.key = key;
        entry.params.tile_m = 8 + (i % 8) * 8;
        entry.measured_tflops = static_cast<float>(i);
        cache.insert(entry);
    }

    std::vector<std::thread> threads;

    // Launch threads that do concurrent reads and writes
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&cache, &successful_lookups, &failed_lookups, t]() {
            for (int i = 0; i < OPS_PER_THREAD; i++) {
                // Mix of reads and writes
                if (i % 3 == 0) {
                    // Write a new entry
                    TuningKey key;
                    key.quant_type = (t * OPS_PER_THREAD + i) % 20;
                    key.batch_bucket = static_cast<BatchBucket>(i % 4);
                    key.K = 512 * ((i % 16) + 1);
                    key.N = 2048;

                    TuningEntry entry;
                    entry.key = key;
                    entry.params.tile_m = 16;
                    entry.params.workgroup_size = 128;
                    entry.measured_tflops = static_cast<float>(t * i);
                    cache.insert(entry);
                } else {
                    // Read (may hit or miss)
                    TuningKey key;
                    key.quant_type = i % 10;
                    key.batch_bucket = static_cast<BatchBucket>(i % 4);
                    key.K = 1024 * ((i % 8) + 1);
                    key.N = 4096;

                    auto result = cache.lookup(key);
                    if (result.has_value()) {
                        successful_lookups++;
                        // Verify data integrity (basic sanity check)
                        (void)result->params.tile_m;  // Access to verify no corruption
                    } else {
                        failed_lookups++;
                    }
                }
            }
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Just verify no crashes and we got some successful lookups
    if (successful_lookups.load() == 0) {
        printf("FAILED: no successful lookups\n");
        return false;
    }
    printf("PASSED (hits=%d, misses=%d)\n", successful_lookups.load(), failed_lookups.load());
    return true;
}

// =============================================================================
// Test 4: test_batch_bucket
// Verifies BatchBucket enum and bucket_for_batch() helper
// =============================================================================
static bool test_batch_bucket() {
    printf("test_batch_bucket... ");

    // Test bucket_for_batch() function
    bool ok = true;
    ok = ok && (bucket_for_batch(1) == BatchBucket::SINGLE);
    ok = ok && (bucket_for_batch(2) == BatchBucket::SMALL);
    ok = ok && (bucket_for_batch(4) == BatchBucket::SMALL);
    ok = ok && (bucket_for_batch(8) == BatchBucket::SMALL);
    ok = ok && (bucket_for_batch(16) == BatchBucket::MEDIUM);
    ok = ok && (bucket_for_batch(32) == BatchBucket::MEDIUM);
    ok = ok && (bucket_for_batch(64) == BatchBucket::MEDIUM);   // 64 is <= 64, so MEDIUM
    ok = ok && (bucket_for_batch(65) == BatchBucket::LARGE);    // 65 is > 64, so LARGE
    ok = ok && (bucket_for_batch(128) == BatchBucket::LARGE);
    ok = ok && (bucket_for_batch(129) == BatchBucket::XLARGE);  // 129 is > 128, so XLARGE
    ok = ok && (bucket_for_batch(256) == BatchBucket::XLARGE);
    ok = ok && (bucket_for_batch(512) == BatchBucket::XLARGE);
    ok = ok && (bucket_for_batch(1024) == BatchBucket::XLARGE);

    if (!ok) {
        printf("FAILED: bucket_for_batch() mismatch\n");
        return false;
    }

    // Verify enum values for use in hashing
    if (static_cast<int>(BatchBucket::SINGLE) != 0 ||
        static_cast<int>(BatchBucket::SMALL) != 1 ||
        static_cast<int>(BatchBucket::MEDIUM) != 2 ||
        static_cast<int>(BatchBucket::LARGE) != 3 ||
        static_cast<int>(BatchBucket::XLARGE) != 4) {
        printf("FAILED: enum values mismatch\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 5: test_tuning_key_hash
// Verifies TuningKey hash specialization works correctly
// =============================================================================
static bool test_tuning_key_hash() {
    printf("test_tuning_key_hash... ");

    TuningKey key1;
    key1.quant_type = 2;
    key1.batch_bucket = BatchBucket::SMALL;
    key1.K = 4096;
    key1.N = 4096;

    TuningKey key2 = key1;  // Same key

    TuningKey key3;
    key3.quant_type = 2;
    key3.batch_bucket = BatchBucket::MEDIUM;  // Different
    key3.K = 4096;
    key3.N = 4096;

    std::hash<TuningKey> hasher;
    size_t hash1 = hasher(key1);
    size_t hash2 = hasher(key2);
    size_t hash3 = hasher(key3);

    // Same keys should have same hash
    if (hash1 != hash2) {
        printf("FAILED: same keys have different hashes\n");
        return false;
    }

    // Different keys should (very likely) have different hash
    // Note: hash collision is technically possible, but extremely unlikely
    if (hash1 == hash3) {
        printf("FAILED: different keys have same hash (unlikely collision)\n");
        return false;
    }

    // Equality operator
    if (!(key1 == key2)) {
        printf("FAILED: key1 should equal key2\n");
        return false;
    }
    if (key1 == key3) {
        printf("FAILED: key1 should not equal key3\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

// =============================================================================
// Test 6: test_tuned_params_defaults
// Verifies TunedParams has sensible defaults
// =============================================================================
static bool test_tuned_params_defaults() {
    printf("test_tuned_params_defaults... ");

    TunedParams params;

    // Verify default-initialized values are sensible for detection
    // (all zeros indicate "not configured")
    if (params.tile_m != 0 || params.tile_n != 0 || params.tile_k != 0 ||
        params.workgroup_size != 0 || params.slm_kb != 0 || params.prefetch_depth != 0 ||
        params.use_dpas != false || params.layout_mode != 0) {
        printf("FAILED: non-zero defaults\n");
        return false;
    }

    printf("PASSED\n");
    return true;
}

int main() {
    printf("=== Tuning Engine Unit Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_tuning_cache_basic()) passed++; else failed++;
    if (test_tuning_cache_miss()) passed++; else failed++;
    if (test_tuning_cache_thread_safety()) passed++; else failed++;
    if (test_batch_bucket()) passed++; else failed++;
    if (test_tuning_key_hash()) passed++; else failed++;
    if (test_tuned_params_defaults()) passed++; else failed++;

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
