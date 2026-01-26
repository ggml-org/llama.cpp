//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for OneDNNFallback wrapper
// These tests verify the interface, caching behavior, and error handling
// without requiring actual oneDNN execution.

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include "../ggml/src/ggml-sycl/onednn-fallback.hpp"

using namespace ggml_sycl;

// Test counter for reporting
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << __func__ << ": " << msg << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        if (test_func()) { \
            tests_passed++; \
            std::cout << "[PASS] " << #test_func << std::endl; \
        } else { \
            tests_failed++; \
            std::cout << "[FAIL] " << #test_func << std::endl; \
        } \
    } while(0)

// =============================================================================
// Test 1: Initialization behavior
// =============================================================================
bool test_init() {
    OneDNNFallback fallback;

    // Should not be initialized by default
    TEST_ASSERT(fallback.is_initialized() == false,
        "Should not be initialized by default");

    // Initialize with null should fail
    bool result = fallback.init(nullptr);
    TEST_ASSERT(result == false, "Init with nullptr should return false");
    TEST_ASSERT(fallback.is_initialized() == false,
        "Should not be initialized after null init");

    // Initialize with valid pointer should succeed
    int dummy_queue;
    result = fallback.init(&dummy_queue);
    TEST_ASSERT(result == true, "Init with valid pointer should return true");
    TEST_ASSERT(fallback.is_initialized() == true,
        "Should be initialized after valid init");
    TEST_ASSERT(fallback.get_queue() == &dummy_queue,
        "get_queue should return the initialized queue");

    return true;
}

// =============================================================================
// Test 2: M dimension bucketing
// =============================================================================
bool test_m_bucketing() {
    // Edge cases
    TEST_ASSERT(OneDNNFallback::bucket_m(0) == 0, "bucket_m(0) should be 0");
    TEST_ASSERT(OneDNNFallback::bucket_m(-1) == 0, "bucket_m(-1) should be 0");

    // Exact values for M <= 8
    TEST_ASSERT(OneDNNFallback::bucket_m(1) == 1, "bucket_m(1) should be 1");
    TEST_ASSERT(OneDNNFallback::bucket_m(4) == 4, "bucket_m(4) should be 4");
    TEST_ASSERT(OneDNNFallback::bucket_m(8) == 8, "bucket_m(8) should be 8");

    // Multiples of 8 for M in (8, 32]
    TEST_ASSERT(OneDNNFallback::bucket_m(9) == 16, "bucket_m(9) should be 16");
    TEST_ASSERT(OneDNNFallback::bucket_m(10) == 16, "bucket_m(10) should be 16");
    TEST_ASSERT(OneDNNFallback::bucket_m(16) == 16, "bucket_m(16) should be 16");
    TEST_ASSERT(OneDNNFallback::bucket_m(17) == 24, "bucket_m(17) should be 24");
    TEST_ASSERT(OneDNNFallback::bucket_m(24) == 24, "bucket_m(24) should be 24");
    TEST_ASSERT(OneDNNFallback::bucket_m(32) == 32, "bucket_m(32) should be 32");

    // Multiples of 32 for M in (32, 128]
    TEST_ASSERT(OneDNNFallback::bucket_m(33) == 64, "bucket_m(33) should be 64");
    TEST_ASSERT(OneDNNFallback::bucket_m(50) == 64, "bucket_m(50) should be 64");
    TEST_ASSERT(OneDNNFallback::bucket_m(64) == 64, "bucket_m(64) should be 64");
    TEST_ASSERT(OneDNNFallback::bucket_m(65) == 96, "bucket_m(65) should be 96");
    TEST_ASSERT(OneDNNFallback::bucket_m(100) == 128, "bucket_m(100) should be 128");
    TEST_ASSERT(OneDNNFallback::bucket_m(128) == 128, "bucket_m(128) should be 128");

    // Multiples of 64 for M > 128
    TEST_ASSERT(OneDNNFallback::bucket_m(129) == 192, "bucket_m(129) should be 192");
    TEST_ASSERT(OneDNNFallback::bucket_m(200) == 256, "bucket_m(200) should be 256");
    TEST_ASSERT(OneDNNFallback::bucket_m(256) == 256, "bucket_m(256) should be 256");
    TEST_ASSERT(OneDNNFallback::bucket_m(300) == 320, "bucket_m(300) should be 320");

    return true;
}

// =============================================================================
// Test 3: Suitability check for oneDNN
// =============================================================================
bool test_suitability() {
    // Suitable dimensions (M >= 8, N >= 64, K >= 64)
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(8, 64, 64) == true,
        "8x64x64 should be suitable");
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(16, 128, 256) == true,
        "16x128x256 should be suitable");
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(1024, 1024, 1024) == true,
        "1024x1024x1024 should be suitable");

    // M too small
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(1, 64, 64) == false,
        "1x64x64 should not be suitable (M too small)");
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(7, 64, 64) == false,
        "7x64x64 should not be suitable (M too small)");

    // N too small
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(8, 32, 64) == false,
        "8x32x64 should not be suitable (N too small)");
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(8, 63, 64) == false,
        "8x63x64 should not be suitable (N too small)");

    // K too small
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(8, 64, 32) == false,
        "8x64x32 should not be suitable (K too small)");
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(8, 64, 63) == false,
        "8x64x63 should not be suitable (K too small)");

    // All too small
    TEST_ASSERT(OneDNNFallback::is_suitable_for_onednn(1, 1, 1) == false,
        "1x1x1 should not be suitable");

    return true;
}

// =============================================================================
// Test 4: Compute bound check
// =============================================================================
bool test_compute_bound() {
    // Large matmul should be compute-bound
    TEST_ASSERT(OneDNNFallback::is_compute_bound(1024, 1024, 1024) == true,
        "1024x1024x1024 should be compute-bound");
    TEST_ASSERT(OneDNNFallback::is_compute_bound(512, 512, 512) == true,
        "512x512x512 should be compute-bound");

    // Small matmul is typically memory-bound
    TEST_ASSERT(OneDNNFallback::is_compute_bound(1, 64, 64) == false,
        "1x64x64 should be memory-bound");
    TEST_ASSERT(OneDNNFallback::is_compute_bound(8, 64, 64) == false,
        "8x64x64 should be memory-bound");

    return true;
}

// =============================================================================
// Test 5: Cache starts empty
// =============================================================================
bool test_cache_empty() {
    OneDNNFallback fallback;

    TEST_ASSERT(fallback.get_cache_size() == 0,
        "Cache should start empty");
    TEST_ASSERT(fallback.get_cache_hits() == 0,
        "Cache hits should start at 0");
    TEST_ASSERT(fallback.get_cache_misses() == 0,
        "Cache misses should start at 0");
    TEST_ASSERT(fallback.get_cache_hit_rate() == 0.0f,
        "Cache hit rate should be 0 when empty");
    TEST_ASSERT(fallback.get_total_scratchpad_size() == 0,
        "Total scratchpad should be 0 when empty");

    return true;
}

// =============================================================================
// Test 6: Matmul without initialization fails
// =============================================================================
bool test_matmul_without_init() {
    OneDNNFallback fallback;

    float a[1], b[1], c[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    auto result = fallback.matmul(a, b, c, params);

    TEST_ASSERT(result.success == false,
        "Matmul should fail without initialization");
    TEST_ASSERT(result.error_msg != nullptr,
        "Error message should be set");

    return true;
}

// =============================================================================
// Test 7: Matmul with null pointers fails
// =============================================================================
bool test_matmul_null_pointers() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float buffer[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    // Test null A
    auto result = fallback.matmul(nullptr, buffer, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with null A should fail");

    // Test null B
    result = fallback.matmul(buffer, nullptr, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with null B should fail");

    // Test null C
    result = fallback.matmul(buffer, buffer, nullptr, params);
    TEST_ASSERT(result.success == false, "Matmul with null C should fail");

    return true;
}

// =============================================================================
// Test 8: Matmul with invalid dimensions fails
// =============================================================================
bool test_matmul_invalid_dims() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float buffer[1];
    OneDNNMatmulParams params;

    // Test M = 0
    params = make_f32_matmul_params(0, 64, 64);
    auto result = fallback.matmul(buffer, buffer, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with M=0 should fail");

    // Test N = 0
    params = make_f32_matmul_params(8, 0, 64);
    result = fallback.matmul(buffer, buffer, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with N=0 should fail");

    // Test K = 0
    params = make_f32_matmul_params(8, 64, 0);
    result = fallback.matmul(buffer, buffer, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with K=0 should fail");

    // Test negative dimension
    params = make_f32_matmul_params(-1, 64, 64);
    result = fallback.matmul(buffer, buffer, buffer, params);
    TEST_ASSERT(result.success == false, "Matmul with M=-1 should fail");

    return true;
}

// =============================================================================
// Test 9: Cache miss then hit behavior
// =============================================================================
bool test_cache_hit() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    // First call should be a cache miss
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(fallback.get_cache_misses() == 1,
        "First call should be a cache miss");
    TEST_ASSERT(fallback.get_cache_hits() == 0,
        "First call should not be a cache hit");
    TEST_ASSERT(fallback.get_cache_size() == 1,
        "Cache should have one entry after first call");

    // Second call with same params should be a cache hit
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(fallback.get_cache_misses() == 1,
        "Second call should not increase cache misses");
    TEST_ASSERT(fallback.get_cache_hits() == 1,
        "Second call should be a cache hit");
    TEST_ASSERT(fallback.get_cache_size() == 1,
        "Cache size should still be 1");

    // Third call - another hit
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(fallback.get_cache_hits() == 2,
        "Third call should be another cache hit");

    return true;
}

// =============================================================================
// Test 10: Clear cache resets state
// =============================================================================
bool test_clear_cache() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    // Create some cache entries
    fallback.matmul(a, b, c, params);
    fallback.matmul(a, b, c, params);

    TEST_ASSERT(fallback.get_cache_size() == 1, "Should have 1 entry");
    TEST_ASSERT(fallback.get_cache_hits() == 1, "Should have 1 hit");
    TEST_ASSERT(fallback.get_cache_misses() == 1, "Should have 1 miss");

    // Clear the cache
    fallback.clear_cache();

    TEST_ASSERT(fallback.get_cache_size() == 0,
        "Cache should be empty after clear");
    TEST_ASSERT(fallback.get_cache_hits() == 0,
        "Cache hits should reset after clear");
    TEST_ASSERT(fallback.get_cache_misses() == 0,
        "Cache misses should reset after clear");

    // Verify cache is functional after clear
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(fallback.get_cache_size() == 1,
        "Should have new entry after clear");
    TEST_ASSERT(fallback.get_cache_misses() == 1,
        "Should have new miss after clear");

    return true;
}

// =============================================================================
// Test 11: Different dimensions create different cache entries
// =============================================================================
bool test_different_dims() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];

    OneDNNMatmulParams params1 = make_f32_matmul_params(8, 64, 64);
    OneDNNMatmulParams params2 = make_f32_matmul_params(16, 128, 64);
    OneDNNMatmulParams params3 = make_f32_matmul_params(8, 64, 128);

    fallback.matmul(a, b, c, params1);
    fallback.matmul(a, b, c, params2);
    fallback.matmul(a, b, c, params3);

    TEST_ASSERT(fallback.get_cache_size() == 3,
        "Should have 3 different cache entries");
    TEST_ASSERT(fallback.get_cache_misses() == 3,
        "Should have 3 cache misses");

    // Accessing existing entries should be hits
    fallback.matmul(a, b, c, params1);
    fallback.matmul(a, b, c, params2);
    fallback.matmul(a, b, c, params3);

    TEST_ASSERT(fallback.get_cache_size() == 3,
        "Should still have 3 cache entries");
    TEST_ASSERT(fallback.get_cache_hits() == 3,
        "Should have 3 cache hits");

    return true;
}

// =============================================================================
// Test 12: Same bucket = same cache entry
// =============================================================================
bool test_same_bucket() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];

    // M=10 and M=15 both bucket to 16
    OneDNNMatmulParams params1 = make_f32_matmul_params(10, 64, 64);
    OneDNNMatmulParams params2 = make_f32_matmul_params(15, 64, 64);

    fallback.matmul(a, b, c, params1);  // Miss, buckets to M=16
    fallback.matmul(a, b, c, params2);  // Hit, also buckets to M=16

    TEST_ASSERT(fallback.get_cache_size() == 1,
        "Both should use same cache entry (same bucket)");
    TEST_ASSERT(fallback.get_cache_misses() == 1,
        "Should have only 1 miss");
    TEST_ASSERT(fallback.get_cache_hits() == 1,
        "Should have 1 hit");

    // M=9 also buckets to 16
    OneDNNMatmulParams params3 = make_f32_matmul_params(9, 64, 64);
    fallback.matmul(a, b, c, params3);

    TEST_ASSERT(fallback.get_cache_size() == 1,
        "M=9 should also use same cache entry");
    TEST_ASSERT(fallback.get_cache_hits() == 2,
        "Should now have 2 hits");

    return true;
}

// =============================================================================
// Test 13: Different buckets = different cache entries
// =============================================================================
bool test_different_buckets() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];

    // M=8 buckets to 8, M=9 buckets to 16
    OneDNNMatmulParams params1 = make_f32_matmul_params(8, 64, 64);
    OneDNNMatmulParams params2 = make_f32_matmul_params(9, 64, 64);

    fallback.matmul(a, b, c, params1);
    fallback.matmul(a, b, c, params2);

    TEST_ASSERT(fallback.get_cache_size() == 2,
        "Different buckets should create different entries");
    TEST_ASSERT(fallback.get_cache_misses() == 2,
        "Should have 2 misses");

    return true;
}

// =============================================================================
// Test 14: Different data types create different cache entries
// =============================================================================
bool test_different_dtypes() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];

    OneDNNMatmulParams params_f32 = make_f32_matmul_params(8, 64, 64);
    OneDNNMatmulParams params_f16 = make_f16_matmul_params(8, 64, 64);
    OneDNNMatmulParams params_mixed = make_mixed_matmul_params(8, 64, 64);

    fallback.matmul(a, b, c, params_f32);
    fallback.matmul(a, b, c, params_f16);
    fallback.matmul(a, b, c, params_mixed);

    TEST_ASSERT(fallback.get_cache_size() == 3,
        "Different data types should create different entries");

    return true;
}

// =============================================================================
// Test 15: Hit rate calculation
// =============================================================================
bool test_hit_rate() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    // 1 miss
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(std::abs(fallback.get_cache_hit_rate() - 0.0f) < 0.01f,
        "Hit rate should be 0% after 1 miss");

    // 1 miss, 1 hit = 50%
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(std::abs(fallback.get_cache_hit_rate() - 0.5f) < 0.01f,
        "Hit rate should be 50% (1 hit, 1 miss)");

    // 1 miss, 2 hits = 66.7%
    fallback.matmul(a, b, c, params);
    TEST_ASSERT(std::abs(fallback.get_cache_hit_rate() - 0.667f) < 0.01f,
        "Hit rate should be ~67% (2 hits, 1 miss)");

    // 1 miss, 3 hits = 75%
    fallback.matmul(a, b, c, params);
    float rate = fallback.get_cache_hit_rate();
    TEST_ASSERT(rate > 0.74f && rate < 0.76f,
        "Hit rate should be 75% (3 hits, 1 miss)");

    return true;
}

// =============================================================================
// Test 16: Successful matmul returns timing info
// =============================================================================
bool test_matmul_timing() {
    OneDNNFallback fallback;
    int dummy;
    fallback.init(&dummy);

    float a[1], b[1], c[1];
    OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

    auto result = fallback.matmul(a, b, c, params);

    TEST_ASSERT(result.success == true, "Matmul should succeed");
    TEST_ASSERT(result.time_ms >= 0.0, "Timing should be non-negative");
    TEST_ASSERT(result.error_msg == nullptr, "Error message should be null on success");

    return true;
}

// =============================================================================
// Test 17: Destructor cleans up properly
// =============================================================================
bool test_destructor() {
    {
        OneDNNFallback fallback;
        int dummy;
        fallback.init(&dummy);

        float a[1], b[1], c[1];
        OneDNNMatmulParams params = make_f32_matmul_params(8, 64, 64);

        // Create some cache entries
        fallback.matmul(a, b, c, params);
        fallback.matmul(a, b, c, make_f32_matmul_params(16, 64, 64));

        // Destructor will be called here
    }

    // If we get here without crashing, destructor worked
    return true;
}

// =============================================================================
// Test 18: Convenience functions create correct params
// =============================================================================
bool test_convenience_functions() {
    auto f32 = make_f32_matmul_params(8, 64, 128);
    TEST_ASSERT(f32.M == 8, "F32 M should be 8");
    TEST_ASSERT(f32.N == 64, "F32 N should be 64");
    TEST_ASSERT(f32.K == 128, "F32 K should be 128");
    TEST_ASSERT(f32.dt_a == OneDNNDataType::F32, "F32 dt_a should be F32");
    TEST_ASSERT(f32.dt_b == OneDNNDataType::F32, "F32 dt_b should be F32");
    TEST_ASSERT(f32.dt_c == OneDNNDataType::F32, "F32 dt_c should be F32");

    auto f16 = make_f16_matmul_params(16, 128, 256);
    TEST_ASSERT(f16.M == 16, "F16 M should be 16");
    TEST_ASSERT(f16.N == 128, "F16 N should be 128");
    TEST_ASSERT(f16.K == 256, "F16 K should be 256");
    TEST_ASSERT(f16.dt_a == OneDNNDataType::F16, "F16 dt_a should be F16");
    TEST_ASSERT(f16.dt_b == OneDNNDataType::F16, "F16 dt_b should be F16");
    TEST_ASSERT(f16.dt_c == OneDNNDataType::F16, "F16 dt_c should be F16");

    auto mixed = make_mixed_matmul_params(32, 64, 64);
    TEST_ASSERT(mixed.M == 32, "Mixed M should be 32");
    TEST_ASSERT(mixed.dt_a == OneDNNDataType::F16, "Mixed dt_a should be F16");
    TEST_ASSERT(mixed.dt_b == OneDNNDataType::F16, "Mixed dt_b should be F16");
    TEST_ASSERT(mixed.dt_c == OneDNNDataType::F32, "Mixed dt_c should be F32");

    return true;
}

// =============================================================================
// Test 19: Large M value bucketing
// =============================================================================
bool test_large_m_bucketing() {
    // Test bucketing for larger M values
    TEST_ASSERT(OneDNNFallback::bucket_m(1000) == 1024, "bucket_m(1000) should be 1024");
    TEST_ASSERT(OneDNNFallback::bucket_m(1024) == 1024, "bucket_m(1024) should be 1024");
    TEST_ASSERT(OneDNNFallback::bucket_m(1025) == 1088, "bucket_m(1025) should be 1088");
    TEST_ASSERT(OneDNNFallback::bucket_m(4096) == 4096, "bucket_m(4096) should be 4096");

    return true;
}

// =============================================================================
// Test 20: Key hash uniqueness
// =============================================================================
bool test_key_hash() {
    OneDNNPrimitiveKeyHash hasher;

    OneDNNPrimitiveKey key1 = {8, 64, 64, 1, 1, 1};
    OneDNNPrimitiveKey key2 = {8, 64, 64, 1, 1, 1};
    OneDNNPrimitiveKey key3 = {16, 64, 64, 1, 1, 1};
    OneDNNPrimitiveKey key4 = {8, 128, 64, 1, 1, 1};
    OneDNNPrimitiveKey key5 = {8, 64, 64, 2, 1, 1};

    // Same keys should have same hash
    TEST_ASSERT(hasher(key1) == hasher(key2),
        "Same keys should have same hash");

    // Different keys should (likely) have different hashes
    // Note: hash collisions are possible but unlikely
    TEST_ASSERT(hasher(key1) != hasher(key3),
        "Different M should produce different hash");
    TEST_ASSERT(hasher(key1) != hasher(key4),
        "Different N should produce different hash");
    TEST_ASSERT(hasher(key1) != hasher(key5),
        "Different dtype should produce different hash");

    // Test key equality
    TEST_ASSERT(key1 == key2, "Same keys should be equal");
    TEST_ASSERT(!(key1 == key3), "Different keys should not be equal");

    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main() {
    std::cout << "\n=== oneDNN Fallback Unit Tests ===\n" << std::endl;

    RUN_TEST(test_init);
    RUN_TEST(test_m_bucketing);
    RUN_TEST(test_suitability);
    RUN_TEST(test_compute_bound);
    RUN_TEST(test_cache_empty);
    RUN_TEST(test_matmul_without_init);
    RUN_TEST(test_matmul_null_pointers);
    RUN_TEST(test_matmul_invalid_dims);
    RUN_TEST(test_cache_hit);
    RUN_TEST(test_clear_cache);
    RUN_TEST(test_different_dims);
    RUN_TEST(test_same_bucket);
    RUN_TEST(test_different_buckets);
    RUN_TEST(test_different_dtypes);
    RUN_TEST(test_hit_rate);
    RUN_TEST(test_matmul_timing);
    RUN_TEST(test_destructor);
    RUN_TEST(test_convenience_functions);
    RUN_TEST(test_large_m_bucketing);
    RUN_TEST(test_key_hash);

    std::cout << "\n==================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    std::cout << "==================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
