//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for expert_cache with LRU/frequency eviction
// Part of tiered memory architecture for MoE models

#include "expert-cache.hpp"
#include "pinned-pool.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

// Pinned pool requires at least 8GB budget (CHUNK_SIZE)
// We use HOST_BUDGET = 8GB for all tests
constexpr size_t HOST_BUDGET = 8ULL * 1024 * 1024 * 1024;

// Test basic cache operations: register, get, hit/miss
static void test_expert_cache_basic() {
    printf("test_expert_cache_basic: ");

    sycl::queue q;

    // Setup: 500MB VRAM budget
    constexpr size_t VRAM_BUDGET = 500ULL * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;  // 50MB per expert

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Pre-populate pinned pool with "expert data"
    std::vector<void *> host_ptrs;
    for (int i = 0; i < 20; i++) {
        void * ptr = pool.allocate(EXPERT_SIZE);
        assert(ptr != nullptr && "Failed to allocate from pinned pool");

        // Fill with pattern for verification
        memset(ptr, i + 1, EXPERT_SIZE);
        host_ptrs.push_back(ptr);

        cache.register_expert(/*layer=*/0, /*expert=*/i, ptr, EXPERT_SIZE);
    }

    // Access experts 0-9 (should cache to VRAM, 10 * 50MB = 500MB)
    for (int i = 0; i < 10; i++) {
        void * vram_ptr = cache.get_expert(0, i, EXPERT_SIZE);
        assert(vram_ptr != nullptr && "Should return VRAM pointer");
        (void) vram_ptr;  // Silence unused warning
    }

    // All should be cached
    for (int i = 0; i < 10; i++) {
        assert(cache.is_cached_in_vram(0, i) && "Expert should be in VRAM");
    }

    // Verify stats
    assert(cache.cache_misses() == 10 && "Should have 10 misses (cold start)");
    assert(cache.cache_hits() == 0 && "Should have 0 hits so far");

    // Access expert 0 again - should be a hit
    cache.get_expert(0, 0, EXPERT_SIZE);
    assert(cache.cache_hits() == 1 && "Should have 1 hit now");

    // Access expert 10 - should evict least recently used (expert 1)
    // Expert 0 was just accessed, so it's most recent
    cache.get_expert(0, 10, EXPERT_SIZE);

    // Expert 10 should be cached
    assert(cache.is_cached_in_vram(0, 10) && "Expert 10 should be cached");

    // One of experts 1-9 should be evicted (whichever has lowest score)
    int evicted_count = 0;
    for (int i = 1; i < 10; i++) {
        if (!cache.is_cached_in_vram(0, i)) {
            evicted_count++;
        }
    }
    assert(evicted_count == 1 && "Exactly one expert should be evicted");

    // Expert 0 should still be cached (was accessed twice, highest frequency)
    assert(cache.is_cached_in_vram(0, 0) && "Expert 0 should remain cached");

    printf("PASSED\n");
}

// Test frequency-based eviction preference
static void test_expert_cache_frequency() {
    printf("test_expert_cache_frequency: ");

    sycl::queue q;

    // 250MB budget = 5 experts at 50MB each
    constexpr size_t VRAM_BUDGET = 250ULL * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Register 10 experts
    std::vector<void *> host_ptrs;
    for (int i = 0; i < 10; i++) {
        void * ptr = pool.allocate(EXPERT_SIZE);
        assert(ptr != nullptr && "Failed to allocate from pinned pool");
        host_ptrs.push_back(ptr);
        cache.register_expert(0, i, ptr, EXPERT_SIZE);
    }

    // Access expert 0 many times (high frequency)
    for (int j = 0; j < 100; j++) {
        cache.get_expert(0, 0, EXPERT_SIZE);
    }

    // Access experts 1-4 once each (fills VRAM: 5 * 50MB = 250MB)
    for (int i = 1; i < 5; i++) {
        cache.get_expert(0, i, EXPERT_SIZE);
    }

    // Verify all are cached
    for (int i = 0; i < 5; i++) {
        assert(cache.is_cached_in_vram(0, i) && "All 5 should be cached");
    }

    // Access expert 5 - should evict lowest score
    // Expert 0 has 100 accesses, experts 1-4 have 1 access each
    // Frequency weight is 70%, so experts 1-4 are candidates for eviction
    cache.get_expert(0, 5, EXPERT_SIZE);

    // Expert 0 should still be cached (high frequency protects it)
    assert(cache.is_cached_in_vram(0, 0) && "Expert 0 should remain cached");

    // Expert 5 should be cached
    assert(cache.is_cached_in_vram(0, 5) && "Expert 5 should be cached");

    // One of experts 1-4 should be evicted
    int cached_1_to_4 = 0;
    for (int i = 1; i <= 4; i++) {
        if (cache.is_cached_in_vram(0, i)) {
            cached_1_to_4++;
        }
    }
    assert(cached_1_to_4 == 3 && "Should have 3 of experts 1-4 cached");

    printf("PASSED\n");
}

// Test statistics tracking
static void test_expert_cache_stats() {
    printf("test_expert_cache_stats: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 200 * 1024 * 1024;  // 200MB
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;   // 50MB

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Register experts
    for (int i = 0; i < 5; i++) {
        void * ptr = pool.allocate(EXPERT_SIZE);
        assert(ptr != nullptr && "Failed to allocate from pinned pool");
        cache.register_expert(0, i, ptr, EXPERT_SIZE);
    }

    // Verify initial stats
    assert(cache.cache_hits() == 0);
    assert(cache.cache_misses() == 0);
    assert(cache.vram_used() == 0);

    // First access = miss
    cache.get_expert(0, 0, EXPERT_SIZE);
    assert(cache.cache_misses() == 1);
    assert(cache.cache_hits() == 0);

    // Second access = hit
    cache.get_expert(0, 0, EXPERT_SIZE);
    assert(cache.cache_misses() == 1);
    assert(cache.cache_hits() == 1);

    // VRAM used should be 50MB
    assert(cache.vram_used() == EXPERT_SIZE);

    // Add three more experts
    cache.get_expert(0, 1, EXPERT_SIZE);
    cache.get_expert(0, 2, EXPERT_SIZE);
    cache.get_expert(0, 3, EXPERT_SIZE);

    assert(cache.cache_misses() == 4);             // 1 + 3 new
    assert(cache.vram_used() == 4 * EXPERT_SIZE);  // 200MB
    assert(cache.entries_count() == 4);

    // Budget check
    assert(cache.vram_budget() == VRAM_BUDGET);

    printf("PASSED\n");
}

// Test prefetch functionality
static void test_expert_cache_prefetch() {
    printf("test_expert_cache_prefetch: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 200 * 1024 * 1024;  // 200MB
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;   // 50MB

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Register experts
    for (int i = 0; i < 10; i++) {
        void * ptr = pool.allocate(EXPERT_SIZE);
        assert(ptr != nullptr && "Failed to allocate from pinned pool");
        cache.register_expert(0, i, ptr, EXPERT_SIZE);
    }

    // Prefetch experts 0-3
    std::vector<ggml_sycl::expert_key> to_prefetch = {
        { 0, 0 },
        { 0, 1 },
        { 0, 2 },
        { 0, 3 }
    };
    cache.prefetch(to_prefetch, EXPERT_SIZE);
    cache.wait_prefetch();

    // All should be in VRAM now
    for (int i = 0; i < 4; i++) {
        assert(cache.is_cached_in_vram(0, i) && "Prefetched expert should be in VRAM");
    }

    // Prefetch should not count as hits or misses (it's speculative)
    // Actually our implementation does count them for VRAM tracking but not as misses
    // The entries have access_count=0 (prefetch doesn't increment)

    // Now access them - should be hits
    for (int i = 0; i < 4; i++) {
        cache.get_expert(0, i, EXPERT_SIZE);
    }
    assert(cache.cache_hits() == 4 && "Prefetched accesses should be hits");

    printf("PASSED\n");
}

// Test multi-layer experts
static void test_expert_cache_multi_layer() {
    printf("test_expert_cache_multi_layer: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 200 * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Register experts across multiple layers (use fewer to fit in 8GB)
    for (int layer = 0; layer < 4; layer++) {
        for (int expert = 0; expert < 8; expert++) {
            void * ptr = pool.allocate(EXPERT_SIZE);
            assert(ptr != nullptr && "Failed to allocate from pinned pool");
            cache.register_expert(layer, expert, ptr, EXPERT_SIZE);
        }
    }

    // Access layer 0, expert 0
    cache.get_expert(0, 0, EXPERT_SIZE);
    assert(cache.is_cached_in_vram(0, 0));

    // Access layer 1, expert 0 (different from layer 0, expert 0)
    cache.get_expert(1, 0, EXPERT_SIZE);
    assert(cache.is_cached_in_vram(1, 0));

    // Both should be independently cached
    assert(cache.entries_count() == 2);

    // Access layer 2, experts 0-1
    cache.get_expert(2, 0, EXPERT_SIZE);
    cache.get_expert(2, 1, EXPERT_SIZE);

    // All 4 should be cached (200MB total)
    assert(cache.entries_count() == 4);
    assert(cache.vram_used() == 4 * EXPERT_SIZE);

    printf("PASSED\n");
}

// Test unregistered expert handling
static void test_expert_cache_unregistered() {
    printf("test_expert_cache_unregistered: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 100 * 1024 * 1024;
    constexpr size_t EXPERT_SIZE = 50 * 1024 * 1024;

    ggml_sycl::pinned_chunk_pool pool(q, HOST_BUDGET);
    ggml_sycl::expert_cache      cache(q, pool, VRAM_BUDGET);

    // Try to get unregistered expert - should return nullptr
    void * ptr = cache.get_expert(0, 0, EXPERT_SIZE);
    assert(ptr == nullptr && "Unregistered expert should return nullptr");
    (void) ptr;  // Silence unused warning

    printf("PASSED\n");
}

int main() {
    try {
        printf("\n=== Expert Cache Unit Tests ===\n\n");

        test_expert_cache_basic();
        test_expert_cache_frequency();
        test_expert_cache_stats();
        test_expert_cache_prefetch();
        test_expert_cache_multi_layer();
        test_expert_cache_unregistered();

        printf("\nAll tests PASSED!\n\n");
        return 0;
    } catch (const sycl::exception & e) {
        fprintf(stderr, "\nSYCL exception: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "\nTest FAILED: %s\n", e.what());
        return 1;
    }
}
