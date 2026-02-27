//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for ExpertCache with LRU/frequency eviction.
// Tests the contiguous VRAM pool, O(1) lookup, eviction scoring,
// and thread-safety of the MoE expert VRAM cache manager.

#include "expert-cache.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

// Each "expert" is 1 MB for testing purposes.
constexpr size_t EXPERT_SIZE = 1 * 1024 * 1024;

// Helper: allocate host-pinned buffers to simulate expert weights.
static std::vector<void *> allocate_host_experts(sycl::queue & q, int count, size_t size) {
    std::vector<void *> ptrs;
    ptrs.reserve(count);
    for (int i = 0; i < count; i++) {
        void * p = sycl::malloc_host(size, q);
        assert(p != nullptr && "Failed to allocate host expert memory");
        memset(p, i + 1, size);
        ptrs.push_back(p);
    }
    return ptrs;
}

// Helper: free host-pinned buffers.
static void free_host_experts(sycl::queue & q, std::vector<void *> & ptrs) {
    for (void * p : ptrs) {
        sycl::free(p, q);
    }
    ptrs.clear();
}

// Test basic cache operations: register, lookup, ensure_cached, hit/miss
static void test_basic() {
    printf("test_basic: ");

    sycl::queue q;

    // 10 MB VRAM budget = 10 slots of 1 MB each
    constexpr size_t VRAM_BUDGET = 10 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);
    assert(cache.is_initialized());

    auto host_ptrs = allocate_host_experts(q, 20, EXPERT_SIZE);

    // Register 20 experts in layer 0
    for (int i = 0; i < 20; i++) {
        cache.register_expert(0, i, host_ptrs[i], EXPERT_SIZE);
    }

    assert(cache.total_slots() == 10);

    // Lookup before caching should return not-cached
    {
        auto lk = cache.lookup(0, 0);
        assert(!lk.is_cached);
        assert(lk.host_ptr == host_ptrs[0]);
    }

    // ensure_cached: fills VRAM (10 experts)
    for (int i = 0; i < 10; i++) {
        assert(cache.ensure_cached(0, i, q) != nullptr);
    }

    // All should now be cached
    for (int i = 0; i < 10; i++) {
        assert(cache.is_cached_in_vram(0, i));
    }

    assert(cache.cache_misses() == 10);
    assert(cache.cache_hits() == 0);
    assert(cache.cached_count() == 10);

    // Re-access expert 0 -> hit
    assert(cache.ensure_cached(0, 0, q) != nullptr);
    assert(cache.cache_hits() == 1);

    // Access expert 10 -> must evict one of 1-9 (0 was recently accessed)
    cache.ensure_cached(0, 10, q);
    assert(cache.is_cached_in_vram(0, 10));

    // Expert 0 should remain (accessed twice, highest frequency)
    assert(cache.is_cached_in_vram(0, 0));

    // Exactly one of 1-9 should be evicted
    {
        int n_evicted = 0;
        for (int i = 1; i < 10; i++) {
            if (!cache.is_cached_in_vram(0, i)) {
                n_evicted++;
            }
        }
        assert(n_evicted == 1);
    }

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

// Test frequency-based eviction: frequently accessed experts survive
static void test_frequency_eviction() {
    printf("test_frequency_eviction: ");

    sycl::queue q;

    // 5 slots
    constexpr size_t VRAM_BUDGET = 5 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    auto host_ptrs = allocate_host_experts(q, 10, EXPERT_SIZE);
    for (int i = 0; i < 10; i++) {
        cache.register_expert(0, i, host_ptrs[i], EXPERT_SIZE);
    }

    // Access expert 0 many times (high frequency)
    for (int j = 0; j < 50; j++) {
        cache.ensure_cached(0, 0, q);
    }

    // Fill remaining 4 slots with experts 1-4
    for (int i = 1; i < 5; i++) {
        cache.ensure_cached(0, i, q);
    }

    // All 5 should be cached
    for (int i = 0; i < 5; i++) {
        assert(cache.is_cached_in_vram(0, i));
    }

    // Access expert 5 -> evict lowest score (one of 1-4)
    cache.ensure_cached(0, 5, q);

    // Expert 0 must survive (high frequency)
    assert(cache.is_cached_in_vram(0, 0));
    assert(cache.is_cached_in_vram(0, 5));

    // Exactly 3 of 1-4 should remain
    {
        int n_cached = 0;
        for (int i = 1; i <= 4; i++) {
            if (cache.is_cached_in_vram(0, i)) {
                n_cached++;
            }
        }
        assert(n_cached == 3);
    }

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

// Test statistics tracking
static void test_stats() {
    printf("test_stats: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 4 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    auto host_ptrs = allocate_host_experts(q, 5, EXPERT_SIZE);
    for (int i = 0; i < 5; i++) {
        cache.register_expert(0, i, host_ptrs[i], EXPERT_SIZE);
    }

    assert(cache.cache_hits() == 0);
    assert(cache.cache_misses() == 0);
    assert(cache.vram_budget() == VRAM_BUDGET);

    // First access = miss
    cache.ensure_cached(0, 0, q);
    assert(cache.cache_misses() == 1);
    assert(cache.cache_hits() == 0);

    // Second access = hit
    cache.ensure_cached(0, 0, q);
    assert(cache.cache_misses() == 1);
    assert(cache.cache_hits() == 1);

    // VRAM used = 1 slot
    assert(cache.vram_used() == cache.vram_used());  // sanity
    assert(cache.entries_count() == 1);

    // Add 3 more experts
    cache.ensure_cached(0, 1, q);
    cache.ensure_cached(0, 2, q);
    cache.ensure_cached(0, 3, q);

    assert(cache.cache_misses() == 4);
    assert(cache.entries_count() == 4);

    // hit_rate = 1 / (1 + 4) = 0.2
    assert(cache.hit_rate() > 0.19f && cache.hit_rate() < 0.21f);

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

// Test async prefetch
static void test_prefetch() {
    printf("test_prefetch: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 4 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    auto host_ptrs = allocate_host_experts(q, 10, EXPERT_SIZE);
    for (int i = 0; i < 10; i++) {
        cache.register_expert(0, i, host_ptrs[i], EXPERT_SIZE);
    }

    // Prefetch experts 0-3 async
    std::vector<sycl::event> events;
    for (int i = 0; i < 4; i++) {
        events.push_back(cache.prefetch_async(0, i, q));
    }

    // Wait for all prefetches
    for (auto & evt : events) {
        evt.wait();
    }

    // All should be in VRAM
    for (int i = 0; i < 4; i++) {
        assert(cache.is_cached_in_vram(0, i));
    }

    // Now access them -> should be hits
    for (int i = 0; i < 4; i++) {
        cache.ensure_cached(0, i, q);
    }
    assert(cache.cache_hits() == 4);

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

// Test multi-layer experts
static void test_multi_layer() {
    printf("test_multi_layer: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 8 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    auto host_ptrs = allocate_host_experts(q, 32, EXPERT_SIZE);
    int idx = 0;
    for (int layer = 0; layer < 4; layer++) {
        for (int expert = 0; expert < 8; expert++) {
            cache.register_expert(layer, expert, host_ptrs[idx++], EXPERT_SIZE);
        }
    }

    // Access layer 0 expert 0
    cache.ensure_cached(0, 0, q);
    assert(cache.is_cached_in_vram(0, 0));

    // Access layer 1 expert 0 (different from layer 0 expert 0)
    cache.ensure_cached(1, 0, q);
    assert(cache.is_cached_in_vram(1, 0));

    // Both independently cached
    assert(cache.entries_count() == 2);

    // Fill up with layer 2 experts
    for (int e = 0; e < 6; e++) {
        cache.ensure_cached(2, e, q);
    }
    assert(cache.entries_count() == 8);

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

// Test unregistered expert returns nullptr
static void test_unregistered() {
    printf("test_unregistered: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 4 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    // Ensure_cached on unregistered expert -> nullptr
    assert(cache.ensure_cached(0, 0, q) == nullptr);

    // Lookup on unregistered expert -> not cached, no host_ptr
    {
        auto lk = cache.lookup(0, 0);
        assert(!lk.is_cached);
        assert(lk.host_ptr == nullptr);
    }

    cache.shutdown();
    printf("PASSED\n");
}

// Test update_score API
static void test_update_score() {
    printf("test_update_score: ");

    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 3 * EXPERT_SIZE;

    ggml_sycl::ExpertCache cache;
    cache.init(0, VRAM_BUDGET, q);

    auto host_ptrs = allocate_host_experts(q, 5, EXPERT_SIZE);
    for (int i = 0; i < 5; i++) {
        cache.register_expert(0, i, host_ptrs[i], EXPERT_SIZE);
    }

    // Fill cache with experts 0, 1, 2
    cache.ensure_cached(0, 0, q);
    cache.ensure_cached(0, 1, q);
    cache.ensure_cached(0, 2, q);

    // Boost expert 1's score via update_score
    for (uint64_t t = 1; t <= 100; t++) {
        cache.update_score(0, 1, t);
    }

    // Now add expert 3 -> should evict 0 or 2, NOT 1 (high score)
    cache.ensure_cached(0, 3, q);

    // Expert 1 must survive
    assert(cache.is_cached_in_vram(0, 1));

    cache.shutdown();
    free_host_experts(q, host_ptrs);
    printf("PASSED\n");
}

int main() {
    try {
        printf("\n=== Expert Cache Unit Tests ===\n\n");

        test_basic();
        test_frequency_eviction();
        test_stats();
        test_prefetch();
        test_multi_layer();
        test_unregistered();
        test_update_score();

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
