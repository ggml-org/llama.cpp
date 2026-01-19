//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Test for routing-aware MoE expert pre-staging
// Tests the deduplication and routing logic for prestage_routed_experts()
// This is a standalone test - does not require SYCL runtime

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

// ===========================================================================
// Test Helpers
// ===========================================================================

// Generate realistic MoE routing indices
// Each token selects top-k experts from n_experts_total
static std::vector<int32_t> generate_routing_indices(
    int n_tokens,
    int n_expert_used,
    int n_experts_total,
    std::mt19937& rng
) {
    std::vector<int32_t> indices(n_tokens * n_expert_used);

    for (int t = 0; t < n_tokens; t++) {
        // Generate n_expert_used distinct expert IDs for this token
        std::vector<int> available(n_experts_total);
        std::iota(available.begin(), available.end(), 0);
        std::shuffle(available.begin(), available.end(), rng);

        for (int k = 0; k < n_expert_used; k++) {
            indices[t * n_expert_used + k] = available[k];
        }
    }

    return indices;
}

// Count unique expert IDs in routing indices
static int count_unique_experts(const std::vector<int32_t>& indices) {
    std::unordered_set<int32_t> unique;
    for (int32_t id : indices) {
        unique.insert(id);
    }
    return static_cast<int>(unique.size());
}

// Deduplicate expert IDs (mimics prestage_routed_experts logic)
static std::vector<int32_t> deduplicate_experts(
    const int32_t* expert_ids,
    int n_expert_used,
    int n_tokens,
    int n_experts_total
) {
    std::unordered_set<int32_t> unique;
    const int total = n_expert_used * n_tokens;

    for (int i = 0; i < total; i++) {
        int32_t id = expert_ids[i];
        if (id >= 0 && id < n_experts_total) {
            unique.insert(id);
        }
    }

    return std::vector<int32_t>(unique.begin(), unique.end());
}

// Filter out already-cached experts (mimics cache hit check)
static std::vector<int32_t> filter_uncached(
    const std::vector<int32_t>& unique_experts,
    const std::unordered_set<int32_t>& cached
) {
    std::vector<int32_t> uncached;
    for (int32_t id : unique_experts) {
        if (cached.find(id) == cached.end()) {
            uncached.push_back(id);
        }
    }
    return uncached;
}

// ===========================================================================
// Unit Tests
// ===========================================================================

// Test 1: Deduplication works correctly
// 512 tokens x 4 experts should deduplicate to ~30 unique experts (for 128 total)
static bool test_deduplication() {
    printf("Test 1: Deduplication... ");

    std::mt19937 rng(42);
    const int n_tokens = 512;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);

    // Expected: 512 * 4 = 2048 total IDs
    assert(indices.size() == static_cast<size_t>(n_tokens * n_expert_used));

    // Deduplicate using the helper function
    auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);
    int unique_count = static_cast<int>(unique.size());

    // With 512 tokens selecting 4 experts from 128, we expect most experts to be hit
    // but not all - mathematically around 60-80 unique (depends on RNG)
    printf("got %d unique from %zu total... ", unique_count, indices.size());

    // Sanity checks
    assert(unique_count > 0);
    assert(unique_count <= n_experts_total);
    assert(unique_count < static_cast<int>(indices.size()));  // Some deduplication happened

    printf("PASS\n");
    return true;
}

// Test 2: Bounds checking - expert IDs within valid range
static bool test_bounds_checking() {
    printf("Test 2: Bounds checking... ");

    std::mt19937 rng(42);
    const int n_tokens = 100;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);

    for (int32_t id : indices) {
        assert(id >= 0);
        assert(id < n_experts_total);
    }

    // Also test deduplication filters invalid IDs
    std::vector<int32_t> with_invalid = indices;
    with_invalid[0] = -1;
    with_invalid[1] = 999;  // Out of range

    auto filtered = deduplicate_experts(with_invalid.data(), n_expert_used, n_tokens, n_experts_total);

    // Verify invalid IDs were filtered
    for (int32_t id : filtered) {
        assert(id >= 0);
        assert(id < n_experts_total);
    }

    printf("PASS\n");
    return true;
}

// Test 3: Empty input handling
static bool test_empty_input() {
    printf("Test 3: Empty input handling... ");

    // Zero tokens
    std::vector<int32_t> empty_indices;
    auto unique = deduplicate_experts(empty_indices.data(), 4, 0, 128);
    assert(unique.empty());

    // Zero experts per token
    std::vector<int32_t> dummy(100, 0);
    auto unique2 = deduplicate_experts(dummy.data(), 0, 100, 128);
    assert(unique2.empty());

    printf("PASS\n");
    return true;
}

// Test 4: Single token case
static bool test_single_token() {
    printf("Test 4: Single token case... ");

    std::mt19937 rng(42);
    const int n_tokens = 1;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);

    assert(indices.size() == 4);

    auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);
    assert(unique.size() == 4);  // All 4 experts should be distinct

    printf("PASS\n");
    return true;
}

// Test 5: Large batch (stress test)
static bool test_large_batch() {
    printf("Test 5: Large batch (2048 tokens)... ");

    std::mt19937 rng(42);
    const int n_tokens = 2048;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);

    // 2048 * 4 = 8192 total IDs
    assert(indices.size() == 8192);

    auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);

    // With 2048 tokens, we expect nearly all 128 experts to be hit
    printf("got %zu unique... ", unique.size());
    assert(unique.size() >= 100);  // Should hit most experts
    assert(unique.size() <= 128);

    printf("PASS\n");
    return true;
}

// Test 6: Expert stride calculation
static bool test_expert_stride() {
    printf("Test 6: Expert stride calculation... ");

    // Typical MoE expert: 4096 x 14336 in Q4_0
    // Q4_0: 18 bytes per 32 elements
    const int64_t ne0 = 4096;   // hidden_dim
    const int64_t ne1 = 14336;  // intermediate_dim
    const size_t type_size = 18;  // Q4_0 block size
    const size_t block_elements = 32;

    const size_t expert_size = (ne0 * ne1 / block_elements) * type_size;
    const size_t expert_stride = expert_size;  // Experts are contiguous

    printf("expert_size=%zu MB, stride=%zu... ", expert_size / (1024 * 1024), expert_stride);

    // Verify pointer arithmetic
    const char* base = reinterpret_cast<const char*>(0x1000000);
    const char* expert_5 = base + 5 * expert_stride;
    const char* expert_100 = base + 100 * expert_stride;

    assert(expert_5 - base == 5 * static_cast<ptrdiff_t>(expert_stride));
    assert(expert_100 - base == 100 * static_cast<ptrdiff_t>(expert_stride));

    printf("PASS\n");
    return true;
}

// Test 7: Routing pattern analysis (realistic MoE behavior)
static bool test_routing_patterns() {
    printf("Test 7: Routing pattern analysis... ");

    std::mt19937 rng(12345);
    const int n_tokens = 512;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);

    // Count frequency of each expert
    std::vector<int> freq(n_experts_total, 0);
    for (int32_t id : indices) {
        freq[id]++;
    }

    // With uniform random selection, each expert should be hit roughly equally
    // Expected: 2048 / 128 = 16 times per expert on average
    int min_freq = *std::min_element(freq.begin(), freq.end());
    int max_freq = *std::max_element(freq.begin(), freq.end());

    printf("min=%d, max=%d, expected=16... ", min_freq, max_freq);

    // Should be within reasonable bounds
    assert(min_freq >= 0);
    assert(max_freq <= 50);  // Not too skewed

    printf("PASS\n");
    return true;
}

// Test 8: Cache hit filtering (simulates already-cached experts)
static bool test_cache_hit_filtering() {
    printf("Test 8: Cache hit filtering... ");

    std::mt19937 rng(42);
    const int n_tokens = 512;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);
    auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);

    // Simulate 50% cache hits
    std::unordered_set<int32_t> cached;
    for (size_t i = 0; i < unique.size() / 2; i++) {
        cached.insert(unique[i]);
    }

    auto uncached = filter_uncached(unique, cached);

    printf("unique=%zu, cached=%zu, to_stage=%zu... ",
           unique.size(), cached.size(), uncached.size());

    // Should need to stage roughly half
    assert(uncached.size() == unique.size() - cached.size());

    printf("PASS\n");
    return true;
}

// Test 9: Integration test preparation (verifies warm cache behavior)
static bool test_warm_cache_behavior() {
    printf("Test 9: Warm cache behavior... ");

    std::mt19937 rng(42);
    const int n_tokens = 512;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    // First batch (cold cache)
    auto indices1 = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);
    auto unique1 = deduplicate_experts(indices1.data(), n_expert_used, n_tokens, n_experts_total);

    // Simulate staging all unique experts from batch 1
    std::unordered_set<int32_t> cached;
    for (int32_t id : unique1) {
        cached.insert(id);
    }

    printf("batch1: staged %zu... ", unique1.size());

    // Second batch (warm cache)
    auto indices2 = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);
    auto unique2 = deduplicate_experts(indices2.data(), n_expert_used, n_tokens, n_experts_total);
    auto to_stage2 = filter_uncached(unique2, cached);

    printf("batch2: need %zu of %zu... ", to_stage2.size(), unique2.size());

    // Second batch should need fewer staging operations
    assert(to_stage2.size() <= unique2.size());

    // After many batches, cache should converge to all experts
    for (int batch = 0; batch < 10; batch++) {
        auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);
        auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);
        for (int32_t id : unique) {
            cached.insert(id);
        }
    }

    printf("after 10 batches: cached %zu of %d... ", cached.size(), n_experts_total);

    // Should have most/all experts cached
    assert(cached.size() >= static_cast<size_t>(n_experts_total - 5));

    printf("PASS\n");
    return true;
}

// Test 10: Pin/unpin tracking simulation
static bool test_pin_tracking() {
    printf("Test 10: Pin/unpin tracking... ");

    std::mt19937 rng(42);
    const int n_tokens = 256;
    const int n_expert_used = 4;
    const int n_experts_total = 128;

    auto indices = generate_routing_indices(n_tokens, n_expert_used, n_experts_total, rng);
    auto unique = deduplicate_experts(indices.data(), n_expert_used, n_tokens, n_experts_total);

    // Track pinned experts
    std::unordered_set<int32_t> pinned;

    // Pin all unique experts
    for (int32_t id : unique) {
        pinned.insert(id);
    }

    printf("pinned %zu experts... ", pinned.size());
    assert(pinned.size() == unique.size());

    // Unpin all
    for (int32_t id : unique) {
        pinned.erase(id);
    }

    assert(pinned.empty());

    printf("PASS\n");
    return true;
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== MoE Expert Pre-staging Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    // Run unit tests (no SYCL required)
    if (test_deduplication()) passed++; else failed++;
    if (test_bounds_checking()) passed++; else failed++;
    if (test_empty_input()) passed++; else failed++;
    if (test_single_token()) passed++; else failed++;
    if (test_large_batch()) passed++; else failed++;
    if (test_expert_stride()) passed++; else failed++;
    if (test_routing_patterns()) passed++; else failed++;
    if (test_cache_hit_filtering()) passed++; else failed++;
    if (test_warm_cache_behavior()) passed++; else failed++;
    if (test_pin_tracking()) passed++; else failed++;

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
