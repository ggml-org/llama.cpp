//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Unit tests for KV cache manager with per-head granularity
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-vr5)
//
// TDD: These tests define the interface. Implementation comes AFTER tests fail.
//

#include "ggml-sycl/chunk-manager.hpp"
#include "ggml-sycl/eviction-policy.hpp"
#include "ggml-sycl/kv-cache-manager.hpp"
#include "ggml.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Test framework helpers
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_CASE(name)                                              \
    static void test_##name();                                       \
    static struct test_##name##_register {                           \
        test_##name##_register() {                                   \
            printf("Running test: %s\n", #name);                     \
            try {                                                    \
                test_##name();                                       \
                g_tests_passed++;                                    \
                printf("  PASSED: %s\n", #name);                     \
            } catch (const std::exception & e) {                     \
                g_tests_failed++;                                    \
                printf("  FAILED: %s - %s\n", #name, e.what());      \
            } catch (...) {                                          \
                g_tests_failed++;                                    \
                printf("  FAILED: %s - unknown exception\n", #name); \
            }                                                        \
        }                                                            \
    } test_##name##_instance;                                        \
    static void test_##name()

#define ASSERT_TRUE(cond)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond); \
        }                                                                        \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b)                                                                     \
    do {                                                                                    \
        if ((a) != (b)) {                                                                   \
            throw std::runtime_error(std::string("Assertion failed: ") + #a + " == " + #b); \
        }                                                                                   \
    } while (0)

#define ASSERT_NE(a, b)                                                                     \
    do {                                                                                    \
        if ((a) == (b)) {                                                                   \
            throw std::runtime_error(std::string("Assertion failed: ") + #a + " != " + #b); \
        }                                                                                   \
    } while (0)

#define ASSERT_GT(a, b)                                                                    \
    do {                                                                                   \
        if (!((a) > (b))) {                                                                \
            throw std::runtime_error(std::string("Assertion failed: ") + #a + " > " + #b); \
        }                                                                                  \
    } while (0)

using namespace ggml_sycl;

// =============================================================================
// Test 1: Basic configuration and structure
// =============================================================================
TEST_CASE(kv_manager_configuration) {
    // KVCacheManager should be configurable with model parameters
    KVCacheManager kv_mgr;

    // Configure for a typical model: 32 layers, 32 heads, 128 head_dim, f16
    const uint32_t  num_layers = 32;
    const uint32_t  num_heads  = 32;
    const size_t    head_dim   = 128;
    const ggml_type kv_type    = GGML_TYPE_F16;

    kv_mgr.configure(num_layers, num_heads, head_dim, kv_type);

    // Verify configuration
    ASSERT_EQ(kv_mgr.get_num_layers(), num_layers);
    ASSERT_EQ(kv_mgr.get_num_heads(), num_heads);
    ASSERT_EQ(kv_mgr.get_head_dim(), head_dim);
    ASSERT_EQ(kv_mgr.get_kv_type(), kv_type);

    // Check bytes per head calculation
    // For interleaved K+V: head_dim * sizeof(f16) * 2 (K and V)
    // Per token: 128 * 2 * 2 = 512 bytes per token per head
    const size_t expected_bytes_per_token_per_head = head_dim * ggml_type_size(kv_type) * 2;
    ASSERT_EQ(kv_mgr.get_bytes_per_token_per_head(), expected_bytes_per_token_per_head);
}

// =============================================================================
// Test 2: Per-head allocation and tracking
// =============================================================================
TEST_CASE(kv_per_head_allocation) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(4, 8, 64, GGML_TYPE_F16);  // 4 layers, 8 heads, 64 head_dim

    // Allocate KV for a single head
    const uint32_t layer      = 0;
    const uint32_t head       = 0;
    const size_t   seq_pos    = 0;
    const size_t   num_tokens = 128;

    KVHandle handle = kv_mgr.allocate(layer, head, seq_pos, num_tokens);

    // Handle should be valid
    ASSERT_TRUE(kv_mgr.is_valid_handle(handle));

    // Should be able to query the allocation
    ASSERT_EQ(kv_mgr.get_layer_id(handle), layer);
    ASSERT_EQ(kv_mgr.get_head_id(handle), head);
    ASSERT_EQ(kv_mgr.get_seq_start(handle), seq_pos);
    ASSERT_EQ(kv_mgr.get_seq_len(handle), num_tokens);

    // Size should match: tokens * bytes_per_token_per_head
    const size_t expected_size = num_tokens * kv_mgr.get_bytes_per_token_per_head();
    ASSERT_EQ(kv_mgr.get_allocation_size(handle), expected_size);
}

// =============================================================================
// Test 3: Interleaved K/V layout (K0,V0,K1,V1,...)
// =============================================================================
TEST_CASE(kv_interleaved_layout) {
    KVCacheManager kv_mgr;
    const uint32_t num_heads = 4;
    const size_t   head_dim  = 8;  // Small for easy testing
    const size_t   seq_len   = 2;  // 2 tokens

    kv_mgr.configure(1, num_heads, head_dim, GGML_TYPE_F32);

    // Create grouped layout data [all K, all V]
    // K section: filled with 1.0f
    // V section: filled with 2.0f
    const size_t head_k_size  = head_dim * seq_len;           // floats
    const size_t total_floats = num_heads * head_k_size * 2;  // K + V

    std::vector<float> grouped(total_floats);
    // Fill K section (first half) with 1.0f
    std::fill(grouped.begin(), grouped.begin() + total_floats / 2, 1.0f);
    // Fill V section (second half) with 2.0f
    std::fill(grouped.begin() + total_floats / 2, grouped.end(), 2.0f);

    std::vector<float> interleaved(total_floats);
    kv_mgr.convert_to_interleaved(grouped.data(), interleaved.data(), seq_len);

    // Verify interleaved layout: [K0, V0, K1, V1, K2, V2, K3, V3]
    // Each K and V section is head_dim * seq_len floats
    for (uint32_t h = 0; h < num_heads; h++) {
        // K section for head h
        float * k_ptr = &interleaved[h * head_k_size * 2];
        // V section for head h (immediately after K)
        float * v_ptr = &interleaved[h * head_k_size * 2 + head_k_size];

        // All K values should be 1.0f
        for (size_t i = 0; i < head_k_size; i++) {
            ASSERT_EQ(k_ptr[i], 1.0f);
        }
        // All V values should be 2.0f
        for (size_t i = 0; i < head_k_size; i++) {
            ASSERT_EQ(v_ptr[i], 2.0f);
        }
    }
}

// =============================================================================
// Test 4: Get K and V data pointers
// =============================================================================
TEST_CASE(kv_get_k_v_data) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate KV for a head
    KVHandle handle = kv_mgr.allocate(0, 0, 0, 64);

    // Get K and V pointers
    void * k_ptr = kv_mgr.get_k_data(handle);
    void * v_ptr = kv_mgr.get_v_data(handle);

    // Both should be valid
    ASSERT_NE(k_ptr, nullptr);
    ASSERT_NE(v_ptr, nullptr);

    // V should be after K (interleaved layout)
    // K size = head_dim * seq_len * sizeof(f16)
    const size_t k_size = 64 * 64 * sizeof(uint16_t);  // head_dim * seq_len * f16
    ASSERT_EQ(static_cast<uint8_t *>(v_ptr), static_cast<uint8_t *>(k_ptr) + k_size);
}

// =============================================================================
// Test 5: Priority transitions (cold <-> active)
// =============================================================================
TEST_CASE(kv_priority_transitions) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(4, 8, 64, GGML_TYPE_F16);

    // Allocate some heads
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);
    KVHandle h1 = kv_mgr.allocate(0, 1, 0, 64);
    KVHandle h2 = kv_mgr.allocate(0, 2, 0, 64);

    // Initially all should be cold (P4_COLD or similar low priority)
    // Using P1_ACTIVE_KV for active, something lower for cold
    ASSERT_EQ(kv_mgr.get_priority(h0), EvictionPriority::P4_COLD_EXPERT);
    ASSERT_EQ(kv_mgr.get_priority(h1), EvictionPriority::P4_COLD_EXPERT);
    ASSERT_EQ(kv_mgr.get_priority(h2), EvictionPriority::P4_COLD_EXPERT);
    ASSERT_FALSE(kv_mgr.is_pinned(h0));

    // Mark heads 0 and 1 as active (being used in attention)
    kv_mgr.mark_heads_active(0, { 0, 1 });

    // Active heads should have P1_ACTIVE_KV priority and be pinned
    ASSERT_EQ(kv_mgr.get_priority(h0), EvictionPriority::P1_ACTIVE_KV);
    ASSERT_EQ(kv_mgr.get_priority(h1), EvictionPriority::P1_ACTIVE_KV);
    ASSERT_EQ(kv_mgr.get_priority(h2), EvictionPriority::P4_COLD_EXPERT);  // Still cold
    ASSERT_TRUE(kv_mgr.is_pinned(h0));
    ASSERT_TRUE(kv_mgr.is_pinned(h1));
    ASSERT_FALSE(kv_mgr.is_pinned(h2));

    // Mark heads cold again (attention done)
    kv_mgr.mark_heads_cold(0, { 0, 1 });

    // Should be back to cold priority, unpinned
    ASSERT_EQ(kv_mgr.get_priority(h0), EvictionPriority::P4_COLD_EXPERT);
    ASSERT_EQ(kv_mgr.get_priority(h1), EvictionPriority::P4_COLD_EXPERT);
    ASSERT_FALSE(kv_mgr.is_pinned(h0));
    ASSERT_FALSE(kv_mgr.is_pinned(h1));
}

// =============================================================================
// Test 6: LRU timestamp updates on access
// =============================================================================
TEST_CASE(kv_lru_timestamp_updates) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate heads
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);
    KVHandle h1 = kv_mgr.allocate(0, 1, 0, 64);

    // Get initial timestamps
    uint64_t t0_before = kv_mgr.get_last_access(h0);
    uint64_t t1_before = kv_mgr.get_last_access(h1);

    // Access h0
    kv_mgr.touch(h0);

    // h0's timestamp should be updated
    uint64_t t0_after = kv_mgr.get_last_access(h0);
    ASSERT_GT(t0_after, t0_before);

    // h1's timestamp should be unchanged
    ASSERT_EQ(kv_mgr.get_last_access(h1), t1_before);
}

// =============================================================================
// Test 7: Eviction order respects priority and recency
// =============================================================================
TEST_CASE(kv_eviction_order) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate several heads
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);  // Will be cold
    KVHandle h1 = kv_mgr.allocate(0, 1, 0, 64);  // Will be hot
    KVHandle h2 = kv_mgr.allocate(0, 2, 0, 64);  // Will be cold, accessed later

    // Make h1 active (high priority)
    kv_mgr.mark_heads_active(0, { 1 });

    // Touch h2 to make it more recent than h0
    kv_mgr.touch(h2);

    // Get eviction candidates
    auto victims = kv_mgr.get_eviction_candidates(1);

    // h0 should be evicted first:
    // - h1 is active (P1), can't be evicted
    // - h0 and h2 are cold (P4), but h0 is older
    ASSERT_EQ(victims.size(), 1);
    ASSERT_EQ(victims[0], h0);

    // Get 2 candidates
    victims = kv_mgr.get_eviction_candidates(2);

    // Should get h0 then h2 (h1 is pinned/active)
    ASSERT_EQ(victims.size(), 2);
    ASSERT_EQ(victims[0], h0);
    ASSERT_EQ(victims[1], h2);
}

// =============================================================================
// Test 8: Get layer KV pointers for batch attention
// =============================================================================
TEST_CASE(kv_get_layer_kv) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate all heads for layer 0
    for (uint32_t h = 0; h < 4; h++) {
        kv_mgr.allocate(0, h, 0, 128);
    }

    // Get K/V pointers for specific heads
    std::vector<uint32_t> heads_to_get = { 0, 2, 3 };
    auto                  kv_ptrs      = kv_mgr.get_layer_kv(0, heads_to_get, 0, 128);

    // Should get 3 KVPtrs
    ASSERT_EQ(kv_ptrs.size(), 3);

    // All pointers should be valid
    for (const auto & ptrs : kv_ptrs) {
        ASSERT_NE(ptrs.k, nullptr);
        ASSERT_NE(ptrs.v, nullptr);
    }

    // Pointers should be different for each head
    ASSERT_NE(kv_ptrs[0].k, kv_ptrs[1].k);
    ASSERT_NE(kv_ptrs[1].k, kv_ptrs[2].k);
}

// =============================================================================
// Test 9: Context extension (growing KV cache)
// =============================================================================
TEST_CASE(kv_context_extension) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate initial context
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);
    ASSERT_EQ(kv_mgr.get_seq_len(h0), 64);

    // Extend the context (append more tokens)
    KVHandle h0_ext = kv_mgr.extend(h0, 32);  // Add 32 more tokens

    // Extended handle should cover the new range
    ASSERT_EQ(kv_mgr.get_seq_start(h0_ext), 64);  // Starts after original
    ASSERT_EQ(kv_mgr.get_seq_len(h0_ext), 32);

    // Total tokens for this head should be 96
    ASSERT_EQ(kv_mgr.get_total_tokens(0, 0), 96);
}

// =============================================================================
// Test 10: Memory tier tracking
// =============================================================================
TEST_CASE(kv_memory_tier_tracking) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate a head
    KVHandle handle = kv_mgr.allocate(0, 0, 0, 64);

    // Should initially be in HOST or VRAM depending on available memory
    MemoryTier tier = kv_mgr.get_tier(handle);
    ASSERT_TRUE(tier == MemoryTier::VRAM || tier == MemoryTier::HOST);

    // Simulate eviction to HOST
    if (tier == MemoryTier::VRAM) {
        kv_mgr.evict_to_host(handle);
        ASSERT_EQ(kv_mgr.get_tier(handle), MemoryTier::HOST);
    }

    // Simulate prefetch back to VRAM
    kv_mgr.prefetch_to_vram(handle);
    ASSERT_EQ(kv_mgr.get_tier(handle), MemoryTier::VRAM);
}

// =============================================================================
// Test 11: Evict cold heads to free memory
// =============================================================================
TEST_CASE(kv_evict_cold_heads) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate heads in VRAM
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);
    KVHandle h1 = kv_mgr.allocate(0, 1, 0, 64);
    KVHandle h2 = kv_mgr.allocate(0, 2, 0, 64);

    // Mark h1 as hot
    kv_mgr.mark_heads_active(0, { 1 });

    // Force all to VRAM for test
    kv_mgr.prefetch_to_vram(h0);
    kv_mgr.prefetch_to_vram(h1);
    kv_mgr.prefetch_to_vram(h2);

    // Calculate size needed
    size_t head_size    = kv_mgr.get_allocation_size(h0);
    size_t bytes_needed = head_size * 2;

    // Evict cold heads
    size_t evicted = kv_mgr.evict_cold_heads(bytes_needed);

    // Should have evicted at least bytes_needed worth
    ASSERT_TRUE(evicted >= bytes_needed);

    // h0 and h2 should now be in HOST (they were cold)
    ASSERT_EQ(kv_mgr.get_tier(h0), MemoryTier::HOST);
    ASSERT_EQ(kv_mgr.get_tier(h2), MemoryTier::HOST);

    // h1 should still be in VRAM (it was active/hot)
    ASSERT_EQ(kv_mgr.get_tier(h1), MemoryTier::VRAM);
}

// =============================================================================
// Test 12: Layer prefetch
// =============================================================================
TEST_CASE(kv_layer_prefetch) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(4, 4, 64, GGML_TYPE_F16);

    // Allocate heads for layers 0 and 1
    for (uint32_t h = 0; h < 4; h++) {
        kv_mgr.allocate(0, h, 0, 64);
        kv_mgr.allocate(1, h, 0, 64);
    }

    // Evict layer 1 to host
    for (uint32_t h = 0; h < 4; h++) {
        auto handles = kv_mgr.get_handles_for_layer(1);
        for (auto handle : handles) {
            kv_mgr.evict_to_host(handle);
        }
    }

    // Prefetch layer 1
    kv_mgr.prefetch_layer(1);

    // All layer 1 heads should now be in VRAM (or queued for prefetch)
    auto handles = kv_mgr.get_handles_for_layer(1);
    for (auto handle : handles) {
        // Either in VRAM or being prefetched
        MemoryTier tier = kv_mgr.get_tier(handle);
        ASSERT_TRUE(tier == MemoryTier::VRAM || kv_mgr.is_prefetching(handle));
    }
}

// =============================================================================
// Test 13: Invalid handle handling
// =============================================================================
TEST_CASE(kv_invalid_handle) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Create an invalid handle
    KVHandle invalid;
    invalid.id = 0xDEADBEEF;  // Some invalid ID

    // Should not be valid
    ASSERT_FALSE(kv_mgr.is_valid_handle(invalid));

    // Operations on invalid handle should return safe defaults
    ASSERT_EQ(kv_mgr.get_k_data(invalid), nullptr);
    ASSERT_EQ(kv_mgr.get_v_data(invalid), nullptr);
    ASSERT_EQ(kv_mgr.get_tier(invalid), MemoryTier::NONE);
}

// =============================================================================
// Test 14: Multiple sequence positions (sparse KV)
// =============================================================================
TEST_CASE(kv_multiple_sequences) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    // Allocate non-contiguous sequence ranges
    KVHandle h0_a = kv_mgr.allocate(0, 0, 0, 64);    // tokens 0-63
    KVHandle h0_b = kv_mgr.allocate(0, 0, 128, 32);  // tokens 128-159 (skip 64-127)

    // Both should be valid
    ASSERT_TRUE(kv_mgr.is_valid_handle(h0_a));
    ASSERT_TRUE(kv_mgr.is_valid_handle(h0_b));

    // Should be different handles
    ASSERT_NE(h0_a.id, h0_b.id);

    // Query should return appropriate handle
    KVHandle found = kv_mgr.find_handle(0, 0, 130);  // Token 130 is in h0_b
    ASSERT_EQ(found.id, h0_b.id);
}

// =============================================================================
// Test 15: Total memory usage tracking
// =============================================================================
TEST_CASE(kv_memory_usage_tracking) {
    KVCacheManager kv_mgr;
    kv_mgr.configure(2, 4, 64, GGML_TYPE_F16);

    ASSERT_EQ(kv_mgr.get_total_vram_usage(), 0);
    ASSERT_EQ(kv_mgr.get_total_host_usage(), 0);

    // Allocate some heads
    KVHandle h0 = kv_mgr.allocate(0, 0, 0, 64);
    KVHandle h1 = kv_mgr.allocate(0, 1, 0, 64);

    // Move to VRAM
    kv_mgr.prefetch_to_vram(h0);
    kv_mgr.prefetch_to_vram(h1);

    size_t head_size = kv_mgr.get_allocation_size(h0);

    // Should show VRAM usage
    ASSERT_EQ(kv_mgr.get_total_vram_usage(), head_size * 2);
    ASSERT_EQ(kv_mgr.get_total_host_usage(), 0);

    // Evict one to host
    kv_mgr.evict_to_host(h0);

    ASSERT_EQ(kv_mgr.get_total_vram_usage(), head_size);
    ASSERT_EQ(kv_mgr.get_total_host_usage(), head_size);
}

// =============================================================================
// Main entry point
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== KV Cache Manager Unit Tests ===\n");
    printf("Testing per-head granularity KV cache management\n\n");

    // Tests are auto-registered and run via static initialization
    // Just need to print summary

    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", g_tests_passed);
    printf("Failed: %d\n", g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
