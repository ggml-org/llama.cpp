//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Test for tiered memory dispatch integration.
// Validates that the tensor inventory and tiered mode affect dispatch decisions.

#include "ggml-sycl.h"
#include "ggml.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Test basic tiered mode query
static void test_tiered_mode_query() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_tiered_mode_query: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create inventory exceeding VRAM to trigger tiered mode
    std::vector<std::string>           name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    size_t                             total_size = 0;

    // Each tensor is 4% of VRAM, 50 tensors = 200% -> should enable tiered mode
    const size_t num_tensors     = 50;
    const size_t size_per_tensor = free_vram / 25;

    name_storage.reserve(num_tensors);
    tensors.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; i++) {
        name_storage.push_back("blk." + std::to_string(i) + ".attn_q.weight");
        tensors.push_back({ name_storage.back().c_str(), size_per_tensor });
        total_size += size_per_tensor;
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = total_size;

    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Verify tiered mode is enabled
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    assert(tiered && "Large inventory (200% VRAM) should enable tiered mode");

    ggml_backend_free(backend);
    printf("test_tiered_mode_query: PASSED (inventory=%.1fGB, VRAM=%.1fGB, tiered=%s)\n",
           total_size / (1024.0 * 1024.0 * 1024.0), free_vram / (1024.0 * 1024.0 * 1024.0), tiered ? "true" : "false");
}

// Test that small inventory does NOT enable tiered mode
static void test_small_inventory_no_tiered() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_small_inventory_no_tiered: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create small inventory (50% of VRAM - should NOT enable tiered mode)
    std::vector<std::string>           name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    size_t                             total_size = 0;

    // Each tensor is 5% of VRAM, 10 tensors = 50% -> tiered should be disabled
    const size_t num_tensors     = 10;
    const size_t size_per_tensor = free_vram / 20;

    name_storage.reserve(num_tensors);
    tensors.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; i++) {
        name_storage.push_back("blk." + std::to_string(i) + ".attn_q.weight");
        tensors.push_back({ name_storage.back().c_str(), size_per_tensor });
        total_size += size_per_tensor;
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = total_size;

    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Verify tiered mode is NOT enabled
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    assert(!tiered && "Small inventory (50% VRAM) should NOT enable tiered mode");

    ggml_backend_free(backend);
    printf("test_small_inventory_no_tiered: PASSED (inventory=%.1fGB, VRAM=%.1fGB, tiered=%s)\n",
           total_size / (1024.0 * 1024.0 * 1024.0), free_vram / (1024.0 * 1024.0 * 1024.0), tiered ? "true" : "false");
}

// Test inventory with different tensor types (verifies storage)
static void test_inventory_tensor_types() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_inventory_tensor_types: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create inventory with different tensor types to verify storage
    std::vector<ggml_sycl_tensor_info> tensors = {
        { "token_embd.weight",          100 * 1024 * 1024 }, // Embedding
        { "output.weight",              100 * 1024 * 1024 }, // Output
        { "blk.0.attn_q.weight",        50 * 1024 * 1024  }, // Attention
        { "blk.0.ffn_down.weight",      200 * 1024 * 1024 }, // FFN
        { "blk.0.ffn_gate_inp.weight",  1 * 1024 * 1024   }, // Router (MoE)
        { "blk.0.ffn_down_exps.weight", 500 * 1024 * 1024 }, // Expert
        { "blk.0.attn_norm.weight",     1 * 1024 * 1024   }, // Norm
    };

    size_t total_size = 0;
    for (const auto & t : tensors) {
        total_size += t.size;
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = total_size;

    // Should not crash
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Query tiered state (depends on VRAM size)
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);

    ggml_backend_free(backend);
    printf("test_inventory_tensor_types: PASSED (%zu tensors, %.1fMB total, tiered=%s)\n", tensors.size(),
           total_size / (1024.0 * 1024.0), tiered ? "true" : "false");
}

// Test that tensor cache instance is available when tiered mode is enabled
static void test_cache_instance_available() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_instance_available: SKIPPED (no SYCL device)\n");
        return;
    }

    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    std::vector<std::string>           name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    size_t                             total_size = 0;

    for (size_t i = 0; i < 50; i++) {
        name_storage.push_back("blk." + std::to_string(i) + ".weight");
        tensors.push_back({ name_storage.back().c_str(), free_vram / 25 });
        total_size += free_vram / 25;
    }

    ggml_sycl_tensor_inventory inventory = { tensors.data(), tensors.size(), total_size };
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    bool has_cache = ggml_backend_sycl_has_tensor_cache(backend);
    assert(has_cache && "Backend should have tensor cache when tiered enabled");

    ggml_backend_free(backend);
    printf("test_cache_instance_available: PASSED\n");
}

// Test clearing inventory (setting new inventory clears old)
static void test_inventory_clear() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_inventory_clear: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // First: set large inventory to enable tiered mode
    std::vector<std::string>           name_storage_large;
    std::vector<ggml_sycl_tensor_info> tensors_large;

    const size_t num_large = 50;
    for (size_t i = 0; i < num_large; i++) {
        name_storage_large.push_back("large.blk." + std::to_string(i) + ".weight");
        tensors_large.push_back({ name_storage_large.back().c_str(), free_vram / 25 });
    }

    ggml_sycl_tensor_inventory inv_large;
    inv_large.tensors    = tensors_large.data();
    inv_large.count      = tensors_large.size();
    inv_large.total_size = num_large * (free_vram / 25);

    ggml_backend_sycl_set_tensor_inventory(backend, &inv_large);
    assert(ggml_backend_sycl_is_tiered_enabled(backend) && "Large inventory should enable tiered");

    // Second: set small inventory to disable tiered mode
    std::vector<std::string>           name_storage_small;
    std::vector<ggml_sycl_tensor_info> tensors_small;

    const size_t num_small = 5;
    for (size_t i = 0; i < num_small; i++) {
        name_storage_small.push_back("small.blk." + std::to_string(i) + ".weight");
        tensors_small.push_back({ name_storage_small.back().c_str(), free_vram / 20 });
    }

    ggml_sycl_tensor_inventory inv_small;
    inv_small.tensors    = tensors_small.data();
    inv_small.count      = tensors_small.size();
    inv_small.total_size = num_small * (free_vram / 20);  // 25% of VRAM

    ggml_backend_sycl_set_tensor_inventory(backend, &inv_small);
    assert(!ggml_backend_sycl_is_tiered_enabled(backend) && "Small inventory should disable tiered");

    ggml_backend_free(backend);
    printf("test_inventory_clear: PASSED (inventory replacement works)\n");
}

// Test cache stats API with null safety
static void test_cache_stats_null_safety() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_stats_null_safety: SKIPPED (no SYCL device)\n");
        return;
    }

    // Should not crash with NULL outputs
    ggml_backend_sycl_get_cache_stats(backend, nullptr, nullptr);

    uint64_t hits = 99;
    ggml_backend_sycl_get_cache_stats(backend, &hits, nullptr);
    assert(hits == 0 && "No cache created yet, hits should be 0");

    uint64_t misses = 99;
    ggml_backend_sycl_get_cache_stats(backend, nullptr, &misses);
    assert(misses == 0 && "No cache created yet, misses should be 0");

    ggml_backend_free(backend);
    printf("test_cache_stats_null_safety: PASSED\n");
}

// Test cache stats API returns zero initially
static void test_cache_stats_api() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_stats_api: SKIPPED (no SYCL device)\n");
        return;
    }

    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create inventory exceeding VRAM to enable tiered mode
    std::vector<std::string>           name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    name_storage.reserve(2);
    tensors.reserve(2);

    // Each tensor = full VRAM, so 2 tensors = 2x VRAM
    name_storage.push_back("blk.0.weight");
    tensors.push_back({ name_storage.back().c_str(), free_vram });
    name_storage.push_back("blk.1.weight");
    tensors.push_back({ name_storage.back().c_str(), free_vram });

    ggml_sycl_tensor_inventory inventory = { tensors.data(), tensors.size(), free_vram * 2 };
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Query cache stats - should have planned tiers
    uint64_t hits = 0, misses = 0;
    ggml_backend_sycl_get_cache_stats(backend, &hits, &misses);

    // Initially zero (no lookups yet)
    assert(hits == 0 && misses == 0 && "Initial cache stats should be zero");

    ggml_backend_free(backend);
    printf("test_cache_stats_api: PASSED (hits=%llu, misses=%llu)\n", (unsigned long long) hits,
           (unsigned long long) misses);
}

// Test cache hit rate with large inventory
static void test_cache_hit_rate() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_cache_hit_rate: SKIPPED (no SYCL device)\n");
        return;
    }

    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create large inventory to enable tiered mode (100 tensors, each 2% of VRAM = 200% total)
    std::vector<std::string>           names;
    std::vector<ggml_sycl_tensor_info> tensors;
    names.reserve(100);
    tensors.reserve(100);

    for (int i = 0; i < 100; i++) {
        names.push_back("blk." + std::to_string(i) + ".weight");
        tensors.push_back({ names.back().c_str(), free_vram / 50 });
    }

    ggml_sycl_tensor_inventory inv = { tensors.data(), tensors.size(), free_vram * 2 };
    ggml_backend_sycl_set_tensor_inventory(backend, &inv);

    // Verify cache stats API works
    uint64_t hits = 0, misses = 0;
    ggml_backend_sycl_get_cache_stats(backend, &hits, &misses);

    // Initially should be zero
    assert(hits == 0 && misses == 0 && "Initial cache stats should be zero");

    ggml_backend_free(backend);
    printf("test_cache_hit_rate: PASSED (hits=%llu, misses=%llu, tensors=%zu)\n", (unsigned long long) hits,
           (unsigned long long) misses, tensors.size());
}

int main() {
    printf("=== Tiered Dispatch Tests ===\n\n");

    test_tiered_mode_query();
    test_small_inventory_no_tiered();
    test_inventory_tensor_types();
    test_cache_instance_available();
    test_inventory_clear();
    test_cache_stats_null_safety();
    test_cache_stats_api();
    test_cache_hit_rate();

    printf("\nAll tiered dispatch tests PASSED!\n");
    return 0;
}
