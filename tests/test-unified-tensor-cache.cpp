//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unit tests for unified_tensor_cache class
// Tests inventory-based placement, tiered memory auto-enable, and statistics
//

#include "../ggml/src/ggml-sycl/unified-tensor-cache.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace ggml_sycl;

void test_inventory_placement() {
    sycl::queue q;

    // Create cache with 500MB VRAM, 1GB host
    constexpr size_t VRAM_BUDGET = 500 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    // Create inventory with mixed priorities
    tensor_inventory inventory;

    // 100MB embedding (priority 0) - should go to VRAM
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 100 * 1024 * 1024));

    // 200MB attention (priority 1) - should go to VRAM
    inventory.tensors.push_back(make_tensor_info("blk.0.attn_q.weight", 200 * 1024 * 1024));

    // 300MB experts (priority 3) - should go to host (VRAM full after embedding+attention)
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.weight", 300 * 1024 * 1024));

    inventory.total_size = 600 * 1024 * 1024;

    // Set inventory and trigger placement
    cache.set_inventory(inventory);

    // Verify placements
    assert(cache.get_planned_tier(0) == memory_tier::VRAM && "Embedding should be planned for VRAM");
    assert(cache.get_planned_tier(1) == memory_tier::VRAM && "Attention should be planned for VRAM");
    assert(cache.get_planned_tier(2) == memory_tier::PINNED_HOST && "Expert should be planned for host");

    std::cout << "test_inventory_placement: PASSED\n";
}

void test_tiered_auto_enable() {
    sycl::queue q;

    // Small VRAM budget - should auto-enable tiered mode
    constexpr size_t VRAM_BUDGET = 100 * 1024 * 1024;  // 100MB
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    tensor_inventory inventory;
    // 500MB total - exceeds VRAM
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 200 * 1024 * 1024));
    inventory.tensors.push_back(make_tensor_info("blk.0.attn_q.weight", 300 * 1024 * 1024));
    inventory.total_size = 500 * 1024 * 1024;

    cache.set_inventory(inventory);

    assert(cache.is_tiered_enabled() && "Tiered mode should be auto-enabled");

    std::cout << "test_tiered_auto_enable: PASSED\n";
}

void test_tiered_disabled_small_model() {
    sycl::queue q;

    // Large VRAM budget - should NOT enable tiered mode
    constexpr size_t VRAM_BUDGET = 1ULL * 1024 * 1024 * 1024;  // 1GB
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    tensor_inventory inventory;
    // 500MB total - fits in VRAM
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", 200 * 1024 * 1024));
    inventory.tensors.push_back(make_tensor_info("blk.0.attn_q.weight", 300 * 1024 * 1024));
    inventory.total_size = 500 * 1024 * 1024;

    cache.set_inventory(inventory);

    assert(!cache.is_tiered_enabled() && "Tiered mode should NOT be enabled for small model");

    std::cout << "test_tiered_disabled_small_model: PASSED\n";
}

void test_statistics() {
    sycl::queue q;

    constexpr size_t VRAM_BUDGET = 500 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    assert(cache.vram_budget() == VRAM_BUDGET);
    assert(cache.host_budget() == HOST_BUDGET);
    assert(cache.vram_used() == 0);
    assert(cache.cache_hits() == 0);
    assert(cache.cache_misses() == 0);

    std::cout << "test_statistics: PASSED\n";
}

void test_load_tensor_vram() {
    sycl::queue q;

    // Large VRAM budget to fit tensors
    constexpr size_t VRAM_BUDGET = 500 * 1024 * 1024;
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;
    constexpr size_t TENSOR_SIZE = 1024;  // 1KB test tensor

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    // Create small inventory
    tensor_inventory inventory;
    inventory.tensors.push_back(make_tensor_info("token_embd.weight", TENSOR_SIZE));
    inventory.total_size = TENSOR_SIZE;

    cache.set_inventory(inventory);

    // Create source data
    std::vector<uint8_t> src_data(TENSOR_SIZE);
    for (size_t i = 0; i < TENSOR_SIZE; i++) {
        src_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Load tensor
    cache.load_tensor_data(0, src_data.data());

    // Verify it's accessible
    auto loc = cache.get_tensor_with_location(0);
    assert(loc.ptr != nullptr && "Tensor should be loaded");
    assert(loc.tier == memory_tier::VRAM && "Tensor should be in VRAM");

    std::cout << "test_load_tensor_vram: PASSED\n";
}

void test_load_tensor_host_fallback() {
    sycl::queue q;

    // Small VRAM budget - will force host fallback
    constexpr size_t VRAM_BUDGET = 512;   // 512 bytes only
    constexpr size_t HOST_BUDGET = 1ULL * 1024 * 1024 * 1024;
    constexpr size_t TENSOR_SIZE = 1024;  // 1KB - larger than VRAM

    unified_tensor_cache cache(q, VRAM_BUDGET, HOST_BUDGET);

    // Create inventory with tensor larger than VRAM
    tensor_inventory inventory;
    inventory.tensors.push_back(make_tensor_info("blk.0.ffn_down_exps.weight", TENSOR_SIZE));
    inventory.total_size = TENSOR_SIZE;

    cache.set_inventory(inventory);

    // Verify planned tier is host (since it won't fit in VRAM)
    assert(cache.get_planned_tier(0) == memory_tier::PINNED_HOST && "Tensor should be planned for host");

    // Create source data
    std::vector<uint8_t> src_data(TENSOR_SIZE);
    for (size_t i = 0; i < TENSOR_SIZE; i++) {
        src_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Load tensor
    cache.load_tensor_data(0, src_data.data());

    // Verify it's accessible in host memory
    auto loc = cache.get_tensor_with_location(0);
    assert(loc.ptr != nullptr && "Tensor should be loaded");
    assert(loc.tier == memory_tier::PINNED_HOST && "Tensor should be in host memory");

    std::cout << "test_load_tensor_host_fallback: PASSED\n";
}

int main() {
    try {
        test_inventory_placement();
        test_tiered_auto_enable();
        test_tiered_disabled_small_model();
        test_statistics();
        test_load_tensor_vram();
        test_load_tensor_host_fallback();
        std::cout << "\nAll unified_tensor_cache tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
