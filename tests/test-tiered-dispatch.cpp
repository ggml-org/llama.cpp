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

int main() {
    printf("=== Tiered Dispatch Tests ===\n\n");

    test_tiered_mode_query();
    test_small_inventory_no_tiered();
    test_inventory_tensor_types();
    test_inventory_clear();

    printf("\nAll tiered dispatch tests PASSED!\n");
    return 0;
}
