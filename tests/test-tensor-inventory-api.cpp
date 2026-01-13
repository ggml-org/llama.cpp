//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "ggml-sycl.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void test_set_inventory() {
    // Create SYCL backend
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_set_inventory: SKIPPED (no SYCL device)\n");
        return;
    }

    // Create mock inventory
    std::vector<ggml_sycl_tensor_info> tensors = {
        { "token_embd.weight",          100 * 1024 * 1024 },
        { "blk.0.attn_q.weight",        50 * 1024 * 1024  },
        { "blk.0.ffn_down_exps.weight", 200 * 1024 * 1024 },
    };

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = 350 * 1024 * 1024;

    // Should not crash
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Check tiered status (may be true or false depending on VRAM)
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    printf("test_set_inventory: tiered_enabled=%s\n", tiered ? "true" : "false");

    ggml_backend_free(backend);
    printf("test_set_inventory: PASSED\n");
}

static void test_null_safety() {
    // Should not crash with null inputs
    ggml_backend_sycl_set_tensor_inventory(nullptr, nullptr);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend) {
        ggml_backend_sycl_set_tensor_inventory(backend, nullptr);
        ggml_backend_free(backend);
    }

    printf("test_null_safety: PASSED\n");
}

static void test_empty_inventory() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_empty_inventory: SKIPPED (no SYCL device)\n");
        return;
    }

    // Empty inventory
    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = nullptr;
    inventory.count      = 0;
    inventory.total_size = 0;

    // Should not crash with empty inventory
    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Tiered should be disabled for empty inventory (0 < VRAM)
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    assert(!tiered && "Empty inventory should not enable tiered mode");

    ggml_backend_free(backend);
    printf("test_empty_inventory: PASSED\n");
}

static void test_large_inventory() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("test_large_inventory: SKIPPED (no SYCL device)\n");
        return;
    }

    // Get VRAM info
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(0, &free_vram, &total_vram);

    // Create inventory that exceeds 90% of VRAM
    size_t num_tensors     = 100;
    size_t size_per_tensor = free_vram / 50;  // Each tensor is 2% of VRAM, 100 = 200%
    size_t total_size      = num_tensors * size_per_tensor;

    // Use std::vector<std::string> to manage string lifetime safely
    std::vector<std::string>           name_storage;
    std::vector<ggml_sycl_tensor_info> tensors;
    name_storage.reserve(num_tensors);
    tensors.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; i++) {
        name_storage.push_back("blk." + std::to_string(i) + ".weight");
        tensors.push_back({ name_storage.back().c_str(), size_per_tensor });
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = total_size;

    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // Large inventory should enable tiered mode
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    assert(tiered && "Large inventory exceeding VRAM should enable tiered mode");

    ggml_backend_free(backend);
    printf("test_large_inventory: PASSED (tiered=%s, inventory=%.1fGB, VRAM=%.1fGB)\n", tiered ? "true" : "false",
           total_size / (1024.0 * 1024.0 * 1024.0), free_vram / (1024.0 * 1024.0 * 1024.0));
}

int main() {
    printf("=== Tensor Inventory API Tests ===\n\n");

    test_null_safety();
    test_empty_inventory();
    test_set_inventory();
    test_large_inventory();

    printf("\nAll tensor inventory API tests PASSED!\n");
    return 0;
}
