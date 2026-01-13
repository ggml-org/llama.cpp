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

    // Create a large inventory that should exceed VRAM
    std::vector<ggml_sycl_tensor_info> tensors;
    size_t                             total_size = 0;

    // 100 tensors of 1GB each = 100GB total (should exceed any consumer GPU)
    for (int i = 0; i < 100; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.weight", i);
        size_t size = 1ULL * 1024 * 1024 * 1024;  // 1GB
        tensors.push_back({ strdup(name), size });
        total_size += size;
    }

    ggml_sycl_tensor_inventory inventory;
    inventory.tensors    = tensors.data();
    inventory.count      = tensors.size();
    inventory.total_size = total_size;

    ggml_backend_sycl_set_tensor_inventory(backend, &inventory);

    // With 100GB model, tiered should be enabled
    bool tiered = ggml_backend_sycl_is_tiered_enabled(backend);
    printf("test_large_inventory: tiered_enabled=%s (expected: true for 100GB model)\n", tiered ? "true" : "false");
    assert(tiered && "Large inventory should enable tiered mode");

    // Cleanup
    for (auto & t : tensors) {
        free(const_cast<char *>(t.name));
    }
    ggml_backend_free(backend);
    printf("test_large_inventory: PASSED\n");
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
