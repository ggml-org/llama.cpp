//
// Test: Unified Cache Fast Path
//
// TDD tests for reader-writer lock optimization in unified cache.
// Tests verify:
// 1. try_get_cached_fast() returns nullptr for non-existent entries
// 2. try_get_cached_fast() returns cached pointer for READY entries
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>

// Include the unified cache header
#include "../unified-cache.hpp"

// =============================================================================
// Test Helpers
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_SKIP(reason)                          \
    do {                                           \
        g_tests_skipped++;                         \
        fprintf(stderr, "SKIPPED (%s)\n", reason); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

// =============================================================================
// Helper: Create a test cache_id for lookup
// =============================================================================
static ggml_sycl_cache_id make_test_cache_id(uint64_t model_id, uint64_t aux_id) {
    ggml_sycl_cache_id id = {};
    id.valid              = true;
    id.model_id           = model_id;
    id.aux_id             = aux_id;
    id.has_gguf           = false;
    id.file_idx           = 0;
    id.file_offs          = 0;
    id.nbytes             = 1024;
    id.name_hash          = 0;
    id.type               = GGML_TYPE_F32;
    id.tp_sharded         = false;
    id.tp_rank            = 0;
    id.tp_world_size      = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = (i == 0) ? 32 : 1;
        id.tp_local_ne[i]  = id.ne[i];
        id.tp_offset_ne[i] = 0;
    }
    return id;
}

// =============================================================================
// Test 1: try_get_cached_fast() returns nullptr for non-existent entry
//
// Verifies that the fast path returns nullptr for entries that haven't been
// cached yet.
// =============================================================================
static bool test_fast_path_returns_nullptr_for_missing(sycl::queue & q) {
    TEST_BEGIN("test_fast_path_returns_nullptr_for_missing");

    // Create a cache with small budget for testing
    constexpr size_t budget = 1024 * 1024;  // 1 MB
    ggml_sycl::unified_cache cache(q, budget);

    // Create a cache_id for an entry that doesn't exist
    ggml_sycl_cache_id id = make_test_cache_id(/*model_id=*/42, /*aux_id=*/12345);

    // Call the new fast path method - THIS SHOULD FAIL TO COMPILE
    // because try_get_cached_fast() doesn't exist yet
    void * ptr = cache.try_get_cached_fast(id, GGML_LAYOUT_AOS);

    // Verify it returns nullptr for non-existent entry
    TEST_ASSERT(ptr == nullptr, "try_get_cached_fast should return nullptr for missing entry");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: try_get_cached_fast() returns cached pointer for existing entry
//
// This test verifies that after ensure_cached() adds an entry, try_get_cached_fast()
// returns the same pointer.
// =============================================================================
static bool test_fast_path_returns_cached_pointer(sycl::queue & q) {
    TEST_BEGIN("test_fast_path_returns_cached_pointer");

    // Create a cache with small budget for testing
    constexpr size_t budget = 1024 * 1024;  // 1 MB
    ggml_sycl::unified_cache cache(q, budget);

    // Create test data on HOST (ensure_cached copies from host to device)
    constexpr size_t data_size = 1024;
    void * host_data = sycl::malloc_host(data_size, q);
    TEST_ASSERT(host_data != nullptr, "Failed to allocate host memory");

    // Initialize with some data
    std::memset(host_data, 0xAB, data_size);

    // Create a cache_id
    ggml_sycl_cache_id id = make_test_cache_id(/*model_id=*/100, /*aux_id=*/55555);
    id.nbytes = data_size;

    // Use ensure_cached() to add entry to cache
    // ensure_cached() copies from host src_ptr to device and returns device pointer
    void * cached_ptr = cache.ensure_cached(
        id,
        host_data,        // src_ptr (host memory)
        data_size,        // size
        ggml_sycl::cache_entry_type::DENSE_WEIGHT,
        /*layer_id=*/-1,
        /*expert_id=*/-1,
        GGML_LAYOUT_AOS,  // layout
        /*validate_content=*/false
    );
    TEST_ASSERT(cached_ptr != nullptr, "ensure_cached should return non-null pointer");

    // Now call try_get_cached_fast() - should return the same pointer
    void * fast_ptr = cache.try_get_cached_fast(id, GGML_LAYOUT_AOS);
    TEST_ASSERT(fast_ptr != nullptr, "try_get_cached_fast should return non-null for cached entry");
    TEST_ASSERT(fast_ptr == cached_ptr, "try_get_cached_fast should return same pointer as ensure_cached");

    // Clean up host allocation
    sycl::free(host_data, q);

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Unified Cache Fast Path Tests\n");
    fprintf(stderr, "===========================================\n");

    // Select GPU device
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU device found: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    fprintf(stderr, "-------------------------------------------\n");

    sycl::queue q(dev, sycl::property::queue::in_order{});

    // Run tests
    bool all_passed = true;

    all_passed &= test_fast_path_returns_nullptr_for_missing(q);
    all_passed &= test_fast_path_returns_cached_pointer(q);

    // Summary
    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n",
            g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }

    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
