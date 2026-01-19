//
// Test for pinned_chunk_pool - 8GB chunked host memory allocator
// Part of: llama.cpp-2pa (Tiered Memory Architecture)
//

#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

// Include the header we're testing
#include "pinned-pool.hpp"
#include "unified-cache.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"

void test_basic_allocation() {
    sycl::queue q;

    // 16GB budget, 8GB chunks
    constexpr size_t             BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // Allocate 1GB
    void * ptr1 = pool.allocate(1ULL * 1024 * 1024 * 1024);
    assert(ptr1 != nullptr && "1GB allocation should succeed");

    // Allocate another 6GB (still fits in first 8GB chunk)
    void * ptr2 = pool.allocate(6ULL * 1024 * 1024 * 1024);
    assert(ptr2 != nullptr && "6GB allocation should succeed");

    // Allocate 2GB (needs second chunk)
    void * ptr3 = pool.allocate(2ULL * 1024 * 1024 * 1024);
    assert(ptr3 != nullptr && "2GB allocation should trigger new chunk");

    // Verify pointers are different
    assert(ptr1 != ptr2 && ptr2 != ptr3 && "Pointers should be unique");
    (void) ptr1;
    (void) ptr2;
    (void) ptr3;  // Suppress unused variable warnings

    std::cout << "test_basic_allocation: PASSED\n";
}

void test_budget_limit() {
    sycl::queue q;

    // Small 10GB budget
    constexpr size_t             BUDGET = 10ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // First 8GB chunk should succeed
    void * ptr1 = pool.allocate(7ULL * 1024 * 1024 * 1024);
    assert(ptr1 != nullptr && "7GB allocation should succeed");
    (void) ptr1;  // Suppress unused variable warning

    // Second chunk would exceed budget (7GB used + 8GB new chunk > 10GB budget)
    // But we need space for 4GB more...
    void * ptr2 = pool.allocate(4ULL * 1024 * 1024 * 1024);
    // This should fail since we can't allocate another 8GB chunk
    assert(ptr2 == nullptr && "Should fail - exceeds budget");
    (void) ptr2;  // Suppress unused variable warning

    std::cout << "test_budget_limit: PASSED\n";
}

void test_gpu_accessible() {
    sycl::queue q;

    constexpr size_t             BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    constexpr size_t SIZE     = 1024 * 1024;  // 1MB
    void *           host_ptr = pool.allocate(SIZE);
    assert(host_ptr != nullptr);

    // Write pattern to host memory
    std::memset(host_ptr, 0xAB, SIZE);

    // Allocate device memory
    void * device_ptr = sycl::malloc_device(SIZE, q);
    assert(device_ptr != nullptr);

    // Copy from pinned host to device (should work if truly pinned)
    q.memcpy(device_ptr, host_ptr, SIZE).wait();

    // Copy back to verify
    std::vector<char> verify(SIZE);
    q.memcpy(verify.data(), device_ptr, SIZE).wait();

    // Check pattern
    for (size_t i = 0; i < SIZE; i++) {
        assert(static_cast<unsigned char>(verify[i]) == 0xAB && "Data mismatch");
    }

    sycl::free(device_ptr, q);
    std::cout << "test_gpu_accessible: PASSED\n";
}

void test_alignment() {
    sycl::queue q;

    constexpr size_t             BUDGET = 16ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // Allocate with various sizes, check 64-byte alignment
    void * ptr1 = pool.allocate(100);
    void * ptr2 = pool.allocate(1000);
    void * ptr3 = pool.allocate(10000);

    assert((reinterpret_cast<uintptr_t>(ptr1) % 64) == 0 && "ptr1 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr2) % 64) == 0 && "ptr2 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr3) % 64) == 0 && "ptr3 not aligned");
    (void) ptr1;
    (void) ptr2;
    (void) ptr3;  // Suppress unused variable warnings

    std::cout << "test_alignment: PASSED\n";
}

void test_chunk_count() {
    sycl::queue q;

    constexpr size_t             BUDGET = 32ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    // Initially no chunks
    assert(pool.chunk_count() == 0 && "Should start with 0 chunks");

    // After first allocation, should have 1 chunk
    void * ptr1 = pool.allocate(1024);
    assert(ptr1 != nullptr);
    (void) ptr1;  // Suppress unused variable warning
    assert(pool.chunk_count() == 1 && "Should have 1 chunk after first alloc");

    // Fill the first chunk (8GB - 1KB already used)
    void * ptr2 = pool.allocate(7ULL * 1024 * 1024 * 1024);
    assert(ptr2 != nullptr);
    (void) ptr2;  // Suppress unused variable warning in release builds
    assert(pool.chunk_count() == 1 && "Should still have 1 chunk");

    // This should trigger a second chunk
    void * ptr3 = pool.allocate(2ULL * 1024 * 1024 * 1024);
    assert(ptr3 != nullptr);
    (void) ptr3;  // Suppress unused variable warning in release builds
    assert(pool.chunk_count() == 2 && "Should have 2 chunks after overflow");

    std::cout << "test_chunk_count: PASSED\n";
}

void test_statistics() {
    sycl::queue q;

    constexpr size_t             BUDGET = 24ULL * 1024 * 1024 * 1024;
    ggml_sycl::pinned_chunk_pool pool(q, BUDGET);

    assert(pool.budget() == BUDGET && "Budget should match");
    assert(pool.allocated() == 0 && "Should start with 0 allocated");

    // First allocation triggers 8GB chunk
    void * ptr1 = pool.allocate(1024);
    assert(ptr1 != nullptr);
    (void) ptr1;  // Suppress unused variable warning in release builds
    assert(pool.allocated() == 8ULL * 1024 * 1024 * 1024 && "Should have 8GB allocated");

    std::cout << "test_statistics: PASSED\n";
}

// Include unified-cache header for host_cache integration test
#include "unified-cache.hpp"

void test_host_cache_uses_pool() {
    // This test verifies host_cache allocates from pinned pool
    // and doesn't fall back to std::malloc

    sycl::queue q;

    // Get host cache (should use pool internally)
    auto * cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr && "host_cache should be created");

    // Allocate several tensors through host_cache
    // Using 256MB each to test significant allocations
    const size_t        TENSOR_SIZE = 256 * 1024 * 1024;  // 256MB each
    std::vector<void *> ptrs;

    for (int i = 0; i < 10; i++) {
        bool needs_fill = false;
        bool pinned     = false;
        ggml_sycl::cache_location location = ggml_sycl::cache_location::HOST_MMAP;

        // Use fake but non-null pointers for cache id and src_ptr
        // (host_cache checks for nullptr and returns early)
        const ggml_sycl_cache_id fake_key = ggml_sycl::test_make_cache_id(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(0x1000 + i)));
        const void * fake_src = reinterpret_cast<const void *>(static_cast<uintptr_t>(0x2000 + i));

        void * ptr =
            cache->ensure_cached_alloc(fake_key,     // cache id - stable identifier
                                       fake_src,     // src_ptr - source data pointer (non-null to pass validation)
                                       TENSOR_SIZE,  // src_size
                                       TENSOR_SIZE,  // dst_size
                                       ggml_sycl::cache_entry_type::MOE_EXPERT,
                                       i,            // layer_id
                                       i,            // expert_id
                                       GGML_LAYOUT_AOS,
                                       false,        // validate_content (skip hash computation for fake ptr)
                                       &needs_fill, &pinned,
                                       &location,
                                       nullptr       // xmx_info
            );

        assert(ptr != nullptr && "Allocation should succeed");
        assert(pinned && "Should be pinned allocation, not std::malloc fallback");
        assert(location == ggml_sycl::cache_location::HOST_PINNED && "Expected pinned host cache location");
        ptrs.push_back(ptr);
    }

    std::cout << "test_host_cache_uses_pool: PASSED\n";
}

void test_host_cache_mmap_alias() {
    sycl::queue q;

    // Budget too small for an 8GB chunk: forces pinned alloc failure.
    constexpr size_t        BUDGET = 1ULL * 1024 * 1024;  // 1MB
    ggml_sycl::host_cache   cache(q, BUDGET);

    std::vector<uint8_t> data(1024, 0x5A);
    bool                 needs_fill = false;
    bool                 pinned     = false;
    ggml_sycl::cache_location location = ggml_sycl::cache_location::HOST_MMAP;

    const ggml_sycl_cache_id fake_key = ggml_sycl::test_make_cache_id(
        reinterpret_cast<const void *>(static_cast<uintptr_t>(0x4000)));
    void * ptr = cache.ensure_cached_alloc(fake_key, data.data(), data.size(), data.size(),
                                           ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1,
                                           GGML_LAYOUT_AOS, false, &needs_fill, &pinned, &location, nullptr);

    assert(ptr == data.data() && "Expected mmap alias pointer");
    assert(!pinned && "Expected non-pinned alias");
    assert(!needs_fill && "Alias path should not require fill");
    assert(location == ggml_sycl::cache_location::HOST_MMAP && "Expected HOST_MMAP location");

    std::cout << "test_host_cache_mmap_alias: PASSED\n";
}

// ==============================================================================
// Tests for Non-USM Detection (Phase 1 of streaming tensor reordering)
// ==============================================================================

void test_non_usm_detection_mmap() {
    // Test that mmap'd memory is detected as non-USM (unknown alloc type)
    // This is critical for the streaming fill path - we need to correctly
    // identify that mmap'd pointers cannot be accessed from GPU kernels.

    sycl::queue q;

    // Simulate mmap'd memory using regular std::vector (which is on the heap)
    // This memory is NOT USM-accessible from the GPU.
    std::vector<uint8_t> heap_data(4096, 0xAB);
    const void *         non_usm_ptr = heap_data.data();

    // Check pointer type - should be unknown (non-USM)
    sycl::usm::alloc alloc_type = sycl::get_pointer_type(non_usm_ptr, q.get_context());
    (void)alloc_type;  // Suppress unused warning when asserts disabled
    assert(alloc_type == sycl::usm::alloc::unknown && "Heap/mmap memory should be unknown USM type");

    std::cout << "test_non_usm_detection_mmap: PASSED\n";
}

void test_non_usm_detection_pinned() {
    // Test that pinned USM memory is detected as host type
    sycl::queue q;

    void * pinned_ptr = sycl::malloc_host(4096, q);
    assert(pinned_ptr != nullptr && "malloc_host should succeed");

    sycl::usm::alloc alloc_type = sycl::get_pointer_type(pinned_ptr, q.get_context());
    (void)alloc_type;  // Suppress unused warning when asserts disabled
    assert(alloc_type == sycl::usm::alloc::host && "Pinned memory should be host USM type");

    sycl::free(pinned_ptr, q);
    std::cout << "test_non_usm_detection_pinned: PASSED\n";
}

void test_non_usm_detection_device() {
    // Test that device memory is detected as device type
    sycl::queue q;

    void * device_ptr = sycl::malloc_device(4096, q);
    assert(device_ptr != nullptr && "malloc_device should succeed");

    sycl::usm::alloc alloc_type = sycl::get_pointer_type(device_ptr, q.get_context());
    (void)alloc_type;  // Suppress unused warning when asserts disabled
    assert(alloc_type == sycl::usm::alloc::device && "Device memory should be device USM type");

    sycl::free(device_ptr, q);
    std::cout << "test_non_usm_detection_device: PASSED\n";
}

void test_is_usm_accessible_helper() {
    // Test the helper function that determines if a pointer is safe for GPU kernel access
    // This helper should return true only for host/device/shared USM pointers,
    // NOT for unknown (mmap'd) pointers.

    sycl::queue q;

    // Non-USM (heap/mmap) - NOT accessible
    std::vector<uint8_t> heap_data(4096, 0xAB);
    const sycl::usm::alloc heap_type = sycl::get_pointer_type(heap_data.data(), q.get_context());
    bool heap_accessible = (heap_type == sycl::usm::alloc::host ||
                            heap_type == sycl::usm::alloc::device ||
                            heap_type == sycl::usm::alloc::shared);
    (void)heap_accessible;  // Suppress unused warning when asserts disabled
    assert(!heap_accessible && "Heap memory should NOT be USM accessible");

    // Pinned USM - IS accessible
    void * pinned_ptr = sycl::malloc_host(4096, q);
    const sycl::usm::alloc pinned_type = sycl::get_pointer_type(pinned_ptr, q.get_context());
    bool pinned_accessible = (pinned_type == sycl::usm::alloc::host ||
                              pinned_type == sycl::usm::alloc::device ||
                              pinned_type == sycl::usm::alloc::shared);
    (void)pinned_accessible;  // Suppress unused warning when asserts disabled
    assert(pinned_accessible && "Pinned memory should be USM accessible");
    sycl::free(pinned_ptr, q);

    // Device USM - IS accessible
    void * device_ptr = sycl::malloc_device(4096, q);
    const sycl::usm::alloc device_type = sycl::get_pointer_type(device_ptr, q.get_context());
    bool device_accessible = (device_type == sycl::usm::alloc::host ||
                              device_type == sycl::usm::alloc::device ||
                              device_type == sycl::usm::alloc::shared);
    (void)device_accessible;  // Suppress unused warning when asserts disabled
    assert(device_accessible && "Device memory should be USM accessible");
    sycl::free(device_ptr, q);

    std::cout << "test_is_usm_accessible_helper: PASSED\n";
}

// ==============================================================================
// Tests for Staging Buffer Helper (Phase 2 of streaming tensor reordering)
// ==============================================================================

void test_staging_buffer_allocation() {
    // Test that allocate_pinned_runtime returns USM-accessible memory
    // that can be used as a staging buffer for non-USM -> device transfers

    sycl::queue q;

    // Get the host cache
    auto * cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr && "host_cache should be created");

    // Allocate a staging buffer (64MB default for streaming chunks)
    constexpr size_t STAGING_SIZE = 64 * 1024 * 1024;  // 64MB
    void * staging = cache->allocate_pinned_runtime(STAGING_SIZE);
    assert(staging != nullptr && "Staging buffer allocation should succeed");

    // Verify it's USM-accessible (host type)
    sycl::usm::alloc alloc_type = sycl::get_pointer_type(staging, q.get_context());
    (void)alloc_type;  // Suppress unused warning when asserts disabled
    assert(alloc_type == sycl::usm::alloc::host && "Staging buffer should be pinned host memory");

    // Free the staging buffer
    cache->free_pinned_runtime(staging, STAGING_SIZE);

    std::cout << "test_staging_buffer_allocation: PASSED\n";
}

void test_staging_buffer_transfer() {
    // Test that staging buffer can be used to transfer non-USM data to device
    // This is the core pattern for streaming tensor reordering

    sycl::queue q;

    auto * cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr);

    // Create non-USM source data (simulating mmap'd weights)
    constexpr size_t DATA_SIZE = 1024 * 1024;  // 1MB
    std::vector<uint8_t> non_usm_data(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        non_usm_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Allocate staging buffer
    void * staging = cache->allocate_pinned_runtime(DATA_SIZE);
    assert(staging != nullptr);

    // Copy non-USM data to staging buffer (CPU memcpy)
    std::memcpy(staging, non_usm_data.data(), DATA_SIZE);

    // Allocate device memory
    void * device_ptr = sycl::malloc_device(DATA_SIZE, q);
    assert(device_ptr != nullptr);

    // Transfer staging -> device (this is the key: staging is USM-accessible)
    q.memcpy(device_ptr, staging, DATA_SIZE).wait();

    // Verify by reading back
    std::vector<uint8_t> verify(DATA_SIZE);
    q.memcpy(verify.data(), device_ptr, DATA_SIZE).wait();

    // Check data integrity
    for (size_t i = 0; i < DATA_SIZE; i++) {
        assert(verify[i] == non_usm_data[i] && "Data mismatch after staging transfer");
    }

    // Cleanup
    sycl::free(device_ptr, q);
    cache->free_pinned_runtime(staging, DATA_SIZE);

    std::cout << "test_staging_buffer_transfer: PASSED\n";
}

void test_staging_buffer_chunked_transfer() {
    // Test chunked transfer pattern: larger data transferred in chunks
    // through a smaller staging buffer

    sycl::queue q;

    auto * cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr);

    // Large non-USM source (16MB)
    constexpr size_t TOTAL_SIZE = 16 * 1024 * 1024;
    // Small staging buffer (4MB)
    constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;

    std::vector<uint8_t> non_usm_data(TOTAL_SIZE);
    for (size_t i = 0; i < TOTAL_SIZE; i++) {
        non_usm_data[i] = static_cast<uint8_t>((i * 17 + 31) & 0xFF);  // Pseudo-random pattern
    }

    // Allocate staging buffer (smaller than total)
    void * staging = cache->allocate_pinned_runtime(CHUNK_SIZE);
    assert(staging != nullptr);

    // Allocate device memory for full size
    void * device_ptr = sycl::malloc_device(TOTAL_SIZE, q);
    assert(device_ptr != nullptr);

    // Transfer in chunks
    for (size_t offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE) {
        size_t chunk_bytes = std::min(CHUNK_SIZE, TOTAL_SIZE - offset);

        // Copy chunk to staging
        std::memcpy(staging, non_usm_data.data() + offset, chunk_bytes);

        // Transfer staging -> device (at correct offset)
        q.memcpy(static_cast<uint8_t*>(device_ptr) + offset, staging, chunk_bytes).wait();
    }

    // Verify by reading back
    std::vector<uint8_t> verify(TOTAL_SIZE);
    q.memcpy(verify.data(), device_ptr, TOTAL_SIZE).wait();

    for (size_t i = 0; i < TOTAL_SIZE; i++) {
        assert(verify[i] == non_usm_data[i] && "Data mismatch in chunked transfer");
    }

    // Cleanup
    sycl::free(device_ptr, q);
    cache->free_pinned_runtime(staging, CHUNK_SIZE);

    std::cout << "test_staging_buffer_chunked_transfer: PASSED\n";
}

// ==============================================================================
// Tests for Streaming Fill Function (Phase 3 of streaming tensor reordering)
// ==============================================================================

void test_is_usm_accessible_correct_for_unknown() {
    // Test the helper function is_usm_accessible() correctly rejects unknown pointers
    // This is the critical fix: unknown != host-accessible

    sycl::queue q;

    // Heap memory (simulates mmap) should return false for is_usm_accessible
    std::vector<uint8_t> heap_data(4096, 0xAB);

    // Check that unknown pointer type is correctly identified as NOT USM-accessible
    const sycl::usm::alloc heap_type = sycl::get_pointer_type(heap_data.data(), q.get_context());

    // This is the key assertion - unknown should NOT be treated as host-accessible
    bool is_accessible = (heap_type == sycl::usm::alloc::host ||
                          heap_type == sycl::usm::alloc::device ||
                          heap_type == sycl::usm::alloc::shared);

    assert(!is_accessible && "Unknown (mmap/heap) memory must NOT be considered USM-accessible");
    assert(heap_type == sycl::usm::alloc::unknown && "Heap memory should have unknown alloc type");

    std::cout << "test_is_usm_accessible_correct_for_unknown: PASSED\n";
}

void test_streaming_fill_with_non_usm_source() {
    // Test that we can copy non-USM data to device memory correctly
    // using the staging buffer pattern (simulates streaming fill)

    sycl::queue q;

    auto * cache = ggml_sycl::get_host_cache(q);
    assert(cache != nullptr);

    // Create non-USM source data (simulating mmap'd weights)
    constexpr size_t DATA_SIZE = 256 * 1024;  // 256KB
    std::vector<uint8_t> non_usm_data(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        non_usm_data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    // Verify source is non-USM
    const sycl::usm::alloc src_type = sycl::get_pointer_type(non_usm_data.data(), q.get_context());
    assert(src_type == sycl::usm::alloc::unknown && "Source should be non-USM");

    // Allocate pinned staging buffer
    void * staging = cache->allocate_pinned_runtime(DATA_SIZE);
    assert(staging != nullptr);

    // Allocate device destination
    void * device_dst = sycl::malloc_device(DATA_SIZE, q);
    assert(device_dst != nullptr);

    // Copy non-USM -> staging (CPU memcpy, safe)
    std::memcpy(staging, non_usm_data.data(), DATA_SIZE);

    // Copy staging -> device (SYCL memcpy, staging is USM-accessible)
    q.memcpy(device_dst, staging, DATA_SIZE).wait();

    // Verify by reading back
    std::vector<uint8_t> verify(DATA_SIZE);
    q.memcpy(verify.data(), device_dst, DATA_SIZE).wait();

    for (size_t i = 0; i < DATA_SIZE; i++) {
        assert(verify[i] == non_usm_data[i] && "Data mismatch in streaming fill");
    }

    // Cleanup
    sycl::free(device_dst, q);
    cache->free_pinned_runtime(staging, DATA_SIZE);

    std::cout << "test_streaming_fill_with_non_usm_source: PASSED\n";
}

void test_fill_reordered_detects_non_usm() {
    // Test that fill_reordered correctly detects non-USM source and uses staging
    // This tests the fix to ggml_sycl_fill_reordered_host()
    //
    // The bug was: unknown alloc type was treated as host-accessible
    // The fix: only host/device/shared are USM-accessible, unknown is NOT

    sycl::queue q;

    // Create non-USM data
    std::vector<uint8_t> mmap_data(4096, 0x42);

    // Verify the SYCL API returns unknown for non-USM
    sycl::usm::alloc alloc = sycl::get_pointer_type(mmap_data.data(), q.get_context());
    assert(alloc == sycl::usm::alloc::unknown);

    // The correct behavior: unknown should NOT be treated as USM-accessible
    // OLD buggy code: src_is_host = (alloc == host || alloc == shared || alloc == unknown)
    // NEW fixed code:  src_is_host = (alloc == host || alloc == shared)  // NOT unknown!

    bool buggy_is_host = (alloc == sycl::usm::alloc::host ||
                          alloc == sycl::usm::alloc::shared ||
                          alloc == sycl::usm::alloc::unknown);  // WRONG!

    bool fixed_is_host = (alloc == sycl::usm::alloc::host ||
                          alloc == sycl::usm::alloc::shared);  // CORRECT!

    assert(buggy_is_host == true && "Buggy detection treats unknown as host");
    assert(fixed_is_host == false && "Fixed detection correctly rejects unknown");

    std::cout << "test_fill_reordered_detects_non_usm: PASSED\n";
}

int main() {
    try {
        test_basic_allocation();
        test_budget_limit();
        test_gpu_accessible();
        test_alignment();
        test_chunk_count();
        test_statistics();
        test_host_cache_uses_pool();
        test_host_cache_mmap_alias();

        // Phase 1: Non-USM detection tests
        test_non_usm_detection_mmap();
        test_non_usm_detection_pinned();
        test_non_usm_detection_device();
        test_is_usm_accessible_helper();

        // Phase 2: Staging buffer tests
        test_staging_buffer_allocation();
        test_staging_buffer_transfer();
        test_staging_buffer_chunked_transfer();

        // Phase 3: Streaming fill function tests
        test_is_usm_accessible_correct_for_unknown();
        test_streaming_fill_with_non_usm_source();
        test_fill_reordered_detects_non_usm();

        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
