//
// Test for pinned_chunk_pool - 8GB chunked host memory allocator
// Part of: llama.cpp-2pa (Tiered Memory Architecture)
//

#include <cassert>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

// Include the header we're testing
#include "pinned-pool.hpp"

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

        // Use fake but non-null pointers for key_ptr and src_ptr
        // (host_cache checks for nullptr and returns early)
        const void * fake_key = reinterpret_cast<const void *>(static_cast<uintptr_t>(0x1000 + i));
        const void * fake_src = reinterpret_cast<const void *>(static_cast<uintptr_t>(0x2000 + i));

        void * ptr =
            cache->ensure_cached_alloc(fake_key,     // key_ptr - stable identifier
                                       fake_src,     // src_ptr - source data pointer (non-null to pass validation)
                                       TENSOR_SIZE,  // src_size
                                       TENSOR_SIZE,  // dst_size
                                       ggml_sycl::cache_entry_type::MOE_EXPERT,
                                       i,            // layer_id
                                       i,            // expert_id
                                       GGML_LAYOUT_AOS,
                                       false,        // validate_content (skip hash computation for fake ptr)
                                       &needs_fill, &pinned,
                                       nullptr       // xmx_info
            );

        assert(ptr != nullptr && "Allocation should succeed");
        assert(pinned && "Should be pinned allocation, not std::malloc fallback");
        ptrs.push_back(ptr);
    }

    std::cout << "test_host_cache_uses_pool: PASSED\n";
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
        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
