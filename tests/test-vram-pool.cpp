//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../ggml/src/ggml-sycl/vram-pool.hpp"

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace ggml_sycl;

void test_basic_allocation() {
    sycl::queue q;

    // 1GB budget
    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    // Allocate 100MB
    void * ptr1 = pool.allocate(100 * 1024 * 1024, 1);
    assert(ptr1 != nullptr && "100MB allocation should succeed");

    // Allocate another 200MB
    void * ptr2 = pool.allocate(200 * 1024 * 1024, 2);
    assert(ptr2 != nullptr && "200MB allocation should succeed");

    // Verify pointers are different
    assert(ptr1 != ptr2 && "Pointers should be unique");

    // Verify used tracking
    assert(pool.used() == 300 * 1024 * 1024 && "Used should be 300MB");

    std::cout << "test_basic_allocation: PASSED\n";
}

void test_budget_limit() {
    sycl::queue q;

    // Small 100MB budget
    constexpr size_t BUDGET = 100 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    // First 80MB should succeed
    void * ptr1 = pool.allocate(80 * 1024 * 1024, 1);
    assert(ptr1 != nullptr && "80MB allocation should succeed");
    (void) ptr1;  // Suppress unused variable warning

    // Second 80MB should fail (exceeds budget)
    void * ptr2 = pool.allocate(80 * 1024 * 1024, 2);
    assert(ptr2 == nullptr && "Should fail - exceeds budget");
    (void) ptr2;  // Suppress unused variable warning

    std::cout << "test_budget_limit: PASSED\n";
}

void test_deallocation() {
    sycl::queue q;

    constexpr size_t BUDGET = 200 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    // Allocate 150MB
    void * ptr1 = pool.allocate(150 * 1024 * 1024, 1);
    assert(ptr1 != nullptr);
    (void) ptr1;  // Suppress unused variable warning
    assert(pool.used() == 150 * 1024 * 1024);

    // Deallocate
    pool.deallocate(1);
    assert(pool.used() == 0 && "Used should be 0 after deallocation");

    // Should be able to allocate again
    void * ptr2 = pool.allocate(150 * 1024 * 1024, 2);
    assert(ptr2 != nullptr && "Should succeed after deallocation");
    (void) ptr2;  // Suppress unused variable warning

    std::cout << "test_deallocation: PASSED\n";
}

void test_alignment() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    // Allocate with various sizes, check 64-byte alignment
    void * ptr1 = pool.allocate(100, 1);
    void * ptr2 = pool.allocate(1000, 2);
    void * ptr3 = pool.allocate(10000, 3);

    assert((reinterpret_cast<uintptr_t>(ptr1) % 64) == 0 && "ptr1 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr2) % 64) == 0 && "ptr2 not aligned");
    assert((reinterpret_cast<uintptr_t>(ptr3) % 64) == 0 && "ptr3 not aligned");
    (void) ptr1;
    (void) ptr2;
    (void) ptr3;  // Suppress unused variable warnings

    std::cout << "test_alignment: PASSED\n";
}

void test_is_allocated() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    assert(!pool.is_allocated(1) && "Should not be allocated initially");

    void * ptr1 = pool.allocate(1024, 1);
    assert(ptr1 != nullptr);
    (void) ptr1;  // Suppress unused variable warning
    assert(pool.is_allocated(1) && "Should be allocated after allocate()");

    pool.deallocate(1);
    assert(!pool.is_allocated(1) && "Should not be allocated after deallocate()");

    std::cout << "test_is_allocated: PASSED\n";
}

void test_get() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    assert(pool.get(1) == nullptr && "get() should return nullptr for non-existent");

    void * ptr1 = pool.allocate(1024, 1);
    assert(ptr1 != nullptr);
    assert(pool.get(1) == ptr1 && "get() should return allocated pointer");

    pool.deallocate(1);
    assert(pool.get(1) == nullptr && "get() should return nullptr after deallocate");

    std::cout << "test_get: PASSED\n";
}

void test_allocation_count() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    assert(pool.allocation_count() == 0 && "Should start with 0 allocations");

    pool.allocate(1024, 1);
    assert(pool.allocation_count() == 1);

    pool.allocate(2048, 2);
    assert(pool.allocation_count() == 2);

    pool.deallocate(1);
    assert(pool.allocation_count() == 1);

    pool.deallocate(2);
    assert(pool.allocation_count() == 0);

    std::cout << "test_allocation_count: PASSED\n";
}

void test_duplicate_tensor_id() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    // Allocate with tensor_id 1
    void * ptr1 = pool.allocate(1024, 1);
    assert(ptr1 != nullptr);

    // Allocating again with same tensor_id should return same pointer
    void * ptr2 = pool.allocate(1024, 1);
    assert(ptr2 == ptr1 && "Should return existing allocation for same tensor_id");

    // Should only count as one allocation
    assert(pool.allocation_count() == 1);

    std::cout << "test_duplicate_tensor_id: PASSED\n";
}

void test_available() {
    sycl::queue q;

    constexpr size_t BUDGET = 1ULL * 1024 * 1024 * 1024;
    vram_pool        pool(q, BUDGET);

    assert(pool.available() == BUDGET && "Should start with full budget available");

    // Account for 64-byte alignment rounding
    size_t aligned_size = (100 * 1024 * 1024 + 63) & ~63ULL;
    pool.allocate(100 * 1024 * 1024, 1);
    assert(pool.available() == BUDGET - aligned_size && "Available should decrease after allocation");

    pool.deallocate(1);
    assert(pool.available() == BUDGET && "Available should restore after deallocation");

    std::cout << "test_available: PASSED\n";
}

int main() {
    try {
        test_basic_allocation();
        test_budget_limit();
        test_deallocation();
        test_alignment();
        test_is_allocated();
        test_get();
        test_allocation_count();
        test_duplicate_tensor_id();
        test_available();
        std::cout << "\nAll vram_pool tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
