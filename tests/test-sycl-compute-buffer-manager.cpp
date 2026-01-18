// SYCL Compute Buffer Manager unit tests
// Tests for compute buffer management with P0 priority (never evicted)
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-6s5)
//
// TDD: These tests written FIRST, before implementation.
// Implementation must make these tests pass.
//
// Compute Buffer Design:
// - P0 priority: NEVER evicted (pinned=true)
// - Pool-based allocation: Reuse buffers to avoid expensive SYCL mallocs
// - Scratch space: Resizable buffer for temporary operations
// - Automatic cleanup: Free all buffers on manager destruction
//
// Test Cases:
// 1. Compute buffer allocation with P0 priority
// 2. Buffer pool pre-allocation at known sizes
// 3. P0 buffers are never evicted (pinned=true)
// 4. Memory accounting integration
// 5. Double-buffer acquire/release cycle
// 6. Buffer reuse within same inference pass
// 7. Proper cleanup on model unload
// 8. Out-of-memory handling (fail fast, not evict compute)

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Include the compute buffer manager header (to be created)
#include "ggml-sycl/compute-buffer-manager.hpp"

// Helper macros for size literals
#define KB(x) ((x) * 1024ULL)
#define MB(x) ((x) * 1024ULL * 1024ULL)

// Test context: manages SYCL queue initialization
struct TestContext {
    sycl::queue queue;
    bool        valid = false;

    TestContext() {
        try {
            // Try to get a GPU device, fall back to default
            auto gpu_selector = sycl::gpu_selector_v;
            queue             = sycl::queue(gpu_selector, sycl::property::queue::in_order{});
            valid             = true;
            printf("  [TestContext] Using GPU: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());
        } catch (const sycl::exception & e) {
            // No GPU available, use default device
            try {
                queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
                valid = true;
                printf("  [TestContext] Using default device: %s\n",
                       queue.get_device().get_info<sycl::info::device::name>().c_str());
            } catch (const sycl::exception & e2) {
                printf("  [TestContext] FATAL: No SYCL device available: %s\n", e2.what());
                valid = false;
            }
        }
    }
};

static TestContext & get_test_context() {
    static TestContext ctx;
    return ctx;
}

// =============================================================================
// Test 1: Compute buffer allocation returns valid pointer
// =============================================================================
static bool test_compute_buffer_allocation() {
    printf("TEST: test_compute_buffer_allocation\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Allocate a compute buffer
    void * ptr = mgr.allocate(MB(1), "test_buffer");

    if (ptr == nullptr) {
        printf("  FAIL: allocate() returned nullptr\n");
        return false;
    }

    // Verify we can write to the buffer (proves it's valid SYCL memory)
    ctx.queue.memset(ptr, 0, MB(1)).wait();

    // Clean up
    mgr.release(ptr);

    printf("  PASS: compute buffer allocation works\n");
    return true;
}

// =============================================================================
// Test 2: Compute buffers have P0 (CRITICAL) priority
// =============================================================================
static bool test_compute_buffer_p0_priority() {
    printf("TEST: test_compute_buffer_p0_priority\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Allocate a compute buffer
    void * ptr = mgr.allocate(MB(1), "test_compute");

    if (ptr == nullptr) {
        printf("  FAIL: allocate() returned nullptr\n");
        return false;
    }

    // Verify it's registered as P0
    auto priority = mgr.get_priority(ptr);
    if (priority != ggml_sycl::EvictionPriority::P0_COMPUTE) {
        printf("  FAIL: compute buffer should have P0_COMPUTE priority\n");
        return false;
    }

    mgr.release(ptr);

    printf("  PASS: compute buffers have P0 priority\n");
    return true;
}

// =============================================================================
// Test 3: P0 buffers are never evicted (pinned=true)
// =============================================================================
static bool test_compute_buffer_never_evicted() {
    printf("TEST: test_compute_buffer_never_evicted\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Allocate compute buffer
    void * compute_ptr = mgr.allocate(MB(10), "critical_compute");

    if (compute_ptr == nullptr) {
        printf("  FAIL: allocate() returned nullptr\n");
        return false;
    }

    // Verify it's pinned (should never be evicted)
    bool is_pinned = mgr.is_pinned(compute_ptr);
    if (!is_pinned) {
        printf("  FAIL: compute buffer should be pinned (not evictable)\n");
        return false;
    }

    // Attempting to request eviction should return false (nothing to evict)
    bool evicted = mgr.try_evict_for_space(MB(100));
    if (evicted) {
        printf("  FAIL: should not be able to evict P0 compute buffers\n");
        return false;
    }

    // Verify buffer still valid after eviction attempt
    bool still_valid = mgr.is_valid(compute_ptr);
    if (!still_valid) {
        printf("  FAIL: compute buffer should still be valid after eviction attempt\n");
        return false;
    }

    mgr.release(compute_ptr);

    printf("  PASS: P0 buffers are never evicted\n");
    return true;
}

// =============================================================================
// Test 4: Pool reuses buffers instead of reallocating
// =============================================================================
static bool test_compute_buffer_pool_reuse() {
    printf("TEST: test_compute_buffer_pool_reuse\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Allocate and release a buffer
    void * ptr1 = mgr.allocate(MB(1), "first_alloc");
    if (ptr1 == nullptr) {
        printf("  FAIL: first allocate() returned nullptr\n");
        return false;
    }
    mgr.release(ptr1);

    // Allocate same size - should reuse the same buffer
    void * ptr2 = mgr.allocate(MB(1), "second_alloc");

    if (ptr2 != ptr1) {
        printf("  FAIL: expected buffer reuse, got different pointers (ptr1=%p, ptr2=%p)\n", ptr1, ptr2);
        return false;
    }

    // Verify pool hit stats
    if (mgr.num_pool_hits() < 1) {
        printf("  FAIL: expected at least 1 pool hit, got %zu\n", mgr.num_pool_hits());
        return false;
    }

    mgr.release(ptr2);

    printf("  PASS: pool reuses buffers correctly\n");
    return true;
}

// =============================================================================
// Test 5: Pool miss creates new buffer
// =============================================================================
static bool test_compute_buffer_pool_miss() {
    printf("TEST: test_compute_buffer_pool_miss\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // First allocation - pool miss (no buffers in pool)
    void * ptr1 = mgr.allocate(MB(1), "first");

    if (mgr.num_pool_misses() != 1) {
        printf("  FAIL: expected 1 pool miss, got %zu\n", mgr.num_pool_misses());
        return false;
    }

    // Second allocation of different size - pool miss (no buffer of that size)
    void * ptr2 = mgr.allocate(MB(2), "second");

    if (mgr.num_pool_misses() != 2) {
        printf("  FAIL: expected 2 pool misses, got %zu\n", mgr.num_pool_misses());
        return false;
    }

    // Pointers should be different
    if (ptr1 == ptr2) {
        printf("  FAIL: expected different pointers for different sizes\n");
        return false;
    }

    mgr.release(ptr1);
    mgr.release(ptr2);

    printf("  PASS: pool miss creates new buffers\n");
    return true;
}

// =============================================================================
// Test 6: Memory accounting tracks pool usage
// =============================================================================
static bool test_compute_buffer_memory_accounting() {
    printf("TEST: test_compute_buffer_memory_accounting\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Initial usage should be 0
    if (mgr.pool_used_size() != 0) {
        printf("  FAIL: expected 0 initial usage, got %zu\n", mgr.pool_used_size());
        return false;
    }

    // Allocate some buffers
    void * ptr1 = mgr.allocate(MB(1), "buf1");
    void * ptr2 = mgr.allocate(MB(2), "buf2");

    // Usage should reflect allocated buffers
    size_t expected_used = MB(3);
    if (mgr.pool_used_size() != expected_used) {
        printf("  FAIL: expected %zu used, got %zu\n", expected_used, mgr.pool_used_size());
        return false;
    }

    // Release one buffer - used size should decrease
    mgr.release(ptr1);

    expected_used = MB(2);
    if (mgr.pool_used_size() != expected_used) {
        printf("  FAIL: expected %zu used after release, got %zu\n", expected_used, mgr.pool_used_size());
        return false;
    }

    // Total pool size should still include the released buffer
    if (mgr.pool_total_size() < MB(3)) {
        printf("  FAIL: total pool size should include released buffers\n");
        return false;
    }

    mgr.release(ptr2);

    printf("  PASS: memory accounting is accurate\n");
    return true;
}

// =============================================================================
// Test 7: Scratch space grows as needed
// =============================================================================
static bool test_compute_scratch_resize() {
    printf("TEST: test_compute_scratch_resize\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Initial scratch size should be 0
    if (mgr.get_scratch_size() != 0) {
        printf("  FAIL: expected 0 initial scratch size\n");
        return false;
    }

    // Request small scratch
    void * scratch1 = mgr.get_scratch(MB(8));
    if (scratch1 == nullptr) {
        printf("  FAIL: get_scratch() returned nullptr\n");
        return false;
    }
    if (mgr.get_scratch_capacity() < MB(8)) {
        printf("  FAIL: scratch capacity should be >= 8 MB\n");
        return false;
    }

    // Request larger scratch - should grow
    void * scratch2 = mgr.get_scratch(MB(32));
    if (scratch2 == nullptr) {
        printf("  FAIL: get_scratch() for larger size returned nullptr\n");
        return false;
    }
    if (mgr.get_scratch_capacity() < MB(32)) {
        printf("  FAIL: scratch capacity should be >= 32 MB after growth\n");
        return false;
    }

    // Request smaller scratch - should reuse existing (no shrink)
    void * scratch3 = mgr.get_scratch(MB(4));
    if (scratch3 != scratch2) {
        // Pointer might be same or reallocated, but capacity should not shrink
        if (mgr.get_scratch_capacity() < MB(32)) {
            printf("  FAIL: scratch should not shrink when smaller size requested\n");
            return false;
        }
    }

    printf("  PASS: scratch space grows as needed\n");
    return true;
}

// =============================================================================
// Test 8: Double-buffer acquire/release cycle
// =============================================================================
static bool test_compute_double_buffer_cycle() {
    printf("TEST: test_compute_double_buffer_cycle\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Simulate double-buffering: allocate two buffers, use them alternately
    void * buf_a = mgr.allocate(MB(4), "double_buf_a");
    void * buf_b = mgr.allocate(MB(4), "double_buf_b");

    if (buf_a == nullptr || buf_b == nullptr) {
        printf("  FAIL: failed to allocate double buffers\n");
        return false;
    }

    // They should be different buffers
    if (buf_a == buf_b) {
        printf("  FAIL: double buffers should have different pointers\n");
        return false;
    }

    // Simulate several iterations of double-buffering
    for (int i = 0; i < 5; i++) {
        // Use buf_a for compute, buf_b for transfer
        ctx.queue.memset(buf_a, i, MB(4));

        // Swap roles
        std::swap(buf_a, buf_b);
    }

    // Both should still be valid after cycles
    if (!mgr.is_valid(buf_a) || !mgr.is_valid(buf_b)) {
        printf("  FAIL: buffers should remain valid after use cycles\n");
        return false;
    }

    mgr.release(buf_a);
    mgr.release(buf_b);

    printf("  PASS: double-buffer cycle works correctly\n");
    return true;
}

// =============================================================================
// Test 9: Buffer reuse within same inference pass
// =============================================================================
static bool test_compute_buffer_reuse_inference_pass() {
    printf("TEST: test_compute_buffer_reuse_inference_pass\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // First inference pass: allocate buffers
    void * attn_buf = mgr.allocate(MB(2), "attention_scratch");
    void * ffn_buf  = mgr.allocate(MB(4), "ffn_scratch");

    mgr.release(attn_buf);
    mgr.release(ffn_buf);

    size_t misses_after_first_pass = mgr.num_pool_misses();

    // Second inference pass: should reuse buffers
    void * attn_buf2 = mgr.allocate(MB(2), "attention_scratch");
    void * ffn_buf2  = mgr.allocate(MB(4), "ffn_scratch");

    size_t misses_after_second_pass = mgr.num_pool_misses();

    // No new pool misses should occur in second pass
    if (misses_after_second_pass != misses_after_first_pass) {
        printf("  FAIL: second pass should reuse buffers without pool misses\n");
        printf("        misses: first_pass=%zu, second_pass=%zu\n", misses_after_first_pass, misses_after_second_pass);
        return false;
    }

    // Buffers should be the same pointers
    if (attn_buf2 != attn_buf || ffn_buf2 != ffn_buf) {
        printf("  FAIL: second pass should get same buffer pointers\n");
        return false;
    }

    mgr.release(attn_buf2);
    mgr.release(ffn_buf2);

    printf("  PASS: buffer reuse works within inference pass\n");
    return true;
}

// =============================================================================
// Test 10: Proper cleanup on manager destruction
// =============================================================================
static bool test_compute_buffer_cleanup() {
    printf("TEST: test_compute_buffer_cleanup\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    // Create manager in a scope to test destruction
    {
        ggml_sycl::ComputeBufferManager mgr(ctx.queue);

        // Allocate several buffers
        void * ptr1 = mgr.allocate(MB(1), "cleanup_test_1");
        void * ptr2 = mgr.allocate(MB(2), "cleanup_test_2");
        void * ptr3 = mgr.allocate(MB(3), "cleanup_test_3");

        // Don't release - destructor should clean up
        (void) ptr1;
        (void) ptr2;
        (void) ptr3;

        // Also allocate scratch
        mgr.get_scratch(MB(5));

        // Manager destructor will be called here
    }

    // If we get here without crash, cleanup worked
    // (Can't easily verify memory was freed without external tools)

    printf("  PASS: cleanup on destruction works\n");
    return true;
}

// =============================================================================
// Test 11: Out-of-memory handling (fail fast, not evict compute)
// =============================================================================
static bool test_compute_buffer_oom_handling() {
    printf("TEST: test_compute_buffer_oom_handling\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Get device memory to understand limits
    auto   device     = ctx.queue.get_device();
    size_t global_mem = device.get_info<sycl::info::device::global_mem_size>();

    printf("  [info] Device has %zu bytes (%.1f GB) global memory\n", global_mem, global_mem / 1e9);

    // Try to allocate more than device memory - should fail fast
    size_t huge_size      = global_mem * 2;  // 2x device memory
    void * should_be_null = mgr.allocate(huge_size, "huge_impossible");

    if (should_be_null != nullptr) {
        printf("  FAIL: allocation of %zu bytes should have failed\n", huge_size);
        mgr.release(should_be_null);
        return false;
    }

    // After OOM, normal allocation should still work
    void * normal = mgr.allocate(MB(1), "normal_after_oom");
    if (normal == nullptr) {
        printf("  FAIL: normal allocation after OOM should work\n");
        return false;
    }

    mgr.release(normal);

    printf("  PASS: OOM handling works correctly\n");
    return true;
}

// =============================================================================
// Test 12: Allocation statistics tracking
// =============================================================================
static bool test_compute_buffer_statistics() {
    printf("TEST: test_compute_buffer_statistics\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Initially all stats should be 0
    if (mgr.num_allocations() != 0) {
        printf("  FAIL: initial allocations should be 0\n");
        return false;
    }

    // Allocate some buffers
    void * ptr1 = mgr.allocate(MB(1), "stat_test_1");
    void * ptr2 = mgr.allocate(MB(2), "stat_test_2");

    if (mgr.num_allocations() != 2) {
        printf("  FAIL: expected 2 allocations, got %zu\n", mgr.num_allocations());
        return false;
    }

    // Both should be pool misses (new buffers)
    if (mgr.num_pool_misses() != 2) {
        printf("  FAIL: expected 2 pool misses, got %zu\n", mgr.num_pool_misses());
        return false;
    }

    mgr.release(ptr1);
    mgr.release(ptr2);

    // Allocate again - should be pool hits
    void * ptr3 = mgr.allocate(MB(1), "stat_test_3");
    void * ptr4 = mgr.allocate(MB(2), "stat_test_4");

    if (mgr.num_allocations() != 4) {
        printf("  FAIL: expected 4 total allocations, got %zu\n", mgr.num_allocations());
        return false;
    }

    if (mgr.num_pool_hits() != 2) {
        printf("  FAIL: expected 2 pool hits, got %zu\n", mgr.num_pool_hits());
        return false;
    }

    // Pool misses should still be 2
    if (mgr.num_pool_misses() != 2) {
        printf("  FAIL: pool misses should still be 2, got %zu\n", mgr.num_pool_misses());
        return false;
    }

    mgr.release(ptr3);
    mgr.release(ptr4);

    printf("  PASS: statistics tracking is accurate\n");
    return true;
}

// =============================================================================
// Test 13: Multiple managers are independent
// =============================================================================
static bool test_compute_buffer_manager_independence() {
    printf("TEST: test_compute_buffer_manager_independence\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr1(ctx.queue);
    ggml_sycl::ComputeBufferManager mgr2(ctx.queue);

    // Allocate from mgr1
    void * ptr1 = mgr1.allocate(MB(1), "mgr1_buf");

    // mgr2 should have separate pool
    if (mgr2.num_allocations() != 0) {
        printf("  FAIL: mgr2 should have 0 allocations initially\n");
        return false;
    }

    // Allocate from mgr2
    void * ptr2 = mgr2.allocate(MB(1), "mgr2_buf");

    // Pointers should be different
    if (ptr1 == ptr2) {
        printf("  FAIL: different managers should have different pools\n");
        return false;
    }

    // Stats should be independent
    if (mgr1.num_allocations() != 1 || mgr2.num_allocations() != 1) {
        printf("  FAIL: allocation counts should be independent\n");
        return false;
    }

    mgr1.release(ptr1);
    mgr2.release(ptr2);

    printf("  PASS: managers are independent\n");
    return true;
}

// =============================================================================
// Test 14: Buffer reuse respects minimum size (larger buffer can satisfy smaller request)
// =============================================================================
static bool test_compute_buffer_size_flexibility() {
    printf("TEST: test_compute_buffer_size_flexibility\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Allocate large buffer
    void * large_ptr = mgr.allocate(MB(4), "large");
    mgr.release(large_ptr);

    size_t misses_before = mgr.num_pool_misses();

    // Request smaller size - should reuse larger buffer
    void * small_ptr = mgr.allocate(MB(2), "small_from_large");

    // Should be same pointer (reused)
    if (small_ptr != large_ptr) {
        printf("  FAIL: smaller request should reuse larger buffer in pool\n");
        return false;
    }

    // Should be a pool hit, not a miss
    if (mgr.num_pool_misses() != misses_before) {
        printf("  FAIL: should not create new buffer for smaller request\n");
        return false;
    }

    mgr.release(small_ptr);

    printf("  PASS: larger buffers can satisfy smaller requests\n");
    return true;
}

// =============================================================================
// Test 15: Thread safety (concurrent allocations)
// =============================================================================
static bool test_compute_buffer_thread_safety() {
    printf("TEST: test_compute_buffer_thread_safety\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    constexpr int NUM_THREADS       = 4;
    constexpr int ALLOCS_PER_THREAD = 10;

    std::vector<std::thread> threads;
    std::atomic<int>         success_count{ 0 };
    std::atomic<int>         failure_count{ 0 };

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&mgr, &success_count, &failure_count, t]() {
            for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
                char name[64];
                snprintf(name, sizeof(name), "thread_%d_alloc_%d", t, i);

                void * ptr = mgr.allocate(KB(64 + t * 16), name);
                if (ptr != nullptr) {
                    success_count++;
                    // Simulate some work
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    mgr.release(ptr);
                } else {
                    failure_count++;
                }
            }
        });
    }

    for (auto & t : threads) {
        t.join();
    }

    printf("  [info] success=%d, failure=%d\n", success_count.load(), failure_count.load());

    // All allocations should have succeeded
    if (success_count != NUM_THREADS * ALLOCS_PER_THREAD) {
        printf("  FAIL: expected all allocations to succeed\n");
        return false;
    }

    // Total allocations should match
    if (mgr.num_allocations() != NUM_THREADS * ALLOCS_PER_THREAD) {
        printf("  FAIL: allocation count mismatch\n");
        return false;
    }

    printf("  PASS: thread safety verified\n");
    return true;
}

// =============================================================================
// Test 16: Scratch buffer is also P0 priority
// =============================================================================
static bool test_compute_scratch_p0_priority() {
    printf("TEST: test_compute_scratch_p0_priority\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("  SKIP: No SYCL device available\n");
        return true;
    }

    ggml_sycl::ComputeBufferManager mgr(ctx.queue);

    // Get scratch
    void * scratch = mgr.get_scratch(MB(16));

    if (scratch == nullptr) {
        printf("  FAIL: get_scratch() returned nullptr\n");
        return false;
    }

    // Scratch should also be P0 (never evicted)
    if (!mgr.is_scratch_pinned()) {
        printf("  FAIL: scratch buffer should be pinned\n");
        return false;
    }

    printf("  PASS: scratch buffer is P0 priority\n");
    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("=== Compute Buffer Manager Unit Tests ===\n");
    printf("Part of unified memory management (llama.cpp-v3n/llama.cpp-6s5)\n\n");

    auto & ctx = get_test_context();
    if (!ctx.valid) {
        printf("FATAL: No SYCL device available. Cannot run tests.\n");
        return 1;
    }

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_fn)(), const char * name) {
        bool result = test_fn();
        if (result) {
            passed++;
        } else {
            failed++;
            printf("  >>> TEST FAILED: %s\n\n", name);
        }
    };

    run_test(test_compute_buffer_allocation, "test_compute_buffer_allocation");
    run_test(test_compute_buffer_p0_priority, "test_compute_buffer_p0_priority");
    run_test(test_compute_buffer_never_evicted, "test_compute_buffer_never_evicted");
    run_test(test_compute_buffer_pool_reuse, "test_compute_buffer_pool_reuse");
    run_test(test_compute_buffer_pool_miss, "test_compute_buffer_pool_miss");
    run_test(test_compute_buffer_memory_accounting, "test_compute_buffer_memory_accounting");
    run_test(test_compute_scratch_resize, "test_compute_scratch_resize");
    run_test(test_compute_double_buffer_cycle, "test_compute_double_buffer_cycle");
    run_test(test_compute_buffer_reuse_inference_pass, "test_compute_buffer_reuse_inference_pass");
    run_test(test_compute_buffer_cleanup, "test_compute_buffer_cleanup");
    run_test(test_compute_buffer_oom_handling, "test_compute_buffer_oom_handling");
    run_test(test_compute_buffer_statistics, "test_compute_buffer_statistics");
    run_test(test_compute_buffer_manager_independence, "test_compute_buffer_manager_independence");
    run_test(test_compute_buffer_size_flexibility, "test_compute_buffer_size_flexibility");
    run_test(test_compute_buffer_thread_safety, "test_compute_buffer_thread_safety");
    run_test(test_compute_scratch_p0_priority, "test_compute_scratch_p0_priority");

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
