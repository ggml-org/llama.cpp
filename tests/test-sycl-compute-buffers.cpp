// Unit tests for SYCL compute buffer management.
// Tests the ComputeBufferManager class that manages P0 (CRITICAL priority) compute buffers.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#include "ggml-sycl/compute-buffer-manager.hpp"
#include <sycl/sycl.hpp>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        return false; \
    } \
} while (0)

#define RUN_TEST(test_fn) do { \
    printf("Running %s...\n", #test_fn); \
    if (test_fn()) { \
        printf("  PASS\n"); \
        g_tests_passed++; \
    } else { \
        printf("  FAIL\n"); \
        g_tests_failed++; \
    } \
} while (0)

struct TestFixture {
    ggml_backend_t sycl_backend = nullptr;
    sycl::queue* queue = nullptr;
    int device_id = 0;

    bool setup() {
        sycl_backend = ggml_backend_sycl_init(device_id);
        if (!sycl_backend) {
            fprintf(stderr, "Failed to init SYCL backend\n");
            return false;
        }
        try {
            queue = new sycl::queue();
        } catch (const sycl::exception& e) {
            fprintf(stderr, "Failed: %s\n", e.what());
            return false;
        }
        return true;
    }

    void teardown() {
        delete queue;
        queue = nullptr;
        if (sycl_backend) {
            ggml_backend_free(sycl_backend);
            sycl_backend = nullptr;
        }
    }
};

static TestFixture g_fixture;

static bool test_allocate_compute_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    const size_t test_size = 1024 * 1024;
    void* ptr = manager.allocate(test_size, "test_op");
    TEST_ASSERT(ptr != nullptr, "allocate() should return non-null pointer");
    TEST_ASSERT(manager.is_valid(ptr), "allocated buffer should be valid");
    TEST_ASSERT(manager.is_pinned(ptr), "allocated buffer should be pinned");
    TEST_ASSERT(manager.get_priority(ptr) == EvictionPriority::P0_COMPUTE, "buffer should have P0 priority");
    manager.release(ptr);
    return true;
}

static bool test_compute_buffer_never_evicted() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    const size_t large_size = 64 * 1024 * 1024;
    void* ptr = manager.allocate(large_size, "critical_op");
    TEST_ASSERT(ptr != nullptr, "allocate() should succeed for large buffer");
    bool evicted = manager.try_evict_for_space(large_size * 2);
    TEST_ASSERT(!evicted, "P0 buffers should not be evictable");
    TEST_ASSERT(manager.is_valid(ptr), "buffer should remain valid after eviction attempt");
    try {
        g_fixture.queue->memset(ptr, 0xAB, 1024).wait();
    } catch (...) {
        TEST_ASSERT(false, "memset should succeed");
    }
    manager.release(ptr);
    return true;
}

static bool test_release_compute_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    const size_t test_size = 512 * 1024;
    void* ptr = manager.allocate(test_size, "release_test");
    TEST_ASSERT(ptr != nullptr, "allocate() should succeed");
    size_t used_before = manager.pool_used_size();
    TEST_ASSERT(used_before >= test_size, "pool_used_size should reflect allocation");
    manager.release(ptr);
    size_t used_after = manager.pool_used_size();
    TEST_ASSERT(used_after < used_before, "pool_used_size should decrease after release");
    return true;
}

static bool test_reuse_compute_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    const size_t test_size = 2 * 1024 * 1024;
    void* ptr1 = manager.allocate(test_size, "op1");
    TEST_ASSERT(ptr1 != nullptr, "first allocate should succeed");
    manager.release(ptr1);
    void* ptr2 = manager.allocate(test_size, "op2");
    TEST_ASSERT(ptr2 != nullptr, "second allocate should succeed");
    TEST_ASSERT(manager.num_pool_hits() > 0, "should have at least one pool hit");
    TEST_ASSERT(ptr2 == ptr1, "buffer should be reused from pool");
    manager.release(ptr2);
    return true;
}

static bool test_compute_buffer_pool() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    std::vector<void*> buffers;
    const size_t buf_size = 256 * 1024;
    const size_t num_bufs = 8;
    for (size_t i = 0; i < num_bufs; ++i) {
        std::string op_name = "pool_test_" + std::to_string(i);
        void* ptr = manager.allocate(buf_size, op_name.c_str());
        TEST_ASSERT(ptr != nullptr, "allocate should succeed");
        buffers.push_back(ptr);
    }
    size_t total_size = manager.pool_total_size();
    TEST_ASSERT(total_size >= buf_size * num_bufs, "pool_total_size should reflect all allocations");
    for (auto* ptr : buffers) {
        manager.release(ptr);
    }
    TEST_ASSERT(manager.pool_used_size() == 0, "pool_used_size should be 0 after releasing all");
    return true;
}

static bool test_scratch_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    const size_t scratch_size = 4 * 1024 * 1024;
    void* scratch = manager.get_scratch(scratch_size);
    TEST_ASSERT(scratch != nullptr, "get_scratch should succeed");
    // Note: scratch_size is set to aligned capacity when growing
    TEST_ASSERT(manager.get_scratch_size() >= scratch_size, "scratch size should be >= requested");
    TEST_ASSERT(manager.get_scratch_capacity() >= scratch_size, "scratch capacity should be >= size");
    TEST_ASSERT(manager.is_scratch_pinned(), "scratch should be pinned");
    const size_t larger_size = 8 * 1024 * 1024;
    void* larger_scratch = manager.get_scratch(larger_size);
    TEST_ASSERT(larger_scratch != nullptr, "larger get_scratch should succeed");
    TEST_ASSERT(manager.get_scratch_size() >= larger_size, "scratch size should be >= larger request");
    TEST_ASSERT(manager.get_scratch_capacity() >= larger_size, "scratch capacity should grow");
    return true;
}

static bool test_concurrent_access() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    auto worker = [&](int thread_id) {
        for (int i = 0; i < 10; ++i) {
            std::string op = "t" + std::to_string(thread_id) + "_" + std::to_string(i);
            void* ptr = manager.allocate(64 * 1024, op.c_str());
            if (ptr) {
                success_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                manager.release(ptr);
            } else {
                failure_count++;
            }
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) {
        t.join();
    }
    TEST_ASSERT(success_count > 0, "at least some allocations should succeed");
    TEST_ASSERT(manager.pool_used_size() == 0, "all buffers should be released at end");
    printf("    Concurrent: %d successes, %d failures\n", success_count.load(), failure_count.load());
    return true;
}

static bool test_stats_tracking() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.queue);
    TEST_ASSERT(manager.num_allocations() == 0, "initial num_allocations should be 0");
    TEST_ASSERT(manager.num_pool_hits() == 0, "initial num_pool_hits should be 0");
    TEST_ASSERT(manager.num_pool_misses() == 0, "initial num_pool_misses should be 0");
    void* ptr1 = manager.allocate(1024 * 1024, "stats_op1");
    TEST_ASSERT(ptr1 != nullptr, "allocate should succeed");
    TEST_ASSERT(manager.num_allocations() == 1, "should have 1 allocation");
    TEST_ASSERT(manager.num_pool_misses() >= 1, "should have at least 1 pool miss");
    manager.release(ptr1);
    void* ptr2 = manager.allocate(1024 * 1024, "stats_op2");
    TEST_ASSERT(ptr2 != nullptr, "second allocate should succeed");
    TEST_ASSERT(manager.num_pool_hits() >= 1, "should have at least 1 pool hit");
    manager.release(ptr2);
    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }
    printf("SYCL Compute Buffer Management Tests\n");
    printf("=====================================\n\n");
    try {
        sycl::queue q;
        printf("Using device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }
    if (!g_fixture.setup()) {
        fprintf(stderr, "Failed to setup test fixture\n");
        return 1;
    }
    RUN_TEST(test_allocate_compute_buffer);
    RUN_TEST(test_compute_buffer_never_evicted);
    RUN_TEST(test_release_compute_buffer);
    RUN_TEST(test_reuse_compute_buffer);
    RUN_TEST(test_compute_buffer_pool);
    RUN_TEST(test_scratch_buffer);
    RUN_TEST(test_concurrent_access);
    RUN_TEST(test_stats_tracking);
    g_fixture.teardown();
    printf("\n=====================================\n");
    printf("Tests passed: %d\n", g_tests_passed);
    printf("Tests failed: %d\n", g_tests_failed);
    printf("Result: %s\n", g_tests_failed == 0 ? "PASS" : "FAIL");
    return g_tests_failed == 0 ? 0 : 1;
}

#endif
