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

#include "ggml-sycl/common.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml-sycl/compute-buffers.hpp"
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
    ggml_sycl::unified_cache* cache = nullptr;
    sycl::queue* queue = nullptr;
    int device_id = 0;

    bool setup() {
        sycl_backend = ggml_backend_sycl_init(device_id);
        if (!sycl_backend) {
            fprintf(stderr, "Failed to init SYCL backend\n");
            return false;
        }
        cache = ggml_sycl::get_unified_cache_for_device(device_id);
        if (!cache) {
            fprintf(stderr, "Failed to get unified cache\n");
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
    ComputeBufferManager manager(*g_fixture.cache, 4);
    const size_t test_size = 1024 * 1024;
    ComputeBuffer* buffer = manager.acquire(test_size, "test_op");
    TEST_ASSERT(buffer != nullptr, "acquire() should return non-null buffer");
    TEST_ASSERT(buffer->data != nullptr, "buffer->data should be non-null");
    TEST_ASSERT(buffer->size >= test_size, "buffer->size should be >= requested size");
    TEST_ASSERT(buffer->in_use == true, "buffer->in_use should be true after acquire");
    TEST_ASSERT(strcmp(buffer->current_op, "test_op") == 0, "buffer->current_op should match");
    manager.release(buffer);
    TEST_ASSERT(buffer->in_use == false, "buffer->in_use should be false after release");
    return true;
}

static bool test_compute_buffer_never_evicted() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 4);
    const size_t large_size = 64 * 1024 * 1024;
    ComputeBuffer* buffer = manager.acquire(large_size, "critical_op");
    TEST_ASSERT(buffer != nullptr, "acquire() should succeed for large buffer");
    TEST_ASSERT(buffer->in_use == true, "buffer should be in use");
    size_t bytes_freed = g_fixture.cache->evict(large_size * 2);
    (void)bytes_freed;
    TEST_ASSERT(buffer->data != nullptr, "buffer should remain valid after eviction attempt");
    TEST_ASSERT(buffer->in_use == true, "buffer should still be in use after eviction");
    try {
        g_fixture.queue->memset(buffer->data, 0xAB, 1024).wait();
    } catch (...) {
        TEST_ASSERT(false, "memset should succeed");
    }
    manager.release(buffer);
    return true;
}

static bool test_release_compute_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 4);
    const size_t test_size = 512 * 1024;
    ComputeBuffer* buffer = manager.acquire(test_size, "release_test");
    TEST_ASSERT(buffer != nullptr, "acquire() should succeed");
    TEST_ASSERT(manager.bytes_in_use() >= test_size, "bytes_in_use should reflect allocation");
    void* original_ptr = buffer->data;
    manager.release(buffer);
    TEST_ASSERT(buffer->in_use == false, "buffer should not be in use after release");
    TEST_ASSERT(buffer->current_op == nullptr, "current_op should be cleared after release");
    TEST_ASSERT(manager.bytes_free() > 0, "bytes_free should be > 0 after release");
    TEST_ASSERT(buffer->data == original_ptr, "buffer data pointer should be preserved in pool");
    return true;
}

static bool test_reuse_compute_buffer() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 4);
    const size_t test_size = 2 * 1024 * 1024;
    ComputeBuffer* buffer1 = manager.acquire(test_size, "op1");
    TEST_ASSERT(buffer1 != nullptr, "first acquire should succeed");
    void* original_ptr = buffer1->data;
    size_t total_allocs_before = manager.total_allocations();
    manager.release(buffer1);
    ComputeBuffer* buffer2 = manager.acquire(test_size, "op2");
    TEST_ASSERT(buffer2 != nullptr, "second acquire should succeed");
    TEST_ASSERT(manager.pool_hits() > 0, "should have at least one pool hit");
    TEST_ASSERT(buffer2->data == original_ptr, "buffer should be reused from pool");
    TEST_ASSERT(manager.total_allocations() == total_allocs_before, "no new allocation when reusing");
    manager.release(buffer2);
    return true;
}

static bool test_compute_buffer_pool() {
    using namespace ggml_sycl;
    const size_t pool_size = 8;
    ComputeBufferManager manager(*g_fixture.cache, pool_size);
    TEST_ASSERT(manager.pool_size() == pool_size, "pool_size() should return configured size");
    std::vector<ComputeBuffer*> buffers;
    const size_t buf_size = 256 * 1024;
    for (size_t i = 0; i < pool_size; ++i) {
        std::string op_name = "pool_test_" + std::to_string(i);
        ComputeBuffer* buf = manager.acquire(buf_size, op_name.c_str());
        TEST_ASSERT(buf != nullptr, "acquire should succeed within pool capacity");
        buffers.push_back(buf);
    }
    TEST_ASSERT(manager.bytes_free() == 0, "no free bytes when all buffers in use");
    ComputeBuffer* extra = manager.acquire(buf_size, "extra_op");
    if (extra) {
        buffers.push_back(extra);
    }
    for (auto* buf : buffers) {
        manager.release(buf);
    }
    TEST_ASSERT(manager.bytes_in_use() == 0, "bytes_in_use should be 0 after releasing all");
    TEST_ASSERT(manager.peak_usage() > 0, "peak_usage should be tracked");
    return true;
}

static bool test_pool_resize() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 2);
    TEST_ASSERT(manager.pool_size() == 2, "initial pool size should be 2");
    manager.resize_pool(8);
    TEST_ASSERT(manager.pool_size() == 8, "pool size should be 8 after resize");
    std::vector<ComputeBuffer*> buffers;
    for (size_t i = 0; i < 6; ++i) {
        ComputeBuffer* buf = manager.acquire(128 * 1024, "resize_test");
        if (buf) {
            buffers.push_back(buf);
        }
    }
    TEST_ASSERT(buffers.size() >= 2, "should be able to acquire at least 2 buffers");
    for (auto* buf : buffers) {
        manager.release(buf);
    }
    return true;
}

static bool test_concurrent_access() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 16);
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};
    auto worker = [&](int thread_id) {
        for (int i = 0; i < 10; ++i) {
            std::string op = "t" + std::to_string(thread_id) + "_" + std::to_string(i);
            ComputeBuffer* buf = manager.acquire(64 * 1024, op.c_str());
            if (buf) {
                success_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                manager.release(buf);
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
    TEST_ASSERT(success_count > 0, "at least some acquires should succeed");
    TEST_ASSERT(manager.bytes_in_use() == 0, "all buffers should be released at end");
    printf("    Concurrent: %d successes, %d failures\n", success_count.load(), failure_count.load());
    return true;
}

static bool test_stats_tracking() {
    using namespace ggml_sycl;
    ComputeBufferManager manager(*g_fixture.cache, 4);
    TEST_ASSERT(manager.total_allocations() == 0, "initial total_allocations should be 0");
    TEST_ASSERT(manager.pool_hits() == 0, "initial pool_hits should be 0");
    TEST_ASSERT(manager.pool_misses() == 0, "initial pool_misses should be 0");
    TEST_ASSERT(manager.peak_usage() == 0, "initial peak_usage should be 0");
    ComputeBuffer* buf1 = manager.acquire(1024 * 1024, "stats_op1");
    TEST_ASSERT(buf1 != nullptr, "acquire should succeed");
    TEST_ASSERT(manager.total_allocations() == 1, "should have 1 allocation");
    TEST_ASSERT(manager.pool_misses() >= 1, "should have at least 1 pool miss");
    size_t peak1 = manager.peak_usage();
    TEST_ASSERT(peak1 > 0, "peak_usage should be > 0");
    manager.release(buf1);
    ComputeBuffer* buf2 = manager.acquire(1024 * 1024, "stats_op2");
    TEST_ASSERT(buf2 != nullptr, "second acquire should succeed");
    TEST_ASSERT(manager.pool_hits() >= 1, "should have at least 1 pool hit");
    manager.release(buf2);
    TEST_ASSERT(manager.peak_usage() >= peak1, "peak should not decrease");
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
    RUN_TEST(test_pool_resize);
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
