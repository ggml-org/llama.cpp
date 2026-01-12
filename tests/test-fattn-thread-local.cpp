// Unit test for flash-attn thread-local device buffer cleanup
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-fattn-thread-local

#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sycl/sycl.hpp>
#include <thread>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

extern "C" {
int  ggml_sycl_test_seq_id_buffer_instances();
int  ggml_sycl_test_seq_id_buffer_allocs();
void ggml_sycl_test_fattn_set_shutdown(int value);
void ggml_sycl_test_seq_id_buffers_free_all();
bool ggml_sycl_test_seq_id_buffers_touch(sycl::queue * stream);
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    const int instances_before = ggml_sycl_test_seq_id_buffer_instances();
    const int allocs_before    = ggml_sycl_test_seq_id_buffer_allocs();

    auto        queue = std::make_unique<sycl::queue>(sycl::default_selector_v);
    std::thread worker([&queue]() {
        if (!ggml_sycl_test_seq_id_buffers_touch(queue.get())) {
            fprintf(stderr, "Failed to touch seq-id buffers in worker thread\n");
        }
    });
    worker.join();

    const int instances_after = ggml_sycl_test_seq_id_buffer_instances();
    const int allocs_after    = ggml_sycl_test_seq_id_buffer_allocs();

    if (instances_after != instances_before) {
        fprintf(stderr, "Thread-local buffer instances leaked (before=%d after=%d)\n", instances_before,
                instances_after);
        return 1;
    }

    if (allocs_after != allocs_before) {
        fprintf(stderr, "Thread-local device allocations leaked (before=%d after=%d)\n", allocs_before, allocs_after);
        return 1;
    }

    // Validate shutdown guard: free_all should no-op when shutdown flag is set.
    ggml_sycl_test_fattn_set_shutdown(0);
    if (!ggml_sycl_test_seq_id_buffers_touch(&q)) {
        fprintf(stderr, "Failed to touch seq-id buffers for shutdown test\n");
        return 1;
    }

    const int allocs_after_touch = ggml_sycl_test_seq_id_buffer_allocs();
    ggml_sycl_test_fattn_set_shutdown(1);
    ggml_sycl_test_seq_id_buffers_free_all();

    const int allocs_after_noop = ggml_sycl_test_seq_id_buffer_allocs();
    if (allocs_after_noop != allocs_after_touch) {
        fprintf(stderr, "Shutdown guard freed buffers unexpectedly (before=%d after=%d)\n", allocs_after_touch,
                allocs_after_noop);
        return 1;
    }

    ggml_sycl_test_fattn_set_shutdown(0);
    ggml_sycl_test_seq_id_buffers_free_all();

    const int allocs_final = ggml_sycl_test_seq_id_buffer_allocs();
    if (allocs_final != allocs_before) {
        fprintf(stderr, "Seq-id buffers not freed after shutdown reset (before=%d after=%d)\n", allocs_before,
                allocs_final);
        return 1;
    }

    printf("Thread-local buffers cleaned up successfully\n");
    return 0;
}

#endif
