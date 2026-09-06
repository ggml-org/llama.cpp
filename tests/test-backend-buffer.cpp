#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"

#include <array>
#include <atomic>
#include <thread>
#include <vector>

struct test_buffer_context {
    std::array<uint8_t, 64> data = {};
    std::atomic<int> free_count { 0 };
};

// Return a stable name for the test buffer type.
static const char * test_buft_name(ggml_backend_buffer_type_t) {
    return "test";
}

// Record the single backend destruction callback.
static void test_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<test_buffer_context *>(buffer->context);
    context->free_count.fetch_add(1, std::memory_order_relaxed);
}

// Return the fixed storage used by the test buffer.
static void * test_buffer_base(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<test_buffer_context *>(buffer->context);
    return context->data.data();
}

// Create a buffer object over the test context storage.
static ggml_backend_buffer_t test_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * context = static_cast<test_buffer_context *>(buft->context);
    GGML_ASSERT(size <= context->data.size());

    ggml_backend_buffer_i iface = {};
    iface.free_buffer = test_buffer_free;
    iface.get_base = test_buffer_base;
    return ggml_backend_buffer_init(buft, iface, context, size);
}

// Use byte alignment so reference-count tests do not depend on placement.
static size_t test_buft_alignment(ggml_backend_buffer_type_t) {
    return 1;
}

// Mark the test buffer as host-addressable.
static bool test_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

// Build the minimal buffer type needed for lifetime tests.
static ggml_backend_buffer_type make_test_buft(test_buffer_context * context) {
    ggml_backend_buffer_type result = {};
    result.iface.get_name = test_buft_name;
    result.iface.alloc_buffer = test_buft_alloc;
    result.iface.get_alignment = test_buft_alignment;
    result.iface.is_host = test_buft_is_host;
    result.context = context;
    return result;
}

// Verify that only the last release destroys the buffer.
static void test_retain_delays_free() {
    test_buffer_context context;
    auto buft = make_test_buft(&context);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(&buft, context.data.size());
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_retain(nullptr) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_retain(buffer) == buffer);
    GGML_ASSERT(ggml_backend_buffer_retain(buffer) == buffer);

    ggml_backend_buffer_free(buffer);
    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 0);
    ggml_backend_buffer_free(buffer);
    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 0);
    ggml_backend_buffer_free(buffer);
    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 1);
}

// Verify concurrent retain/release pairs while one reference remains live.
static void test_retain_concurrent() {
    test_buffer_context context;
    auto buft = make_test_buft(&context);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(&buft, context.data.size());
    GGML_ASSERT(buffer != nullptr);

    std::vector<std::thread> threads;
    for (int thread = 0; thread < 8; ++thread) {
        threads.emplace_back([buffer] {
            for (int i = 0; i < 10000; ++i) {
                GGML_ASSERT(ggml_backend_buffer_retain(buffer) == buffer);
                ggml_backend_buffer_free(buffer);
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }

    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 0);
    ggml_backend_buffer_free(buffer);
    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 1);
}

int main() {
    test_retain_delays_free();
    test_retain_concurrent();
    return 0;
}
