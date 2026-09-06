#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <array>
#include <atomic>
#include <cstring>
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

// Verify CPU views isolate address bounds and usage while retaining their parent.
static void test_cpu_buffer_view() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t parent_size = 5*alignment;
    ggml_backend_buffer_t parent = ggml_backend_buft_alloc_buffer(buft, parent_size);
    GGML_ASSERT(parent != nullptr && !ggml_backend_buffer_is_view(parent));
    ggml_backend_buffer_clear(parent, 0xa5);
    auto * parent_data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(parent));

    ggml_backend_buffer_t view = ggml_backend_buffer_view(parent, alignment, 2*alignment);
    GGML_ASSERT(view != nullptr && ggml_backend_buffer_is_view(view));
    GGML_ASSERT(ggml_backend_buffer_get_type(view) == buft);
    GGML_ASSERT(ggml_backend_buffer_get_size(view) == 2*alignment);
    GGML_ASSERT(ggml_backend_buffer_get_base(view) == parent_data + alignment);
    ggml_backend_buffer_set_usage(view, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(ggml_backend_buffer_get_usage(view) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(ggml_backend_buffer_get_usage(parent) == GGML_BACKEND_BUFFER_USAGE_ANY);

    ggml_init_params params = {
        /*.mem_size   = */ 2*ggml_tensor_overhead(),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tallocr talloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&talloc, view, 0, ggml_backend_buffer_get_size(view)));
    GGML_ASSERT(ggml_tallocr_alloc(&talloc, tensor) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(tensor->buffer == view);

    const float input[] = {1, 2, 3, 4};
    float output[4] = {};
    ggml_backend_tensor_set(tensor, input, 0, sizeof(input));
    ggml_backend_tensor_get(tensor, output, 0, sizeof(output));
    GGML_ASSERT(memcmp(input, output, sizeof(input)) == 0);

    ggml_backend_buffer_free(parent);
    ggml_backend_buffer_clear(view, 0x3c);
    for (size_t i = 0; i < alignment; ++i) {
        GGML_ASSERT(parent_data[i] == 0xa5);
    }
    for (size_t i = alignment; i < 3*alignment; ++i) {
        GGML_ASSERT(parent_data[i] == 0x3c);
    }
    for (size_t i = 3*alignment; i < parent_size; ++i) {
        GGML_ASSERT(parent_data[i] == 0xa5);
    }
    ggml_backend_buffer_free(view);
}

// Verify nested views retain their immediate parents and compose offsets.
static void test_cpu_buffer_nested_view() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    ggml_backend_buffer_t parent = ggml_backend_buft_alloc_buffer(buft, 5*alignment);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(parent));
    ggml_backend_buffer_t middle = ggml_backend_buffer_view(parent, alignment, 3*alignment);
    ggml_backend_buffer_t leaf = ggml_backend_buffer_view(middle, alignment, alignment);
    GGML_ASSERT(middle != nullptr && leaf != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_base(leaf) == base + 2*alignment);

    ggml_backend_buffer_free(parent);
    ggml_backend_buffer_free(middle);
    ggml_backend_buffer_clear(leaf, 0x6d);
    for (size_t i = 2*alignment; i < 3*alignment; ++i) {
        GGML_ASSERT(base[i] == 0x6d);
    }
    ggml_backend_buffer_free(leaf);
}

// Verify invalid or unsupported views fail without changing parent ownership.
static void test_buffer_view_validation() {
    test_buffer_context context;
    auto test_buft = make_test_buft(&context);
    ggml_backend_buffer_t unsupported = ggml_backend_buft_alloc_buffer(&test_buft, context.data.size());
    GGML_ASSERT(ggml_backend_buffer_view(unsupported, 0, 16) == nullptr);
    ggml_backend_buffer_free(unsupported);
    GGML_ASSERT(context.free_count.load(std::memory_order_relaxed) == 1);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(buft, 4*alignment));
    GGML_ASSERT(ggml_backend_buffer_view(nullptr, 0, alignment) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_view(parent.get(), 0, 0) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_view(parent.get(), 4*alignment, alignment) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_view(parent.get(), 3*alignment, 2*alignment) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_view(parent.get(), 1, alignment) == nullptr);
}
int main() {
    test_retain_delays_free();
    test_retain_concurrent();
    test_cpu_buffer_view();
    test_cpu_buffer_nested_view();
    test_buffer_view_validation();
    return 0;
}
