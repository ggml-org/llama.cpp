#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;

    ggml_backend_buffer_i              buffer_interface;
    ggml_backend_device                device;
    ggml_backend                       backend;
    std::vector<ggml_backend_buffer_t> buffers;
    const ggml_tensor * bound_tensor = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    bool real_data = false;
    bool relative_addresses = false;
    std::map<ggml_backend_buffer_t, std::vector<uint8_t>> data;
    size_t alloc_calls = 0;
    size_t free_calls = 0;
    size_t init_calls = 0;
    size_t shard_size_queries = 0;
    bool expand_quantized_add = false;
    size_t fail_alloc = SIZE_MAX;
    size_t fail_init = SIZE_MAX;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    if (++ctx->alloc_calls == ctx->fail_alloc) {
        return nullptr;
    }
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    if (ctx->real_data) {
        ctx->data[buffer].resize(size);
    }
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return !static_cast<dummy_backend_context *>(buft->context)->relative_addresses;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
    ctx->data.erase(buffer);
    ctx->free_calls++;
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<dummy_backend_context *>(buffer->context);
    return ctx->real_data && !ctx->relative_addresses ? ctx->data.at(buffer).data() : alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *) {
    auto * ctx = static_cast<dummy_backend_context *>(buffer->context);
    return ++ctx->init_calls == ctx->fail_init ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;
}

static uint8_t * dummy_backend_tensor_data(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size) {
    auto * ctx = static_cast<dummy_backend_context *>(buffer->context);
    auto & bytes = ctx->data.at(buffer);
    size_t start = uintptr_t(tensor->data) - uintptr_t(dummy_backend_buffer_get_base(buffer)) + offset;
    GGML_ASSERT(start <= bytes.size() && size <= bytes.size() - start);
    return bytes.data() + start;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (static_cast<dummy_backend_context *>(buffer->context)->real_data) {
        std::memset(dummy_backend_tensor_data(buffer, tensor, offset, size), value, size);
    }
}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (static_cast<dummy_backend_context *>(buffer->context)->real_data) {
        std::memcpy(dummy_backend_tensor_data(buffer, tensor, offset, size), data, size);
    }
}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (static_cast<dummy_backend_context *>(buffer->context)->real_data) {
        std::memcpy(data, dummy_backend_tensor_data(buffer, tensor, offset, size), size);
    }
}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<dummy_backend_context *>(buffer->context);
    if (ctx->real_data) {
        auto & bytes = ctx->data.at(buffer);
        std::fill(bytes.begin(), bytes.end(), value);
    }
}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t) {
    return "dummy_backend";
}

// dummy_backend

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.context->device.context             = b.context.get();
    b.context->device.iface.get_type      = dummy_backend_device_get_type;
    b.context->device.iface.supports_op   = dummy_backend_device_supports_op;
    b.context->device.iface.supports_buft = dummy_backend_device_supports_buft;

    b.context->backend.context        = b.context.get();
    b.context->backend.device         = &b.context->device;
    b.context->backend.iface.get_name = dummy_backend_get_name;

    b.buffer_type.device              = &b.context->device;
    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static void test_backend_graph_optimize(ggml_backend_t, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.context->backend.iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = &backend.context->backend;
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

static bool test_buffer_is_bound(ggml_backend_buffer_t buffer, const ggml_tensor * tensor) {
    auto * ctx = static_cast<dummy_backend_context *>(buffer->context);
    return ctx->bound_tensor == tensor;
}

static void test_tensor_binding_state() {
    auto test_ctx = make_context();
    auto backend = dummy_backend_init(64);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 64));
    ggml_tensor * tensor = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, 1);

    GGML_ASSERT(!ggml_backend_tensor_is_bound(tensor));
    float external = 1.0f;
    tensor->data = &external;
    GGML_ASSERT(ggml_backend_tensor_is_bound(tensor));
    tensor->data = nullptr;
    tensor->buffer = buffer.get();
    GGML_ASSERT(!ggml_backend_tensor_is_bound(tensor));

    static const ggml_backend_buffer_binding_i binding = {test_buffer_is_bound, nullptr};
    buffer->binding = &binding;
    GGML_ASSERT(!ggml_backend_tensor_is_bound(tensor));
    backend.context->bound_tensor = tensor;
    GGML_ASSERT(ggml_backend_tensor_is_bound(tensor));
    GGML_ASSERT(tensor->data == nullptr);
    backend.context->bound_tensor = nullptr;
    tensor->data = &external;
    GGML_ASSERT(!ggml_backend_tensor_is_bound(tensor));

    ggml_tensor * empty = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, 0);
    ggml_backend_buffer_ptr empty_buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 0));
    GGML_ASSERT(!ggml_backend_tensor_is_bound(empty));
    empty->buffer = empty_buffer.get();
    GGML_ASSERT(ggml_backend_tensor_is_bound(empty));
    tensor->data = nullptr;
    tensor->buffer = empty_buffer.get();
    GGML_ASSERT(!ggml_backend_tensor_is_bound(tensor));
}

struct test_binding_context {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(64);
    std::vector<uint8_t> * borrowed_bytes = nullptr;
    std::map<const ggml_tensor *, size_t> offsets;
    int synchronizations = 0;
    int native_inits = 0;
    bool fail_view = false;
};

static bool test_addressless_is_bound(ggml_backend_buffer_t buffer, const ggml_tensor * tensor) {
    auto * ctx = static_cast<test_binding_context *>(buffer->context);
    return ctx->offsets.count(tensor) != 0;
}

static bool test_addressless_is_host(ggml_backend_buffer_type_t) {
    return false;
}

static ggml_backend_buffer_type_t test_device_buffer_type(ggml_backend_dev_t device) {
    return static_cast<dummy_backend_context *>(device->context)->buffer_type;
}

static ggml_status test_addressless_init_view(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = static_cast<test_binding_context *>(buffer->context);
    if (ctx->fail_view) {
        return GGML_STATUS_FAILED;
    }
    GGML_ASSERT(tensor->view_offs <= ggml_nbytes(tensor->view_src));
    GGML_ASSERT(ggml_nbytes(tensor) <= ggml_nbytes(tensor->view_src) - tensor->view_offs);
    ctx->offsets[tensor] = ctx->offsets.at(tensor->view_src) + tensor->view_offs;
    tensor->buffer = buffer;
    tensor->data = nullptr;
    return GGML_STATUS_SUCCESS;
}

static ggml_status test_addressless_native_init(ggml_backend_buffer_t buffer, ggml_tensor *) {
    auto * ctx = static_cast<test_binding_context *>(buffer->context);
    ctx->native_inits++;
    return GGML_STATUS_FAILED;
}

static uint8_t * test_addressless_data(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size) {
    auto * ctx = static_cast<test_binding_context *>(buffer->context);
    GGML_ASSERT(tensor->data == nullptr);
    size_t start = ctx->offsets.at(tensor) + offset;
    auto & bytes = ctx->borrowed_bytes ? *ctx->borrowed_bytes : ctx->bytes;
    GGML_ASSERT(start <= bytes.size() && size <= bytes.size() - start);
    return bytes.data() + start;
}

static void test_addressless_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(test_addressless_data(buffer, tensor, offset, size), data, size);
}

static void test_addressless_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, test_addressless_data(buffer, tensor, offset, size), size);
}

static void test_addressless_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(test_addressless_data(buffer, tensor, offset, size), value, size);
}

static void test_addressless_synchronize(ggml_backend_t backend) {
    auto * ctx = static_cast<test_binding_context *>(backend->context);
    ctx->synchronizations++;
}

static void test_addressless_views_and_transfers() {
    auto test_ctx = make_context();
    auto device = dummy_backend_init(64);
    device.buffer_type.iface.is_host = test_addressless_is_host;
    test_binding_context ctx;
    ggml_backend_buffer_i iface{};
    iface.init_tensor = test_addressless_native_init;
    iface.set_tensor = test_addressless_set;
    iface.get_tensor = test_addressless_get;
    iface.memset_tensor = test_addressless_memset;
    ggml_backend_buffer_ptr buffer(ggml_backend_buffer_init(&device.buffer_type, iface, &ctx, ctx.bytes.size()));
    static const ggml_backend_buffer_binding_i binding = {test_addressless_is_bound, test_addressless_init_view};
    buffer->binding = &binding;
    GGML_ASSERT(ggml_backend_buffer_get_base(buffer.get()) == nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, 16);
    tensor->buffer = buffer.get();
    ctx.offsets[tensor] = 0;
    std::array<float, 16> values;
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = float(i + 1);
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, sizeof(values));
    std::array<float, 16> read{};
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read == values);
    ggml_backend_tensor_memset(tensor, 0, sizeof(float), 2*sizeof(float));
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read[0] == 1.0f && read[1] == 0.0f && read[2] == 0.0f && read[3] == 4.0f);

    std::array<float, 6> rows = {21, 22, 31, 32, 41, 42};
    ggml_backend_tensor_set_2d(tensor, rows.data(), 2*sizeof(float), 2*sizeof(float), 3, 4*sizeof(float), 2*sizeof(float));
    std::array<float, 6> read_rows{};
    ggml_backend_tensor_get_2d(tensor, read_rows.data(), 2*sizeof(float), 2*sizeof(float), 3, 4*sizeof(float), 2*sizeof(float));
    GGML_ASSERT(read_rows == rows);

    ggml_backend backend{};
    backend.context = &ctx;
    backend.iface.synchronize = test_addressless_synchronize;
    ggml_backend_tensor_set_async(&backend, tensor, values.data(), 0, sizeof(values));
    ggml_backend_tensor_get_async(&backend, tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read == values && ctx.synchronizations == 2);
    ggml_backend_tensor_set_2d_async(&backend, tensor, rows.data(), 0, 2*sizeof(float), 3, 4*sizeof(float), 2*sizeof(float));
    ggml_backend_tensor_get_2d_async(&backend, tensor, read_rows.data(), 0, 2*sizeof(float), 3, 4*sizeof(float), 2*sizeof(float));
    GGML_ASSERT(read_rows == rows && ctx.synchronizations == 8);

    ggml_tensor * view = ggml_view_1d(test_ctx.ctx, tensor, 4, 4*sizeof(float));
    GGML_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(view->buffer == buffer.get() && view->data == nullptr);
    ggml_tensor * nested = ggml_view_1d(test_ctx.ctx, view, 2, sizeof(float));
    GGML_ASSERT(ggml_backend_view_init(nested) == GGML_STATUS_SUCCESS);
    std::array<float, 2> replacement = {101, 102};
    ggml_backend_tensor_set(nested, replacement.data(), 0, sizeof(replacement));
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read[5] == 101 && read[6] == 102);
    GGML_ASSERT(nested->buffer == buffer.get() && nested->data == nullptr);

    ggml_tensor * rejected = ggml_view_1d(test_ctx.ctx, tensor, 1, 0);
    ctx.fail_view = true;
    GGML_ASSERT(ggml_backend_view_init(rejected) == GGML_STATUS_FAILED);
    GGML_ASSERT(!ggml_backend_tensor_is_bound(rejected) && rejected->buffer == nullptr);
    ctx.fail_view = false;
    GGML_ASSERT(ggml_backend_view_init(rejected) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ctx.native_inits == 0);

    auto native = dummy_backend_init(64);
    native.context->real_data = true;
    native.context->buffer_type = &native.buffer_type;
    native.context->device.iface.get_buffer_type = test_device_buffer_type;
    native.context->device.iface.get_name = [](ggml_backend_dev_t) { return "dummy_host"; };
    native.context->device.iface.get_description = [](ggml_backend_dev_t) { return "test host storage"; };
    auto native_ctx = make_context();
    ggml_tensor * native_tensor = ggml_new_tensor_1d(native_ctx.ctx, GGML_TYPE_F32, 16);
    ggml_backend_buffer_ptr native_buffer(ggml_backend_alloc_ctx_tensors_from_buft(native_ctx.ctx, &native.buffer_type));
    GGML_ASSERT(native_buffer != nullptr);
    ggml_backend_tensor_copy(tensor, native_tensor);
    GGML_ASSERT(std::memcmp(native_tensor->data, read.data(), sizeof(read)) == 0);
    std::memcpy(native_tensor->data, values.data(), sizeof(values));
    ggml_backend_tensor_copy(native_tensor, tensor);
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read == values);

    backend.iface.cpy_tensor_async = [](ggml_backend_t, ggml_backend_t, const ggml_tensor *, ggml_tensor *) -> bool {
        GGML_ABORT("native async copy received an addressless tensor");
    };
    ggml_backend_tensor_copy_async(&backend, &backend, tensor, native_tensor);
    GGML_ASSERT(ctx.synchronizations == 10);
    ggml_backend_tensor_copy_async(&backend, &backend, native_tensor, tensor);
    GGML_ASSERT(ctx.synchronizations == 12);

    buffer->iface.cpy_tensor = [](ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) -> bool {
        GGML_ABORT("native buffer copy received an addressless tensor");
    };
    GGML_ASSERT(!ggml_backend_buffer_copy_tensor(native_tensor, tensor));
    GGML_ASSERT(!ggml_backend_buffer_copy_tensor(tensor, native_tensor));

    ggml_build_forward_expand(test_ctx.graph, nested);
    auto copy = ggml_backend_graph_copy(&native.context->backend, test_ctx.graph);
    GGML_ASSERT(copy.buffer != nullptr);
    ggml_tensor * copied_view = ggml_graph_node(copy.graph, ggml_graph_n_nodes(copy.graph) - 1);
    GGML_ASSERT(copied_view->data != nullptr);
    GGML_ASSERT(std::memcmp(copied_view->data, values.data() + 5, sizeof(replacement)) == 0);
    ggml_backend_graph_copy_free(copy);

    test_binding_context view_ctx;
    view_ctx.borrowed_bytes = &ctx.bytes;
    ggml_backend_buffer_ptr view_owner(ggml_backend_buffer_init(&device.buffer_type, iface, &view_ctx, 0));
    view_owner->binding = &binding;
    auto * owned_view = ggml_view_1d(test_ctx.ctx, tensor, 2, 2*sizeof(float));
    owned_view->buffer = view_owner.get();
    view_ctx.offsets[owned_view] = 2*sizeof(float);
    GGML_ASSERT(owned_view->buffer != owned_view->view_src->buffer);
    GGML_ASSERT(ggml_backend_tensor_is_bound(owned_view));
    std::array<float, 2> owned_values = {201, 202};
    ggml_backend_tensor_set(owned_view, owned_values.data(), 0, sizeof(owned_values));
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read[2] == 201 && read[3] == 202 && read[1] == 2 && read[4] == 5);
    auto * native_view = ggml_view_1d(native_ctx.ctx, native_tensor, 2, 2*sizeof(float));
    GGML_ASSERT(ggml_backend_view_init(native_view) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy(owned_view, native_view);
    GGML_ASSERT(std::memcmp(native_view->data, owned_values.data(), sizeof(owned_values)) == 0);
    owned_values = {301, 302};
    std::memcpy(native_view->data, owned_values.data(), sizeof(owned_values));
    ggml_backend_tensor_copy(native_view, owned_view);
    ggml_backend_tensor_get(tensor, read.data(), 0, sizeof(read));
    GGML_ASSERT(read[2] == 301 && read[3] == 302);
    GGML_ASSERT(ggml_backend_buffer_get_size(view_owner.get()) == 0);

    ggml_tensor * empty = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, 0);
    ggml_backend_buffer_ptr empty_buffer(ggml_backend_buft_alloc_buffer(&device.buffer_type, 0));
    empty->buffer = empty_buffer.get();
    ggml_tensor * empty_view = ggml_view_1d(test_ctx.ctx, empty, 0, 0);
    GGML_ASSERT(ggml_backend_view_init(empty_view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_tensor_is_bound(empty_view) && empty_view->data == nullptr);
}

static void test_meta_buffer_type_lifetime() {
    static auto backend = dummy_backend_init(64);
    backend.context->device.iface.get_buffer_type = [](ggml_backend_dev_t) { return &backend.buffer_type; };
    backend.context->device.iface.get_name = [](ggml_backend_dev_t) { return "dummy_meta_member"; };
    backend.context->device.iface.get_description = [](ggml_backend_dev_t) { return "test meta member"; };
    ggml_backend_dev_t device = &backend.context->device;
    auto * meta = ggml_backend_meta_device(&device, 1, [](const ggml_tensor *, void *) {
        return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(meta);
    GGML_ASSERT(buft == ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(ggml_backend_buft_get_alignment(buft) == 8);
    GGML_ASSERT(ggml_backend_buft_get_device(buft) == meta);
    GGML_ASSERT(!ggml_backend_buft_is_host(buft));
    auto ctx = make_context();
    ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, 8);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.ctx, buft) == 32);
    ggml_backend_buffer_set buffers{};
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(ctx.ctx, buft, &buffers) == GGML_STATUS_FAILED);
    GGML_ASSERT(buffers.buffers == nullptr && buffers.n_buffers == 0);

    auto * empty = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, 0);
    ggml_backend_buffer_ptr assignment(ggml_backend_buft_alloc_buffer(buft, 0));
    empty->buffer = assignment.get();
    GGML_ASSERT(empty->data == nullptr && !ggml_backend_tensor_is_bound(empty));
    auto * allocation = ggml_backend_buft_get_alloc_interface(buft);
    void * preparation = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 1, nullptr);
    GGML_ASSERT(preparation);
    GGML_ASSERT(allocation->prepare_tensor(preparation, empty) == GGML_STATUS_SUCCESS);
    auto * simple = allocation->get_tensor(preparation, empty, 0);
    buffers.buffers = static_cast<ggml_backend_buffer_t *>(malloc(sizeof(ggml_backend_buffer_t)));
    GGML_ASSERT(buffers.buffers);
    buffers.n_buffers = 1;
    buffers.buffers[0] = ggml_backend_buft_alloc_buffer(allocation->get_domain(buft, 0), 0);
    simple->buffer = buffers.buffers[0];
    GGML_ASSERT(ggml_backend_buffer_init_tensor(simple->buffer, simple) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(simple->data == nullptr && ggml_backend_tensor_is_bound(simple));
    ggml_backend_buffer_ptr owner(allocation->materialize(preparation, &buffers, 1));
    GGML_ASSERT(owner && ggml_backend_buffer_get_size(owner.get()) == 0);
    GGML_ASSERT(ggml_backend_buffer_init_tensor(owner.get(), empty) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(empty->data == nullptr && ggml_backend_tensor_is_bound(empty));
    auto * empty_view = ggml_view_1d(ctx.ctx, empty, 0, 0);
    GGML_ASSERT(ggml_backend_view_init(empty_view) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(empty_view->data == nullptr && ggml_backend_tensor_is_bound(empty_view));
}

static void test_context_buffer_sets() {
    auto ctx_set = make_context();
    auto ctx_legacy = make_context();
    auto set_backend = dummy_backend_init(64);
    auto legacy_backend = dummy_backend_init(64);
    std::array<ggml_tensor *, 3> set_tensors;
    std::array<ggml_tensor *, 3> legacy_tensors;
    for (size_t i = 0; i < set_tensors.size(); i++) {
        set_tensors[i] = ggml_new_tensor_1d(ctx_set.ctx, GGML_TYPE_F32, 8);
        legacy_tensors[i] = ggml_new_tensor_1d(ctx_legacy.ctx, GGML_TYPE_F32, 8);
    }
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_set.ctx, &set_backend.buffer_type) == 96);
    GGML_ASSERT(set_backend.context->alloc_calls == 0);
    ggml_backend_buffer_set buffers{};
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(ctx_set.ctx, &set_backend.buffer_type, &buffers) == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_ptr legacy(ggml_backend_alloc_ctx_tensors_from_buft(ctx_legacy.ctx, &legacy_backend.buffer_type));
    GGML_ASSERT(buffers.n_buffers == 2 && ggml_backend_buffer_is_multi_buffer(legacy.get()));
    GGML_ASSERT(ggml_backend_buffer_get_size(buffers.buffers[0]) == 64);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffers.buffers[1]) == 32);
    GGML_ASSERT(ggml_backend_buffer_get_size(legacy.get()) == 96);
    GGML_ASSERT(set_backend.context->allocated_total() == legacy_backend.context->allocated_total());
    for (size_t i = 0; i < set_tensors.size(); i++) {
        GGML_ASSERT(set_tensors[i]->buffer == buffers.buffers[i/2]);
        GGML_ASSERT(legacy_tensors[i]->buffer == legacy_backend.context->buffers[i/2]);
        GGML_ASSERT(size_t(set_tensors[i]->data) - size_t(alloc_base) == size_t(legacy_tensors[i]->data) - size_t(alloc_base));
    }
    ggml_backend_buffer_set again{};
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(ctx_set.ctx, &set_backend.buffer_type, &again) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(again.buffers == nullptr && again.n_buffers == 0);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx_legacy.ctx, &legacy_backend.buffer_type) == nullptr);
    ggml_backend_buffer_set_free(&buffers);
    ggml_backend_buffer_set_free(&buffers);
    GGML_ASSERT(buffers.buffers == nullptr && buffers.n_buffers == 0);
    GGML_ASSERT(set_backend.context->buffers.empty() && set_backend.context->free_calls == 2);
    legacy.reset();
    GGML_ASSERT(legacy_backend.context->buffers.empty() && legacy_backend.context->free_calls == 2);

    for (bool zero_tensor : {false, true}) {
        auto empty = make_context();
        if (zero_tensor) {
            ggml_new_tensor_1d(empty.ctx, GGML_TYPE_F32, 0);
        }
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(empty.ctx, &set_backend.buffer_type, &buffers) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(buffers.buffers == nullptr && buffers.n_buffers == 0);
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(empty.ctx, &set_backend.buffer_type) == nullptr);
    }

    auto oversized = make_context();
    ggml_new_tensor_1d(oversized.ctx, GGML_TYPE_F32, 32);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(oversized.ctx, &set_backend.buffer_type, &buffers) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(buffers.n_buffers == 1 && ggml_backend_buffer_get_size(buffers.buffers[0]) == 128);
    ggml_backend_buffer_set_free(&buffers);
}

static void test_context_buffer_set_failures() {
    for (bool fail_init : {false, true}) {
        for (size_t nth = 1; nth <= 3; nth++) {
            auto ctx = make_context();
            auto backend = dummy_backend_init(32);
            for (int i = 0; i < 3; i++) {
                ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, 8);
            }
            if (fail_init) {
                backend.context->fail_init = nth;
            } else {
                backend.context->fail_alloc = nth;
            }
            ggml_backend_buffer_set buffers{};
            auto status = ggml_backend_alloc_ctx_tensors_from_buft_set(ctx.ctx, &backend.buffer_type, &buffers);
            GGML_ASSERT(status == (fail_init ? GGML_STATUS_FAILED : GGML_STATUS_ALLOC_FAILED));
            GGML_ASSERT(buffers.buffers == nullptr && buffers.n_buffers == 0);
            GGML_ASSERT(backend.context->buffers.empty());
            GGML_ASSERT(backend.context->free_calls == (fail_init ? nth : nth - 1));
            ggml_backend_buffer_set_free(&buffers);
        }
    }
}

static void test_context_buffer_set_borrowed_view() {
    auto source_ctx = make_context();
    auto new_ctx = make_context();
    auto backend = dummy_backend_init(32);
    ggml_tensor * source = ggml_new_tensor_1d(source_ctx.ctx, GGML_TYPE_F32, 8);
    ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors_from_buft(source_ctx.ctx, &backend.buffer_type));
    ggml_tensor * view = ggml_view_1d(new_ctx.ctx, source, 4, sizeof(float));
    ggml_tensor * owned = ggml_new_tensor_1d(new_ctx.ctx, GGML_TYPE_F32, 8);
    ggml_backend_buffer_set buffers{};
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(new_ctx.ctx, &backend.buffer_type, &buffers) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(buffers.n_buffers == 1 && owned->buffer == buffers.buffers[0]);
    GGML_ASSERT(view->buffer == source_buffer.get());
    ggml_backend_buffer_set_free(&buffers);
    GGML_ASSERT(backend.context->buffers.size() == 1 && backend.context->buffers[0] == source_buffer.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(source_buffer.get()) == 32);
    auto view_ctx = make_context();
    auto * only_view = ggml_view_1d(view_ctx.ctx, source, 4, sizeof(float));
    GGML_ASSERT(only_view->buffer == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(view_ctx.ctx, &backend.buffer_type) == 0);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(view_ctx.ctx, &backend.buffer_type, &buffers) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(buffers.n_buffers == 0 && only_view->buffer == source_buffer.get());
    GGML_ASSERT(backend.context->buffers.size() == 1);
}

static void test_meta_split_preparation() {
    static auto first = dummy_backend_init(256);
    static auto second = dummy_backend_init(256);
    for (auto * backend : {&first, &second}) {
        backend->context->buffer_type = &backend->buffer_type;
        backend->context->device.iface.get_buffer_type = test_device_buffer_type;
        backend->context->device.iface.get_name = [](ggml_backend_dev_t) { return "prepare_member"; };
        backend->context->device.iface.get_description = [](ggml_backend_dev_t) { return "preparation test member"; };
    }
    ggml_backend_dev_t devices[] = {&first.context->device, &second.context->device};
    auto * meta = ggml_backend_meta_device(devices, 2, [](const ggml_tensor * tensor, void *) {
        ggml_backend_meta_split_state result{GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
        if (std::strcmp(tensor->name, "mirror") == 0) {
            result.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        } else if (std::strcmp(tensor->name, "segmented") == 0) {
            result.n_segments = 2;
            result.nr[1] = 1;
            result.ne[0] = 1;
            result.ne[1] = 2;
            result.ne[2] = 3;
            result.ne[3] = 6;
        } else {
            result.axis = std::strcmp(tensor->name, "axis0") == 0 ? GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_1;
            result.ne[0] = tensor->ne[result.axis]/3;
            result.ne[1] = tensor->ne[result.axis] - result.ne[0];
        }
        return result;
    }, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(meta);
    auto * allocation = ggml_backend_buft_get_alloc_interface(buft);
    GGML_ASSERT(allocation && allocation->n_domains(buft) == 2);
    GGML_ASSERT(allocation->get_domain(buft, 0) == &first.buffer_type);
    GGML_ASSERT(allocation->get_domain(buft, 1) == &second.buffer_type);
    GGML_ASSERT(ggml_backend_buft_get_alloc_interface(&first.buffer_type) == nullptr);
    GGML_ASSERT(ggml_backend_buft_get_alloc_interface(ggml_backend_cpu_buffer_type()) == nullptr);
    {
        ggml_backend_buffer_type_t buffer_types[] = {buft, &first.buffer_type, buft};
        ggml_gallocr_ptr domains(ggml_gallocr_new_n(buffer_types, 3));
        GGML_ASSERT(domains && first.context->alloc_calls == 0 && second.context->alloc_calls == 0);
    }
    auto test_ctx = make_context();
    auto * source = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * mirror = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 6);
    ggml_set_name(mirror, "mirror");
    auto * segmented = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 12);
    ggml_set_name(segmented, "segmented");
    auto * empty = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 0, 6);
    auto * small = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 2);
    auto * weight = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_Q4_0, 96, 4);
    auto * input = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 96, 2);
    ggml_set_name(weight, "axis0");
    ggml_set_name(input, "axis0");
    auto * product = ggml_mul_mat(test_ctx.ctx, weight, input);
    auto * scaled = ggml_scale(test_ctx.ctx, source, 0.5f);
    auto * scaled_segments = ggml_scale(test_ctx.ctx, segmented, 0.5f);
    auto * scaled_small = ggml_scale(test_ctx.ctx, small, 0.5f);
    scaled_small->flags |= GGML_TENSOR_FLAG_COMPUTE;
    auto * view = ggml_view_2d(test_ctx.ctx, source, 4, 6, source->nb[1], 4*sizeof(float));
    auto * allocation_view = ggml_view_tensor(test_ctx.ctx, source);
    auto * allocation_dep = ggml_view_tensor(test_ctx.ctx, segmented);
    allocation_dep->src[0] = segmented;
    allocation_dep->src[1] = mirror;
    ggml_backend_buffer_ptr assigned(ggml_backend_buft_alloc_buffer(buft, 0));
    ggml_backend_buffer_set_usage(assigned.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto * tensor : {source, weight, input, segmented, small}) {
        tensor->buffer = assigned.get();
    }
    std::vector<std::pair<ggml_tensor *, std::array<uint8_t, GGML_TENSOR_SIZE>>> before;
    for (auto * tensor = ggml_get_first_tensor(test_ctx.ctx); tensor; tensor = ggml_get_next_tensor(test_ctx.ctx, tensor)) {
        before.emplace_back();
        before.back().first = tensor;
        std::memcpy(before.back().second.data(), tensor, GGML_TENSOR_SIZE);
    }
    auto * weights = ggml_backend_meta_split_context_new(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * compute = ggml_backend_meta_split_context_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    auto ss = ggml_backend_meta_split_context_get(weights, source, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 2 && ss.ne[1] == 4);
    ss = ggml_backend_meta_split_context_get(weights, mirror, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    ss = ggml_backend_meta_split_context_get(weights, segmented, true);
    GGML_ASSERT(ss.n_segments == 2 && ss.ne[0] + ss.ne[2] == 4 && ss.ne[1] + ss.ne[3] == 8);
    ss = ggml_backend_meta_split_context_get(weights, small, true);
    GGML_ASSERT(ss.ne[0] == 0 && ss.ne[1] == 2);
    ss = ggml_backend_meta_split_context_get(weights, empty, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    ss = ggml_backend_meta_split_context_get(compute, scaled, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 2 && ss.ne[1] == 4);
    ss = ggml_backend_meta_split_context_get(compute, scaled_segments, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 4 && ss.ne[1] == 8);
    ss = ggml_backend_meta_split_context_get(compute, view, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 2 && ss.ne[1] == 4);
    ss = ggml_backend_meta_split_context_get(compute, product, false);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL);
    ss = ggml_backend_meta_split_context_get(compute, product, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    GGML_ASSERT(!ggml_backend_tensor_is_bound(source) && !ggml_backend_tensor_is_bound(product));
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);
    for (const auto & entry : before) {
        GGML_ASSERT(std::memcmp(entry.first, entry.second.data(), GGML_TENSOR_SIZE) == 0);
    }

    auto * preparation = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, before.size(), nullptr);
    GGML_ASSERT(preparation != nullptr);
    for (const auto & entry : before) {
        GGML_ASSERT(ggml_backend_meta_preparation_tensor(preparation, entry.first) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_meta_preparation_tensor(preparation, entry.first) == GGML_STATUS_SUCCESS);
        for (size_t device = 0; device < 2; device++) {
            auto * simple = ggml_backend_meta_preparation_get_tensor(preparation, entry.first, device);
            GGML_ASSERT(simple && simple->data == nullptr && simple->buffer == nullptr);
        }
        GGML_ASSERT(std::memcmp(entry.first, entry.second.data(), GGML_TENSOR_SIZE) == 0);
    }
    for (size_t device = 0; device < 2; device++) {
        auto * simple_source = ggml_backend_meta_preparation_get_tensor(preparation, source, device);
        auto * simple_scaled = ggml_backend_meta_preparation_get_tensor(preparation, scaled, device);
        auto * simple_view = ggml_backend_meta_preparation_get_tensor(preparation, view, device);
        auto * simple_weight = ggml_backend_meta_preparation_get_tensor(preparation, weight, device);
        auto * simple_mirror = ggml_backend_meta_preparation_get_tensor(preparation, mirror, device);
        auto * simple_small = ggml_backend_meta_preparation_get_tensor(preparation, scaled_small, device);
        auto * simple_allocation_view = ggml_backend_meta_preparation_get_tensor(preparation, allocation_view, device);
        auto * simple_allocation_dep = ggml_backend_meta_preparation_get_tensor(preparation, allocation_dep, device);
        GGML_ASSERT(simple_source->ne[1] == (device == 0 ? 2 : 4));
        GGML_ASSERT(ggml_nbytes(simple_source) == (device == 0 ? 64 : 128));
        GGML_ASSERT(simple_scaled->src[0] == simple_source);
        GGML_ASSERT(simple_scaled->ne[1] == simple_source->ne[1]);
        GGML_ASSERT(simple_view->view_src == simple_source && simple_view->view_offs == 4*sizeof(float));
        GGML_ASSERT(simple_view->nb[1] == 8*sizeof(float));
        GGML_ASSERT(ggml_nbytes(simple_view) + simple_view->view_offs == ggml_nbytes(simple_source));
        GGML_ASSERT(simple_weight->ne[0] == (device == 0 ? 32 : 64));
        GGML_ASSERT(ggml_nbytes(simple_weight) == (device == 0 ? 72 : 144));
        GGML_ASSERT(ggml_nbytes(simple_mirror) == ggml_nbytes(mirror));
        GGML_ASSERT(bool(simple_small->flags & GGML_TENSOR_FLAG_COMPUTE) == (device != 0));
        GGML_ASSERT(simple_allocation_view->op == GGML_OP_NONE && simple_allocation_view->view_src == simple_source);
        GGML_ASSERT(simple_allocation_view->ne[1] == simple_source->ne[1]);
        GGML_ASSERT(simple_allocation_dep->op == GGML_OP_NONE && simple_allocation_dep->ne[1] == (device == 0 ? 4 : 8));
        GGML_ASSERT(simple_allocation_dep->src[0] == ggml_backend_meta_preparation_get_tensor(preparation, segmented, device));
        GGML_ASSERT(simple_allocation_dep->src[1] == simple_mirror);
    }
    ss = ggml_backend_meta_split_context_get(compute, allocation_dep, true);
    GGML_ASSERT(ss.n_segments == 2 && ss.ne[0] == 1 && ss.ne[1] == 2 && ss.ne[2] == 3 && ss.ne[3] == 6);
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);

    void * generic = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 1, nullptr);
    GGML_ASSERT(generic);
    GGML_ASSERT(allocation->prepare_tensor(generic, source) == GGML_STATUS_SUCCESS);
    for (size_t device = 0; device < allocation->n_domains(buft); device++) {
        auto * simple = allocation->get_tensor(generic, source, device);
        GGML_ASSERT(simple && !simple->buffer && !simple->data);
        GGML_ASSERT(ggml_backend_buft_get_alloc_size(allocation->get_domain(buft, device), simple) == (device == 0 ? 64 : 128));
    }
    allocation->free_preparation(generic);
    generic = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 0, nullptr);
    GGML_ASSERT(generic);
    GGML_ASSERT(allocation->prepare_tensor(generic, source) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(allocation->get_tensor(generic, source, 0) == nullptr);
    allocation->free_preparation(generic);
    for (const auto & entry : before) {
        GGML_ASSERT(std::memcmp(entry.first, entry.second.data(), GGML_TENSOR_SIZE) == 0);
    }
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);

    auto * missing_source = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 2, nullptr);
    GGML_ASSERT(missing_source != nullptr);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(missing_source, scaled) == GGML_STATUS_FAILED);
    GGML_ASSERT(ggml_backend_meta_preparation_get_tensor(missing_source, scaled, 0) == nullptr);
    ggml_backend_meta_preparation_free(missing_source);
    auto * no_capacity = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 0, nullptr);
    GGML_ASSERT(no_capacity != nullptr);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(no_capacity, source) == GGML_STATUS_ALLOC_FAILED);
    ggml_backend_meta_preparation_free(no_capacity);
    GGML_ASSERT(ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, SIZE_MAX, nullptr) == nullptr);
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);

    auto resolved_ctx = make_context();
    auto * resolved_source = ggml_new_tensor_2d(resolved_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * resolved_scaled = ggml_scale(resolved_ctx.ctx, resolved_source, 0.5f);
    auto * resolved_view = ggml_view_2d(resolved_ctx.ctx, resolved_source, 4, 6, resolved_source->nb[1], 4*sizeof(float));
    auto * native_source = ggml_new_tensor_2d(resolved_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * native_scaled = ggml_scale(resolved_ctx.ctx, native_source, 0.5f);
    std::vector<std::pair<ggml_tensor *, std::array<uint8_t, GGML_TENSOR_SIZE>>> resolved_before;
    for (auto * tensor = ggml_get_first_tensor(resolved_ctx.ctx); tensor; tensor = ggml_get_next_tensor(resolved_ctx.ctx, tensor)) {
        resolved_before.emplace_back();
        resolved_before.back().first = tensor;
        std::memcpy(resolved_before.back().second.data(), tensor, GGML_TENSOR_SIZE);
    }
    auto * source_preparation = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 1, nullptr);
    GGML_ASSERT(source_preparation);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(source_preparation, resolved_source) == GGML_STATUS_SUCCESS);
    struct source_context {
        ggml_backend_meta_preparation * preparation;
        ggml_backend_buffer_type_t buft;
        const ggml_tensor * source;
        const ggml_tensor * native;
        bool missing_device = false;
    } source_ctx = {source_preparation, buft, resolved_source, native_source};
    ggml_backend_alloc_source_i sources = {
        [](void * context, const ggml_tensor * tensor, ggml_backend_buffer_usage * usage) {
            auto * ctx = static_cast<source_context *>(context);
            *usage = tensor == ctx->source ? GGML_BACKEND_BUFFER_USAGE_WEIGHTS : GGML_BACKEND_BUFFER_USAGE_COMPUTE;
            return tensor == ctx->native ? ggml_backend_cpu_buffer_type() : ctx->buft;
        },
        [](void * context, const ggml_tensor * tensor, size_t device) -> ggml_tensor * {
            auto * ctx = static_cast<source_context *>(context);
            return tensor == ctx->source && !(ctx->missing_device && device == 1) ?
                ggml_backend_meta_preparation_get_tensor(ctx->preparation, tensor, device) : nullptr;
        },
        &source_ctx,
        nullptr,
    };
    auto * resolved = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 3, &sources);
    GGML_ASSERT(resolved);
    for (auto * tensor : {resolved_source, resolved_scaled, resolved_view, native_source, native_scaled}) {
        GGML_ASSERT(ggml_backend_meta_preparation_tensor(resolved, tensor) == GGML_STATUS_SUCCESS);
    }
    for (size_t device = 0; device < 2; device++) {
        auto * simple_source = ggml_backend_meta_preparation_get_tensor(source_preparation, resolved_source, device);
        auto * simple_scaled = ggml_backend_meta_preparation_get_tensor(resolved, resolved_scaled, device);
        auto * simple_view = ggml_backend_meta_preparation_get_tensor(resolved, resolved_view, device);
        auto * simple_native = ggml_backend_meta_preparation_get_tensor(resolved, native_scaled, device);
        GGML_ASSERT(ggml_backend_meta_preparation_get_tensor(resolved, resolved_source, device) == simple_source);
        GGML_ASSERT(simple_scaled->src[0] == simple_source && simple_scaled->ne[1] == (device == 0 ? 2 : 4));
        GGML_ASSERT(simple_view->view_src == simple_source && simple_view->view_offs == 4*sizeof(float));
        GGML_ASSERT(simple_native->src[0] == native_source && simple_native->ne[1] == 6);
    }
    for (const auto & entry : resolved_before) {
        GGML_ASSERT(std::memcmp(entry.first, entry.second.data(), GGML_TENSOR_SIZE) == 0);
    }
    ggml_backend_meta_preparation_free(resolved);
    GGML_ASSERT(ggml_backend_meta_preparation_get_tensor(source_preparation, resolved_source, 1)->ne[1] == 4);
    source_ctx.missing_device = true;
    for (auto * tensor : {resolved_source, resolved_scaled}) {
        auto * incomplete = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 3, &sources);
        GGML_ASSERT(incomplete);
        GGML_ASSERT(ggml_backend_meta_preparation_tensor(incomplete, tensor) == GGML_STATUS_FAILED);
        GGML_ASSERT(ggml_backend_meta_preparation_get_tensor(incomplete, resolved_source, 0) == nullptr);
        ggml_backend_meta_preparation_free(incomplete);
    }
    source_ctx.missing_device = false;
    ggml_backend_dev_t reversed_devices[] = {devices[1], devices[0]};
    auto * reversed = ggml_backend_meta_device(reversed_devices, 2, [](const ggml_tensor *, void *) {
        return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }, nullptr);
    source_ctx.buft = ggml_backend_dev_buffer_type(reversed);
    auto * incompatible = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 3, &sources);
    GGML_ASSERT(incompatible);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(incompatible, resolved_scaled) == GGML_STATUS_FAILED);
    ggml_backend_meta_preparation_free(incompatible);
    ggml_backend_meta_preparation_free(source_preparation);
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);

    for (auto * backend : {&first, &second}) {
        backend->context->shard_size_queries = 0;
        backend->buffer_type.iface.get_alloc_size = [](ggml_backend_buffer_type_t type, const ggml_tensor * tensor) {
            auto * context = static_cast<dummy_backend_context *>(type->context);
            if (std::strcmp(tensor->name, "graph_shard") == 0 && (tensor->ne[1] == 2 || tensor->ne[1] == 4)) {
                GGML_ASSERT(tensor->data == nullptr && tensor->buffer == nullptr);
                context->shard_size_queries++;
            }
            return ggml_nbytes(tensor) + (context->expand_quantized_add && tensor->op == GGML_OP_ADD && ggml_is_quantized(tensor->type) ? 64 : 0);
        };
    }
    ggml_gallocr_ptr measured(ggml_gallocr_new(buft));
    for (int pass = 0; pass < 2; pass++) {
        auto graph_ctx = make_context();
        auto * graph_source = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 8, 6);
        graph_source->buffer = assigned.get();
        auto * graph_result = ggml_scale(graph_ctx.ctx, graph_source, 0.5f);
        ggml_set_name(graph_result, "graph_shard");
        ggml_set_output(graph_result);
        ggml_build_forward_expand(graph_ctx.graph, graph_result);
        std::array<uint8_t, GGML_TENSOR_SIZE> source_before, result_before;
        std::memcpy(source_before.data(), graph_source, GGML_TENSOR_SIZE);
        std::memcpy(result_before.data(), graph_result, GGML_TENSOR_SIZE);
        size_t required = 0;
        ggml_gallocr_reserve_n_size(measured.get(), graph_ctx.graph, nullptr, nullptr, &required);
        GGML_ASSERT(required == ggml_nbytes(graph_result));
        auto first_plan = ggml_gallocr_get_domain_plan_info(measured.get(), 0, 0);
        auto second_plan = ggml_gallocr_get_domain_plan_info(measured.get(), 0, 1);
        GGML_ASSERT(first_plan.size == 64 && second_plan.size == 128);
        GGML_ASSERT(first_plan.n_chunks == 1 && second_plan.n_chunks == 1);
        GGML_ASSERT(std::memcmp(source_before.data(), graph_source, GGML_TENSOR_SIZE) == 0);
        GGML_ASSERT(std::memcmp(result_before.data(), graph_result, GGML_TENSOR_SIZE) == 0);
        GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);
    }
    {
        auto graph_ctx = make_context();
        auto * native_source = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 8, 6);
        ggml_set_output(native_source);
        auto * result = ggml_scale(graph_ctx.ctx, native_source, 0.5f);
        ggml_build_forward_expand(graph_ctx.graph, result);
        ggml_backend_buffer_type_t types[] = {buft, &first.buffer_type};
        ggml_gallocr_ptr mixed(ggml_gallocr_new_n(types, 2));
        int nodes[] = {0};
        int leafs[] = {1};
        size_t sizes[2] = {};
        ggml_gallocr_reserve_n_size(mixed.get(), graph_ctx.graph, nodes, leafs, sizes);
        GGML_ASSERT(sizes[0] == 192 && sizes[1] == 192);
        GGML_ASSERT(native_source->buffer == nullptr && result->buffer == nullptr);
    }
    {
        auto graph_ctx = make_context();
        auto * a = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 8, 6);
        auto * b = ggml_scale(graph_ctx.ctx, a, 0.5f);
        a->op = GGML_OP_SCALE;
        a->src[0] = b;
        graph_ctx.graph->n_nodes = 2;
        graph_ctx.graph->nodes[0] = a;
        graph_ctx.graph->nodes[1] = b;
        GGML_ASSERT(!ggml_gallocr_reserve(measured.get(), graph_ctx.graph));
        GGML_ASSERT(a->buffer == nullptr && b->buffer == nullptr);
    }
    measured.reset();
    {
        first.context->max_buffer_size = 96;
        second.context->max_buffer_size = 160;
        auto graph_ctx = make_context();
        auto * source = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 8, 6);
        source->buffer = assigned.get();
        auto * a = ggml_scale(graph_ctx.ctx, source, 0.5f);
        auto * view = ggml_view_2d(graph_ctx.ctx, a, 4, 6, a->nb[1], 0);
        auto * b = ggml_cont(graph_ctx.ctx, view);
        ggml_set_output(a);
        ggml_set_output(b);
        ggml_build_forward_expand(graph_ctx.graph, b);
        ggml_gallocr_ptr split_plan(ggml_gallocr_new(buft));
        size_t legacy_size = 0;
        ggml_gallocr_reserve_n_size(split_plan.get(), graph_ctx.graph, nullptr, nullptr, &legacy_size);
        auto d0 = ggml_gallocr_get_domain_plan_info(split_plan.get(), 0, 0);
        auto d1 = ggml_gallocr_get_domain_plan_info(split_plan.get(), 0, 1);
        GGML_ASSERT(d0.n_chunks == 1 && d0.size == 96);
        GGML_ASSERT(d1.n_chunks == 2 && d1.size == 192);
        GGML_ASSERT(!a->buffer && !b->buffer && !view->buffer);
        first.context->max_buffer_size = second.context->max_buffer_size = 256;
    }
    {
        auto graph_ctx = make_context();
        auto * source = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 8, 6);
        source->buffer = assigned.get();
        auto * a = ggml_scale(graph_ctx.ctx, source, 0.5f);
        auto * b = ggml_scale(graph_ctx.ctx, a, 0.5f);
        ggml_set_output(b);
        ggml_build_forward_expand(graph_ctx.graph, b);
        ggml_gallocr_ptr inplace(ggml_gallocr_new(buft));
        size_t legacy_size = 0;
        ggml_gallocr_reserve_n_size(inplace.get(), graph_ctx.graph, nullptr, nullptr, &legacy_size);
        GGML_ASSERT(ggml_gallocr_get_domain_plan_info(inplace.get(), 0, 0).size == 64);
        GGML_ASSERT(ggml_gallocr_get_domain_plan_info(inplace.get(), 0, 1).size == 128);
    }
    {
        second.context->expand_quantized_add = true;
        auto graph_ctx = make_context();
        auto * source = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_Q4_0, 96, 6);
        auto * bias = ggml_new_tensor_2d(graph_ctx.ctx, GGML_TYPE_F32, 96, 6);
        source->buffer = bias->buffer = assigned.get();
        auto * a = ggml_cont(graph_ctx.ctx, source);
        auto * b = ggml_add(graph_ctx.ctx, a, bias);
        ggml_set_output(b);
        ggml_build_forward_expand(graph_ctx.graph, b);
        ggml_gallocr_ptr incompatible(ggml_gallocr_new(buft));
        size_t legacy_size = 0;
        ggml_gallocr_reserve_n_size(incompatible.get(), graph_ctx.graph, nullptr, nullptr, &legacy_size);
        GGML_ASSERT(ggml_gallocr_get_domain_plan_info(incompatible.get(), 0, 0).size == 224);
        GGML_ASSERT(ggml_gallocr_get_domain_plan_info(incompatible.get(), 0, 1).size == 496);
        second.context->expand_quantized_add = false;
    }
    GGML_ASSERT(first.context->alloc_calls == 0 && second.context->alloc_calls == 0);
    GGML_ASSERT(first.context->shard_size_queries >= 2 && second.context->shard_size_queries >= 2);
    first.buffer_type.iface.get_alloc_size = nullptr;
    second.buffer_type.iface.get_alloc_size = nullptr;

    auto bound_ctx = make_context();
    auto * bound = ggml_new_tensor_2d(bound_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * bound_view = ggml_view_2d(bound_ctx.ctx, bound, 4, 6, bound->nb[1], 4*sizeof(float));
    ggml_backend_buffer_ptr bound_buffer(ggml_backend_alloc_ctx_tensors_from_buft(bound_ctx.ctx, buft));
    GGML_ASSERT(bound_buffer != nullptr);
    GGML_ASSERT(first.context->buffers.size() == 1 && second.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(first.context->buffers[0]) == 64);
    GGML_ASSERT(ggml_backend_buffer_get_size(second.context->buffers[0]) == 128);
    auto * legacy_view = ggml_view_2d(bound_ctx.ctx, bound, 4, 6, bound->nb[1], 4*sizeof(float));
    GGML_ASSERT(legacy_view->data != nullptr && legacy_view->buffer == nullptr);
    auto * legacy_preparation = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 1, nullptr);
    GGML_ASSERT(legacy_preparation);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(legacy_preparation, legacy_view) == GGML_STATUS_SUCCESS);
    for (size_t device = 0; device < 2; device++) {
        auto * simple = ggml_backend_meta_preparation_get_tensor(legacy_preparation, legacy_view, device);
        GGML_ASSERT(simple != legacy_view && simple->data == nullptr && simple->buffer == nullptr);
        GGML_ASSERT(simple->ne[1] == (device == 0 ? 2 : 4));
    }
    ggml_backend_meta_preparation_free(legacy_preparation);
    source_ctx.preparation = nullptr;
    source_ctx.source = nullptr;
    auto * bound_preparation = ggml_backend_meta_preparation_new(buft, GGML_BACKEND_BUFFER_USAGE_COMPUTE, 0, &sources);
    GGML_ASSERT(bound_preparation);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(bound_preparation, bound) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_meta_preparation_get_tensor(bound_preparation, bound, 1)->buffer == second.context->buffers[0]);
    ggml_backend_meta_preparation_free(bound_preparation);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(preparation, bound) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_meta_preparation_tensor(preparation, bound_view) == GGML_STATUS_SUCCESS);
    for (size_t device = 0; device < 2; device++) {
        auto * simple_bound = ggml_backend_meta_preparation_get_tensor(preparation, bound, device);
        auto * simple_view = ggml_backend_meta_preparation_get_tensor(preparation, bound_view, device);
        GGML_ASSERT(simple_bound->buffer != nullptr && simple_view->buffer == simple_bound->buffer);
        GGML_ASSERT(size_t(simple_view->data) == size_t(simple_bound->data) + 4*sizeof(float));
    }
    ss = ggml_backend_meta_split_context_get(compute, bound, true);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 2 && ss.ne[1] == 4);
    ggml_backend_buffer_set_usage(bound_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ss = ggml_backend_meta_split_context_get(compute, bound, false);
    GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_1 && ss.ne[0] == 2 && ss.ne[1] == 4);
    ggml_backend_meta_preparation_free(preparation);
    GGML_ASSERT(first.context->buffers.size() == 1 && second.context->buffers.size() == 1);
    for (bool fail_init : {false, true}) {
        auto failed_ctx = make_context();
        ggml_new_tensor_2d(failed_ctx.ctx, GGML_TYPE_F32, 8, 6);
        if (fail_init) {
            second.context->fail_init = second.context->init_calls + 1;
        } else {
            second.context->fail_alloc = second.context->alloc_calls + 1;
        }
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(failed_ctx.ctx, buft) == nullptr);
        GGML_ASSERT(first.context->buffers.size() == 1 && second.context->buffers.size() == 1);
        second.context->fail_init = SIZE_MAX;
        second.context->fail_alloc = SIZE_MAX;
    }
    ggml_backend_meta_split_context_free(compute);
    ggml_backend_meta_split_context_free(weights);
}

static void test_meta_materialization(bool relative_addresses) {
    static auto first = dummy_backend_init(256);
    static auto second = dummy_backend_init(256);
    for (auto * backend : {&first, &second}) {
        backend->context->real_data = true;
        backend->context->relative_addresses = relative_addresses;
        backend->context->alloc_calls = backend->context->free_calls = backend->context->init_calls = 0;
        backend->context->buffer_type = &backend->buffer_type;
        backend->context->device.iface.get_buffer_type = test_device_buffer_type;
        backend->context->device.iface.get_name = [](ggml_backend_dev_t) { return "materialized_member"; };
        backend->context->device.iface.get_description = [](ggml_backend_dev_t) { return "materialization test member"; };
    }
    ggml_backend_dev_t devices[] = {&first.context->device, &second.context->device};
    auto * meta = ggml_backend_meta_device(devices, 2, [](const ggml_tensor * tensor, void *) {
        if (std::strcmp(tensor->name, "mirror") == 0) {
            return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        return ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_1, {2, 4}, {1}, 1};
    }, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(meta);
    auto * allocation = ggml_backend_buft_get_alloc_interface(buft);
    auto test_ctx = make_context();
    auto * split = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * mirror = ggml_new_tensor_2d(test_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * view = ggml_view_1d(test_ctx.ctx, mirror, 4, 8*sizeof(float));
    ggml_set_name(mirror, "mirror");
    void * preparation = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 3, nullptr);
    GGML_ASSERT(preparation);
    for (auto * tensor : {split, mirror, view}) {
        GGML_ASSERT(allocation->prepare_tensor(preparation, tensor) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_buffer_set domains[2] = {};
    for (size_t device = 0; device < 2; device++) {
        domains[device].buffers = static_cast<ggml_backend_buffer_t *>(calloc(2, sizeof(ggml_backend_buffer_t)));
        GGML_ASSERT(domains[device].buffers);
        domains[device].n_buffers = 2;
        domains[device].buffers[0] = ggml_backend_buft_alloc_buffer(allocation->get_domain(buft, device), device == 0 ? 80 : 160);
        domains[device].buffers[1] = ggml_backend_buft_alloc_buffer(allocation->get_domain(buft, device), 256);
        GGML_ASSERT(domains[device].buffers[0] && domains[device].buffers[1]);
        if (relative_addresses) {
            GGML_ASSERT(ggml_backend_buffer_get_base(domains[device].buffers[0]) == ggml_backend_buffer_get_base(domains[device].buffers[1]));
        }
    }
    GGML_ASSERT(allocation->materialize(preparation, domains, 2) == nullptr);
    GGML_ASSERT(domains[0].n_buffers == 2 && domains[1].n_buffers == 2);
    GGML_ASSERT(first.context->free_calls == 0 && second.context->free_calls == 0);
    for (size_t device = 0; device < 2; device++) {
        for (size_t i = 0; i < 2; i++) {
            auto * simple = allocation->get_tensor(preparation, i == 0 ? split : mirror, device);
            auto * buffer = domains[device].buffers[i];
            size_t offset = i == 0 ? (device == 0 ? 8 : 32) : (device == 0 ? 32 : 64);
            GGML_ASSERT(ggml_backend_tensor_alloc(buffer, simple, static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer)) + offset) == GGML_STATUS_SUCCESS);
        }
        auto * simple_view = allocation->get_tensor(preparation, view, device);
        GGML_ASSERT(ggml_backend_view_init(simple_view) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(simple_view->buffer == domains[device].buffers[1]);
    }
    std::swap(domains[0], domains[1]);
    GGML_ASSERT(allocation->materialize(preparation, domains, 2) == nullptr);
    std::swap(domains[0], domains[1]);
    auto * saved_buffer = domains[0].buffers[1];
    domains[0].buffers[1] = domains[0].buffers[0];
    GGML_ASSERT(allocation->materialize(preparation, domains, 2) == nullptr);
    domains[0].buffers[1] = saved_buffer;
    auto * invalid_view = allocation->get_tensor(preparation, view, 1);
    size_t saved_offset = invalid_view->view_offs;
    invalid_view->view_offs = ggml_nbytes(invalid_view->view_src) + 1;
    GGML_ASSERT(allocation->materialize(preparation, domains, 2) == nullptr);
    invalid_view->view_offs = saved_offset;
    ggml_backend_buffer_ptr owner(allocation->materialize(preparation, domains, 2));
    GGML_ASSERT(owner);
    GGML_ASSERT(domains[0].buffers == nullptr && domains[0].n_buffers == 0);
    GGML_ASSERT(domains[1].buffers == nullptr && domains[1].n_buffers == 0);
    GGML_ASSERT(ggml_backend_buffer_get_size(owner.get()) == 752);
    for (size_t device = 0; device < 2; device++) {
        GGML_ASSERT(allocation->n_buffers(owner.get(), device) == 2);
        GGML_ASSERT(ggml_backend_buffer_get_size(allocation->get_buffer(owner.get(), device, 0)) == (device == 0 ? 80 : 160));
        GGML_ASSERT(ggml_backend_buffer_get_size(allocation->get_buffer(owner.get(), device, 1)) == 256);
    }
    GGML_ASSERT(ggml_backend_buffer_get_base(owner.get()) == nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_usage(saved_buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(split->buffer == nullptr && mirror->buffer == nullptr && view->buffer == nullptr);
    auto * raw = ggml_new_tensor_1d(test_ctx.ctx, GGML_TYPE_F32, 1);
    GGML_ASSERT(ggml_backend_tensor_alloc(owner.get(), raw, nullptr) == GGML_STATUS_FAILED);
    auto tallocr = ggml_tallocr_new(owner.get());
    GGML_ASSERT(ggml_tallocr_alloc(&tallocr, raw) == GGML_STATUS_FAILED);
    GGML_ASSERT(raw->buffer == nullptr && raw->data == nullptr);
    split->buffer = owner.get();
    GGML_ASSERT(!ggml_backend_tensor_is_bound(split));
    for (auto * tensor : {split, mirror, view}) {
        GGML_ASSERT(ggml_backend_buffer_init_tensor(owner.get(), tensor) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(tensor->data == nullptr && ggml_backend_tensor_is_bound(tensor));
    }
    {
        auto * cpu_buft = ggml_backend_cpu_buffer_type();
        size_t expected = GGML_PAD(ggml_nbytes(raw), ggml_backend_buft_get_alignment(cpu_buft));
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(test_ctx.ctx, cpu_buft) == expected);
        ggml_backend_buffer_set physical = {};
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(test_ctx.ctx, cpu_buft, &physical) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(physical.n_buffers == 1 && raw->buffer == physical.buffers[0]);
        for (auto * tensor : {split, mirror, view}) {
            GGML_ASSERT(tensor->data == nullptr && tensor->buffer == owner.get() && ggml_backend_tensor_is_bound(tensor));
        }
        ggml_backend_buffer_set_free(&physical);
        GGML_ASSERT(first.context->free_calls == 0 && second.context->free_calls == 0);
    }
    {
        auto graph_ctx = make_context();
        auto * output = ggml_scale(graph_ctx.ctx, split, 0.5f);
        ggml_set_output(output);
        ggml_build_forward_expand(graph_ctx.graph, output);
        ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph_ctx.graph));
        GGML_ASSERT(split->buffer == owner.get() && split->data == nullptr && ggml_backend_tensor_is_bound(split));
        GGML_ASSERT(output->buffer != owner.get() && ggml_backend_tensor_is_bound(output));
        GGML_ASSERT(ggml_gallocr_get_buffer_size(galloc.get(), 0) == ggml_nbytes(output));
    }
    std::array<float, 48> values;
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = float(i + 1);
    }
    std::array<float, 48> result;
    for (auto * tensor : {split, mirror}) {
        ggml_backend_tensor_set(tensor, values.data(), 0, sizeof(values));
        ggml_backend_tensor_get(tensor, result.data(), 0, sizeof(result));
        GGML_ASSERT(values == result);
    }
    std::array<float, 4> view_values;
    ggml_backend_tensor_get(view, view_values.data(), 0, sizeof(view_values));
    GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), values.begin() + 8));
    {
        auto view_ctx = make_context();
        auto * only_view = ggml_view_1d(view_ctx.ctx, mirror, 4, 4*sizeof(float));
        ggml_backend_buffer_set physical = {};
        GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_set(view_ctx.ctx, ggml_backend_cpu_buffer_type(), &physical) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(physical.n_buffers == 0 && only_view->buffer == owner.get() && only_view->data == nullptr);
        ggml_backend_tensor_get(only_view, view_values.data(), 0, sizeof(view_values));
        GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), values.begin() + 4));
    }

    auto * external = ggml_view_1d(test_ctx.ctx, mirror, 4, 16*sizeof(float));
    GGML_ASSERT(ggml_backend_view_init(external) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(external->data == nullptr && ggml_backend_tensor_is_bound(external));
    ggml_backend_tensor_get(external, view_values.data(), 0, sizeof(view_values));
    GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), values.begin() + 16));
    external->buffer = nullptr;
    external->view_offs = 20*sizeof(float);
    GGML_ASSERT(ggml_backend_view_init(external) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(external, view_values.data(), 0, sizeof(view_values));
    GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), values.begin() + 20));
    auto * failed_view = ggml_view_1d(test_ctx.ctx, mirror, 4, 24*sizeof(float));
    second.context->fail_init = second.context->init_calls + 1;
    GGML_ASSERT(ggml_backend_view_init(failed_view) == GGML_STATUS_FAILED);
    GGML_ASSERT(failed_view->buffer == nullptr && failed_view->data == nullptr);
    GGML_ASSERT(ggml_backend_tensor_is_bound(external) && ggml_backend_tensor_is_bound(mirror));
    second.context->fail_init = SIZE_MAX;
    GGML_ASSERT(ggml_backend_view_init(failed_view) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(failed_view, view_values.data(), 0, sizeof(view_values));
    GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), values.begin() + 24));

    auto borrowed_ctx = make_context();
    auto * borrowed = ggml_view_1d(borrowed_ctx.ctx, mirror, 4, 12*sizeof(float));
    void * borrowed_preparation = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 1, nullptr);
    GGML_ASSERT(borrowed_preparation);
    GGML_ASSERT(allocation->prepare_tensor(borrowed_preparation, borrowed) == GGML_STATUS_SUCCESS);
    for (size_t device = 0; device < 2; device++) {
        GGML_ASSERT(ggml_backend_view_init(allocation->get_tensor(borrowed_preparation, borrowed, device)) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_buffer_ptr borrowed_owner(allocation->materialize(borrowed_preparation, domains, 2));
    GGML_ASSERT(borrowed_owner && ggml_backend_buffer_get_size(borrowed_owner.get()) == 0);
    GGML_ASSERT(ggml_backend_buffer_init_tensor(borrowed_owner.get(), borrowed) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(borrowed->data == nullptr && borrowed->buffer != mirror->buffer);
    auto * dependent = ggml_view_1d(borrowed_ctx.ctx, mirror, 4, 28*sizeof(float));
    void * dependent_preparation = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 1, nullptr);
    GGML_ASSERT(dependent_preparation);
    GGML_ASSERT(allocation->prepare_tensor(dependent_preparation, dependent) == GGML_STATUS_SUCCESS);
    for (size_t device = 0; device < 2; device++) {
        GGML_ASSERT(ggml_backend_view_init(allocation->get_tensor(dependent_preparation, dependent, device)) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_buffer_ptr dependent_owner(allocation->materialize(dependent_preparation, domains, 2));
    GGML_ASSERT(dependent_owner);
    GGML_ASSERT(ggml_backend_buffer_init_tensor(dependent_owner.get(), dependent) == GGML_STATUS_SUCCESS);
    view_values.fill(123);
    ggml_backend_tensor_set(borrowed, view_values.data(), 0, sizeof(view_values));
    ggml_backend_buffer_clear(borrowed_owner.get(), 0);
    ggml_backend_tensor_get(mirror, result.data(), 0, sizeof(result));
    GGML_ASSERT(std::equal(view_values.begin(), view_values.end(), result.begin() + 12));
    ggml_backend_buffer_reset(borrowed_owner.get());
    GGML_ASSERT(!ggml_backend_tensor_is_bound(borrowed));
    GGML_ASSERT(ggml_backend_buffer_init_tensor(borrowed_owner.get(), borrowed) == GGML_STATUS_FAILED);
    borrowed_owner.reset();
    GGML_ASSERT(ggml_backend_tensor_is_bound(dependent) && ggml_backend_tensor_is_bound(mirror));
    GGML_ASSERT(first.context->free_calls == 0 && second.context->free_calls == 0);
    ggml_backend_buffer_clear(owner.get(), 0);
    ggml_backend_alloc_source_i replacing = {};
    replacing.replaced_buffer = owner.get();
    void * replacement = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 3, &replacing);
    GGML_ASSERT(replacement);
    for (auto * tensor : {split, mirror, view}) {
        GGML_ASSERT(allocation->prepare_tensor(replacement, tensor) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(tensor->buffer == owner.get() && ggml_backend_tensor_is_bound(tensor));
        for (size_t device = 0; device < 2; device++) {
            auto * simple = allocation->get_tensor(replacement, tensor, device);
            GGML_ASSERT(simple && simple->buffer == nullptr && simple->data == nullptr);
        }
    }
    allocation->free_preparation(replacement);
    for (auto * backend : {&first, &second}) {
        for (const auto & bytes : backend->context->data) {
            GGML_ASSERT(std::all_of(bytes.second.begin(), bytes.second.end(), [](uint8_t value) { return value == 0; }));
        }
    }
    ggml_backend_buffer_set invalid_result[2] = {};
    invalid_result[1].n_buffers = 1;
    GGML_ASSERT(allocation->release_buffers(owner.get(), invalid_result, 2) == GGML_STATUS_FAILED);
    GGML_ASSERT(allocation->release_buffers(owner.get(), domains, 1) == GGML_STATUS_FAILED);
    GGML_ASSERT(ggml_backend_tensor_is_bound(mirror) && ggml_backend_buffer_get_size(owner.get()) == 752);
    GGML_ASSERT(allocation->release_buffers(owner.get(), domains, 2) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_buffer_get_size(owner.get()) == 0);
    GGML_ASSERT(allocation->n_buffers(owner.get(), 0) == 0 && allocation->n_buffers(owner.get(), 1) == 0);
    GGML_ASSERT(domains[0].n_buffers == 2 && domains[1].n_buffers == 2);
    GGML_ASSERT(!ggml_backend_tensor_is_bound(mirror) && !ggml_backend_tensor_is_bound(external));
    GGML_ASSERT(!ggml_backend_tensor_is_bound(dependent));
    GGML_ASSERT(ggml_backend_buffer_init_tensor(dependent_owner.get(), dependent) == GGML_STATUS_FAILED);
    dependent_owner.reset();
    owner.reset();
    GGML_ASSERT(first.context->free_calls == 0 && second.context->free_calls == 0);
    auto reused_ctx = make_context();
    auto * reused_split = ggml_new_tensor_2d(reused_ctx.ctx, GGML_TYPE_F32, 8, 6);
    auto * reused_mirror = ggml_new_tensor_2d(reused_ctx.ctx, GGML_TYPE_F32, 8, 6);
    ggml_set_name(reused_mirror, "mirror");
    void * reused_preparation = allocation->new_preparation(buft, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, 2, nullptr);
    GGML_ASSERT(reused_preparation);
    for (auto * tensor : {reused_split, reused_mirror}) {
        GGML_ASSERT(allocation->prepare_tensor(reused_preparation, tensor) == GGML_STATUS_SUCCESS);
    }
    for (size_t device = 0; device < 2; device++) {
        for (size_t chunk = 0; chunk < 2; chunk++) {
            auto * tensor = allocation->get_tensor(reused_preparation, chunk == 0 ? reused_split : reused_mirror, device);
            auto * buffer = domains[device].buffers[chunk];
            GGML_ASSERT(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
        }
    }
    ggml_backend_buffer_ptr reused_owner(allocation->materialize(reused_preparation, domains, 2));
    GGML_ASSERT(reused_owner && ggml_backend_buffer_get_size(reused_owner.get()) == 752);
    for (auto * tensor : {reused_split, reused_mirror}) {
        GGML_ASSERT(ggml_backend_buffer_init_tensor(reused_owner.get(), tensor) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(tensor->data == nullptr && ggml_backend_tensor_is_bound(tensor));
        ggml_backend_tensor_set(tensor, values.data(), 0, sizeof(values));
        ggml_backend_tensor_get(tensor, result.data(), 0, sizeof(result));
        GGML_ASSERT(result == values);
    }
    GGML_ASSERT(first.context->alloc_calls == 2 && second.context->alloc_calls == 2);
    reused_owner.reset();
    GGML_ASSERT(first.context->free_calls == 2 && second.context->free_calls == 2);
    GGML_ASSERT(first.context->buffers.empty() && second.context->buffers.empty());
}

static void test_graph_measurement_binding() {
    for (bool placeholder : {false, true}) {
        auto backend = dummy_backend_init(64);
        auto test_ctx = make_context();
        auto * input = make_input_1d(test_ctx.ctx, 8);
        ggml_backend_buffer_ptr assigned;
        if (placeholder) {
            assigned.reset(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 0));
            input->buffer = assigned.get();
        }
        auto * output = ggml_scale(test_ctx.ctx, input, 2.0f);
        ggml_set_output(output);
        ggml_build_forward_expand(test_ctx.graph, output);
        ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
        std::array<uint8_t, GGML_TENSOR_SIZE> before;
        std::memcpy(before.data(), input, GGML_TENSOR_SIZE);
        size_t size = 0;
        ggml_gallocr_reserve_n_size(galloc.get(), test_ctx.graph, nullptr, nullptr, &size);
        GGML_ASSERT(size == 32 && backend.context->alloc_calls == 0);
        GGML_ASSERT(std::memcmp(before.data(), input, GGML_TENSOR_SIZE) == 0);
        GGML_ASSERT(!ggml_backend_tensor_is_bound(input) && !ggml_backend_tensor_is_bound(output));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), test_ctx.graph) == !placeholder);
        if (placeholder) {
            GGML_ASSERT(input->buffer == assigned.get() && input->data == nullptr);
            GGML_ASSERT(output->buffer == nullptr && output->data == nullptr);
        } else {
            GGML_ASSERT(ggml_backend_tensor_is_bound(input) && ggml_backend_tensor_is_bound(output));
        }
        galloc.reset();
        GGML_ASSERT(backend.context->buffers.empty());
    }
}

static void test_graph_init_failure() {
    for (size_t fail_at : {size_t(1), size_t(2)}) {
        auto backend = dummy_backend_init(64);
        backend.context->fail_init = fail_at;
        auto test_ctx = make_context();
        auto * input = make_input_1d(test_ctx.ctx, 8);
        auto * output = ggml_scale(test_ctx.ctx, input, 2.0f);
        ggml_set_output(output);
        ggml_build_forward_expand(test_ctx.graph, output);
        ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(!ggml_gallocr_alloc_graph(galloc.get(), test_ctx.graph));
        GGML_ASSERT(backend.context->init_calls == fail_at);
        galloc.reset();
        GGML_ASSERT(backend.context->buffers.empty() && backend.context->free_calls == backend.context->alloc_calls);
    }
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    run("test_graph_measurement_binding", test_graph_measurement_binding);
    run("test_graph_init_failure", test_graph_init_failure);
    run("test_meta_materialization", []() { test_meta_materialization(false); });
    run("test_meta_materialization_same_base", []() { test_meta_materialization(true); });
    run("test_meta_split_preparation", test_meta_split_preparation);
    run("test_context_buffer_sets", test_context_buffer_sets);
    run("test_context_buffer_set_failures", test_context_buffer_set_failures);
    run("test_context_buffer_set_borrowed_view", test_context_buffer_set_borrowed_view);
    run("test_meta_buffer_type_lifetime", test_meta_buffer_type_lifetime);
    run("test_tensor_binding_state", test_tensor_binding_state);
    run("test_addressless_views_and_transfers", test_addressless_views_and_transfers);
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    return 0;
}
