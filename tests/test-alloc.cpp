#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
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

    bool custom_alloc_buffer_n_called   = false;
    bool custom_get_alloc_size_n_called = false;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
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

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    ctx->custom_alloc_buffer_n_called = true;

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 64);
    if (buffer == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < n_tensors; i++) {
        tensors[i]->buffer = buffer;
        tensors[i]->data   = (char *) ggml_backend_buffer_get_base(buffer);
    }
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alloc_size_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    GGML_UNUSED(tensors);
    GGML_UNUSED(n_tensors);
    ctx->custom_get_alloc_size_n_called = true;
    return 64;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t /*buft*/) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t /*buft*/, const ggml_tensor * /*x*/, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t /*buft*/, uint8_t /*val*/) {}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor * /*x*/) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t /*backend*/) {
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
        auto * buft     = ggml_backend_buffer_get_type(t->buffer);
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

static void test_backend_graph_optimize(ggml_backend_t /*backend*/, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
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

// Check that the size reported by ggml_backend_alloc_ctx_tensors_from_buft_size
// matches the actual size of the buffer allocated for the ctx tensors
static ggml_backend_buffer_ptr check_size_matches(ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors.push_back(t);
    }

    const size_t expected_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(buft, tensors.data(), (int) tensors.size()) == expected_size);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    GGML_ASSERT((buffer != nullptr) == (expected_size != 0));
    if (buffer) {
        GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == expected_size);
    }
    return buffer;
}

// Check that all tensors are placed in a single buffer when they fit within the
// backend's max size
static void test_buft_alloc_buffer_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { x[0], x[1], x[2] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(!ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer == buffer.get());
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that tensors are split across multiple underlying buffers when they don't
// fit within the backend's max size
static void test_buft_alloc_buffer_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that allocating tensors with zero total size returns nullptr
static void test_buft_alloc_buffer_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 0);
}

// Check that tensors already allocated to a buffer are left in place, and only the
// remaining tensors are allocated to a new buffer
static void test_buft_alloc_buffer_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 8);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, a, 2) == nullptr);

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
}

// Check that views don't require any extra memory and share the base tensor's data
static void test_buft_alloc_buffer_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    GGML_ASSERT(base->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(extra->buffer == buffer.get());
    GGML_ASSERT(view->data == base->data);
}

// Check that the reported size matches the allocated size in various scenarios:
// single buffer, multi buffer, and views
static void test_alloc_ctx_tensors_from_buft_size_matches() {
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX, 8);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 4);
        x[1] = make_input_with_size(ctx, 4);
        x[2] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
    {
        dummy_backend backend      = dummy_backend_init(16);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 8);
        x[1] = make_input_with_size(ctx, 8);
        x[2] = make_input_with_size(ctx, 8);
        x[3] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 32);
    }
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * base  = make_input_1d(ctx, 4);
        ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
        ggml_tensor * extra = make_input_1d(ctx, 2);
        assign_names(ctx);

        GGML_UNUSED(view);
        GGML_UNUSED(extra);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
}

// Check that a backend-provided alloc_buffer_n implementation takes precedence over
// the default one
static void test_buft_alloc_buffer_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n = dummy_backend_buffer_type_alloc_buffer_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
    GGML_ASSERT(x[0]->buffer == buffer.get());
    GGML_ASSERT(x[1]->buffer == buffer.get());
}

// Check that get_alloc_size_n predicts a single buffer allocation
static void test_buft_get_alloc_size_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that get_alloc_size_n accounts for splitting into multiple buffers
static void test_buft_get_alloc_size_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 4) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a tensor larger than max_size is still accounted for
static void test_buft_get_alloc_size_n_single_tensor_exceeds_max() {
    dummy_backend backend      = dummy_backend_init(8);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x = make_input_with_size(ctx, 16);
    assign_names(ctx);

    ggml_tensor * tensors[1] = { x };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 1) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 1));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that zero-size tensors report a total allocation size of 0
static void test_buft_get_alloc_size_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
}

// Check that already-allocated tensors are not counted again
static void test_buft_get_alloc_size_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 8);

    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
}

// Check that views do not add to the projected allocation size
static void test_buft_get_alloc_size_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 24);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
}

// Check that n_tensors == 0 is handled
static void test_buft_get_alloc_size_n_n_tensors_zero() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, nullptr, 0) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, nullptr, 0) == nullptr);
}

// Check that get_alloc_size_n respects the backend alignment
static void test_buft_get_alloc_size_n_alignment() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX, 16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 4);
    x[1] = make_input_with_size(ctx, 4);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a backend-provided get_alloc_size_n implementation takes precedence over
// the default one
static void test_buft_get_alloc_size_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n   = dummy_backend_buffer_type_alloc_buffer_n_custom;
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 64);
    GGML_ASSERT(backend.context->custom_get_alloc_size_n_called);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
}

// Check that the custom get_alloc_size_n override is used by the context size helper
static void test_buft_get_alloc_size_n_custom_override_ctx_size() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    GGML_UNUSED(x);

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 64);
}

// Check that querying the projected allocation size does not allocate anything
static void test_buft_get_alloc_size_n_does_not_allocate() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);
    GGML_ASSERT(backend.context->buffers.empty());
    GGML_ASSERT(x[0]->data == nullptr);
    GGML_ASSERT(x[1]->data == nullptr);
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
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
    run("test_buft_alloc_buffer_n_single_buffer", test_buft_alloc_buffer_n_single_buffer);
    run("test_buft_alloc_buffer_n_multi_buffer", test_buft_alloc_buffer_n_multi_buffer);
    run("test_buft_alloc_buffer_n_zero_size", test_buft_alloc_buffer_n_zero_size);
    run("test_buft_alloc_buffer_n_already_allocated", test_buft_alloc_buffer_n_already_allocated);
    run("test_buft_alloc_buffer_n_views", test_buft_alloc_buffer_n_views);
    run("test_alloc_ctx_tensors_from_buft_size_matches", test_alloc_ctx_tensors_from_buft_size_matches);
    run("test_buft_alloc_buffer_n_custom_override", test_buft_alloc_buffer_n_custom_override);
    run("test_buft_get_alloc_size_n_single_buffer", test_buft_get_alloc_size_n_single_buffer);
    run("test_buft_get_alloc_size_n_multi_buffer", test_buft_get_alloc_size_n_multi_buffer);
    run("test_buft_get_alloc_size_n_single_tensor_exceeds_max", test_buft_get_alloc_size_n_single_tensor_exceeds_max);
    run("test_buft_get_alloc_size_n_zero_size", test_buft_get_alloc_size_n_zero_size);
    run("test_buft_get_alloc_size_n_already_allocated", test_buft_get_alloc_size_n_already_allocated);
    run("test_buft_get_alloc_size_n_views", test_buft_get_alloc_size_n_views);
    run("test_buft_get_alloc_size_n_n_tensors_zero", test_buft_get_alloc_size_n_n_tensors_zero);
    run("test_buft_get_alloc_size_n_alignment", test_buft_get_alloc_size_n_alignment);
    run("test_buft_get_alloc_size_n_custom_override", test_buft_get_alloc_size_n_custom_override);
    run("test_buft_get_alloc_size_n_custom_override_ctx_size", test_buft_get_alloc_size_n_custom_override_ctx_size);
    run("test_buft_get_alloc_size_n_does_not_allocate", test_buft_get_alloc_size_n_does_not_allocate);
    return 0;
}
