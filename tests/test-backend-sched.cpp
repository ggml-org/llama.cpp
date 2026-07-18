#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <ggml-backend-impl.h>
#include <ggml-impl.h>

#include <cstdlib>
#include <cstring>
#include <vector>

struct mock_context {
    int allocations = 0;
    int frees = 0;
    int transfers = 0;
    int computes = 0;
    int synchronizes = 0;
    int fail_alloc_after = -1;
    bool fail_attach = false;
    int fail_attach_after = -1;
    bool fail_compute = false;
    size_t free_memory = 1u << 30;
    size_t total_memory = 1u << 30;
    std::vector<ggml_tensor *> transferred;
    std::vector<ggml_tensor *> computed_transients;
};

struct mock_buffer_context {
    mock_context * owner;
    void * data;
};

static const char * mock_buft_name(ggml_backend_buffer_type_t) { return "mock-device"; }
static size_t mock_buft_alignment(ggml_backend_buffer_type_t) { return 16; }
static size_t mock_buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) { return ggml_nbytes(tensor) + 16; }
static bool mock_buft_is_host(ggml_backend_buffer_type_t) { return false; }
static void mock_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (mock_buffer_context *) buffer->context;
    ctx->owner->frees++;
    free(ctx->data);
    delete ctx;
}
static void * mock_buffer_base(ggml_backend_buffer_t buffer) { return ((mock_buffer_context *) buffer->context)->data; }
static ggml_status mock_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *) {
    auto * owner = ((mock_buffer_context *) buffer->context)->owner;
    if (owner->fail_attach || owner->fail_attach_after == 0) {
        return GGML_STATUS_FAILED;
    }
    if (owner->fail_attach_after > 0) {
        owner->fail_attach_after--;
    }
    return GGML_STATUS_SUCCESS;
}
static void mock_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (mock_buffer_context *) buffer->context;
    ctx->owner->transfers++;
    ctx->owner->transferred.push_back(tensor);
    memcpy((char *) tensor->data + offset, data, size);
}
static void mock_buffer_get(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}
static void mock_buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}
static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(mock_buffer_base(buffer), value, buffer->size);
}
static ggml_backend_buffer_t mock_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * owner = (mock_context *) buft->context;
    if (owner->fail_alloc_after == 0) {
        return nullptr;
    }
    if (owner->fail_alloc_after > 0) {
        owner->fail_alloc_after--;
    }
    auto * ctx = new mock_buffer_context{owner, malloc(size)};
    if (ctx->data == nullptr) {
        delete ctx;
        return nullptr;
    }
    owner->allocations++;
    ggml_backend_buffer_i iface{};
    iface.free_buffer = mock_buffer_free;
    iface.get_base = mock_buffer_base;
    iface.init_tensor = mock_buffer_init_tensor;
    iface.set_tensor = mock_buffer_set;
    iface.get_tensor = mock_buffer_get;
    iface.memset_tensor = mock_buffer_memset;
    iface.clear = mock_buffer_clear;
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

static const char * mock_backend_name(ggml_backend_t) { return "mock"; }
static void mock_set_async(ggml_backend_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    mock_buffer_set(tensor->buffer, tensor, data, offset, size);
}
static void mock_synchronize(ggml_backend_t backend) { ((mock_context *) backend->context)->synchronizes++; }
static ggml_status mock_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = (mock_context *) backend->context;
    ctx->computes++;
    for (int i = 0; i < graph->n_nodes; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = graph->nodes[i]->src[j];
            if (src != nullptr && (src->flags & GGML_TENSOR_FLAG_NO_ALLOC)) {
                GGML_ASSERT(src->buffer != nullptr && src->data != nullptr);
                ctx->computed_transients.push_back(src);
            }
        }
    }
    return ctx->fail_compute ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;
}
static const char * mock_dev_name(ggml_backend_dev_t) { return "mock-device"; }
static const char * mock_dev_desc(ggml_backend_dev_t) { return "deterministic scheduler mock"; }
static void mock_dev_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (mock_context *) dev->context;
    *free = ctx->free_memory;
    *total = ctx->total_memory;
}
static enum ggml_backend_dev_type mock_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void mock_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = { mock_dev_name(dev), mock_dev_desc(dev), 1u << 30, 1u << 30, GGML_BACKEND_DEVICE_TYPE_GPU, nullptr, {} };
}
static bool mock_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool mock_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { return buft->device == dev; }
static bool mock_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

struct mock_backend {
    mock_context ctx;
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    ggml_backend backend{};

    mock_backend() {
        device.context = &ctx;
        device.iface.get_name = mock_dev_name;
        device.iface.get_description = mock_dev_desc;
        device.iface.get_memory = mock_dev_memory;
        device.iface.get_type = mock_dev_type;
        device.iface.get_props = mock_dev_props;
        device.iface.supports_op = mock_supports_op;
        device.iface.supports_buft = mock_supports_buft;
        device.iface.offload_op = mock_offload_op;
        buft.device = &device;
        buft.context = &ctx;
        buft.iface.get_name = mock_buft_name;
        buft.iface.alloc_buffer = mock_buft_alloc;
        buft.iface.get_alignment = mock_buft_alignment;
        buft.iface.get_alloc_size = mock_buft_alloc_size;
        buft.iface.is_host = mock_buft_is_host;
        backend.device = &device;
        backend.context = &ctx;
        backend.iface.get_name = mock_backend_name;
        backend.iface.set_tensor_async = mock_set_async;
        backend.iface.synchronize = mock_synchronize;
        backend.iface.graph_compute = mock_compute;
    }
};

struct test_env {
    mock_backend mock;
    ggml_backend_t cpu = nullptr;
    ggml_context_ptr weight_ctx;
    ggml_backend_buffer_t weights = nullptr;

    test_env(size_t tensor_count) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        GGML_ASSERT(cpu != nullptr);
        ggml_init_params params{ tensor_count * ggml_tensor_overhead(), nullptr, true };
        weight_ctx.reset(ggml_init(params));
    }

    ggml_tensor * weight_1d(int64_t ne) {
        return ggml_new_tensor_1d(weight_ctx.get(), GGML_TYPE_F32, ne);
    }

    void allocate_weights() {
        weights = ggml_backend_alloc_ctx_tensors(weight_ctx.get(), cpu);
        GGML_ASSERT(weights != nullptr);
        ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    ~test_env() {
        ggml_backend_buffer_free(weights);
        ggml_backend_free(cpu);
    }
};

static ggml_backend_sched_t make_sched(test_env & env, bool parallel = false) {
    ggml_backend_t backends[] = { &env.mock.backend, env.cpu };
    ggml_backend_buffer_type_t bufts[] = { &env.mock.buft, ggml_backend_get_default_buffer_type(env.cpu) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 64, parallel, true);
    ggml_backend_sched_set_force_weight_offload(sched, true);
    return sched;
}

static void assert_detached(const mock_context & ctx) {
    for (ggml_tensor * tensor : ctx.computed_transients) {
        GGML_ASSERT(tensor->buffer == nullptr && tensor->data == nullptr);
    }
}

static int count_transfers(const mock_context & ctx, ggml_tensor * tensor) {
    int result = 0;
    for (ggml_tensor * transferred : ctx.transferred) {
        result += transferred == tensor;
    }
    return result;
}

static ggml_backend_sched_transient_metrics get_metrics(ggml_backend_sched_t sched) {
    ggml_backend_sched_transient_metrics metrics{};
    GGML_ASSERT(ggml_backend_sched_get_transient_metrics(sched, &metrics));
    return metrics;
}

static void enable_residency(ggml_backend_sched_t sched, test_env & env, size_t resident_bytes) {
    constexpr size_t MIB = 1024u * 1024u;
    env.mock.ctx.free_memory = 512 * MIB + resident_bytes;
    env.mock.ctx.total_memory = 1024 * MIB;
    size_t window = 0;
    size_t reserve = 0;
    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, env.mock.ctx.free_memory, env.mock.ctx.total_memory, SIZE_MAX, &window, &reserve));
    GGML_ASSERT(window == resident_bytes && reserve == 512 * MIB);
    ggml_backend_sched_set_weight_residency(sched, &env.mock.backend, true);
}

static void assert_manual_ledgers_zero(ggml_backend_sched_t sched) {
    const auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.current_resident_bytes == 0 && metrics.current_resident_records == 0);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);
}

static void assert_zero_metrics(ggml_backend_sched_t sched) {
    const auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.peak_transient_bytes == 0);
    GGML_ASSERT(metrics.current_transient_records == 0 && metrics.peak_transient_records == 0);
    GGML_ASSERT(metrics.graph_compute_count == 0 && metrics.graph_compute_failure_count == 0);
    GGML_ASSERT(metrics.callback_early_stop_count == 0 && metrics.ledger_mismatch_count == 0 && metrics.counter_overflow_count == 0);
    for (int b = 0; b < metrics.n_backends; ++b) {
        const auto & row = metrics.backends[b];
        GGML_ASSERT(row.allocation_requested_bytes == 0 && row.allocation_admitted_bytes == 0 && row.allocation_rejected_bytes == 0);
        GGML_ASSERT(row.allocation_count == 0 && row.upload_count == 0 && row.splits_seen_count == 0 && row.transient_split_count == 0);
    }
}

static bool stop_after_first_callback(ggml_tensor *, bool ask, void *) {
    return ask;
}

static void test_transient_success_and_failures() {
    test_env env(4);
    ggml_tensor * weight_a = env.weight_1d(16);
    ggml_tensor * weight_b = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_add(graph_ctx.get(), weight_a, weight_b);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    const int persistent = env.mock.ctx.allocations - env.mock.ctx.frees;

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.transfers == 2);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    auto metrics = get_metrics(sched);
    const auto & first = metrics.backends[0];
    GGML_ASSERT(metrics.graph_compute_count == 1 && metrics.graph_compute_failure_count == 0);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);
    GGML_ASSERT(metrics.peak_transient_bytes == 160 && metrics.peak_transient_records == 2);
    GGML_ASSERT(first.allocation_requested_bytes == 160 && first.allocation_admitted_bytes == 160 && first.allocation_rejected_bytes == 0);
    GGML_ASSERT(first.allocation_count == 2 && first.upload_count == 2);
    GGML_ASSERT(first.uploaded_logical_bytes == 128 && first.uploaded_backend_bytes == 160);
    GGML_ASSERT(first.transfer_completion_wait_count == 1 && first.compute_completion_wait_count == 1);
    GGML_ASSERT(first.transient_split_count == 1 && first.drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_NORMAL] == 1);

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.transfers == 4);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_count == 2 && metrics.peak_transient_bytes == 160);
    GGML_ASSERT(metrics.backends[0].allocation_requested_bytes == 320 && metrics.backends[0].upload_count == 4);

    env.mock.ctx.fail_alloc_after = 1;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_failure_count == 1);
    GGML_ASSERT(metrics.backends[0].allocation_failure_count == 1);
    GGML_ASSERT(metrics.backends[0].allocation_rejected_bytes == 80);
    GGML_ASSERT(metrics.backends[0].drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ALLOCATION_FAILURE] == 1);
    env.mock.ctx.fail_alloc_after = -1;

    env.mock.ctx.fail_attach = true;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_FAILED);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_failure_count == 2);
    GGML_ASSERT(metrics.backends[0].drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ATTACHMENT_FAILURE] == 1);
    env.mock.ctx.fail_attach = false;

    ggml_backend_sched_set_eval_callback(sched, stop_after_first_callback, nullptr);
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.callback_early_stop_count == 1);
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);

    env.mock.ctx.fail_compute = true;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_FAILED);
    assert_detached(env.mock.ctx);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_failure_count == 3);
    GGML_ASSERT(metrics.backends[0].drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_COMPUTE_FAILURE] == 1);
    GGML_ASSERT(metrics.ledger_mismatch_count == 0 && metrics.counter_overflow_count == 0);
    env.mock.ctx.fail_compute = false;

    ggml_backend_sched_reset(sched);
    assert_detached(env.mock.ctx);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_two_device_splits_recopy_tied_weight() {
    test_env env(4);
    ggml_tensor * tied = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * first = ggml_add(graph_ctx.get(), tied, tied);
    ggml_tensor * cpu_barrier = ggml_sqr(graph_ctx.get(), first);
    ggml_tensor * second = ggml_add(graph_ctx.get(), tied, cpu_barrier);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, second);

    ggml_backend_sched_t sched = make_sched(env);
    ggml_backend_sched_set_tensor_backend(sched, first, &env.mock.backend);
    ggml_backend_sched_set_tensor_backend(sched, cpu_barrier, env.cpu);
    ggml_backend_sched_set_tensor_backend(sched, second, &env.mock.backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == 3);

    ggml_tensor * tied_copy = first->src[0];
    GGML_ASSERT(tied_copy == first->src[1]);
    GGML_ASSERT(tied_copy == second->src[0]);
    GGML_ASSERT(tied_copy != tied && (tied_copy->flags & GGML_TENSOR_FLAG_NO_ALLOC));
    const int persistent = env.mock.ctx.allocations - env.mock.ctx.frees;

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(count_transfers(env.mock.ctx, tied_copy) == 2);
    GGML_ASSERT(tied_copy->buffer == nullptr && tied_copy->data == nullptr);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_count == 1);
    GGML_ASSERT(metrics.backends[0].shared_reload_count == 1);
    GGML_ASSERT(metrics.backends[0].transient_split_count == 2);
    GGML_ASSERT(metrics.peak_transient_bytes == 80 && metrics.peak_transient_records == 1);
    GGML_ASSERT(metrics.backends[0].drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_NORMAL] == 2);

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(count_transfers(env.mock.ctx, tied_copy) == 4);
    GGML_ASSERT(tied_copy->buffer == nullptr && tied_copy->data == nullptr);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_count == 2 && metrics.backends[0].shared_reload_count == 2);
    GGML_ASSERT(metrics.backends[0].upload_count == 4);

    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_parallel_fails_closed_to_ordinary_ownership() {
    test_env env(2);
    ggml_tensor * weight = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_sqr(graph_ctx.get(), weight);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env, true);
    GGML_ASSERT(ggml_backend_sched_get_n_copies(sched) > 1);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_tensor * copy = node->src[0];
    GGML_ASSERT(copy != weight);
    GGML_ASSERT((copy->flags & GGML_TENSOR_FLAG_NO_ALLOC) == 0);
    GGML_ASSERT(copy->buffer != nullptr && copy->data != nullptr);
    const int persistent = env.mock.ctx.allocations - env.mock.ctx.frees;

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(copy->buffer != nullptr && copy->data != nullptr);
    GGML_ASSERT(env.mock.ctx.allocations - env.mock.ctx.frees == persistent);
    assert_zero_metrics(sched);

    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_view_fails_closed_to_ordinary_ownership() {
    test_env env(4);
    ggml_tensor * weight = env.weight_1d(16);
    ggml_tensor * view = ggml_view_1d(env.weight_ctx.get(), weight, 8, 0);
    env.allocate_weights();

    ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_sqr(graph_ctx.get(), view);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_tensor * copy = node->src[0];
    GGML_ASSERT(copy != view);
    GGML_ASSERT((copy->flags & GGML_TENSOR_FLAG_NO_ALLOC) == 0);
    GGML_ASSERT(copy->buffer != nullptr && copy->data != nullptr);
    assert_zero_metrics(sched);

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(copy->buffer != nullptr && copy->data != nullptr);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_disabled_offload_has_zero_metrics() {
    test_env env(2);
    ggml_tensor * weight = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_sqr(graph_ctx.get(), weight);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    ggml_backend_sched_set_force_weight_offload(sched, false);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    assert_zero_metrics(sched);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_limit_rejections() {
    test_env env(4);
    ggml_tensor * weight_a = env.weight_1d(16);
    ggml_tensor * weight_b = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_add(graph_ctx.get(), weight_a, weight_b);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    ggml_backend_sched_set_max_weight_bytes_per_split(sched, 79);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].allocation_limit_rejection_count == 1);
    GGML_ASSERT(metrics.backends[0].oversized_tensor_rejection_count == 1);
    GGML_ASSERT(metrics.backends[0].allocation_rejected_bytes == 80 && metrics.current_transient_records == 0);
    ggml_backend_sched_set_max_weight_bytes_per_split(sched, 120);
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].allocation_limit_rejection_count == 2);
    GGML_ASSERT(metrics.backends[0].oversized_tensor_rejection_count == 1);
    GGML_ASSERT(metrics.backends[0].allocation_admitted_bytes == 80 && metrics.backends[0].allocation_rejected_bytes == 160);
    GGML_ASSERT(metrics.backends[0].drain_count[GGML_BACKEND_SCHED_TRANSIENT_DRAIN_ALLOCATION_FAILURE] == 1);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_weight_window_admission() {
    constexpr size_t MIB = 1024u * 1024u;
    test_env env(4);
    ggml_tensor * weight_a = env.weight_1d(16);
    ggml_tensor * weight_b = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_add(graph_ctx.get(), weight_a, weight_b);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    size_t window = 0;
    size_t reserve = 0;
    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 900 * MIB, 1000 * MIB, SIZE_MAX, &window, &reserve));
    GGML_ASSERT(reserve == 512 * MIB && window == 388 * MIB);
    auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].weight_window_configured && metrics.backends[0].weight_window_memory_valid);
    GGML_ASSERT(metrics.backends[0].weight_window_limit_bytes == window);

    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 8u * 1024u * MIB, 16u * 1024u * MIB, 1024 * MIB, &window, &reserve));
    GGML_ASSERT(reserve == ((size_t) 16 * 1024 * MIB) / 10 + 1 && window == 1024 * MIB);

    env.mock.ctx.free_memory = 512 * MIB + 160;
    env.mock.ctx.total_memory = 1024 * MIB;
    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 512 * MIB + 160, 1024 * MIB, SIZE_MAX, &window, &reserve));
    GGML_ASSERT(window == 160 && reserve == 512 * MIB);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    const int allocations_before_exact = env.mock.ctx.allocations;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.allocations == allocations_before_exact + 2);
    assert_detached(env.mock.ctx);

    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 512 * MIB + 79, 1024 * MIB, SIZE_MAX, &window, &reserve));
    GGML_ASSERT(window == 79);
    const int allocations_before_window_rejection = env.mock.ctx.allocations;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(env.mock.ctx.allocations == allocations_before_window_rejection);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].allocation_limit_rejection_count == 1);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);

    env.mock.ctx.free_memory = 512 * MIB + 160;
    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 512 * MIB + 160, 1024 * MIB, SIZE_MAX, &window, &reserve));
    env.mock.ctx.free_memory = 512 * MIB + 79;
    const int allocations_before_live_rejection = env.mock.ctx.allocations;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(env.mock.ctx.allocations == allocations_before_live_rejection);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].allocation_live_guard_rejection_count == 1);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);

    env.mock.ctx.free_memory = 0;
    env.mock.ctx.total_memory = 0;
    GGML_ASSERT(!ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, 0, 0, SIZE_MAX, &window, &reserve));
    GGML_ASSERT(window == 0 && reserve == 512 * MIB);
    const int allocations_before_unknown = env.mock.ctx.allocations;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(env.mock.ctx.allocations == allocations_before_unknown);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].allocation_unknown_memory_rejection_count == 1);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);

    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_persistent_residency_reuse_and_drain() {
    test_env env(4);
    ggml_tensor * weight_a = env.weight_1d(16);
    ggml_tensor * weight_b = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_add(graph_ctx.get(), weight_a, weight_b);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    enable_residency(sched, env, 160);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 2);
    auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.current_resident_bytes == 160 && metrics.current_resident_records == 2);
    GGML_ASSERT(metrics.backends[0].residency_miss_count == 2);
    GGML_ASSERT(metrics.backends[0].residency_upload_count == 2);
    GGML_ASSERT(metrics.backends[0].peak_manually_owned_bytes == 160);

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 2);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].residency_hit_count == 2);
    GGML_ASSERT(metrics.backends[0].residency_upload_count == 2);

    ggml_backend_sched_set_weight_residency(sched, &env.mock.backend, false);
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.current_resident_bytes == 0 && metrics.current_resident_records == 0);
    GGML_ASSERT(metrics.current_transient_bytes == 0 && metrics.current_transient_records == 0);
    GGML_ASSERT(node->src[0]->buffer == nullptr && node->src[0]->data == nullptr);
    GGML_ASSERT(node->src[1]->buffer == nullptr && node->src[1]->data == nullptr);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_warm_resident_callback_and_compute_failure() {
    test_env env(2);
    ggml_tensor * weight = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_sqr(graph_ctx.get(), weight);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    enable_residency(sched, env, 80);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_tensor * copy = node->src[0];

    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 1 && copy->buffer != nullptr && copy->data != nullptr);

    ggml_backend_sched_set_eval_callback(sched, stop_after_first_callback, nullptr);
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.callback_early_stop_count == 1);
    GGML_ASSERT(metrics.backends[0].residency_hit_count == 1);
    GGML_ASSERT(metrics.backends[0].residency_upload_count == 1 && env.mock.ctx.transfers == 1);
    GGML_ASSERT(metrics.current_resident_records == 1 && copy->buffer != nullptr && copy->data != nullptr);
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);

    env.mock.ctx.fail_compute = true;
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_FAILED);
    env.mock.ctx.fail_compute = false;
    metrics = get_metrics(sched);
    GGML_ASSERT(metrics.graph_compute_failure_count == 1);
    GGML_ASSERT(metrics.backends[0].residency_hit_count == 2);
    GGML_ASSERT(metrics.backends[0].residency_upload_count == 1 && env.mock.ctx.transfers == 1);
    GGML_ASSERT(copy->buffer == nullptr && copy->data == nullptr);
    assert_manual_ledgers_zero(sched);

    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_resident_admission_failures_drain_existing() {
    for (int failure = 0; failure < 2; ++failure) {
        test_env env(6);
        ggml_tensor * weight_a = env.weight_1d(16);
        ggml_tensor * weight_b = env.weight_1d(16);
        env.allocate_weights();

        ggml_init_params graph_params{ 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr graph_ctx(ggml_init(graph_params));
        ggml_tensor * first = ggml_sqr(graph_ctx.get(), weight_a);
        ggml_tensor * barrier = ggml_sqr(graph_ctx.get(), first);
        ggml_tensor * second = ggml_add(graph_ctx.get(), weight_b, barrier);
        ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
        ggml_build_forward_expand(graph, second);

        ggml_backend_sched_t sched = make_sched(env);
        ggml_backend_sched_set_tensor_backend(sched, first, &env.mock.backend);
        ggml_backend_sched_set_tensor_backend(sched, barrier, env.cpu);
        ggml_backend_sched_set_tensor_backend(sched, second, &env.mock.backend);
        enable_residency(sched, env, 160);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == 3);
        ggml_tensor * copy_a = first->src[0];
        ggml_tensor * copy_b = second->src[0];

        if (failure == 0) {
            env.mock.ctx.fail_alloc_after = 1;
            GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ALLOC_FAILED);
        } else {
            env.mock.ctx.fail_attach_after = 1;
            GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_FAILED);
        }
        GGML_ASSERT(copy_a->buffer == nullptr && copy_a->data == nullptr);
        GGML_ASSERT(copy_b->buffer == nullptr && copy_b->data == nullptr);
        assert_manual_ledgers_zero(sched);
        const auto metrics = get_metrics(sched);
        GGML_ASSERT(metrics.graph_compute_failure_count == 1);
        GGML_ASSERT(metrics.backends[0].residency_miss_count >= 1);
        GGML_ASSERT(metrics.backends[0].residency_hit_count == 0);

        ggml_backend_sched_free(sched);
        GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
    }
}

static void test_resident_graph_rebuild_and_source_mismatch() {
    test_env env(2);
    ggml_tensor * weight = env.weight_1d(16);
    env.allocate_weights();

    ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * node = ggml_sqr(graph_ctx.get(), weight);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, node);

    ggml_backend_sched_t sched = make_sched(env);
    enable_residency(sched, env, 80);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_tensor * old_copy = node->src[0];
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 1 && old_copy->buffer != nullptr);

    ggml_init_params rebuilt_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr rebuilt_ctx(ggml_init(rebuilt_params));
    ggml_tensor * rebuilt_node = ggml_sqr(rebuilt_ctx.get(), weight);
    ggml_cgraph * rebuilt_graph = ggml_new_graph(rebuilt_ctx.get());
    ggml_build_forward_expand(rebuilt_graph, rebuilt_node);
    ggml_backend_sched_set_tensor_backend(sched, rebuilt_node, &env.mock.backend);
    const int frees_before_rebuild = env.mock.ctx.frees;
    GGML_ASSERT(ggml_backend_sched_reserve(sched, rebuilt_graph));
    GGML_ASSERT(env.mock.ctx.frees == frees_before_rebuild + 1);
    assert_manual_ledgers_zero(sched);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);

    ggml_init_params mismatch_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr mismatch_ctx(ggml_init(mismatch_params));
    ggml_tensor * mismatch_node = ggml_sqr(mismatch_ctx.get(), weight);
    ggml_cgraph * mismatch_graph = ggml_new_graph(mismatch_ctx.get());
    ggml_build_forward_expand(mismatch_graph, mismatch_node);
    ggml_backend_sched_t mismatch_sched = make_sched(env);
    enable_residency(mismatch_sched, env, 80);
    ggml_backend_sched_set_tensor_backend(mismatch_sched, mismatch_node, &env.mock.backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(mismatch_sched, mismatch_graph));
    ggml_tensor * rebuilt_copy = mismatch_node->src[0];
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(mismatch_sched, mismatch_graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 2 && rebuilt_copy->buffer != nullptr);

    std::vector<uint8_t> alternate_source(ggml_nbytes(weight), 0x5a);
    void * original_data = weight->data;
    weight->data = alternate_source.data();
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(mismatch_sched, mismatch_graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(env.mock.ctx.transfers == 3);
    const auto metrics = get_metrics(mismatch_sched);
    GGML_ASSERT(metrics.backends[0].residency_miss_count == 2);
    GGML_ASSERT(metrics.backends[0].residency_hit_count == 0);
    GGML_ASSERT(metrics.backends[0].residency_upload_count == 2);
    GGML_ASSERT(metrics.current_resident_records == 1);
    weight->data = original_data;

    ggml_backend_sched_set_weight_residency(mismatch_sched, &env.mock.backend, false);
    GGML_ASSERT(rebuilt_copy->buffer == nullptr && rebuilt_copy->data == nullptr);
    assert_manual_ledgers_zero(mismatch_sched);
    ggml_backend_sched_free(mismatch_sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

static void test_warm_resident_reset_and_direct_destruction() {
    for (int lifecycle = 0; lifecycle < 2; ++lifecycle) {
        test_env env(2);
        ggml_tensor * weight = env.weight_1d(16);
        env.allocate_weights();

        ggml_init_params graph_params{ 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr graph_ctx(ggml_init(graph_params));
        ggml_tensor * node = ggml_sqr(graph_ctx.get(), weight);
        ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
        ggml_build_forward_expand(graph, node);

        ggml_backend_sched_t sched = make_sched(env);
        enable_residency(sched, env, 80);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        ggml_tensor * copy = node->src[0];
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(copy->buffer != nullptr && get_metrics(sched).current_resident_records == 1);

        if (lifecycle == 0) {
            ggml_backend_sched_reset(sched);
            GGML_ASSERT(copy->buffer == nullptr && copy->data == nullptr);
            assert_manual_ledgers_zero(sched);
            ggml_backend_sched_free(sched);
        } else {
            ggml_backend_sched_free(sched);
        }
        GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
    }
}

static void test_persistent_residency_lru_eviction() {
    constexpr size_t MIB = 1024u * 1024u;
    test_env env(6);
    ggml_tensor * weight_a = env.weight_1d(16);
    ggml_tensor * weight_b = env.weight_1d(16);
    env.allocate_weights();
    ggml_init_params graph_params{ 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    ggml_tensor * first = ggml_sqr(graph_ctx.get(), weight_a);
    ggml_tensor * barrier = ggml_sqr(graph_ctx.get(), first);
    ggml_tensor * second = ggml_add(graph_ctx.get(), weight_b, barrier);
    ggml_cgraph * graph = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(graph, second);

    ggml_backend_sched_t sched = make_sched(env);
    ggml_backend_sched_set_tensor_backend(sched, first, &env.mock.backend);
    ggml_backend_sched_set_tensor_backend(sched, barrier, env.cpu);
    ggml_backend_sched_set_tensor_backend(sched, second, &env.mock.backend);
    env.mock.ctx.free_memory = 512 * MIB + 80;
    env.mock.ctx.total_memory = 1024 * MIB;
    size_t window = 0;
    size_t reserve = 0;
    GGML_ASSERT(ggml_backend_sched_set_weight_window(
        sched, &env.mock.backend, env.mock.ctx.free_memory, env.mock.ctx.total_memory, SIZE_MAX, &window, &reserve));
    ggml_backend_sched_set_weight_residency(sched, &env.mock.backend, true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    const auto metrics = get_metrics(sched);
    GGML_ASSERT(metrics.backends[0].residency_miss_count == 2);
    GGML_ASSERT(metrics.backends[0].residency_eviction_count == 1);
    GGML_ASSERT(metrics.current_resident_bytes == 80 && metrics.backends[0].peak_manually_owned_bytes == 80);
    ggml_backend_sched_reset(sched);
    GGML_ASSERT(get_metrics(sched).current_resident_bytes == 0);
    ggml_backend_sched_free(sched);
    GGML_ASSERT(env.mock.ctx.allocations == env.mock.ctx.frees);
}

int main() {
    test_transient_success_and_failures();
    test_two_device_splits_recopy_tied_weight();
    test_parallel_fails_closed_to_ordinary_ownership();
    test_view_fails_closed_to_ordinary_ownership();
    test_disabled_offload_has_zero_metrics();
    test_limit_rejections();
    test_weight_window_admission();
    test_persistent_residency_reuse_and_drain();
    test_warm_resident_callback_and_compute_failure();
    test_resident_admission_failures_drain_existing();
    test_resident_graph_rebuild_and_source_mismatch();
    test_warm_resident_reset_and_direct_destruction();
    test_persistent_residency_lru_eviction();
    return 0;
}
