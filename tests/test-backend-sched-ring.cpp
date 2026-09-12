#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstring>
#include <functional>
#include <vector>

struct test_backend_context {
    int synchronize_count = 0;
    size_t completed = 0;
    std::vector<std::function<void()>> pending;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    std::vector<int> input_slots;
    std::vector<uint64_t> graph_uids;
};

struct test_event_context {
    test_backend_context * backend = nullptr;
    size_t position = 0;
};

static const char * test_backend_name(ggml_backend_t) {
    return "test";
}

static void test_backend_complete(test_backend_context & context, size_t position) {
    GGML_ASSERT(position <= context.pending.size());
    while (context.completed < position) {
        context.pending[context.completed++]();
    }
}

static void test_backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    context->synchronize_count++;
    test_backend_complete(*context, context->pending.size());
}

static ggml_status test_backend_graph_compute(ggml_backend_t, ggml_cgraph *) {
    return GGML_STATUS_SUCCESS;
}

static ggml_status test_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    context->input_slots.push_back(graph->input_slot);
    context->graph_uids.push_back(graph->uid);
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_VIEW) {
            continue;
        }
        GGML_ASSERT(node->op == GGML_OP_SCALE);
        GGML_ASSERT(node->type == GGML_TYPE_F32 && node->src[0]->type == GGML_TYPE_F32);
        const float * src = static_cast<const float *>(node->src[0]->data);
        float * dst = static_cast<float *>(node->data);
        const int64_t n = ggml_nelements(node);
        float scale;
        memcpy(&scale, node->op_params, sizeof(scale));
        context->pending.push_back([src, dst, n, scale]() {
            for (int64_t j = 0; j < n; j++) {
                dst[j] = src[j] * scale;
            }
        });
    }
    return GGML_STATUS_SUCCESS;
}

static void test_backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    const void * src = static_cast<const char *>(tensor->data) + offset;
    context->pending.push_back([src, data, size]() { memcpy(data, src, size); });
}

static bool test_backend_copy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto * context_src = static_cast<test_backend_context *>(backend_src->context);
    auto * context_dst = static_cast<test_backend_context *>(backend_dst->context);
    const void * data_src = src->data;
    void * data_dst = dst->data;
    const size_t size = ggml_nbytes(src);
    context_src->pending.push_back([data_src, data_dst, size]() { memcpy(data_dst, data_src, size); });
    const size_t position = context_src->pending.size();
    context_dst->pending.push_back([context_src, position]() { test_backend_complete(*context_src, position); });
    return true;
}

static ggml_backend_event_t test_device_event_new(ggml_backend_dev_t device) {
    return new ggml_backend_event{device, new test_event_context{}};
}

static void test_device_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    delete static_cast<test_event_context *>(event->context);
    delete event;
}

static void test_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    auto * ev = static_cast<test_event_context *>(event->context);
    ev->backend = context;
    ev->position = context->pending.size();
}

static void test_device_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t event) {
    auto * ev = static_cast<test_event_context *>(event->context);
    if (ev->backend != nullptr) {
        test_backend_complete(*ev->backend, ev->position);
    }
}

static void test_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    const auto ev = *static_cast<test_event_context *>(event->context);
    if (ev.backend != nullptr) {
        context->pending.push_back([ev]() { test_backend_complete(*ev.backend, ev.position); });
    }
}

static const char * test_device_name(ggml_backend_dev_t) {
    return "test";
}

static enum ggml_backend_dev_type test_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static enum ggml_backend_dev_type test_device_type_async(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void test_device_props_async(ggml_backend_dev_t, ggml_backend_dev_props * props) {
    *props = {};
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async = true;
    props->caps.events = true;
}

static bool test_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool test_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == static_cast<test_backend_context *>(device->context)->buft;
}

static void test_backend_init(ggml_backend & backend, ggml_backend_device & device, test_backend_context & context) {
    device.context            = &context;
    device.iface.get_name      = test_device_name;
    device.iface.get_type      = test_device_type;
    device.iface.supports_op   = test_device_supports_op;
    device.iface.supports_buft = test_device_supports_buft;

    backend.iface.get_name      = test_backend_name;
    backend.iface.synchronize   = test_backend_synchronize;
    backend.iface.graph_compute = test_backend_graph_compute;
    backend.device              = &device;
    backend.context             = &context;
}

static void test_backend_init_async(ggml_backend & backend, ggml_backend_device & device, test_backend_context & context) {
    test_backend_init(backend, device, context);
    device.iface.get_type          = test_device_type_async;
    device.iface.get_props         = test_device_props_async;
    device.iface.event_new         = test_device_event_new;
    device.iface.event_free        = test_device_event_free;
    device.iface.event_synchronize = test_device_event_synchronize;
    backend.iface.graph_compute    = test_backend_graph_compute_async;
    backend.iface.get_tensor_async = test_backend_get_tensor_async;
    backend.iface.cpy_tensor_async = test_backend_copy_tensor_async;
    backend.iface.event_record     = test_backend_event_record;
    backend.iface.event_wait       = test_backend_event_wait;
}

static void test_single_copy() {
    test_backend_context context;
    ggml_backend_device device = {};
    ggml_backend backend = {};
    test_backend_init(backend, device, context);

    ggml_backend_t backends[] = { &backend };
    ggml_backend_buffer_type_t bufts[] = { ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 1, 16, false, false));

    ggml_init_params params = {};
    params.mem_size         = 4*ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc         = true;
    ggml_context_ptr ctx(ggml_init(params));

    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_tensor * output = ggml_scale(ctx.get(), input, 2.0f);
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(context.synchronize_count == 0);

    ggml_backend_sched_prepare_inputs(sched.get());
    GGML_ASSERT(context.synchronize_count == 0);
}

static void test_shared_input_ring() {
    test_backend_context context, context_cpu;
    ggml_backend_device device = {}, device_cpu = {};
    ggml_backend backend = {}, backend_cpu = {};
    test_backend_init_async(backend, device, context);
    test_backend_init(backend_cpu, device_cpu, context_cpu);

    ggml_backend_t backends[] = { &backend, &backend_cpu };
    ggml_backend_buffer_type_t bufts[] = { ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 16, false, false));
    const int n_copies = ggml_backend_sched_get_n_copies(sched.get());
    GGML_ASSERT(n_copies > 1);

    ggml_init_params params = {};
    params.mem_size = 4*ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    ggml_tensor * output = ggml_scale(ctx.get(), input, 2.0f);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_sched_set_tensor_backend(sched.get(), output, &backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), input) == &backend_cpu);
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), output) == &backend);
    GGML_ASSERT(output->src[0] == input);

    const int synchronize_count = context.synchronize_count;
    const int n_submissions = 2*n_copies + 2;
    std::vector<float> results(n_submissions, -1.0f);
    std::vector<void *> addresses(n_submissions);
    for (int i = 0; i < n_submissions; i++) {
        ggml_backend_sched_prepare_inputs(sched.get());
        if (i > 0) {
            GGML_ASSERT(context.completed + 1 < context.pending.size());
        }
        addresses[i] = input->data;
        const float value = i + 1;
        ggml_backend_tensor_set(input, &value, 0, sizeof(value));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(&backend, output, &results[i], 0, sizeof(results[i]));
    }
    GGML_ASSERT(context.synchronize_count == synchronize_count);
    ggml_backend_sched_synchronize(sched.get());

    for (int i = 0; i < n_submissions; i++) {
        GGML_ASSERT(results[i] == 2.0f*(i + 1));
        GGML_ASSERT(addresses[i] == addresses[i % n_copies]);
        GGML_ASSERT(context.input_slots[i] == i % n_copies);
        if (i >= n_copies) {
            GGML_ASSERT(context.graph_uids[i] != context.graph_uids[i - n_copies]);
        }
    }
    for (int i = 0; i < n_copies; i++) {
        for (int j = 0; j < i; j++) {
            GGML_ASSERT(addresses[i] != addresses[j]);
        }
    }
}

static void test_pipeline_copies(bool use_views) {
    test_backend_context contexts[3];
    ggml_backend_device devices[3] = {};
    ggml_backend backends[3] = {};
    ggml_backend_buffer_type device_bufts[2] = { *ggml_backend_cpu_buffer_type(), *ggml_backend_cpu_buffer_type() };
    for (int i = 0; i < 2; i++) {
        device_bufts[i].iface.is_host = nullptr;
        device_bufts[i].device = &devices[i];
        contexts[i].buft = &device_bufts[i];
        test_backend_init_async(backends[i], devices[i], contexts[i]);
    }
    test_backend_init(backends[2], devices[2], contexts[2]);

    ggml_backend_t backend_ptrs[] = { &backends[0], &backends[1], &backends[2] };
    ggml_backend_buffer_type_t bufts[] = { &device_bufts[0], &device_bufts[1], ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backend_ptrs, bufts, 3, 32, true, false));
    const int n_copies = ggml_backend_sched_get_n_copies(sched.get());
    GGML_ASSERT(n_copies > 1);

    ggml_init_params params = {};
    params.mem_size = 8*ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, use_views ? 2 : 1);
    ggml_set_input(input);
    ggml_tensor * input_view = use_views ? ggml_view_1d(ctx.get(), input, 1, sizeof(float)) : input;
    ggml_tensor * first = ggml_scale(ctx.get(), input_view, 2.0f);
    ggml_tensor * first_view = use_views ? ggml_view_1d(ctx.get(), first, 1, 0) : first;
    ggml_tensor * second = ggml_scale(ctx.get(), first_view, 3.0f);
    ggml_tensor * third = ggml_scale(ctx.get(), second, 5.0f);
    ggml_tensor * output = ggml_scale(ctx.get(), first_view, 7.0f);
    ggml_set_output(third);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, third);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched.get(), first, &backends[0]);
    ggml_backend_sched_set_tensor_backend(sched.get(), second, &backends[1]);
    ggml_backend_sched_set_tensor_backend(sched.get(), third, &backends[0]);
    ggml_backend_sched_set_tensor_backend(sched.get(), output, &backends[1]);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched.get()) == 4);
    GGML_ASSERT(first->src[0] != input_view);
    GGML_ASSERT(second->src[0] != first_view);
    GGML_ASSERT(output->src[0] == second->src[0]);

    const int synchronize_count = contexts[0].synchronize_count + contexts[1].synchronize_count;
    const int n_submissions = 2*n_copies + 2;
    std::vector<float> results(n_submissions, -1.0f);
    std::vector<float> results_third(n_submissions, -1.0f);
    for (int i = 0; i < n_submissions; i++) {
        ggml_backend_sched_prepare_inputs(sched.get());
        if (i > 0) {
            GGML_ASSERT(contexts[1].completed + 1 < contexts[1].pending.size());
        }
        const float values[] = { use_views ? -(i + 1.0f) : i + 1.0f, i + 1.0f };
        ggml_backend_tensor_set(input, values, 0, ggml_nbytes(input));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(&backends[0], third, &results_third[i], 0, sizeof(float));
        ggml_backend_tensor_get_async(&backends[1], output, &results[i], 0, sizeof(float));
    }
    GGML_ASSERT(contexts[0].synchronize_count + contexts[1].synchronize_count == synchronize_count);
    ggml_backend_sched_synchronize(sched.get());
    for (int i = 0; i < n_submissions; i++) {
        GGML_ASSERT(results_third[i] == 30.0f*(i + 1));
        GGML_ASSERT(results[i] == 14.0f*(i + 1));
        for (int b = 0; b < 2; b++) {
            for (int j = 0; j < 2; j++) {
                const int k = 2*i + j;
                GGML_ASSERT(contexts[b].input_slots[k] == i % n_copies);
                if (i >= n_copies) {
                    GGML_ASSERT(contexts[b].graph_uids[k] != contexts[b].graph_uids[k - 2*n_copies]);
                }
            }
        }
    }
}

static void test_graph_cache_metadata() {
    ggml_init_params params = {};
    params.mem_size = 4*ggml_tensor_overhead() + 2*ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    GGML_ASSERT(graph->input_slot == 0);
    ggml_build_forward_expand(graph, ggml_scale(ctx.get(), input, 2.0f));
    graph->input_slot = 1;
    graph->uid = 1;

    ggml_cgraph view = ggml_graph_view(graph, 0, graph->n_nodes);
    GGML_ASSERT(view.input_slot == graph->input_slot);
    GGML_ASSERT(view.uid == 0);

    ggml_cgraph * copy = ggml_new_graph(ctx.get());
    copy->uid = 2;
    ggml_graph_cpy(graph, copy);
    GGML_ASSERT(copy->input_slot == graph->input_slot);
    GGML_ASSERT(copy->uid == 0);
    ggml_graph_clear(copy);
    GGML_ASSERT(copy->input_slot == 0);
    GGML_ASSERT(copy->uid == 0);
}

int main() {
    test_single_copy();
    test_shared_input_ring();
    test_pipeline_copies(false);
    test_pipeline_copies(true);
    test_graph_cache_metadata();
    return 0;
}
