#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstring>
#include <functional>
#include <vector>

struct test_backend_context {
    int synchronize_count = 0;
    size_t completed = 0;
    std::vector<std::function<void()>> pending;
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
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
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

static bool test_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type();
}

static void test_backend_init(ggml_backend & backend, ggml_backend_device & device, test_backend_context & context) {
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
    test_backend_init(backend, device, context);
    test_backend_init(backend_cpu, device_cpu, context_cpu);

    device.iface.get_type          = test_device_type_async;
    device.iface.get_props         = test_device_props_async;
    device.iface.event_new         = test_device_event_new;
    device.iface.event_free        = test_device_event_free;
    device.iface.event_synchronize = test_device_event_synchronize;
    backend.iface.graph_compute    = test_backend_graph_compute_async;
    backend.iface.get_tensor_async = test_backend_get_tensor_async;
    backend.iface.event_record     = test_backend_event_record;

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
    }
    for (int i = 0; i < n_copies; i++) {
        for (int j = 0; j < i; j++) {
            GGML_ASSERT(addresses[i] != addresses[j]);
        }
    }
}

int main() {
    test_single_copy();
    test_shared_input_ring();
    return 0;
}
