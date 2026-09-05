#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml.h"

struct test_backend_context {
    int synchronize_count = 0;
};

static const char * test_backend_name(ggml_backend_t) {
    return "test";
}

static void test_backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<test_backend_context *>(backend->context);
    context->synchronize_count++;
}

static ggml_status test_backend_graph_compute(ggml_backend_t, ggml_cgraph *) {
    return GGML_STATUS_SUCCESS;
}

static const char * test_device_name(ggml_backend_dev_t) {
    return "test";
}

static enum ggml_backend_dev_type test_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool test_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool test_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type();
}

int main() {
    test_backend_context context;

    ggml_backend_device device = {};
    device.iface.get_name      = test_device_name;
    device.iface.get_type      = test_device_type;
    device.iface.supports_op   = test_device_supports_op;
    device.iface.supports_buft = test_device_supports_buft;

    ggml_backend backend        = {};
    backend.iface.get_name      = test_backend_name;
    backend.iface.synchronize   = test_backend_synchronize;
    backend.iface.graph_compute = test_backend_graph_compute;
    backend.device              = &device;
    backend.context             = &context;

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

    return 0;
}
