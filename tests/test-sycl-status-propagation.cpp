#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include "../ggml/src/ggml-sycl/ggml-sycl-test.h"

#include <array>
#include <cstdio>

namespace {

struct test_resources {
    ggml_backend_t        backend = nullptr;
    ggml_context *        context = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;

    ~test_resources() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

static bool expect_status(const char * step, ggml_status actual, ggml_status expected) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: expected %s, got %s\n",
                 step, ggml_status_to_string(expected), ggml_status_to_string(actual));
    return false;
}

} // namespace

int main() {
    test_resources resources;
    resources.backend = ggml_backend_sycl_init(0);
    if (resources.backend == nullptr) {
        std::fprintf(stderr, "failed to initialize SYCL backend 0\n");
        return 1;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    resources.context = ggml_init(params);
    if (resources.context == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    ggml_tensor * input  = ggml_new_tensor_1d(resources.context, GGML_TYPE_F32, 4);
    ggml_tensor * output = ggml_add(resources.context, input, input);
    ggml_cgraph * graph  = ggml_new_graph(resources.context);
    ggml_build_forward_expand(graph, output);

    resources.buffer = ggml_backend_alloc_ctx_tensors(resources.context, resources.backend);
    if (resources.buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate SYCL tensor buffer\n");
        return 1;
    }

    const std::array<float, 4> input_data = { 1.0f, -2.0f, 3.0f, -4.0f };
    ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(input_data));

    if (!expect_status("initial graph compute",
                       ggml_backend_graph_compute(resources.backend, graph),
                       GGML_STATUS_SUCCESS)) {
        return 1;
    }

    if (!ggml_backend_sycl_test_inject_sync_failure_once(resources.backend)) {
        std::fprintf(stderr, "failed to arm one-shot SYCL synchronization failure\n");
        return 1;
    }

    if (!expect_status("injected graph compute",
                       ggml_backend_graph_compute(resources.backend, graph),
                       GGML_STATUS_FAILED)) {
        return 1;
    }

    // The synchronous wrapper consumes the status it reports: nothing stays pending.
    if (!expect_status("status consumed by synchronous wrapper",
                       ggml_backend_sycl_consume_last_status(resources.backend),
                       GGML_STATUS_SUCCESS)) {
        return 1;
    }

    if (!expect_status("graph compute after one-shot failure",
                       ggml_backend_graph_compute(resources.backend, graph),
                       GGML_STATUS_SUCCESS)) {
        return 1;
    }

    // The scheduler synchronizes a backend between splits without consuming
    // its status. A failure that surfaces there must survive the next graph
    // submission until a consumer sees it.
    if (!ggml_backend_sycl_test_inject_sync_failure_once(resources.backend)) {
        std::fprintf(stderr, "failed to arm one-shot SYCL synchronization failure\n");
        return 1;
    }
    ggml_backend_synchronize(resources.backend);

    if (!expect_status("graph submission after unconsumed sync failure",
                       ggml_backend_graph_compute_async(resources.backend, graph),
                       GGML_STATUS_SUCCESS)) {
        return 1;
    }
    ggml_backend_synchronize(resources.backend);

    if (!expect_status("sync failure survives graph submission",
                       ggml_backend_sycl_consume_last_status(resources.backend),
                       GGML_STATUS_FAILED)) {
        return 1;
    }

    if (!expect_status("graph compute after consumed sync failure",
                       ggml_backend_graph_compute(resources.backend, graph),
                       GGML_STATUS_SUCCESS)) {
        return 1;
    }

    return 0;
}
