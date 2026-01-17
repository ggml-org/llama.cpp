// Stress test for unpin_on_event via binbcast ops.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-unpin-event --mode=safe
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-unpin-event --mode=barrier
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-unpin-event --mode=compare

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

namespace {

enum class event_mode {
    SAFE,
    BARRIER,
    COMPARE,
};

const char * mode_name(event_mode mode) {
    switch (mode) {
        case event_mode::SAFE:
            return "safe";
        case event_mode::BARRIER:
            return "barrier";
        case event_mode::COMPARE:
            return "compare";
        default:
            return "unknown";
    }
}

event_mode parse_mode(const char * mode_str) {
    if (!mode_str) {
        return event_mode::SAFE;
    }
    if (std::strcmp(mode_str, "safe") == 0) {
        return event_mode::SAFE;
    }
    if (std::strcmp(mode_str, "barrier") == 0) {
        return event_mode::BARRIER;
    }
    if (std::strcmp(mode_str, "compare") == 0 || std::strcmp(mode_str, "both") == 0) {
        return event_mode::COMPARE;
    }
    return event_mode::SAFE;
}

int parse_iters(const char * iters_str, int default_iters) {
    if (!iters_str || !*iters_str) {
        return default_iters;
    }
    const int value = std::atoi(iters_str);
    return value > 0 ? value : default_iters;
}

const char * get_arg(int argc, char ** argv, const char * prefix) {
    const size_t prefix_len = std::strlen(prefix);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix, prefix_len) == 0) {
            return argv[i] + prefix_len;
        }
    }
    return nullptr;
}

static bool run_binbcast_stress(event_mode mode, int iters) {
    const char * env_mode = (mode == event_mode::SAFE) ? "safe" : "barrier";
    setenv("GGML_SYCL_BINBCAST_EVENT_MODE", env_mode, 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    if (!host_buft) {
        fprintf(stderr, "Failed to get SYCL host buffer type\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    const int64_t ne0 = 1024;
    const int64_t ne1 = 4;

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(weight, (std::string("weight.") + env_mode).c_str());

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(input, (std::string("input.") + env_mode).c_str());

    ggml_tensor * out = ggml_mul(ctx, input, weight);
    ggml_set_name(out, (std::string("out.") + env_mode).c_str());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_buf_size);
    if (!weight_buf) {
        fprintf(stderr, "Failed to allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    std::vector<float> weight_data(ggml_nelements(weight));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    std::vector<float> input_data(ggml_nelements(input));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 13) * 0.02f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    bool ok = true;
    for (int i = 0; i < iters; ++i) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[%s] graph compute failed at iter %d (%d)\n", env_mode, i, (int) status);
            ok = false;
            break;
        }
    }

    ggml_backend_sycl_submit_barrier(backend);
    ggml_backend_synchronize(backend);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    const char * mode_arg  = get_arg(argc, argv, "--mode=");
    const char * iters_arg = get_arg(argc, argv, "--iters=");

    event_mode mode = parse_mode(mode_arg ? mode_arg : std::getenv("GGML_SYCL_UNPIN_EVENT_MODE"));
    int        iters = parse_iters(iters_arg ? iters_arg : std::getenv("GGML_SYCL_UNPIN_ITERS"), 200);

    bool ok = true;
    if (mode == event_mode::COMPARE) {
        printf("Mode: compare (safe then barrier), iters=%d\n", iters);
        ok = run_binbcast_stress(event_mode::SAFE, iters);
        if (ok) {
            ok = run_binbcast_stress(event_mode::BARRIER, iters);
        }
    } else {
        printf("Mode: %s, iters=%d\n", mode_name(mode), iters);
        ok = run_binbcast_stress(mode, iters);
    }

    printf("\nUnified cache unpin event test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
