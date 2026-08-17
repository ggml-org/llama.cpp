// Tests for GGML_OP_SLEEP, on every backend that supports it. Two properties are checked:
//   - the op is a pass-through: the output must be a bit-for-bit copy of the input
//   - the op does not return early: a chain of n sleeps must take at least n*us
// Only a lower bound is asserted for the duration. A busy-wait cannot finish early, so that bound is
// deterministic, whereas an upper bound would be flaky on a loaded machine.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

enum test_result {
    TEST_OK,
    TEST_FAIL,
    TEST_SKIP,
};

static test_result test_sleep(ggml_backend_t backend, ggml_type type, int64_t ne, int32_t us, int n_nodes) {
    const size_t graph_size = n_nodes + 16;

    ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(params_static);

    ggml_tensor * x = ggml_new_tensor_1d(ctx_static, type, ne);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_init_params params_compute = {
        /*.mem_size   =*/ n_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(params_compute);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, graph_size, false);

    ggml_tensor * out = x;
    for (int i = 0; i < n_nodes; i++) {
        out = ggml_sleep(ctx_compute, out, us);
    }
    ggml_set_name(out, "out");
    ggml_build_forward_expand(gf, out);

    // the backends log to stdout while computing, so the result line is only printed once it is complete
    char desc[128];
    snprintf(desc, sizeof(desc), "  %-6s %-4s ne=%-9" PRId64 " us=%-7d n_nodes=%d  ",
            ggml_backend_name(backend), ggml_type_name(type), ne, us, n_nodes);

    test_result result = TEST_OK;

    if (!ggml_backend_supports_op(backend, out)) {
        printf("%snot supported\n", desc);
        result = TEST_SKIP;
    } else {
        ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, backend);

        // random bytes rather than random floats, so that bit patterns a value-wise copy could
        // mangle, such as NaNs and denormals, are covered as well
        std::vector<uint8_t> data_in(ggml_nbytes(x));
        std::mt19937 rng(1234);
        for (size_t i = 0; i < data_in.size(); i++) {
            data_in[i] = rng() & 0xFF;
        }
        ggml_backend_tensor_set(x, data_in.data(), 0, data_in.size());

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(galloc, gf);

        // warm up, so that one-time costs such as lazy CUDA module loading are not timed
        ggml_backend_graph_compute(backend, gf);

        const int64_t t_start_us = ggml_time_us();
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        const int64_t t_us = ggml_time_us() - t_start_us;

        std::vector<uint8_t> data_out(ggml_nbytes(out));
        ggml_backend_tensor_get(out, data_out.data(), 0, data_out.size());

        const int64_t t_min_us = (int64_t) n_nodes * us;

        const bool ok_status = status == GGML_STATUS_SUCCESS;
        const bool ok_data   = memcmp(data_in.data(), data_out.data(), data_in.size()) == 0;
        const bool ok_time   = t_us >= t_min_us;

        printf("%selapsed=%7" PRId64 " us (>= %6" PRId64 ")  data: %-4s  time: %s\n",
                desc, t_us, t_min_us, ok_data ? "OK" : "FAIL", ok_time ? "OK" : "FAIL");

        if (!ok_status) {
            printf("    compute failed: %s\n", ggml_status_to_string(status));
        }

        result = ok_status && ok_data && ok_time ? TEST_OK : TEST_FAIL;

        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(buf_static);
    }

    ggml_free(ctx_compute);
    ggml_free(ctx_static);

    return result;
}

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    int n_ok   = 0;
    int n_fail = 0;
    int n_skip = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            printf("  failed to initialize %s\n", ggml_backend_dev_name(dev));
            n_fail++;
            continue;
        }

        // a zero duration must still produce a valid copy, longer ones are checked against the clock,
        // and the last case verifies that a chain of sleeps accumulates instead of collapsing into one
        const struct {
            ggml_type type;
            int64_t   ne;
            int32_t   us;
            int       n_nodes;
        } cases[] = {
            { GGML_TYPE_F32,       1,     0, 1 },
            { GGML_TYPE_F32,    4096, 10000, 1 },
            { GGML_TYPE_F16,    1024, 10000, 1 },
            { GGML_TYPE_F32, 1 << 20,  5000, 4 },
        };

        for (const auto & c : cases) {
            switch (test_sleep(backend, c.type, c.ne, c.us, c.n_nodes)) {
                case TEST_OK:   n_ok++;   break;
                case TEST_FAIL: n_fail++; break;
                case TEST_SKIP: n_skip++; break;
            }
        }

        ggml_backend_free(backend);
    }

    printf("%d passed, %d failed, %d skipped\n", n_ok, n_fail, n_skip);

    return n_fail > 0;
}
