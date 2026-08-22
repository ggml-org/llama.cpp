#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

static bool run_case(ggml_backend_t backend, const std::array<int64_t, 4> & ne, float scale, float & max_abs_err) {
    ggml_init_params params = {
        /* .mem_size   = */ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * dy = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne.data());
    ggml_tensor * y  = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne.data());
    ggml_set_input(dy);
    ggml_set_input(y);
    ggml_set_output(dy);

    ggml_tensor * dx = ggml_soft_max_ext_back(ctx.get(), dy, y, scale, 0.0f);
    ggml_set_output(dx);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dx);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), graph)) {
        fprintf(stderr, "graph allocation failed\n");
        return false;
    }
    if (dx->data != y->data || dx->data == dy->data) {
        fprintf(stderr, "allocator did not alias dst with src1\n");
        return false;
    }

    const int64_t nc = ne[0];
    const int64_t nr = ggml_nrows(dy);
    const int64_t n  = ggml_nelements(dy);
    std::vector<float> dy_data(n);
    std::vector<float> y_data(n);
    std::vector<float> expected(n);
    std::vector<float> actual(n);

    for (int64_t row = 0; row < nr; row++) {
        float sum = 0.0f;
        for (int64_t col = 0; col < nc; col++) {
            const int64_t i = row*nc + col;
            dy_data[i] = ((col + 3*row) % 17 - 8) * 0.125f;
            y_data[i]  = 2.0f*(col + 1)/(nc*(nc + 1));
            sum += dy_data[i]*y_data[i];
        }
        for (int64_t col = 0; col < nc; col++) {
            const int64_t i = row*nc + col;
            expected[i] = scale*(dy_data[i] - sum)*y_data[i];
        }
    }

    ggml_backend_tensor_set(dy, dy_data.data(), 0, ggml_nbytes(dy));
    ggml_backend_tensor_set(y, y_data.data(), 0, ggml_nbytes(y));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return false;
    }
    ggml_backend_tensor_get(dx, actual.data(), 0, ggml_nbytes(dx));

    max_abs_err = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        max_abs_err = std::max(max_abs_err, std::abs(expected[i] - actual[i]));
    }
    return max_abs_err <= 1e-6f;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backend);

    const std::array<std::array<int64_t, 4>, 2> cases = {{
        {4, 2, 1, 1},
        {1024, 4, 2, 1},
    }};

    int passed = 0;
    for (const auto & ne : cases) {
        float max_abs_err = 0.0f;
        const bool ok = run_case(backend.get(), ne, 0.7f, max_abs_err);
        printf("soft_max_back ne=[%lld,%lld,%lld,%lld] max_abs_err=%.9g %s\n",
                (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
                max_abs_err, ok ? "PASSED" : "FAILED");
        passed += ok;
    }

    printf("%zu cases: %d passed, %zu failed\n", cases.size(), passed, cases.size() - passed);
    return passed == (int) cases.size() ? 0 : 1;
}
