// Repro for SYCL ADD on strided views (broadcast-style MoE paths).
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-add-strided-view

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool test_add_strided_view() {
    printf("Test: SYCL ADD with strided views\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    const int64_t ne0 = 2880;
    const int64_t ne1 = 32;
    const int64_t ne2 = 2;
    const int64_t ne3 = 1;

    // Base tensor [ne0, ne1, ne2, ne3] to emulate MoE intermediate layout.
    struct ggml_tensor * base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_name(base, "base");

    // Views shaped [ne0, 2, 1] with stride from higher dimensions.
    const size_t         nb1_view = base->nb[2];
    const size_t         nb2_view = base->nb[3];
    struct ggml_tensor * view0    = ggml_view_3d(ctx, base, ne0, 2, 1, nb1_view, nb2_view, 0);
    struct ggml_tensor * view1    = ggml_view_3d(ctx, base, ne0, 2, 1, nb1_view, nb2_view, base->nb[0] * ne0);
    ggml_set_name(view0, "view0");
    ggml_set_name(view1, "view1");

    struct ggml_tensor * mul = ggml_mul(ctx, view0, view0);
    ggml_set_name(mul, "mul");

    struct ggml_tensor * add1 = ggml_add(ctx, mul, view1);
    ggml_set_name(add1, "add1");

    struct ggml_tensor * out = ggml_add(ctx, add1, view1);
    ggml_set_name(out, "out");

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Initialize base data with a simple ramp.
    std::vector<float> base_data(ggml_nelements(base));
    for (size_t i = 0; i < base_data.size(); i++) {
        base_data[i] = static_cast<float>(i % 97) * 0.01f;
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed (%d)\n", status);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_sycl_submit_barrier(backend);
    ggml_backend_tensor_get_async(backend, out, out_data.data(), 0, out_data.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    int non_zero = 0;
    for (float v : out_data) {
        if (std::isfinite(v) && v != 0.0f) {
            non_zero++;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (non_zero == 0) {
        printf("  FAIL: Output is all zeros or non-finite\n");
        return false;
    }

    printf("  PASS: Output readback succeeded (%d/%zu non-zero)\n", non_zero, out_data.size());
    return true;
}

int main() {
    int passed = 0;
    int failed = 0;

    if (test_add_strided_view()) {
        passed++;
    } else {
        failed++;
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
