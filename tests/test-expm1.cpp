// Precision regression test for GGML_UNARY_OP_EXPM1.
// Verifies that the CPU implementation uses expm1f() semantics and does not
// suffer from catastrophic cancellation near x = 0.

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool test_expm1_precision() {
    // Inputs chosen to stress-test cancellation at different scales.
    // For x near zero, expf(x)-1.0f collapses to 0.0f due to float rounding,
    // while expm1f(x) returns the correct value ~x.
    const std::vector<float> inputs = { 1e-1f, 1e-3f, 1e-5f, 1e-7f, 0.0f };
    const int n = (int)inputs.size();

    ggml_init_params params {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * res = ggml_unary(ctx, src, GGML_UNARY_OP_EXPM1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();

    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};

    ggml_backend_tensor_set(src, inputs.data(), 0, n * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> output(n);
    ggml_backend_tensor_get(res, output.data(), 0, n * sizeof(float));

    bool passed = true;
    for (int i = 0; i < n; i++) {
        const float expected = expm1f(inputs[i]);
        const float got      = output[i];
        const float err      = fabsf(got - expected);
        const float tol      = 1e-6f;
        if (err > tol) {
            printf("FAIL expm1(%e): expected %e, got %e (err %e > tol %e)\n",
                   inputs[i], expected, got, err, tol);
            passed = false;
        }
    }

    if (passed) {
        printf("test_expm1_precision: PASSED\n");
    }
    return passed;
}

int main() {
    return test_expm1_precision() ? 0 : 1;
}
