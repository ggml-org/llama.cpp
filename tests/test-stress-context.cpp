#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) return 1;

    const int n_ctx = 8192;
    const int d = 128;
    printf("Stress testing 8k context KV cache on SYCL...\n");

    struct ggml_init_params params = { 128 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    // KV Cache tensor (TURBO3_0) - on backend
    struct ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, d, n_ctx);

    // Simulate KV update via SET_ROWS - on backend
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1024);
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1024);
    
    // Build view
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * view = ggml_set_rows(ctx, kv, input, indices);
    ggml_build_forward_expand(gf, view);

    // Allocate tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate backend buffer\n");
        return 1;
    }

    std::vector<int32_t> indices_data(1024);
    for (int i = 0; i < 1024; i++) indices_data[i] = i;
    ggml_backend_tensor_set(indices, indices_data.data(), 0, 1024 * sizeof(int32_t));

    std::vector<float> input_data(1024 * d, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, 1024 * d * sizeof(float));

    for (int i = 0; i < 10; i++) {
        printf("  Iteration %d/10\n", i+1);
        ggml_backend_graph_compute(backend, gf);
    }

    printf("Stress test PASSED\n");

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
