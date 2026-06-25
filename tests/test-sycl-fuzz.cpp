#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <random>

// Fuzzing set_rows with random indices and types
int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) return 1;

    const int n_ctx = 1024;
    const int d = 128;
    printf("Starting fuzzing test (10k iterations)...\n");

    struct ggml_init_params params = { 10 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, d, n_ctx);
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 32);
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32);
    
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, n_ctx - 1);

    // Initialize indices
    std::vector<int32_t> indices_data(1024);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * view = ggml_set_rows(ctx, kv, input, indices);
    ggml_build_forward_expand(gf, view);

    for (int i = 0; i < 10000; i++) {
        if (i % 1000 == 0) printf("  Iteration %d/10000\n", i);
        std::vector<int32_t> idx(32);
        for (int k = 0; k < 32; k++) idx[k] = dist(rng);
        ggml_backend_tensor_set(indices, idx.data(), 0, 32 * sizeof(int32_t));

        ggml_backend_graph_compute(backend, gf);
    }
    printf("Fuzzing test PASSED\n");
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
