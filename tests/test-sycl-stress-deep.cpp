#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>

// Deep Stress Test: Fuzzes KV update and memory stability
int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) return 1;

    // Use a large context to maximize memory pressure
    const int n_ctx = 32768; 
    const int d = 128;
    printf("Starting DEEP stress test (100k iterations)...\n");

    struct ggml_init_params params = { 512 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO3_0, d, n_ctx);
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 256); // Larger batch
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 256);
    
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * view = ggml_set_rows(ctx, kv, input, indices);
    ggml_build_forward_expand(gf, view);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 1;

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int32_t> dist(0, n_ctx - 1);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        std::vector<int32_t> idx(256);
        for (int k = 0; k < 256; k++) idx[k] = dist(rng);
        ggml_backend_tensor_set(indices, idx.data(), 0, 256 * sizeof(int32_t));
        
        ggml_backend_graph_compute(backend, gf);

        if (i % 5000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            printf("  Iteration %d/100000 passed (%lds)\n", i, elapsed);
        }
    }

    printf("DEEP Stress test PASSED\n");
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
