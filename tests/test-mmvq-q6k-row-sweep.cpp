// tests/test-mmvq-q6k-row-sweep.cpp
// Test Q6_K 56-block with varying row counts to find the failure threshold

#include "../ggml/src/ggml-quants.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static bool run_mul_mat_backend(ggml_backend_t       backend,
                                ggml_type            weight_type,
                                const void *         weight_data,
                                size_t               weight_size,
                                const float *        input_data,
                                int                  n_embd,
                                int                  n_rows,
                                std::vector<float> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    ggml_set_name(weight, "weight");

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_name(input, "input");

    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    ggml_backend_tensor_set(input, input_data, 0, n_embd * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    enum ggml_status status  = ggml_backend_graph_compute(backend, graph);
    bool             success = (status == GGML_STATUS_SUCCESS);
    if (success) {
        output.resize(n_rows);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

static bool run_test(ggml_backend_t gpu, ggml_backend_t cpu, int nrows) {
    const int ncols  = 14336;  // 56 blocks
    const int blocks = ncols / QK_K;

    std::mt19937                          rng(2027 + nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_f((size_t) nrows * ncols);
    for (float & v : weight_f) {
        v = dist(rng);
    }

    std::vector<block_q6_K> weight_q((size_t) nrows * blocks);
    for (int r = 0; r < nrows; ++r) {
        quantize_row_q6_K_ref(weight_f.data() + (size_t) r * ncols, weight_q.data() + (size_t) r * blocks, ncols);
    }

    std::vector<float> input(ncols);
    for (float & v : input) {
        v = dist(rng);
    }

    std::vector<float> gpu_out, cpu_out;
    if (!run_mul_mat_backend(gpu, GGML_TYPE_Q6_K, weight_q.data(), weight_q.size() * sizeof(block_q6_K), input.data(),
                             ncols, nrows, gpu_out)) {
        return false;
    }
    if (!run_mul_mat_backend(cpu, GGML_TYPE_Q6_K, weight_q.data(), weight_q.size() * sizeof(block_q6_K), input.data(),
                             ncols, nrows, cpu_out)) {
        return false;
    }

    int   errors   = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < nrows; ++i) {
        float diff  = fabsf(gpu_out[i] - cpu_out[i]);
        float denom = fmaxf(1.0f, fabsf(cpu_out[i]));
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 0.5f && diff / denom > 0.2f) {
            errors++;
        }
    }

    bool pass = (errors == 0);
    printf("nrows=%4d: errors=%d max_diff=%.4f %s\n", nrows, errors, max_diff, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== Q6_K 56-block Row Sweep ===\n");
    setenv("GGML_SYCL_LAYOUT_OVERRIDE", "coalesced", 1);

    ggml_backend_t gpu = ggml_backend_sycl_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) {
        printf("Backend init failed\n");
        return 1;
    }

    int row_counts[] = { 1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    for (int r : row_counts) {
        run_test(gpu, cpu, r);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return 0;
}
