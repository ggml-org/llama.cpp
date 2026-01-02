// tests/test-dmmv-q6k-coalesced.cpp
// Test that Q6_K DMMV with coalesced layout produces same output as CPU backend.
//
// Build: cmake --build build --target test-dmmv-q6k-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-dmmv-q6k-coalesced

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Coalesced tile configuration (must match common.hpp)
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
constexpr int WARP_SIZE = GGML_SYCL_WARP_SIZE;
constexpr int MMVQ_COALESCED_TILE_BLOCKS = WARP_SIZE;

static bool run_mul_mat_backend(ggml_backend_t backend, ggml_type weight_type,
                                const void * weight_data, size_t weight_size,
                                const float * input_data, int n_embd, int n_rows,
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

    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    ggml_backend_tensor_set(input, input_data, 0, n_embd * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool success = (status == GGML_STATUS_SUCCESS);
    if (success) {
        output.resize(n_rows);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

static bool run_dmmv_q6k_coalesced_case(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
                                       int ncols, int nrows, bool verbose) {
    if (ncols % QK_K != 0) {
        printf("  SKIP: ncols not multiple of QK_K\n");
        return true;
    }
    const int blocks_per_row = ncols / QK_K;
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("  SKIP: blocks_per_row=%d not multiple of %d (coalesced tile)\n",
               blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;
    }

    std::mt19937 rng(2027 + ncols + nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float((size_t)nrows * ncols);
    for (float & v : weight_float) v = dist(rng);

    std::vector<block_q6_K> weight_quant((size_t)nrows * blocks_per_row);
    for (int row = 0; row < nrows; ++row) {
        quantize_row_q6_K_ref(weight_float.data() + (size_t)row * ncols,
                              weight_quant.data() + (size_t)row * blocks_per_row,
                              ncols);
    }

    std::vector<float> input_data(ncols);
    for (float & v : input_data) v = dist(rng);

    const size_t weight_bytes = weight_quant.size() * sizeof(block_q6_K);

    std::vector<float> gpu_out;
    std::vector<float> cpu_out;
    if (!run_mul_mat_backend(gpu_backend, GGML_TYPE_Q6_K, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, gpu_out)) {
        printf("  FAIL: GPU backend compute failed\n");
        return false;
    }
    if (!run_mul_mat_backend(cpu_backend, GGML_TYPE_Q6_K, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, cpu_out)) {
        printf("  FAIL: CPU backend compute failed\n");
        return false;
    }

    const float abs_tol = 0.5f;
    const float rel_tol = 0.2f;
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int errors = 0;
    for (int i = 0; i < nrows; ++i) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        float denom = fmaxf(1.0f, fabsf(cpu_out[i]));
        float rel = diff / denom;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            errors++;
        }
    }

    const bool pass = (errors == 0);
    if (verbose || !pass) {
        printf("  Q6_K coalesced ncols=%d nrows=%d: errors=%d max_diff=%.6f max_rel=%.6f %s\n",
               ncols, nrows, errors, max_diff, max_rel, pass ? "PASS" : "FAIL");
    }
    return pass;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    printf("=== Q6_K DMMV Coalesced Tests ===\n");

    setenv("GGML_SYCL_REORDER_MODE", "coalesced", 1);
    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);

    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return 0;
    }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("SKIP: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 0;
    }

    bool ok = true;
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 1, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 8, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 32, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 128, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 16384, 8, true);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("=== Summary: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
