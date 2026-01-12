// Targeted test: ensure SoA layout pointer is used from unified cache for Q8_0 DMMV.
// Strategy:
// 1) Upload AoS weights and force SoA layout caching.
// 2) Mutate AoS host data after caching.
// 3) Run DMMV and verify output matches original weights (not mutated AoS).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include "common.hpp"

#define QK8_0 32

typedef struct {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                                 ggml_backend_buffer_usage usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static void fill_q8_0_ones(block_q8_0_test * blocks, int nblocks) {
    const ggml_fp16_t d = ggml_fp32_to_fp16(1.0f);
    for (int i = 0; i < nblocks; ++i) {
        blocks[i].d = d;
        for (int j = 0; j < QK8_0; ++j) {
            blocks[i].qs[j] = 1;
        }
    }
}

static void compute_reference(const block_q8_0_test * weights, const float * input, float * output,
                              int ncols, int nrows) {
    const int blocks_per_row = ncols / QK8_0;
    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        const block_q8_0_test * row_blocks = weights + row * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q8_0_test * blk = row_blocks + b;
            const float d = ggml_fp16_to_fp32(blk->d);
            const float * x = input + b * QK8_0;
            for (int j = 0; j < QK8_0; ++j) {
                sum += d * (float) blk->qs[j] * x[j];
            }
        }
        output[row] = sum;
    }
}

int main() {
    setenv("GGML_SYCL_LAYOUT_OVERRIDE", "soa", 1);
    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);

    ggml_backend_sycl_set_unified_cache_budget_pct(90);
    ggml_backend_sycl_set_unified_cache_host_budget_pct(90);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return 1;
    }

    const int ncols = 256;
    const int nrows = 4;
    const int batch = 1;
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks = nrows * blocks_per_row;

    ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "cache_weight_q8_0");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    ggml_set_name(input, "cache_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    ggml_backend_buffer_type_t weight_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);

    ggml_backend_buffer_t weight_buf =
        alloc_tensor_buffer(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t input_buf =
        alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t output_buf =
        alloc_tensor_buffer(dev_buft, output, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!weight_buf || !input_buf || !output_buf) {
        fprintf(stderr, "Failed to allocate buffers\n");
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (output_buf) ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }

    std::vector<block_q8_0_test> weight_data(nblocks);
    fill_q8_0_ones(weight_data.data(), nblocks);

    std::vector<float> input_data(ncols * batch, 1.0f);
    std::vector<float> ref_output(nrows, 0.0f);
    compute_reference(weight_data.data(), input_data.data(), ref_output.data(), ncols, nrows);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    const char * source = nullptr;
    void * layout_ptr = ggml_sycl_get_layout_ptr_for(weight, 0, GGML_LAYOUT_SOA, &source);
    if (!layout_ptr) {
        fprintf(stderr, "Failed to resolve SoA layout pointer (source=%s)\n", source ? source : "null");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    if (layout_ptr == weight->data) {
        fprintf(stderr, "SoA layout pointer unexpectedly equals AoS data pointer\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::memset(weight->data, 0, ggml_nbytes(weight));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph compute failed\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(output_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> gpu_output(nrows, 0.0f);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, nrows * sizeof(float));

    float max_diff = 0.0f;
    float max_rel = 0.0f;
    float min_abs = std::fabs(gpu_output[0]);
    for (int i = 0; i < nrows; ++i) {
        const float ref = ref_output[i];
        const float diff = std::fabs(gpu_output[i] - ref);
        max_diff = std::max(max_diff, diff);
        const float rel = std::fabs(ref) > 1e-6f ? diff / std::fabs(ref) : diff;
        max_rel = std::max(max_rel, rel);
        min_abs = std::min(min_abs, std::fabs(gpu_output[i]));
    }

    const float rel_tol = 1e-3f;
    const float abs_tol = 1e-2f;
    bool pass = (max_rel < rel_tol || max_diff < abs_tol) && min_abs > 1.0f;

    printf("Max diff: %.6e, max rel: %.6e, min abs: %.6f\n", max_diff, max_rel, min_abs);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(output_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return pass ? 0 : 1;
}
