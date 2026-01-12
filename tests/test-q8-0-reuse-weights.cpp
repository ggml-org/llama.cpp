/*
 * Q8_0 Weight Reuse Test
 *
 * Tests if the SoA reordering is preserved when the same weight buffer
 * is used for multiple operations (MMQ then DMMV, like real inference).
 *
 * Build: cmake --build build --target test-q8-0-reuse-weights
 * Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-q8-0-reuse-weights
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"

#define QK8_0 32

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } fu = {f};
    uint32_t sign = (fu.u >> 16) & 0x8000;
    int32_t exponent = ((fu.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fu.u >> 13) & 0x3FF;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        union { float f; uint32_t u; } fu;
        fu.u = sign | 0x7F800000 | (mant << 13);
        return fu.f;
    }
    union { float f; uint32_t u; } fu;
    fu.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return fu.f;
}

static void quantize_to_q8_0(const float* src, block_q8_0_test* dst, int n) {
    int nblocks = n / QK8_0;
    for (int ib = 0; ib < nblocks; ib++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(src[ib * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 127.0f / amax : 0.0f;
        dst[ib].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            float v = src[ib * QK8_0 + j] * id;
            dst[ib].qs[j] = (int8_t)roundf(v);
        }
    }
}

static void cpu_matmul_q8_0(const block_q8_0_test* weight, const float* input,
                             float* output, int ncols, int nrows, int batch) {
    int blocks_per_row = ncols / QK8_0;
    for (int b = 0; b < batch; b++) {
        for (int row = 0; row < nrows; row++) {
            float sum = 0.0f;
            for (int ib = 0; ib < blocks_per_row; ib++) {
                const block_q8_0_test* blk = &weight[row * blocks_per_row + ib];
                float d = fp16_to_fp32(blk->d);
                for (int j = 0; j < QK8_0; j++) {
                    int col = ib * QK8_0 + j;
                    float w = (float)blk->qs[j] * d;
                    sum += w * input[b * ncols + col];
                }
            }
            output[b * nrows + row] = sum;
        }
    }
}

int main(int argc, char** argv) {
    printf("=== Q8_0 Weight Reuse Test ===\n");
    printf("Tests if SoA reordering persists across multiple operations\n\n");

    // Initialize backends
    ggml_backend_t gpu = ggml_backend_sycl_init(0);
    if (!gpu) {
        printf("FAIL: Could not initialize SYCL backend\n");
        return 1;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        printf("FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(gpu);
        return 1;
    }

    printf("GPU: %s\n", ggml_backend_name(gpu));
    printf("CPU: %s\n\n", ggml_backend_name(cpu));

    // Use production-like dimensions
    const int ncols = 4096;
    const int nrows = 14336;  // FFN dimension
    const int prompt_batch = 16;  // MMQ
    const int decode_batch = 1;   // DMMV

    int nblocks = nrows * (ncols / QK8_0);

    // Generate random weights
    std::vector<float> weight_float(nrows * ncols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < weight_float.size(); i++) {
        weight_float[i] = dist(rng);
    }

    // Quantize weights
    std::vector<block_q8_0_test> weight_q8(nblocks);
    quantize_to_q8_0(weight_float.data(), weight_q8.data(), (int)weight_float.size());

    // Create GPU weight buffer ONCE (like production)
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu);
    size_t weight_size = nblocks * sizeof(block_q8_0_test) * 2 + 4096;
    ggml_backend_buffer_t gpu_weight_buf = ggml_backend_buft_alloc_buffer(gpu_buft, weight_size);
    ggml_backend_buffer_set_usage(gpu_weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    printf("=== Phase 1: Load weights (triggers SoA reorder) ===\n");

    // Create a temporary context just for setting weights
    struct ggml_init_params params = { 32 * 1024 * 1024, NULL, true };
    struct ggml_context* weight_ctx = ggml_init(params);
    struct ggml_tensor* weight_tensor = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_backend_tensor_alloc(gpu_weight_buf, weight_tensor, (void*)ggml_backend_buffer_get_base(gpu_weight_buf));

    printf("  Setting weight data (GPU reorders to SoA)...\n");
    ggml_backend_tensor_set(weight_tensor, weight_q8.data(), 0, nblocks * sizeof(block_q8_0_test));
    printf("  Weight tensor set. Buffer base: %p\n", ggml_backend_buffer_get_base(gpu_weight_buf));

    printf("  Weight tensor data ptr: %p\n", weight_tensor->data);

    bool all_passed = true;

    // Phase 2: MMQ (prompt processing) - use existing weight buffer
    printf("\n=== Phase 2: MMQ (batch=%d) with reused weight buffer ===\n", prompt_batch);
    {
        struct ggml_context* ctx = ggml_init(params);

        struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, prompt_batch);
        struct ggml_tensor* output = ggml_mul_mat(ctx, weight_tensor, input);

        // Allocate compute buffer
        size_t compute_size = ggml_nbytes(input) + ggml_nbytes(output) + 4096;
        ggml_backend_buffer_t compute_buf = ggml_backend_buft_alloc_buffer(gpu_buft, compute_size);
        uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buf);
        ggml_backend_tensor_alloc(compute_buf, input, base);
        ggml_backend_tensor_alloc(compute_buf, output, base + ggml_nbytes(input) + 512);

        // Generate input
        std::vector<float> input_data(ncols * prompt_batch);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = dist(rng);
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Manual reference
        std::vector<float> manual_output(nrows * prompt_batch);
        cpu_matmul_q8_0(weight_q8.data(), input_data.data(), manual_output.data(), ncols, nrows, prompt_batch);

        // Execute GPU
        printf("  Executing GPU compute...\n");
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(gpu, graph);

        // Get result
        std::vector<float> gpu_result(nrows * prompt_batch);
        ggml_backend_tensor_get(output, gpu_result.data(), 0, gpu_result.size() * sizeof(float));

        // Compare (GPU vs Manual)
        float max_err = 0;
        int errors = 0;
        for (size_t i = 0; i < gpu_result.size(); i++) {
            float diff = fabsf(gpu_result[i] - manual_output[i]);
            if (diff > max_err) max_err = diff;
            float ref = fabsf(manual_output[i]);
            if (diff > 0.5f && (ref < 1e-6f || diff / ref > 0.1f)) errors++;
        }

        printf("  Max GPU-Manual error: %.6f\n", max_err);
        printf("  Errors (>0.5 or >10%%): %d/%zu\n", errors, gpu_result.size());

        // Print first values
        printf("  First 4: Manual=[%.4f, %.4f, %.4f, %.4f] GPU=[%.4f, %.4f, %.4f, %.4f]\n",
               manual_output[0], manual_output[1], manual_output[2], manual_output[3],
               gpu_result[0], gpu_result[1], gpu_result[2], gpu_result[3]);

        // MMQ will have precision differences (different accumulation order)
        // So we accept larger tolerance here
        printf("  MMQ: %s (precision differences expected)\n", max_err < 1.0f ? "OK" : "FAIL");

        ggml_backend_buffer_free(compute_buf);
        ggml_free(ctx);
    }

    // Phase 3: DMMV (token generation) - reuse SAME weight buffer
    printf("\n=== Phase 3: DMMV (batch=%d) with reused weight buffer ===\n", decode_batch);
    for (int step = 0; step < 5; step++) {
        printf("  --- Decode step %d ---\n", step);

        struct ggml_context* ctx = ggml_init(params);

        struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, decode_batch);
        struct ggml_tensor* output = ggml_mul_mat(ctx, weight_tensor, input);

        // Allocate compute buffer
        size_t compute_size = ggml_nbytes(input) + ggml_nbytes(output) + 4096;
        ggml_backend_buffer_t compute_buf = ggml_backend_buft_alloc_buffer(gpu_buft, compute_size);
        uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buf);
        ggml_backend_tensor_alloc(compute_buf, input, base);
        ggml_backend_tensor_alloc(compute_buf, output, base + ggml_nbytes(input) + 512);

        // Generate input
        std::vector<float> input_data(ncols * decode_batch);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = dist(rng);
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Manual reference
        std::vector<float> manual_output(nrows * decode_batch);
        cpu_matmul_q8_0(weight_q8.data(), input_data.data(), manual_output.data(), ncols, nrows, decode_batch);

        // Execute GPU
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(gpu, graph);

        // Get result
        std::vector<float> gpu_result(nrows * decode_batch);
        ggml_backend_tensor_get(output, gpu_result.data(), 0, gpu_result.size() * sizeof(float));

        // Compare (GPU vs Manual) - DMMV should match exactly
        float max_err = 0;
        int errors = 0;
        for (size_t i = 0; i < gpu_result.size(); i++) {
            float diff = fabsf(gpu_result[i] - manual_output[i]);
            if (diff > max_err) max_err = diff;
            if (diff > 0.01f) errors++;  // Tight tolerance for DMMV
        }

        printf("    Max GPU-Manual error: %.6f\n", max_err);
        printf("    Errors (>0.01): %d/%zu\n", errors, gpu_result.size());

        // Print first values
        printf("    First 4: Manual=[%.4f, %.4f, %.4f, %.4f] GPU=[%.4f, %.4f, %.4f, %.4f]\n",
               manual_output[0], manual_output[1], manual_output[2], manual_output[3],
               gpu_result[0], gpu_result[1], gpu_result[2], gpu_result[3]);

        bool passed = (max_err < 0.01f);
        printf("    DMMV step %d: %s\n", step, passed ? "PASS" : "FAIL");
        if (!passed) all_passed = false;

        ggml_backend_buffer_free(compute_buf);
        ggml_free(ctx);
    }

    // Cleanup
    ggml_backend_buffer_free(gpu_weight_buf);
    ggml_free(weight_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    printf("\n========================================\n");
    printf("RESULT: %s\n", all_passed ? "ALL PASSED" : "FAILED");
    printf("========================================\n");

    return all_passed ? 0 : 1;
}
