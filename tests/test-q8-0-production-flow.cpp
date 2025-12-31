/*
 * Q8_0 Production Flow Test
 *
 * Simulates actual inference: MMQ for prompt processing, DMMV for token generation.
 * Uses production ggml backends with extensive debug output.
 *
 * This test mimics a simplified transformer layer:
 * 1. Prompt phase: Process multiple tokens at once (MMQ, batch > 1)
 * 2. Decode phase: Generate tokens one at a time (DMMV, batch = 1)
 *
 * Build: cmake --build build --target test-q8-0-production-flow
 * Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-q8-0-production-flow
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
    uint16_t d;           // delta (fp16 as uint16_t)
    int8_t qs[QK8_0];     // quants
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

// Quantize float array to Q8_0
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

// Manual CPU reference for Q8_0 matmul
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

struct TestContext {
    ggml_backend_t gpu;
    ggml_backend_t cpu;
    bool verbose;
};

// Run a single matmul on both CPU and GPU, compare results
static bool run_matmul_compare(TestContext& ctx, const char* phase,
                                const block_q8_0_test* weight_data, int nblocks,
                                const float* input_data,
                                int ncols, int nrows, int batch) {
    printf("\n=== %s: ncols=%d nrows=%d batch=%d ===\n", phase, ncols, nrows, batch);

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(ctx.gpu);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(ctx.cpu);

    // Create contexts
    struct ggml_init_params params = { 32 * 1024 * 1024, NULL, true };
    struct ggml_context* gpu_ctx = ggml_init(params);
    struct ggml_context* cpu_ctx = ggml_init(params);
    if (!gpu_ctx || !cpu_ctx) {
        printf("  ERROR: Failed to create contexts\n");
        return false;
    }

    // Create tensors
    struct ggml_tensor* gpu_weight = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* gpu_input = ggml_new_tensor_2d(gpu_ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* gpu_output = ggml_mul_mat(gpu_ctx, gpu_weight, gpu_input);

    struct ggml_tensor* cpu_weight = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* cpu_input = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* cpu_output = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);

    // Allocate GPU buffers
    size_t weight_size = ggml_nbytes(gpu_weight) * 2 + 4096;
    ggml_backend_buffer_t gpu_weight_buf = ggml_backend_buft_alloc_buffer(gpu_buft, weight_size);
    ggml_backend_buffer_set_usage(gpu_weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(gpu_weight_buf, gpu_weight, (void*)ggml_backend_buffer_get_base(gpu_weight_buf));

    size_t compute_size = ggml_nbytes(gpu_input) + ggml_nbytes(gpu_output) + 4096;
    ggml_backend_buffer_t gpu_compute_buf = ggml_backend_buft_alloc_buffer(gpu_buft, compute_size);
    uint8_t* gpu_compute_base = (uint8_t*)ggml_backend_buffer_get_base(gpu_compute_buf);
    ggml_backend_tensor_alloc(gpu_compute_buf, gpu_input, gpu_compute_base);
    ggml_backend_tensor_alloc(gpu_compute_buf, gpu_output, gpu_compute_base + ggml_nbytes(gpu_input) + 512);

    // Allocate CPU buffers
    ggml_backend_buffer_t cpu_weight_buf = ggml_backend_buft_alloc_buffer(cpu_buft, weight_size);
    ggml_backend_buffer_set_usage(cpu_weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(cpu_weight_buf, cpu_weight, (void*)ggml_backend_buffer_get_base(cpu_weight_buf));

    ggml_backend_buffer_t cpu_compute_buf = ggml_backend_buft_alloc_buffer(cpu_buft, compute_size);
    uint8_t* cpu_compute_base = (uint8_t*)ggml_backend_buffer_get_base(cpu_compute_buf);
    ggml_backend_tensor_alloc(cpu_compute_buf, cpu_input, cpu_compute_base);
    ggml_backend_tensor_alloc(cpu_compute_buf, cpu_output, cpu_compute_base + ggml_nbytes(cpu_input) + 512);

    // Set data (this triggers SoA reorder for GPU weights)
    printf("  Setting weight data (GPU will reorder to SoA)...\n");
    ggml_backend_tensor_set(gpu_weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(cpu_weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));

    printf("  Setting input data...\n");
    ggml_backend_tensor_set(gpu_input, input_data, 0, ncols * batch * sizeof(float));
    ggml_backend_tensor_set(cpu_input, input_data, 0, ncols * batch * sizeof(float));

    // Manual CPU reference
    std::vector<float> manual_output(nrows * batch);
    cpu_matmul_q8_0(weight_data, input_data, manual_output.data(), ncols, nrows, batch);

    // Execute GPU
    printf("  Executing GPU compute...\n");
    struct ggml_cgraph* gpu_graph = ggml_new_graph(gpu_ctx);
    ggml_build_forward_expand(gpu_graph, gpu_output);
    enum ggml_status gpu_status = ggml_backend_graph_compute(ctx.gpu, gpu_graph);

    // Execute CPU
    printf("  Executing CPU compute...\n");
    struct ggml_cgraph* cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    enum ggml_status cpu_status = ggml_backend_graph_compute(ctx.cpu, cpu_graph);

    if (gpu_status != GGML_STATUS_SUCCESS || cpu_status != GGML_STATUS_SUCCESS) {
        printf("  ERROR: Compute failed (GPU=%d, CPU=%d)\n", gpu_status, cpu_status);
        ggml_backend_buffer_free(gpu_compute_buf);
        ggml_backend_buffer_free(gpu_weight_buf);
        ggml_backend_buffer_free(cpu_compute_buf);
        ggml_backend_buffer_free(cpu_weight_buf);
        ggml_free(gpu_ctx);
        ggml_free(cpu_ctx);
        return false;
    }

    // Get results
    std::vector<float> gpu_result(nrows * batch);
    std::vector<float> cpu_result(nrows * batch);
    ggml_backend_tensor_get(gpu_output, gpu_result.data(), 0, nrows * batch * sizeof(float));
    ggml_backend_tensor_get(cpu_output, cpu_result.data(), 0, nrows * batch * sizeof(float));

    // Compare results
    printf("  Comparing results...\n");
    int errors = 0;
    float max_gpu_cpu_err = 0, max_gpu_manual_err = 0;
    int worst_gpu_cpu_idx = -1, worst_gpu_manual_idx = -1;

    for (int i = 0; i < nrows * batch; i++) {
        float gpu_cpu_diff = fabsf(gpu_result[i] - cpu_result[i]);
        float gpu_manual_diff = fabsf(gpu_result[i] - manual_output[i]);

        if (gpu_cpu_diff > max_gpu_cpu_err) {
            max_gpu_cpu_err = gpu_cpu_diff;
            worst_gpu_cpu_idx = i;
        }
        if (gpu_manual_diff > max_gpu_manual_err) {
            max_gpu_manual_err = gpu_manual_diff;
            worst_gpu_manual_idx = i;
        }

        // Error threshold: absolute > 0.1 or relative > 5%
        float ref = fabsf(manual_output[i]);
        if (gpu_manual_diff > 0.1f && (ref < 1e-6f || gpu_manual_diff / ref > 0.05f)) {
            errors++;
        }
    }

    // Print summary
    printf("  Results:\n");
    printf("    Max GPU-CPU error:    %.6f at idx %d\n", max_gpu_cpu_err, worst_gpu_cpu_idx);
    printf("    Max GPU-Manual error: %.6f at idx %d\n", max_gpu_manual_err, worst_gpu_manual_idx);
    printf("    Errors (>0.1 or >5%%): %d/%d\n", errors, nrows * batch);

    // Print first few values
    printf("  First 8 output values:\n");
    printf("    Idx |    Manual |       CPU |       GPU | GPU-Manual\n");
    printf("    ----|-----------|-----------|-----------|----------\n");
    for (int i = 0; i < std::min(8, nrows * batch); i++) {
        printf("    %3d | %9.4f | %9.4f | %9.4f | %10.6f\n",
               i, manual_output[i], cpu_result[i], gpu_result[i],
               gpu_result[i] - manual_output[i]);
    }

    // If there are errors, print worst cases
    if (errors > 0 && ctx.verbose) {
        printf("  Worst error cases:\n");
        std::vector<std::pair<float, int>> sorted_errors;
        for (int i = 0; i < nrows * batch; i++) {
            float err = fabsf(gpu_result[i] - manual_output[i]);
            sorted_errors.push_back({err, i});
        }
        std::sort(sorted_errors.begin(), sorted_errors.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        for (int k = 0; k < std::min(5, (int)sorted_errors.size()); k++) {
            int i = sorted_errors[k].second;
            printf("    idx=%d: Manual=%.4f GPU=%.4f diff=%.6f\n",
                   i, manual_output[i], gpu_result[i], sorted_errors[k].first);
        }
    }

    // Cleanup
    ggml_backend_buffer_free(gpu_compute_buf);
    ggml_backend_buffer_free(gpu_weight_buf);
    ggml_backend_buffer_free(cpu_compute_buf);
    ggml_backend_buffer_free(cpu_weight_buf);
    ggml_free(gpu_ctx);
    ggml_free(cpu_ctx);

    bool passed = (errors == 0);
    printf("  %s: %s\n", phase, passed ? "PASS" : "FAIL");
    return passed;
}

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = true;
    }

    printf("=== Q8_0 Production Flow Test ===\n");
    printf("Simulates: MMQ (prompt) -> DMMV (decode) pattern\n\n");

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

    TestContext ctx = { gpu, cpu, verbose };

    // Use Mistral-like dimensions
    const int hidden_dim = 4096;
    const int ffn_dim = 14336;
    const int vocab_size = 32000;

    // Test configurations
    struct TestCase {
        const char* name;
        int ncols;
        int nrows;
        int prompt_batch;  // MMQ batch size
        int decode_batch;  // DMMV batch size (always 1)
    };

    TestCase tests[] = {
        // Small tests first
        {"small_attn",  128,  128, 8, 1},
        {"small_ffn",   128,  256, 8, 1},

        // Production-like dimensions
        {"attn_q",      hidden_dim, hidden_dim, 16, 1},  // Q projection
        {"ffn_gate",    hidden_dim, ffn_dim,    16, 1},  // FFN gate
        {"ffn_down",    ffn_dim,    hidden_dim, 16, 1},  // FFN down
        {"output",      hidden_dim, vocab_size, 16, 1},  // Output projection
    };

    int total_passed = 0;
    int total_tests = 0;

    for (auto& test : tests) {
        printf("\n========================================\n");
        printf("Test: %s (ncols=%d, nrows=%d)\n", test.name, test.ncols, test.nrows);
        printf("========================================\n");

        // Generate random weights
        int nblocks = test.nrows * (test.ncols / QK8_0);
        std::vector<float> weight_float(test.nrows * test.ncols);
        std::mt19937 rng(42 + test.ncols * 1000 + test.nrows);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < weight_float.size(); i++) {
            weight_float[i] = dist(rng);
        }

        // Quantize weights
        std::vector<block_q8_0_test> weight_q8(nblocks);
        quantize_to_q8_0(weight_float.data(), weight_q8.data(), (int)weight_float.size());

        // Test 1: MMQ path (prompt processing)
        {
            std::vector<float> prompt_input(test.ncols * test.prompt_batch);
            for (size_t i = 0; i < prompt_input.size(); i++) {
                prompt_input[i] = dist(rng);
            }

            char phase_name[64];
            snprintf(phase_name, sizeof(phase_name), "MMQ (batch=%d)", test.prompt_batch);
            bool passed = run_matmul_compare(ctx, phase_name, weight_q8.data(), nblocks,
                                              prompt_input.data(), test.ncols, test.nrows,
                                              test.prompt_batch);
            total_tests++;
            if (passed) total_passed++;
        }

        // Test 2: DMMV path (token generation) - multiple decode steps
        for (int decode_step = 0; decode_step < 3; decode_step++) {
            std::vector<float> decode_input(test.ncols * test.decode_batch);
            for (size_t i = 0; i < decode_input.size(); i++) {
                decode_input[i] = dist(rng);
            }

            char phase_name[64];
            snprintf(phase_name, sizeof(phase_name), "DMMV step %d (batch=%d)", decode_step, test.decode_batch);
            bool passed = run_matmul_compare(ctx, phase_name, weight_q8.data(), nblocks,
                                              decode_input.data(), test.ncols, test.nrows,
                                              test.decode_batch);
            total_tests++;
            if (passed) total_passed++;
        }
    }

    printf("\n========================================\n");
    printf("SUMMARY: %d/%d tests passed\n", total_passed, total_tests);
    printf("========================================\n");

    if (total_passed < total_tests) {
        printf("\nFAILED TESTS DETECTED!\n");
        printf("This indicates the Q8_0 SoA kernel has bugs.\n");
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    return (total_passed == total_tests) ? 0 : 1;
}
