// Test for MMVQ Q6_K SoA with row slicing
// This test verifies the actual MMVQ kernel produces correct results with SoA layout.
//
// MMVQ is used for batch sizes 2-32 (between DMMV batch=1 and MMQ batch>32).
//
// Build: cmake --build build --target test-mmvq-q6k-soa-rowslice
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmvq-q6k-soa-rowslice

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"

#define QK_K 256

// CPU reference: compute full matmul Q6_K @ F32
static void cpu_mul_mat_q6k_f32(const void* x_data, const float* y,
                                 float* dst, int nrows, int ncols, int n_tokens) {
    const block_q6_K* x = (const block_q6_K*)x_data;
    const int blocks_per_row = ncols / QK_K;

    std::vector<float> x_f32(ncols);

    for (int row = 0; row < nrows; row++) {
        dequantize_row_q6_K(&x[row * blocks_per_row], x_f32.data(), ncols);

        for (int tok = 0; tok < n_tokens; tok++) {
            float sum = 0.0f;
            for (int k = 0; k < ncols; k++) {
                sum += x_f32[k] * y[tok * ncols + k];
            }
            // Output is transposed: dst[tok, row]
            dst[tok * nrows + row] = sum;
        }
    }
}

// Test MMVQ with specific batch size
static bool test_mmvq_q6k_batch(int n_tokens, int n_rows, int n_cols, bool verbose) {
    printf("\n--- MMVQ Q6_K batch=%d rows=%d cols=%d ---\n", n_tokens, n_rows, n_cols);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("  FAIL: Could not create ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    // Weight: [n_cols, n_rows] Q6_K
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_cols, n_rows);
    ggml_set_name(weight, "test_q6k_weight");

    // Input: [n_cols, n_tokens] F32
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_set_name(input, "test_input");

    // Output: MUL_MAT(weight, input) -> [n_rows, n_tokens]
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "test_output");

    // Allocate weight buffer (triggers SoA reordering)
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate weight data - use proper quantization like test-mmq-q6k-gpu
    const int blocks_per_row = n_cols / QK_K;
    const int total_blocks = n_rows * blocks_per_row;
    std::mt19937 rng(42 + n_tokens);  // Different seed per test
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Create float data first
    const int weight_floats = n_rows * n_cols;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    // Quantize to Q6_K properly
    std::vector<block_q6_K> weight_data(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_data.data(), n_rows, n_cols, nullptr);

    // Upload weights (triggers SoA transformation)
    ggml_backend_tensor_set(weight, weight_data.data(), 0, total_blocks * sizeof(block_q6_K));

    // Generate input
    std::vector<float> input_data(n_cols * n_tokens);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    for (auto& v : input_data) v = input_dist(rng);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Build and execute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_graph_compute(backend, graph);

    // Get GPU output
    std::vector<float> gpu_output(n_rows * n_tokens);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // CPU reference
    std::vector<float> cpu_output(n_rows * n_tokens);
    cpu_mul_mat_q6k_f32(weight_data.data(), input_data.data(),
                        cpu_output.data(), n_rows, n_cols, n_tokens);

    // Compare
    int mismatches = 0;
    float max_diff = 0.0f;
    int first_bad_tok = -1, first_bad_row = -1;

    for (int tok = 0; tok < n_tokens; tok++) {
        for (int row = 0; row < n_rows; row++) {
            int idx = tok * n_rows + row;
            float diff = std::fabs(gpu_output[idx] - cpu_output[idx]);
            float ref_mag = std::fabs(cpu_output[idx]);
            float rel_err = ref_mag > 1e-6 ? diff / ref_mag : diff;

            if (diff > 0.1f && rel_err > 0.15f) {
                if (first_bad_tok < 0) {
                    first_bad_tok = tok;
                    first_bad_row = row;
                }
                mismatches++;
                max_diff = std::max(max_diff, diff);
            }
        }
    }

    bool passed = (mismatches == 0);
    printf("  Result: %s (%d/%d mismatches, max_diff=%.4f)\n",
           passed ? "PASS" : "FAIL", mismatches, n_rows * n_tokens, max_diff);

    if (!passed && verbose && first_bad_tok >= 0) {
        printf("  First mismatch: tok=%d row=%d\n", first_bad_tok, first_bad_row);
        int idx = first_bad_tok * n_rows + first_bad_row;
        printf("    GPU=%.6f CPU=%.6f diff=%.6f\n",
               gpu_output[idx], cpu_output[idx],
               std::fabs(gpu_output[idx] - cpu_output[idx]));
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return passed;
}

int main() {
    printf("=== Q6_K MMVQ SoA Row Slice Test ===\n");
    printf("Tests actual MMVQ kernel (batch 2-32) with various tensor sizes.\n");

    int passed = 0, failed = 0;

    // Test 1: Small tensor, batch=4 (MMVQ range)
    if (test_mmvq_q6k_batch(4, 256, 512, true)) passed++; else failed++;

    // Test 2: Medium tensor, batch=8
    if (test_mmvq_q6k_batch(8, 512, 1024, true)) passed++; else failed++;

    // Test 3: Large tensor, batch=16
    if (test_mmvq_q6k_batch(16, 2048, 1024, true)) passed++; else failed++;

    // Test 4: Batch=2 (edge of MMVQ range)
    if (test_mmvq_q6k_batch(2, 256, 512, true)) passed++; else failed++;

    // Test 5: Batch=32 (upper edge of MMVQ range)
    if (test_mmvq_q6k_batch(32, 512, 1024, true)) passed++; else failed++;

    // Test 6: Production-like dimensions (Mistral 7B)
    if (test_mmvq_q6k_batch(8, 4096, 4096, true)) passed++; else failed++;

    printf("\n=================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
