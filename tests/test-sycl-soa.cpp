// SYCL SoA Test: Validates auto-convert path produces correct results
// Tests: CPU reference vs SYCL with SoA auto-conversion
//
// Build: cmake --build build --target test-sycl-soa
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-soa

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"

static float max_diff(const float* a, const float* b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static void fill_random(float * data, int n, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) data[i] = dist(rng);
}

// Run MUL_MAT on a backend and get the result
// mark_as_weights: if true, marks buffer as WEIGHTS which triggers SoA auto-convert on SYCL
static bool run_mul_mat(ggml_backend_t backend, ggml_type weight_type,
                        const void* weight_data, size_t weight_size,
                        const float* input_data, int k, int n_rows, int n_tokens,
                        std::vector<float>& output, bool mark_as_weights = false) {

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    // Create weight tensor [k, n_rows]
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, k, n_rows);
    ggml_set_name(weight, "weight");

    // Create input tensor [k, n_tokens]
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_tokens);
    ggml_set_name(input, "input");

    // Output: weight^T @ input -> [n_rows, n_tokens]
    struct ggml_tensor* out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    // Allocate weight buffer
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    if (mark_as_weights) {
        // This triggers SoA auto-convert on SYCL backend
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer for input and output
    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    // Set weight data (triggers SoA conversion if buffer is marked as WEIGHTS)
    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);

    // Set input data
    ggml_backend_tensor_set(input, input_data, 0, k * n_tokens * sizeof(float));

    // Build and run graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    bool success = (status == GGML_STATUS_SUCCESS);
    if (success) {
        output.resize(n_rows * n_tokens);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

// Test a specific quant type for DMMV (batch=1) and MMQ (batch>1)
static bool test_quant_type(const char* type_name, ggml_type type, int k, int n_rows) {
    printf("\n=== Testing %s ===\n", type_name);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        printf("  SKIP: SYCL backend not available\n");
        return true;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("  SKIP: CPU backend not available\n");
        ggml_backend_free(sycl_backend);
        return true;
    }

    // Get quantization traits
    const auto* qfns = ggml_get_type_traits(type);
    const auto* qfns_cpu = ggml_get_type_traits_cpu(type);
    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Quant functions not available\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return true;
    }

    // Generate and quantize weights
    std::vector<float> weight_f32(n_rows * k);
    fill_random(weight_f32.data(), n_rows * k, 42);

    size_t block_size = qfns->blck_size;
    size_t type_size = qfns->type_size;
    size_t nblocks = (n_rows * k) / block_size;
    size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_f32.data(), weight_quant.data(), n_rows * k);

    bool all_pass = true;

    // Test 1: DMMV (batch=1)
    {
        printf("  Testing DMMV (batch=1)...\n");
        const int n_tokens = 1;

        std::vector<float> input_data(k * n_tokens);
        fill_random(input_data.data(), k * n_tokens, 123);

        std::vector<float> cpu_out, sycl_out;

        // CPU reference (no SoA)
        bool cpu_ok = run_mul_mat(cpu_backend, type, weight_quant.data(), weight_bytes,
                                   input_data.data(), k, n_rows, n_tokens, cpu_out, false);

        // SYCL with SoA auto-convert (mark_as_weights=true)
        bool sycl_ok = run_mul_mat(sycl_backend, type, weight_quant.data(), weight_bytes,
                                    input_data.data(), k, n_rows, n_tokens, sycl_out, true);

        if (!cpu_ok || !sycl_ok) {
            printf("    FAIL: Compute error\n");
            all_pass = false;
        } else {
            float diff = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());
            printf("    max_diff: %.6f\n", diff);

            const float tolerance = 0.5f;
            if (diff > tolerance) {
                printf("    FAIL: SYCL differs from CPU (max=%.6f > %.6f)\n", diff, tolerance);
                // Print first few values for debugging
                printf("    First 5 values:\n");
                for (int i = 0; i < 5 && i < (int)cpu_out.size(); i++) {
                    printf("      [%d] CPU=%.4f, SYCL=%.4f, diff=%.4f\n",
                           i, cpu_out[i], sycl_out[i], fabsf(cpu_out[i] - sycl_out[i]));
                }
                all_pass = false;
            } else {
                printf("    PASS\n");
            }
        }
    }

    // Test 2: MMQ (batch=8)
    {
        printf("  Testing MMQ (batch=8)...\n");
        const int n_tokens = 8;

        std::vector<float> input_data(k * n_tokens);
        fill_random(input_data.data(), k * n_tokens, 456);

        std::vector<float> cpu_out, sycl_out;

        // CPU reference (no SoA)
        bool cpu_ok = run_mul_mat(cpu_backend, type, weight_quant.data(), weight_bytes,
                                   input_data.data(), k, n_rows, n_tokens, cpu_out, false);

        // SYCL with SoA auto-convert (mark_as_weights=true)
        bool sycl_ok = run_mul_mat(sycl_backend, type, weight_quant.data(), weight_bytes,
                                    input_data.data(), k, n_rows, n_tokens, sycl_out, true);

        if (!cpu_ok || !sycl_ok) {
            printf("    FAIL: Compute error\n");
            all_pass = false;
        } else {
            float diff = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());
            printf("    max_diff: %.6f\n", diff);

            const float tolerance = 0.5f;
            if (diff > tolerance) {
                printf("    FAIL: SYCL differs from CPU (max=%.6f > %.6f)\n", diff, tolerance);
                // Print first few values for debugging
                printf("    First 5 values:\n");
                for (int i = 0; i < 5 && i < (int)cpu_out.size(); i++) {
                    printf("      [%d] CPU=%.4f, SYCL=%.4f, diff=%.4f\n",
                           i, cpu_out[i], sycl_out[i], fabsf(cpu_out[i] - sycl_out[i]));
                }
                all_pass = false;
            } else {
                printf("    PASS\n");
            }
        }
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);

    return all_pass;
}

int main() {
    printf("SYCL SoA Auto-Convert Test\n");
    printf("==========================\n");
    printf("Compares CPU reference vs SYCL with SoA auto-conversion.\n");
    printf("Tests both DMMV (batch=1) and MMQ (batch>1) paths.\n");

    ggml_cpu_init();

    // Test dimensions matching Mistral 7B
    const int k = 4096;       // embedding dim
    const int n_rows = 11008; // FFN hidden dim

    int failed = 0;

    if (!test_quant_type("Q8_0", GGML_TYPE_Q8_0, k, n_rows)) failed++;
    if (!test_quant_type("Q4_0", GGML_TYPE_Q4_0, k, n_rows)) failed++;
    if (!test_quant_type("Q6_K", GGML_TYPE_Q6_K, k, n_rows)) failed++;

    printf("\n==========================\n");
    if (failed == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("%d TESTS FAILED\n", failed);
        return 1;
    }
}
