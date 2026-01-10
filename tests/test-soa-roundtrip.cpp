// SoA Round-Trip Test: Verify SYCL SoA kernels produce same results as CPU reference
// Tests actual GPU DMMV/MMQ kernels for Q4_0, Q8_0, and Q6_K
//
// Build: cmake --build build --target test-soa-roundtrip
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-soa-roundtrip
//
// This test runs MUL_MAT on both SYCL (with SoA) and CPU (reference),
// then compares results. They must match within tolerance.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Use actual ggml headers - production functions
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

// Calculate max absolute difference
static float max_diff(const float* a, const float* b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static void get_tolerances(ggml_type type, bool is_mmq, float & abs_tol, float & rel_tol) {
    abs_tol = 0.1f;
    rel_tol = 0.15f;

    if (type == GGML_TYPE_Q4_0) {
        abs_tol = 0.8f;
        rel_tol = 0.30f;
    } else if (type == GGML_TYPE_Q4_K) {
        abs_tol = 1.0f;
        rel_tol = 0.35f;
    } else if (type == GGML_TYPE_Q5_K) {
        abs_tol = 0.8f;
        rel_tol = 0.30f;
    } else if (type == GGML_TYPE_Q6_K && !is_mmq) {
        abs_tol = 0.5f;
        rel_tol = 0.20f;
    }
}

static void dequantize_row_q8_1_ref(const block_q8_1 * x, float * y, int64_t k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_1; ++j) {
            y[i * QK8_1 + j] = d * x[i].qs[j];
        }
    }
}

static void build_input_reference(ggml_type weight_type, int n_embd, int n_tokens,
                                  const float * input_data, std::vector<float> & input_ref) {
    input_ref.resize(n_embd * n_tokens);
    if (n_tokens == 1) {
        if (weight_type == GGML_TYPE_Q4_0 || weight_type == GGML_TYPE_Q8_0) {
            const int blocks = n_embd / QK8_0;
            std::vector<block_q8_0> q8(blocks);
            quantize_row_q8_0_ref(input_data, q8.data(), n_embd);
            dequantize_row_q8_0(q8.data(), input_ref.data(), n_embd);
            return;
        }
        // Q6_K DMMV uses float input
        memcpy(input_ref.data(), input_data, n_embd * sizeof(float));
        return;
    }

    const int blocks = n_embd / QK8_1;
    std::vector<block_q8_1> q8(n_tokens * blocks);
    for (int t = 0; t < n_tokens; ++t) {
        quantize_row_q8_1_ref(input_data + t * n_embd, q8.data() + t * blocks, n_embd);
        dequantize_row_q8_1_ref(q8.data() + t * blocks, input_ref.data() + t * n_embd, n_embd);
    }
}

static bool run_reference_mul_mat(ggml_type weight_type, const void * weight_data,
                                  const float * input_data, int n_embd, int n_rows, int n_tokens,
                                  std::vector<float> & output) {
    std::vector<float> input_ref;
    build_input_reference(weight_type, n_embd, n_tokens, input_data, input_ref);
    output.assign(n_rows * n_tokens, 0.0f);

    int blocks_per_row = 0;
    switch (weight_type) {
        case GGML_TYPE_Q4_0:
            blocks_per_row = n_embd / QK4_0;
            break;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            blocks_per_row = n_embd / QK_K;
            break;
        case GGML_TYPE_Q8_0:
            blocks_per_row = n_embd / QK8_0;
            break;
        case GGML_TYPE_Q6_K:
            blocks_per_row = n_embd / QK_K;
            break;
        default:
            return false;
    }
    std::vector<float> weight_f32(n_embd);

    for (int row = 0; row < n_rows; ++row) {
        const int row_offset = row * blocks_per_row;
        switch (weight_type) {
            case GGML_TYPE_Q4_0:
                dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(weight_data) + row_offset,
                                    weight_f32.data(), n_embd);
                break;
            case GGML_TYPE_Q4_K:
                dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(weight_data) + row_offset,
                                    weight_f32.data(), n_embd);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(weight_data) + row_offset,
                                    weight_f32.data(), n_embd);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(weight_data) + row_offset,
                                    weight_f32.data(), n_embd);
                break;
            case GGML_TYPE_Q6_K:
                dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(weight_data) + row_offset,
                                    weight_f32.data(), n_embd);
                break;
            default:
                return false;
        }

        for (int t = 0; t < n_tokens; ++t) {
            float sum = 0.0f;
            const float * x = input_ref.data() + t * n_embd;
            for (int k = 0; k < n_embd; ++k) {
                sum += weight_f32[k] * x[k];
            }
            output[t * n_rows + row] = sum;
        }
    }

    return true;
}

// Run MUL_MAT using the actual ggml backend (invokes real kernels)
static bool run_mul_mat(ggml_backend_t backend, ggml_type weight_type,
                        const void* weight_data, size_t weight_size,
                        const float* input_data, int n_embd, int n_rows, int n_tokens,
                        std::vector<float>& output) {

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    // Create weight tensor [n_embd, n_rows] - this gets SoA reordered on SYCL
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    ggml_set_name(weight, "weight");

    // Create input tensor [n_embd, n_tokens]
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "input");

    // Create output via MUL_MAT: weight^T @ input -> [n_rows, n_tokens]
    struct ggml_tensor* out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    // Allocate weight buffer (triggers SoA reordering for SYCL backend)
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    // Mark as weights buffer - this triggers SoA reordering on init_tensor
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
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

    // Set weight data using backend API (triggers reordering for SYCL)
    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);

    // Set input data
    ggml_backend_tensor_set(input, input_data, 0, n_embd * n_tokens * sizeof(float));

    // Build compute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Execute graph using ACTUAL backend kernels (DMMV for n_tokens=1, MMQ for n_tokens>1)
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

// Test a specific quantization type with DMMV path (batch=1)
static bool test_dmmv(const char* type_name, ggml_type type, int n_embd, int n_rows) {
    printf("\n=== Testing %s DMMV (batch=1) ===\n", type_name);

    // Initialize backends
    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    // Get quantization functions to generate test data
    const auto* qfns = ggml_get_type_traits(type);
    const auto* qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Quantization functions not available for %s\n", type_name);
        ggml_backend_free(sycl_backend);
        return true;
    }

    // Generate random float data for weights
    std::vector<float> weight_float(n_rows * n_embd);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : weight_float) v = dist(rng);

    // Quantize using production ggml functions
    size_t block_size = qfns->blck_size;
    size_t type_size = qfns->type_size;
    size_t n_elements = n_rows * n_embd;
    size_t nblocks = n_elements / block_size;
    size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_float.data(), weight_quant.data(), n_elements);

    // Generate random input data
    const int n_tokens = 1;  // DMMV path
    std::vector<float> input_data(n_embd * n_tokens);
    for (auto& v : input_data) v = dist(rng);

    // Run on SYCL (invokes actual SoA DMMV kernel)
    std::vector<float> sycl_output;
    printf("  Running on SYCL (SoA DMMV kernel)...\n");
    bool sycl_ok = run_mul_mat(sycl_backend, type, weight_quant.data(), weight_bytes,
                               input_data.data(), n_embd, n_rows, n_tokens,
                               sycl_output);
    if (!sycl_ok) {
        printf("  FAIL: SYCL compute failed\n");
        ggml_backend_free(sycl_backend);
        return false;
    }

    // Run CPU reference that mirrors SYCL quantized input path
    std::vector<float> cpu_output;
    printf("  Running on CPU (quantized reference)...\n");
    if (!run_reference_mul_mat(type, weight_quant.data(), input_data.data(), n_embd, n_rows, n_tokens, cpu_output)) {
        printf("  FAIL: CPU reference compute failed\n");
        ggml_backend_free(sycl_backend);
        return false;
    }
    ggml_backend_free(sycl_backend);

    // Compare results
    float max_d = max_diff(sycl_output.data(), cpu_output.data(), sycl_output.size());

    // Check for garbage output
    int sycl_nonzero = 0, cpu_nonzero = 0;
    bool has_nan = false;
    for (size_t i = 0; i < sycl_output.size(); i++) {
        if (sycl_output[i] != 0.0f) sycl_nonzero++;
        if (cpu_output[i] != 0.0f) cpu_nonzero++;
        if (std::isnan(sycl_output[i]) || std::isinf(sycl_output[i])) has_nan = true;
    }

    printf("  SYCL: %d/%zu non-zero values\n", sycl_nonzero, sycl_output.size());
    printf("  CPU:  %d/%zu non-zero values\n", cpu_nonzero, cpu_output.size());
    printf("  Max difference: %.6f\n", max_d);

    // Print first few values for debugging
    printf("  First 5 values:\n");
    for (int i = 0; i < 5 && i < (int)sycl_output.size(); i++) {
        printf("    [%d] SYCL=%.6f, CPU=%.6f, diff=%.6f\n",
               i, sycl_output[i], cpu_output[i], fabsf(sycl_output[i] - cpu_output[i]));
    }

    // Tolerance for quantization + float precision
    float abs_tol = 0.0f;
    float rel_tol = 0.0f;
    get_tolerances(type, false, abs_tol, rel_tol);

    if (has_nan) {
        printf("  FAIL: SYCL output contains NaN/Inf\n");
        return false;
    }
    if (sycl_nonzero == 0) {
        printf("  FAIL: SYCL output is all zeros\n");
        return false;
    }
    float max_rel = 0.0f;
    for (size_t i = 0; i < sycl_output.size(); ++i) {
        const float diff = fabsf(sycl_output[i] - cpu_output[i]);
        const float denom = fmaxf(1.0f, fabsf(cpu_output[i]));
        max_rel = fmaxf(max_rel, diff / denom);
    }

    if (max_d > abs_tol && max_rel > rel_tol) {
        printf("  FAIL: Max difference %.6f exceeds tolerance %.6f\n", max_d, abs_tol);
        return false;
    }

    printf("  PASS: SYCL output matches CPU reference\n");
    return true;
}

// Test a specific quantization type with MMQ path (batch>1)
static bool test_mmq(const char* type_name, ggml_type type, int n_embd, int n_rows) {
    printf("\n=== Testing %s MMQ (batch=8) ===\n", type_name);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    const auto* qfns = ggml_get_type_traits(type);
    const auto* qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Quantization functions not available for %s\n", type_name);
        ggml_backend_free(sycl_backend);
        return true;
    }

    // Generate and quantize weights
    std::vector<float> weight_float(n_rows * n_embd);
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : weight_float) v = dist(rng);

    size_t block_size = qfns->blck_size;
    size_t type_size = qfns->type_size;
    size_t nblocks = (n_rows * n_embd) / block_size;
    size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_float.data(), weight_quant.data(), n_rows * n_embd);

    // Generate input data for batch processing
    const int n_tokens = 8;  // MMQ path
    std::vector<float> input_data(n_embd * n_tokens);
    for (auto& v : input_data) v = dist(rng);

    std::vector<float> sycl_output, cpu_output;

    printf("  Running on SYCL (SoA MMQ kernel)...\n");
    bool sycl_ok = run_mul_mat(sycl_backend, type, weight_quant.data(), weight_bytes,
                               input_data.data(), n_embd, n_rows, n_tokens,
                               sycl_output);

    printf("  Running on CPU (quantized reference)...\n");
    bool cpu_ok = run_reference_mul_mat(type, weight_quant.data(),
                                        input_data.data(), n_embd, n_rows, n_tokens,
                                        cpu_output);
    ggml_backend_free(sycl_backend);

    if (!sycl_ok || !cpu_ok) {
        printf("  FAIL: Compute failed\n");
        return false;
    }

    float max_d = max_diff(sycl_output.data(), cpu_output.data(), sycl_output.size());

    int sycl_nonzero = 0;
    bool has_nan = false;
    for (size_t i = 0; i < sycl_output.size(); i++) {
        if (sycl_output[i] != 0.0f) sycl_nonzero++;
        if (std::isnan(sycl_output[i]) || std::isinf(sycl_output[i])) has_nan = true;
    }

    printf("  SYCL: %d/%zu non-zero, max_diff=%.6f\n", sycl_nonzero, sycl_output.size(), max_d);

    float abs_tol = 0.0f;
    float rel_tol = 0.0f;
    get_tolerances(type, true, abs_tol, rel_tol);
    float max_rel = 0.0f;
    for (size_t i = 0; i < sycl_output.size(); ++i) {
        const float diff = fabsf(sycl_output[i] - cpu_output[i]);
        const float denom = fmaxf(1.0f, fabsf(cpu_output[i]));
        max_rel = fmaxf(max_rel, diff / denom);
    }

    if (has_nan || sycl_nonzero == 0 || (max_d > abs_tol && max_rel > rel_tol)) {
        printf("  FAIL\n");
        return false;
    }

    printf("  PASS\n");
    return true;
}

int main() {
    printf("SoA Round-Trip Test: SYCL SoA Kernels vs CPU Reference\n");
    printf("=========================================================\n");
    printf("This test invokes the ACTUAL GPU kernels (DMMV/MMQ) and compares\n");
    printf("results against the CPU backend (which doesn't use SoA).\n");

    // Initialize ggml CPU for quantization functions
    ggml_cpu_init();

    // Test dimensions (matching real model usage)
    const int n_embd = 4096;
    const int n_rows = 11008;

    int num_failed = 0;

    // === Q4_0 Tests (invokes actual SoA DMMV/MMQ kernels) ===
    if (!test_dmmv("Q4_0", GGML_TYPE_Q4_0, n_embd, n_rows)) num_failed++;
    if (!test_mmq("Q4_0", GGML_TYPE_Q4_0, n_embd, n_rows)) num_failed++;

    // === Q4_K Tests ===
    if (!test_dmmv("Q4_K", GGML_TYPE_Q4_K, n_embd, n_rows)) num_failed++;
    if (!test_mmq("Q4_K", GGML_TYPE_Q4_K, n_embd, n_rows)) num_failed++;

    // === Q5_K Tests ===
    if (!test_dmmv("Q5_K", GGML_TYPE_Q5_K, n_embd, n_rows)) num_failed++;
    if (!test_mmq("Q5_K", GGML_TYPE_Q5_K, n_embd, n_rows)) num_failed++;

    // === Q8_0 Tests ===
    if (!test_dmmv("Q8_0", GGML_TYPE_Q8_0, n_embd, n_rows)) num_failed++;
    if (!test_mmq("Q8_0", GGML_TYPE_Q8_0, n_embd, n_rows)) num_failed++;

    // === Q6_K Tests (reference - should pass since Q6_K SoA works) ===
    if (!test_dmmv("Q6_K", GGML_TYPE_Q6_K, n_embd, n_rows)) num_failed++;
    if (!test_mmq("Q6_K", GGML_TYPE_Q6_K, n_embd, n_rows)) num_failed++;

    printf("\n=========================================================\n");
    if (num_failed == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("%d tests FAILED\n", num_failed);
        return 1;
    }
}
