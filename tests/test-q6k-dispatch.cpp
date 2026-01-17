// Test for Q6_K MMVQ dispatch using ACTUAL ggml backend API
// This test exercises the full dispatch path including SoA/AoS handling
//
// Build: cmake --build build --target test-q6k-dispatch
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-q6k-dispatch
//
// Environment variables:
//   GGML_SYCL_DISABLE_GRAPH=1 - Disable SYCL graphs
//   (default)                - SoA optimization enabled

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Use actual ggml headers
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

// Include quants header for quantization/dequantization functions
#include "ggml-quants.h"

// Constants from ggml-common.h
#define QK_K 256
#define QK8_1 32

static bool parse_layout_arg(const char * arg, ggml_layout_mode & out) {
    if (!arg) {
        return false;
    }
    if (strcmp(arg, "aos") == 0) {
        out = GGML_LAYOUT_AOS;
        return true;
    }
    if (strcmp(arg, "soa") == 0) {
        out = GGML_LAYOUT_SOA;
        return true;
    }
    if (strcmp(arg, "coalesced") == 0) {
        out = GGML_LAYOUT_COALESCED;
        return true;
    }
    if (strcmp(arg, "xmx_tiled") == 0) {
        out = GGML_LAYOUT_XMX_TILED;
        return true;
    }
    if (strcmp(arg, "xmx_gemm_tiled") == 0) {
        out = GGML_LAYOUT_XMX_GEMM_TILED;
        return true;
    }
    return false;
}

static const char * layout_mode_name(ggml_layout_mode mode) {
    switch (mode) {
        case GGML_LAYOUT_AOS:
            return "aos";
        case GGML_LAYOUT_SOA:
            return "soa";
        case GGML_LAYOUT_COALESCED:
            return "coalesced";
        case GGML_LAYOUT_XMX_TILED:
            return "xmx_tiled";
        case GGML_LAYOUT_XMX_GEMM_TILED:
            return "xmx_gemm_tiled";
        default:
            return "unknown";
    }
}

// CPU reference: compute dot product of Q6_K row with F32 vector
// Uses dequantization for accuracy
static float cpu_dot_q6k_f32(const void* x_data, const float* y, int ncols) {
    // Dequantize Q6_K to float
    std::vector<float> x_f32(ncols);
    dequantize_row_q6_K((const block_q6_K*)x_data, x_f32.data(), ncols);

    // Compute dot product
    float sum = 0.0f;
    for (int i = 0; i < ncols; i++) {
        sum += x_f32[i] * y[i];
    }
    return sum;
}

// Test 1: Basic Q6_K MUL_MAT with single token (MMVQ path)
bool test_q6k_mul_mat_single_token() {
    printf("Test 1: Q6_K MUL_MAT single token (MMVQ dispatch path)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }
    printf("  Backend: %s\n", ggml_backend_name(backend));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Use realistic dimensions from Mistral 7B
    const int n_embd = 4096;
    const int n_vocab = 32000;
    const int n_tokens = 1;  // Single token = MMVQ path

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create weight tensor (Q6_K)
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_vocab);
    ggml_set_name(weight, "lm_head");

    // Create input tensor (F32)
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "hidden_state");

    // Create output tensor
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "logits");

    // Allocate weight buffer (with SoA reordering if enabled)
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

    // Create random test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create float data for quantization
    const int weight_floats = n_vocab * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    // Quantize to Q6_K using production function
    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_vocab * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_vocab, n_embd, nullptr);

    // Set weight data
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    // Create input data
    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Build and execute graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("  Executing MUL_MAT graph...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Get GPU output
    std::vector<float> gpu_output(n_vocab);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_vocab * sizeof(float));

    // Compute CPU reference for first few rows
    printf("  Computing CPU reference...\n");
    const int test_rows = std::min(16, n_vocab);
    int mismatches = 0;
    float max_rel_error = 0.0f;

    for (int row = 0; row < test_rows; row++) {
        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[row * blocks_per_row], input_f32.data(), n_embd);
        float gpu_val = gpu_output[row];

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        max_rel_error = std::max(max_rel_error, rel_error);

        if (rel_error > 0.01f) {  // 1% tolerance
            printf("  Row %d: GPU=%.6f CPU=%.6f diff=%.6f (%.2f%%)\n",
                   row, gpu_val, cpu_val, abs_diff, rel_error * 100);
            mismatches++;
        }
    }

    printf("  Max relative error: %.4f%%\n", max_rel_error * 100);
    printf("  Weight extra ptr: %p\n", weight->extra);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d/%d rows have >1%% error\n", mismatches, test_rows);
        return false;
    }

    printf("  PASS: All %d test rows match CPU reference within 1%%\n", test_rows);
    return true;
}

// Test 2: Q6_K with dimensions matching actual Mistral model layers
bool test_q6k_mistral_dimensions() {
    printf("\nTest 2: Q6_K with Mistral 7B layer dimensions\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Mistral 7B FFN dimensions
    const int n_embd = 4096;
    const int n_ff = 14336;  // Mistral 7B FFN intermediate size
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // FFN gate weight (Q6_K)
    struct ggml_tensor* gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff);
    ggml_set_name(gate, "ffn.gate");

    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "hidden");

    struct ggml_tensor* output = ggml_mul_mat(ctx, gate, input);
    ggml_set_name(output, "gate_out");

    // Allocate buffers
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, gate);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, gate, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate test data
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int weight_floats = n_ff * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) {
        weight_f32[i] = dist(rng);
    }

    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_ff * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_ff, n_embd, nullptr);

    ggml_backend_tensor_set(gate, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) {
        input_f32[i] = dist(rng);
    }
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Get GPU output and compare
    std::vector<float> gpu_output(n_ff);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_ff * sizeof(float));

    // Test specific rows (first, middle, last)
    int test_indices[] = {0, 100, 1000, 5000, 10000, n_ff - 1};
    int mismatches = 0;
    float max_rel_error = 0.0f;

    printf("  Sample row comparisons:\n");
    for (int idx : test_indices) {
        if (idx >= n_ff) continue;

        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[idx * blocks_per_row], input_f32.data(), n_embd);
        float gpu_val = gpu_output[idx];

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        max_rel_error = std::max(max_rel_error, rel_error);

        const char* status_str = (rel_error <= 0.01f) ? "OK" : "FAIL";
        printf("    Row %5d: GPU=%10.4f CPU=%10.4f err=%.4f%% %s\n",
               idx, gpu_val, cpu_val, rel_error * 100, status_str);

        if (rel_error > 0.01f) mismatches++;
    }

    printf("  Max relative error: %.4f%%\n", max_rel_error * 100);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d sample rows have >1%% error\n", mismatches);
        return false;
    }

    printf("  PASS: All sample rows match CPU reference\n");
    return true;
}

// Test 3: Compare output determinism (run twice, compare)
bool test_q6k_determinism() {
    printf("\nTest 3: Q6_K output determinism\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_embd = 4096;
    const int n_rows = 1024;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_rows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Generate and set data
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int weight_floats = n_rows * n_embd;
    std::vector<float> weight_f32(weight_floats);
    for (int i = 0; i < weight_floats; i++) weight_f32[i] = dist(rng);

    const int blocks_per_row = n_embd / QK_K;
    const int total_blocks = n_rows * blocks_per_row;
    std::vector<block_q6_K> weight_q6k(total_blocks);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_rows, n_embd, nullptr);
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, total_blocks * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd);
    for (int i = 0; i < n_embd; i++) input_f32[i] = dist(rng);
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Run twice
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    std::vector<float> output1(n_rows), output2(n_rows);

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(output, output1.data(), 0, n_rows * sizeof(float));

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(output, output2.data(), 0, n_rows * sizeof(float));

    // Compare
    int diffs = 0;
    for (int i = 0; i < n_rows; i++) {
        if (output1[i] != output2[i]) {
            if (diffs < 5) {
                printf("  Row %d: run1=%.8f run2=%.8f\n", i, output1[i], output2[i]);
            }
            diffs++;
        }
    }

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (diffs > 0) {
        printf("  FAIL: %d/%d rows differ between runs (non-deterministic!)\n", diffs, n_rows);
        return false;
    }

    printf("  PASS: Output is deterministic\n");
    return true;
}

// Test 4: Small dimensions edge case
bool test_q6k_small_dimensions() {
    printf("\nTest 4: Q6_K small dimension edge case\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Minimum valid Q6_K dimensions (256 elements = 1 block)
    const int n_embd = 256;  // QK_K
    const int n_rows = 16;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_rows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    // Allocate
    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void*)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Use simple predictable values
    std::vector<float> weight_f32(n_rows * n_embd);
    for (int i = 0; i < n_rows * n_embd; i++) {
        weight_f32[i] = 0.01f * ((i % 100) - 50);  // -0.5 to 0.49
    }

    const int blocks_per_row = n_embd / QK_K;  // = 1
    std::vector<block_q6_K> weight_q6k(n_rows * blocks_per_row);
    quantize_q6_K(weight_f32.data(), weight_q6k.data(), n_rows, n_embd, nullptr);
    ggml_backend_tensor_set(weight, weight_q6k.data(), 0, n_rows * blocks_per_row * sizeof(block_q6_K));

    std::vector<float> input_f32(n_embd, 1.0f);  // All ones
    ggml_backend_tensor_set(input, input_f32.data(), 0, n_embd * sizeof(float));

    // Execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    std::vector<float> gpu_output(n_rows);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, n_rows * sizeof(float));

    // Compare all rows
    int mismatches = 0;
    float max_rel_error = 0.0f;

    for (int row = 0; row < n_rows; row++) {
        float cpu_val = cpu_dot_q6k_f32(&weight_q6k[row * blocks_per_row], input_f32.data(), n_embd);
        float gpu_val = gpu_output[row];

        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_error = (std::abs(cpu_val) > 1e-6f) ? abs_diff / std::abs(cpu_val) : abs_diff;
        max_rel_error = std::max(max_rel_error, rel_error);

        if (rel_error > 0.01f) {
            printf("  Row %d: GPU=%.6f CPU=%.6f err=%.2f%%\n", row, gpu_val, cpu_val, rel_error * 100);
            mismatches++;
        }
    }

    printf("  Max relative error: %.4f%%\n", max_rel_error * 100);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (mismatches > 0) {
        printf("  FAIL: %d/%d rows have >1%% error\n", mismatches, n_rows);
        return false;
    }

    printf("  PASS: Small dimension edge case works correctly\n");
    return true;
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("Q6_K Dispatch Unit Test\n");
    printf("========================================\n\n");

    // Check environment
    const char * disable_graph = getenv("GGML_SYCL_DISABLE_GRAPH");
    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    bool has_override = false;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (!arg) {
            continue;
        }
        const char * value = nullptr;
        if (strncmp(arg, "--layout=", 9) == 0) {
            value = arg + 9;
        } else if (strcmp(arg, "--layout") == 0 && i + 1 < argc) {
            value = argv[++i];
        }
        if (value && parse_layout_arg(value, override_layout)) {
            ggml_sycl::test_set_layout_override(override_layout);
            has_override = true;
            break;
        } else if (value) {
            printf("WARNING: unknown --layout=%s (ignoring)\n", value);
        }
    }
    printf("Environment:\n");
    printf("  Layout override: %s\n", has_override ? layout_mode_name(override_layout) : "(auto)");
    printf("  GGML_SYCL_DISABLE_GRAPH: %s\n", disable_graph ? disable_graph : "(not set, graphs enabled)");
    printf("\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    // Run tests
    bool result;

    result = test_q6k_mul_mat_single_token();
    if (result) passed++; else failed++;

    result = test_q6k_mistral_dimensions();
    if (result) passed++; else failed++;

    result = test_q6k_determinism();
    if (result) passed++; else failed++;

    result = test_q6k_small_dimensions();
    if (result) passed++; else failed++;

    // Summary
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    printf("========================================\n");

    if (has_override) {
        ggml_sycl::test_clear_layout_override();
    }

    return (failed > 0) ? 1 : 0;
}
