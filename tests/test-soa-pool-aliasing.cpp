// Test for SoA buffer pool aliasing bug
//
// This test verifies that compute buffer pool allocations work correctly
// when SoA weight reordering is enabled. The bug manifests as:
// - Activation buffers returning zeros when SoA is enabled
// - Same pointer addresses returning different data with SoA on vs off
//
// Build: cmake --build build --target test-soa-pool-aliasing
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-soa-pool-aliasing
//
// Compare:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-soa-pool-aliasing  # Auto layout
//   GGML_SYCL_LAYOUT_OVERRIDE=aos ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-soa-pool-aliasing  # AoS only

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

// Simulate Mistral 7B dimensions
static const int N_EMBD = 4096;
static const int N_FF = 14336;
static const int N_TOKENS = 1;  // Decode phase

// Check if output contains valid (non-zero) values
static bool check_output_valid(const float* data, int n, const char* name, bool verbose = false) {
    float abs_sum = 0.0f;
    float max_val = 0.0f;
    int non_zero = 0;

    for (int i = 0; i < n; i++) {
        float v = std::fabs(data[i]);
        abs_sum += v;
        if (v > max_val) max_val = v;
        if (v > 1e-10f) non_zero++;
    }

    bool valid = (abs_sum > 1e-6f) && (non_zero > 0);

    if (verbose || !valid) {
        printf("  %s: n=%d abs_sum=%.6f max=%.6f non_zero=%d => %s\n",
               name, n, abs_sum, max_val, non_zero, valid ? "VALID" : "ZEROS/INVALID");
        if (n >= 4) {
            printf("    first 4: [%.6f, %.6f, %.6f, %.6f]\n",
                   data[0], data[1], data[2], data[3]);
        }
    }

    return valid;
}

// Test: Sequential MUL_MAT operations (simulates transformer layer)
// This is the critical test - it checks that activation buffers from the pool
// contain valid data after SoA weight allocation
bool test_sequential_matmul() {
    printf("\n=== Test: Sequential MUL_MAT with pool buffer reuse ===\n");
    printf("This simulates a transformer layer with Q/K/V/O projections\n\n");

    // Check layout override mode
    const char * override_env = getenv("GGML_SYCL_LAYOUT_OVERRIDE");
    const bool   aos_only     = (override_env && strcmp(override_env, "aos") == 0);
    printf("Layout override: %s\n", override_env ? override_env : "(auto)");
    printf("Reorder enabled: %s\n", aos_only ? "no" : "yes");

    // Initialize SYCL backend
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return true;
    }
    printf("Backend: %s\n\n", ggml_backend_name(backend));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Create ggml context
    struct ggml_init_params params;
    params.mem_size   = 64 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);

    // Create weight tensors (Q8_0 - will trigger SoA reordering)
    // These simulate attention Q/K/V/O projection weights
    struct ggml_tensor* wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);
    struct ggml_tensor* wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);
    struct ggml_tensor* wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);
    struct ggml_tensor* wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD);

    // Create input/intermediate tensors (F32 - will use pool buffers)
    struct ggml_tensor* hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);

    ggml_set_name(wq, "wq");
    ggml_set_name(wk, "wk");
    ggml_set_name(wv, "wv");
    ggml_set_name(wo, "wo");
    ggml_set_name(hidden, "hidden");

    // Allocate weight buffer (with SoA reordering)
    size_t weight_size = 4 * ggml_backend_buft_get_alloc_size(buft, wq);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_size);
    if (!weight_buffer) {
        printf("FAIL: Could not allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    printf("Allocated weight buffer: %zu bytes\n", weight_size);

    // Allocate compute buffer (for input/intermediate)
    size_t compute_size = 64 * 1024 * 1024;  // 64 MB compute buffer
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, compute_size);
    if (!compute_buffer) {
        printf("FAIL: Could not allocate compute buffer\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    printf("Allocated compute buffer: %zu bytes\n\n", compute_size);

    // Use gallocr for tensor allocation (simpler than scheduler for this test)

    // Initialize weight data with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Q8_0 block structure
    const int block_size = 32;  // QK8_0
    const int n_blocks = (N_EMBD * N_EMBD) / block_size;

    // Allocate temporary buffer for weight initialization
    size_t q8_block_bytes = sizeof(int8_t) * block_size + sizeof(uint16_t);  // qs + d
    std::vector<uint8_t> weight_data(n_blocks * q8_block_bytes);

    // Fill with pattern that produces known outputs
    for (int b = 0; b < n_blocks; b++) {
        uint8_t* block = weight_data.data() + b * q8_block_bytes;
        // Set scale d = 0.01 (fp16)
        uint16_t* d = reinterpret_cast<uint16_t*>(block);
        *d = 0x211E;  // ~0.01 in fp16
        // Set qs values
        int8_t* qs = reinterpret_cast<int8_t*>(block + sizeof(uint16_t));
        for (int i = 0; i < block_size; i++) {
            qs[i] = static_cast<int8_t>((b + i) % 127);
        }
    }

    // Use gallocr for tensor allocation
    ggml_gallocr_t allocr = ggml_gallocr_new(buft);

    // Create compute graph: hidden -> mul_mat(wq) -> q_out
    // This tests that the activation buffer (hidden) has valid data after SoA weight allocation
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // Build sequential matmul chain
    struct ggml_tensor* q_out = ggml_mul_mat(ctx, wq, hidden);
    struct ggml_tensor* k_out = ggml_mul_mat(ctx, wk, hidden);
    struct ggml_tensor* v_out = ggml_mul_mat(ctx, wv, hidden);
    // Combine outputs (simplified)
    struct ggml_tensor* combined = ggml_add(ctx, ggml_add(ctx, q_out, k_out), v_out);
    struct ggml_tensor* final_out = ggml_mul_mat(ctx, wo, combined);

    ggml_set_name(q_out, "q_out");
    ggml_set_name(k_out, "k_out");
    ggml_set_name(v_out, "v_out");
    ggml_set_name(combined, "combined");
    ggml_set_name(final_out, "final_out");

    ggml_build_forward_expand(gf, final_out);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Reserve memory for graph
    if (!ggml_gallocr_reserve(allocr, gf)) {
        printf("FAIL: Could not reserve graph memory\n");
        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Allocate graph tensors
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        printf("FAIL: Could not allocate graph tensors\n");
        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Initialize hidden state with known non-zero values
    std::vector<float> hidden_data(N_EMBD * N_TOKENS);
    for (int i = 0; i < N_EMBD; i++) {
        hidden_data[i] = dist(rng);
    }
    printf("Input hidden state abs_sum: %.6f\n",
           std::accumulate(hidden_data.begin(), hidden_data.end(), 0.0f,
                          [](float a, float b) { return a + std::fabs(b); }));

    // Set tensor data
    ggml_backend_tensor_set(wq, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(wk, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(wv, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(wo, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));

    printf("Set all tensor data\n\n");

    // Execute graph
    printf("Executing graph...\n");
    ggml_backend_graph_compute(backend, gf);
    printf("Graph execution complete\n\n");

    // Read back results
    std::vector<float> q_result(N_EMBD * N_TOKENS);
    std::vector<float> k_result(N_EMBD * N_TOKENS);
    std::vector<float> v_result(N_EMBD * N_TOKENS);
    std::vector<float> final_result(N_EMBD * N_TOKENS);

    ggml_backend_tensor_get(q_out, q_result.data(), 0, q_result.size() * sizeof(float));
    ggml_backend_tensor_get(k_out, k_result.data(), 0, k_result.size() * sizeof(float));
    ggml_backend_tensor_get(v_out, v_result.data(), 0, v_result.size() * sizeof(float));
    ggml_backend_tensor_get(final_out, final_result.data(), 0, final_result.size() * sizeof(float));

    printf("Checking outputs:\n");
    bool q_valid = check_output_valid(q_result.data(), N_EMBD * N_TOKENS, "q_out", true);
    bool k_valid = check_output_valid(k_result.data(), N_EMBD * N_TOKENS, "k_out", true);
    bool v_valid = check_output_valid(v_result.data(), N_EMBD * N_TOKENS, "v_out", true);
    bool final_valid = check_output_valid(final_result.data(), N_EMBD * N_TOKENS, "final_out", true);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("\n");
    if (q_valid && k_valid && v_valid && final_valid) {
        printf("PASS: All outputs contain valid non-zero values\n");
        return true;
    } else {
        printf("FAIL: Some outputs contain zeros or invalid values\n");
        printf("  This indicates the buffer pool returned corrupted data when SoA is enabled\n");
        return false;
    }
}

// Test: Multiple iterations to check pool buffer cycling
bool test_pool_buffer_cycling() {
    printf("\n=== Test: Pool buffer cycling with multiple iterations ===\n");
    printf("This tests that pool buffers work correctly across multiple graph executions\n\n");

    const char * override_env = getenv("GGML_SYCL_LAYOUT_OVERRIDE");
    const bool   aos_only     = (override_env && strcmp(override_env, "aos") == 0);
    printf("Layout override: %s\n", override_env ? override_env : "(auto)");
    printf("Reorder enabled: %s\n", aos_only ? "no" : "yes");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params;
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);

    // Smaller dimensions for faster iteration
    const int dim = 1024;
    const int n_iter = 10;

    // Create weight tensor (Q8_0)
    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, dim, dim);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);

    ggml_set_name(weight, "weight");
    ggml_set_name(input, "input");

    // Create allocator
    ggml_gallocr_t allocr = ggml_gallocr_new(buft);

    // Build simple graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "output");
    ggml_build_forward_expand(gf, output);

    if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
        printf("FAIL: Graph allocation failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Initialize data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int block_size = 32;
    const int n_blocks = (dim * dim) / block_size;
    size_t q8_block_bytes = sizeof(int8_t) * block_size + sizeof(uint16_t);
    std::vector<uint8_t> weight_data(n_blocks * q8_block_bytes);

    for (int b = 0; b < n_blocks; b++) {
        uint8_t* block = weight_data.data() + b * q8_block_bytes;
        uint16_t* d = reinterpret_cast<uint16_t*>(block);
        *d = 0x211E;  // ~0.01 in fp16
        int8_t* qs = reinterpret_cast<int8_t*>(block + sizeof(uint16_t));
        for (int i = 0; i < block_size; i++) {
            qs[i] = static_cast<int8_t>((b + i) % 127);
        }
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());

    int pass_count = 0;
    int fail_count = 0;

    printf("Running %d iterations...\n", n_iter);

    for (int iter = 0; iter < n_iter; iter++) {
        // Set input with different data each iteration
        std::vector<float> input_data(dim);
        for (int i = 0; i < dim; i++) {
            input_data[i] = dist(rng);
        }

        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Execute
        ggml_backend_graph_compute(backend, gf);

        // Check output
        std::vector<float> output_data(dim);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        char name[32];
        snprintf(name, sizeof(name), "iter_%d", iter);

        if (check_output_valid(output_data.data(), dim, name, false)) {
            pass_count++;
        } else {
            fail_count++;
            printf("  Iteration %d FAILED (zeros detected)\n", iter);
        }
    }

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("\nResults: %d/%d iterations passed\n", pass_count, n_iter);

    if (fail_count == 0) {
        printf("PASS: All iterations produced valid outputs\n");
        return true;
    } else {
        printf("FAIL: %d iterations produced zeros/invalid output\n", fail_count);
        return false;
    }
}

int main(int /*argc*/, char** /*argv*/) {
    printf("===========================================\n");
    printf("SoA Buffer Pool Aliasing Test\n");
    printf("===========================================\n");
    printf("\n");
    printf("This test checks for buffer aliasing issues when SoA\n");
    printf("weight reordering is enabled. The bug causes activation\n");
    printf("buffers from the compute pool to contain zeros.\n");
    printf("\n");

    int n_pass = 0;
    int n_fail = 0;
    int n_skip = 0;

    // Test 1: Sequential matmul
    bool result1 = test_sequential_matmul();
    if (result1) n_pass++; else n_fail++;

    // Test 2: Pool buffer cycling
    bool result2 = test_pool_buffer_cycling();
    if (result2) n_pass++; else n_fail++;

    printf("\n===========================================\n");
    printf("Summary: %d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    printf("===========================================\n");

    return (n_fail > 0) ? 1 : 0;
}
