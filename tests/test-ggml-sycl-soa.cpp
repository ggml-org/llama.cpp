// Test for SoA Q4_0 bug using ACTUAL ggml backend API
// This test uses real ggml functions to test the integration path
//
// Build: cmake --build build --target test-ggml-sycl-soa
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-ggml-sycl-soa
//
// Test helper:
//   Use --layout=<aos|soa|coalesced|xmx_tiled|xmx_gemm_tiled> to set a test-only layout override.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Use actual ggml headers
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"

// Q4_0 block structure (from ggml-common.h)
#define QK4_0 32

typedef struct {
    uint16_t d;              // delta (fp16 as uint16_t for byte-level access)
    uint8_t  qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0_test;

static_assert(sizeof(block_q4_0_test) == 18, "block_q4_0 size mismatch");

static const char * layout_override_label() {
    ggml_layout_mode layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl::test_get_layout_override(&layout)) {
        return "Auto (layout policy)";
    }
    switch (layout) {
        case GGML_LAYOUT_AOS:
            return "AoS (test override)";
        case GGML_LAYOUT_SOA:
            return "SoA (test override)";
        case GGML_LAYOUT_COALESCED:
            return "Coalesced (test override)";
        case GGML_LAYOUT_XMX_TILED:
            return "XMX tiled (test override)";
        case GGML_LAYOUT_XMX_GEMM_TILED:
            return "XMX GEMM tiled (test override)";
        default:
            return "Unknown override";
    }
}

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

static bool prime_layout_choice(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * weight) {
    if (!backend || !ctx || !weight) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const int64_t              ncols = weight->ne[0];
    if (ncols <= 0) {
        return false;
    }

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    if (!compute_buffer) {
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    std::vector<float> input_data((size_t) ncols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    ggml_backend_buffer_free(compute_buffer);
    return status == GGML_STATUS_SUCCESS;
}

static bool read_layout_bytes(const ggml_tensor * tensor,
                              int                 device,
                              layout_mode         layout,
                              std::vector<uint8_t> & out,
                              const char **       out_source = nullptr) {
    const char * source = nullptr;
    void *       layout_ptr = ggml_sycl_get_layout_ptr_for(tensor, device, layout, &source);
    if (!layout_ptr) {
        if (out_source) {
            *out_source = source;
        }
        return false;
    }

    const size_t bytes = ggml_nbytes(tensor);
    out.resize(bytes);
    sycl::queue & q = dpct::dev_mgr::instance().get_device(device).default_queue();
    const sycl::usm::alloc alloc = sycl::get_pointer_type(layout_ptr, q.get_context());
    if (alloc == sycl::usm::alloc::device) {
        q.memcpy(out.data(), layout_ptr, bytes).wait();
    } else {
        std::memcpy(out.data(), layout_ptr, bytes);
    }
    if (out_source) {
        *out_source = source;
    }
    return true;
}

// Helper to create Q4_0 quantized data with known values
void create_q4_0_test_data(block_q4_0_test * data, int nrows, int ncols, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int nblocks = nrows * (ncols / QK4_0);
    for (int i = 0; i < nblocks; i++) {
        // Use scale = 1.0 in fp16 format (0x3C00)
        data[i].d = 0x3C00;
        // Fill with predictable pattern
        for (int j = 0; j < QK4_0 / 2; j++) {
            // Values that give known dequantized outputs
            // Each byte: low nibble + high nibble
            data[i].qs[j] = (uint8_t) (((i + j) % 16) | (((i + j + 1) % 16) << 4));
        }
    }
}

// Test 1: Basic buffer initialization with SoA
bool test_buffer_init() {
    printf("Test 1: Buffer initialization with ggml_backend_buffer_init_tensor\n");

    // Initialize SYCL backend
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;  // Skip, not fail
    }
    printf("  Backend: %s\n", ggml_backend_name(backend));

    // Get buffer type
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Create ggml context for tensor metadata
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,  // We'll use backend buffer
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create a Q4_0 weight tensor
    const int            n_rows = 64;
    const int            n_cols = 2048;
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "test_weight");

    // Allocate buffer and initialize tensor
    size_t                buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t buffer   = ggml_backend_buft_alloc_buffer(buft, buf_size);
    if (!buffer) {
        printf("  FAIL: Could not allocate buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Mark as weights buffer (triggers SoA reordering)
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Set tensor buffer
    ggml_backend_tensor_alloc(buffer, weight, (void *) ggml_backend_buffer_get_base(buffer));

    // Create test data
    const int                    nblocks = n_rows * (n_cols / QK4_0);
    std::vector<block_q4_0_test> test_data(nblocks);
    create_q4_0_test_data(test_data.data(), n_rows, n_cols, 42);

    // Set tensor data using ACTUAL backend API
    // This should trigger init_tensor which does SoA reordering
    ggml_backend_tensor_set(weight, test_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Read back and verify we can get data
    std::vector<uint8_t> readback(nblocks * sizeof(block_q4_0_test));
    ggml_backend_tensor_get(weight, readback.data(), 0, readback.size());

    // Check that tensor->extra was set (indicates SoA was applied)
    bool has_extra = (weight->extra != nullptr);
    printf("  tensor->extra: %s\n", has_extra ? "set (SoA)" : "NULL (AoS)");

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  PASS: Buffer initialization completed\n");
    return true;
}

// Test 2: MUL_MAT graph execution with Q4_0 weights
bool test_mul_mat_graph() {
    printf("Test 2: MUL_MAT graph execution with Q4_0 weights\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Dimensions matching actual model usage
    const int n_embd   = 4096;
    const int n_ff     = 11008;
    const int n_tokens = 1;  // Decode phase - single token

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create tensors
    // Weight: Q4_0 [n_ff, n_embd] - this gets SoA reordered
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
    ggml_set_name(weight, "ffn_weight");

    // Input: F32 [n_embd, n_tokens] - this is inp_embd equivalent
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "inp_embd");

    // Output: F32 [n_ff, n_tokens]
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "ffn_output");

    // Allocate weight buffer (with SoA)
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer (for input and output)
    size_t                input_size       = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size      = ggml_backend_buft_get_alloc_size(buft, output);
    size_t                compute_buf_size = input_size + output_size + 1024;  // Extra alignment padding
    ggml_backend_buffer_t compute_buffer   = ggml_backend_buft_alloc_buffer(buft, compute_buf_size);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    // Allocate tensors in compute buffer
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Initialize weight data
    const int                    nblocks = n_ff * (n_embd / QK4_0);
    std::vector<block_q4_0_test> weight_data(nblocks);
    create_q4_0_test_data(weight_data.data(), n_ff, n_embd, 42);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Initialize input data with non-zero values
    std::vector<float> input_data(n_embd * n_tokens, 1.0f);
    for (int i = 0; i < n_embd; i++) {
        input_data[i] = 0.1f * (i % 10);  // Simple pattern
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Build compute graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    // Execute graph using ACTUAL backend API
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

    // Get output
    std::vector<float> output_data(n_ff * n_tokens);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    // Check for all-zeros (the bug symptom)
    int   non_zero = 0;
    float sum      = 0.0f;
    for (size_t i = 0; i < output_data.size(); i++) {
        if (output_data[i] != 0.0f) {
            non_zero++;
        }
        sum += fabsf(output_data[i]);
    }

    printf("  Output: %d/%zu non-zero values, sum=%.4f\n", non_zero, output_data.size(), sum);
    printf("  Weight extra: %p\n", weight->extra);
    printf("  Input extra: %p\n", input->extra);

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (non_zero == 0) {
        printf("  FAIL: Output is all zeros (bug detected!)\n");
        return false;
    }

    printf("  PASS: MUL_MAT produced non-zero output\n");
    return true;
}

// Test 3: Simulate prompt → decode transition
// This is the actual bug scenario: prompt phase works, decode phase fails
bool test_prompt_decode_transition() {
    printf("Test 3: Prompt → Decode transition (bug scenario)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_embd = 4096;
    const int n_ff   = 11008;

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Weight tensor (persists across phases)
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
    ggml_set_name(weight, "ffn_weight");

    // Allocate weight buffer
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    // Initialize weight data
    const int                    nblocks = n_ff * (n_embd / QK4_0);
    std::vector<block_q4_0_test> weight_data(nblocks);
    create_q4_0_test_data(weight_data.data(), n_ff, n_embd, 42);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    bool prompt_ok = false;
    bool decode_ok = false;

    // === PHASE 1: PROMPT (multiple tokens) ===
    {
        printf("  Phase 1: Prompt (batch=8 tokens)\n");
        const int n_tokens = 8;

        struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(input, "prompt_input");
        struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        ggml_set_name(output, "prompt_output");

        // Allocate compute buffer
        size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
        size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
        ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
        ggml_backend_tensor_alloc(compute_buffer, input, base);
        ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

        // Set input data
        std::vector<float> input_data(n_embd * n_tokens);
        for (int i = 0; i < n_embd * n_tokens; i++) {
            input_data[i] = 0.1f * (i % 10);
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(backend, graph);

        // Check output
        std::vector<float> output_data(n_ff * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        int non_zero = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
        }

        printf("    Prompt output: %d/%zu non-zero\n", non_zero, output_data.size());
        prompt_ok = (non_zero > 0);

        // Reset compute buffer (simulates graph reset between phases)
        ggml_backend_buffer_reset(compute_buffer);
        ggml_backend_buffer_free(compute_buffer);
    }

    // === PHASE 2: DECODE (single token) ===
    {
        printf("  Phase 2: Decode (batch=1 token)\n");
        const int n_tokens = 1;  // Single token decode

        struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(input, "decode_input");
        struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        ggml_set_name(output, "decode_output");

        // Allocate NEW compute buffer (simulates real behavior)
        size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
        size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
        ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
        ggml_backend_tensor_alloc(compute_buffer, input, base);
        ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

        // Set input data
        std::vector<float> input_data(n_embd * n_tokens);
        for (int i = 0; i < n_embd; i++) {
            input_data[i] = 0.1f * (i % 10);  // Same pattern as prompt
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Build and execute graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(backend, graph);

        // Check output
        std::vector<float> output_data(n_ff * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        int non_zero = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
        }

        printf("    Decode output: %d/%zu non-zero\n", non_zero, output_data.size());
        decode_ok = (non_zero > 0);

        ggml_backend_buffer_free(compute_buffer);
    }

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (prompt_ok && !decode_ok) {
        printf("  FAIL: Prompt OK but Decode produces zeros (BUG REPRODUCED!)\n");
        return false;
    } else if (!prompt_ok && !decode_ok) {
        printf("  FAIL: Both phases produce zeros\n");
        return false;
    } else if (prompt_ok && decode_ok) {
        printf("  PASS: Both phases produce non-zero output\n");
        return true;
    } else {
        printf("  WARN: Prompt failed but Decode passed (unexpected)\n");
        return false;
    }
}

// Test 4: Compare SoA vs AoS output (requires running twice with env var)
bool test_compare_outputs() {
    printf("Test 4: Numerical output verification\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Smaller dimensions for quick test
    const int n_embd   = 256;
    const int n_ff     = 512;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Set weight data
    const int                    nblocks = n_ff * (n_embd / QK4_0);
    std::vector<block_q4_0_test> weight_data(nblocks);
    create_q4_0_test_data(weight_data.data(), n_ff, n_embd, 42);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Set input to all 1.0
    std::vector<float> input_data(n_embd * n_tokens, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Execute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    // Get output
    std::vector<float> output_data(n_ff * n_tokens);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    // Compute statistics
    float sum = 0.0f, min_val = 1e10f, max_val = -1e10f;
    int   non_zero = 0;
    for (size_t i = 0; i < output_data.size(); i++) {
        float v = output_data[i];
        if (v != 0.0f) {
            non_zero++;
        }
        sum += v;
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }
    float avg = sum / output_data.size();

    printf("  Output stats: non_zero=%d/%zu, min=%.4f, max=%.4f, avg=%.4f\n", non_zero, output_data.size(), min_val,
           max_val, avg);

    // Print first few values for comparison
    printf("  First 10 values: ");
    for (int i = 0; i < 10 && i < (int) output_data.size(); i++) {
        printf("%.3f ", output_data[i]);
    }
    printf("\n");

    printf("  Mode: %s\n", layout_override_label());

    // Cleanup
    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (non_zero == 0) {
        printf("  FAIL: All zeros\n");
        return false;
    }

    printf("  PASS: Output verified\n");
    return true;
}

// Test 5: Actual Mistral 7B dimensions - tests each layer type
bool test_mistral_dimensions() {
    printf("Test 5: Actual Mistral 7B layer dimensions\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Mistral 7B actual dimensions
    const int n_embd   = 4096;
    const int n_ff     = 14336;  // FFN intermediate
    const int n_vocab  = 32000;  // Token embedding rows
    const int n_tokens = 1;      // Decode phase

    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Test structures representing actual layer shapes
    struct test_case {
        const char * name;
        int          cols;                  // n_embd in weight
        int          rows;                  // output dimension
    } tests[] = {
        { "token_embd", n_embd, n_vocab }, // [4096, 32000]
        { "q_proj",     n_embd, n_embd  }, // [4096, 4096]
        { "k_proj",     n_embd, 1024    }, // [4096, 1024] GQA
        { "v_proj",     n_embd, 1024    }, // [4096, 1024] GQA
        { "o_proj",     n_embd, n_embd  }, // [4096, 4096]
        { "gate_proj",  n_embd, n_ff    }, // [4096, 14336]
        { "up_proj",    n_embd, n_ff    }, // [4096, 14336]
        { "down_proj",  n_ff,   n_embd  }, // [14336, 4096]
    };

    int tests_passed = 0;
    int tests_failed = 0;

    for (const auto & tc : tests) {
        printf("  Testing %s [%d, %d]...", tc.name, tc.cols, tc.rows);

        struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, tc.cols, tc.rows);
        struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.cols, n_tokens);
        struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

        // Allocate weight buffer
        size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
        ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

        // Allocate compute buffer
        size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
        size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
        ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
        ggml_backend_tensor_alloc(compute_buffer, input, base);
        ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

        // Initialize weight data
        const int                    nblocks = tc.rows * (tc.cols / QK4_0);
        std::vector<block_q4_0_test> weight_data(nblocks);
        create_q4_0_test_data(weight_data.data(), tc.rows, tc.cols, 42);
        ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

        // Set input
        std::vector<float> input_data(tc.cols * n_tokens, 1.0f);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        // Execute
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(backend, graph);

        // Check output
        std::vector<float> output_data(tc.rows * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        int non_zero = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
        }

        if (non_zero > 0) {
            printf(" PASS (%d non-zero)\n", non_zero);
            tests_passed++;
        } else {
            printf(" FAIL (all zeros)\n");
            tests_failed++;
        }

        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  Subtests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0;
}

// Test 6: Q4_0 vs Q8_0 comparison (Q8_0 is known to work, Q4_0 fails)
// Q8_0 block structure
#define QK8_0 32

typedef struct {
    uint16_t d;          // delta (fp16)
    int8_t   qs[QK8_0];  // quants
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

void create_q8_0_test_data(block_q8_0_test * data, int nrows, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);
    const int    nblocks = nrows * (ncols / QK8_0);
    for (int i = 0; i < nblocks; i++) {
        data[i].d = 0x3C00;  // 1.0 in fp16
        for (int j = 0; j < QK8_0; j++) {
            data[i].qs[j] = (int8_t) ((i + j) % 256 - 128);
        }
    }
}

bool test_q4_vs_q8() {
    printf("Test 6: Q4_0 vs Q8_0 comparison (same dimensions)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Use actual FFN dimensions
    const int n_embd   = 4096;
    const int n_ff     = 14336;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    bool q4_ok = false, q8_ok = false;

    // Test Q4_0
    {
        printf("  Q4_0 test...\n");
        struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
        struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

        size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
        ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

        size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
        size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
        ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
        ggml_backend_tensor_alloc(compute_buffer, input, base);
        ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

        const int                    nblocks = n_ff * (n_embd / QK4_0);
        std::vector<block_q4_0_test> weight_data(nblocks);
        create_q4_0_test_data(weight_data.data(), n_ff, n_embd, 42);
        ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

        std::vector<float> input_data(n_embd * n_tokens, 1.0f);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(backend, graph);

        std::vector<float> output_data(n_ff * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        int   non_zero = 0;
        float sum      = 0.0f;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
            sum += output_data[i];
        }
        printf("    Q4_0: %d/%zu non-zero, sum=%.2f, first=[%.4f,%.4f,%.4f]\n", non_zero, output_data.size(), sum,
               output_data[0], output_data[1], output_data[2]);
        q4_ok = (non_zero > 0);

        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
    }

    // Test Q8_0
    {
        printf("  Q8_0 test...\n");
        struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_embd, n_ff);
        struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

        size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
        ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

        size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
        size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
        ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
        ggml_backend_tensor_alloc(compute_buffer, input, base);
        ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

        const int                    nblocks = n_ff * (n_embd / QK8_0);
        std::vector<block_q8_0_test> weight_data(nblocks);
        create_q8_0_test_data(weight_data.data(), n_ff, n_embd, 42);
        ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q8_0_test));

        std::vector<float> input_data(n_embd * n_tokens, 1.0f);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_backend_graph_compute(backend, graph);

        std::vector<float> output_data(n_ff * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        int   non_zero = 0;
        float sum      = 0.0f;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
            sum += output_data[i];
        }
        printf("    Q8_0: %d/%zu non-zero, sum=%.2f, first=[%.4f,%.4f,%.4f]\n", non_zero, output_data.size(), sum,
               output_data[0], output_data[1], output_data[2]);
        q8_ok = (non_zero > 0);

        ggml_backend_buffer_free(weight_buffer);
        ggml_backend_buffer_free(compute_buffer);
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    if (q4_ok && q8_ok) {
        printf("  PASS: Both Q4_0 and Q8_0 produce non-zero output\n");
        return true;
    } else if (!q4_ok && q8_ok) {
        printf("  FAIL: Q8_0 works but Q4_0 fails (BUG SIGNATURE!)\n");
        return false;
    } else if (q4_ok && !q8_ok) {
        printf("  FAIL: Q4_0 works but Q8_0 fails (unexpected)\n");
        return false;
    } else {
        printf("  FAIL: Both fail\n");
        return false;
    }
}

// Test 7: Sequential MUL_MAT operations (simulate multi-layer)
bool test_sequential_ops() {
    printf("Test 7: Sequential MUL_MAT operations (multi-layer simulation)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_embd   = 4096;
    const int n_ff     = 14336;
    const int n_layers = 4;  // Simulate 4 layers
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create weight tensors for all layers
    std::vector<ggml_tensor *>         gate_weights(n_layers);
    std::vector<ggml_tensor *>         up_weights(n_layers);
    std::vector<ggml_tensor *>         down_weights(n_layers);
    std::vector<ggml_backend_buffer_t> weight_buffers;

    for (int l = 0; l < n_layers; l++) {
        char name[64];
        snprintf(name, sizeof(name), "gate_%d", l);
        gate_weights[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
        ggml_set_name(gate_weights[l], name);

        snprintf(name, sizeof(name), "up_%d", l);
        up_weights[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff);
        ggml_set_name(up_weights[l], name);

        snprintf(name, sizeof(name), "down_%d", l);
        down_weights[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_ff, n_embd);
        ggml_set_name(down_weights[l], name);
    }

    // Allocate and initialize all weights
    for (int l = 0; l < n_layers; l++) {
        for (auto * w : { gate_weights[l], up_weights[l], down_weights[l] }) {
            size_t                buf_size = ggml_backend_buft_get_alloc_size(buft, w);
            ggml_backend_buffer_t buf      = ggml_backend_buft_alloc_buffer(buft, buf_size);
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_alloc(buf, w, (void *) ggml_backend_buffer_get_base(buf));
            weight_buffers.push_back(buf);

            int                          nblocks = w->ne[1] * (w->ne[0] / QK4_0);
            std::vector<block_q4_0_test> data(nblocks);
            create_q4_0_test_data(data.data(), w->ne[1], w->ne[0], 42 + l);
            ggml_backend_tensor_set(w, data.data(), 0, nblocks * sizeof(block_q4_0_test));
        }
    }

    // Create input tensor
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(input, "input");

    // Build graph: input -> layer0 -> layer1 -> ... -> output
    struct ggml_tensor * x = input;
    for (int l = 0; l < n_layers; l++) {
        // Simplified FFN: down(gate(x) * up(x))
        struct ggml_tensor * gate_out = ggml_mul_mat(ctx, gate_weights[l], x);
        struct ggml_tensor * up_out   = ggml_mul_mat(ctx, up_weights[l], x);
        struct ggml_tensor * mul_out  = ggml_mul(ctx, gate_out, up_out);
        x                             = ggml_mul_mat(ctx, down_weights[l], mul_out);
    }
    struct ggml_tensor * output = x;

    // Allocate compute buffer for all intermediate tensors
    // This is simplified - real impl would use graph allocator
    size_t                compute_size   = 256 * 1024 * 1024;  // 256 MB
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, compute_size);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    // Build and execute graph
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, output);

    // Allocate tensors using graph allocator
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Set input data
    std::vector<float> input_data(n_embd * n_tokens, 0.1f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Execute
    printf("  Executing %d-layer graph...\n", n_layers);
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    bool pass = false;
    if (status == GGML_STATUS_SUCCESS) {
        std::vector<float> output_data(n_embd * n_tokens);
        ggml_backend_sycl_submit_barrier(backend);
        ggml_backend_tensor_get_async(backend, output, output_data.data(), 0, output_data.size() * sizeof(float));
        ggml_backend_synchronize(backend);

        int   non_zero = 0;
        float sum      = 0.0f;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
            sum += output_data[i];
        }
        printf("  Output: %d/%zu non-zero, sum=%.4f\n", non_zero, output_data.size(), sum);
        pass = (non_zero > 0);
    } else {
        printf("  Graph compute failed with status %d\n", status);
    }

    // Cleanup
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(compute_buffer);
    for (auto * buf : weight_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (pass) {
        printf("  PASS: Multi-layer graph produced non-zero output\n");
    } else {
        printf("  FAIL: Multi-layer graph failed\n");
    }
    return pass;
}

// CPU reference dequantization for Q4_0
void dequantize_q4_0_cpu(const block_q4_0_test * x, float * y, int nblocks) {
    for (int i = 0; i < nblocks; i++) {
        // Convert fp16 to float
        uint16_t d_bits = x[i].d;
        float    d;
        // Simple fp16 to float conversion
        int      sign = (d_bits >> 15) & 1;
        int      exp  = (d_bits >> 10) & 0x1F;
        int      mant = d_bits & 0x3FF;
        if (exp == 0) {
            d = sign ? -0.0f : 0.0f;
        } else if (exp == 31) {
            d = sign ? -INFINITY : INFINITY;
        } else {
            d = (sign ? -1.0f : 1.0f) * ldexpf(1.0f + mant / 1024.0f, exp - 15);
        }

        for (int j = 0; j < QK4_0 / 2; j++) {
            const int x0                 = (x[i].qs[j] & 0x0F) - 8;
            const int x1                 = (x[i].qs[j] >> 4) - 8;
            y[i * QK4_0 + j]             = x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
}

// CPU reference matrix-vector multiplication
void mul_mat_vec_cpu(const float * weight, const float * input, float * output, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; row++) {
        float sum = 0.0f;
        for (int col = 0; col < n_cols; col++) {
            sum += weight[row * n_cols + col] * input[col];
        }
        output[row] = sum;
    }
}

// Test 8: Q4_0 DMMV correctness with CPU reference
// This tests actual value correctness, not just non-zero output
bool test_q4_0_correctness() {
    printf("Test 8: Q4_0 DMMV correctness (GPU vs CPU reference)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Test with actual Mistral dimensions - DMMV path (single token decode)
    const int n_cols   = 4096;  // Input dimension (Mistral n_embd)
    const int n_rows   = 4096;  // Output dimension (Mistral n_embd for q_proj)
    const int n_tokens = 1;     // Single token = DMMV path

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create tensors - use unique name for Test 8 identification
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST8_WEIGHT");
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate weight buffer
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    // DEBUG: Print tensor pointers for comparison with kernel debug
    printf("  [DEBUG] weight->data = %p\n", weight->data);
    printf("  [DEBUG] weight->extra = %p\n", weight->extra);
    printf("  [DEBUG] buffer base = %p\n", ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create Q4_0 weight data with known pattern that produces non-zero output
    const int                    nblocks        = n_rows * (n_cols / QK4_0);
    const int                    blocks_per_row = n_cols / QK4_0;  // 128
    std::vector<block_q4_0_test> weight_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        weight_data[i].d = 0x3C00;  // 1.0 in fp16
        for (int j = 0; j < QK4_0 / 2; j++) {
            // Pattern: all 15s (gives +7 after -8 subtraction)
            // Each byte: low=15, high=15 -> both nibbles are 15
            // After -8: 15-8=7, so all values are +7
            // Sum per block = 32 * 7 = 224, times 1.0 scale = 224.0
            weight_data[i].qs[j] = 0xFF;  // low=15, high=15 -> both become +7
        }
    }

    // Upload weights
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // ===== DEBUG: Read back GPU memory to verify SoA layout =====
    printf("\n  === SoA Memory Layout Debug ===\n");
    size_t tensor_size = ggml_nbytes(weight);
    printf("  Tensor size: %zu bytes (%d blocks x 18 = %zu)\n", tensor_size, nblocks, (size_t) nblocks * 18);
    printf("  blocks_per_row: %d\n", blocks_per_row);

    // Expected SoA layout:
    // - qs section: n_rows * n_cols / 2 bytes = 4096 * 4096 / 2 = 8388608 bytes at offset 0
    // - d section: nblocks * 2 bytes = 524288 * 2 = 1048576 bytes at offset 8388608
    const size_t expected_qs_size  = (size_t) n_rows * n_cols / 2;
    const size_t expected_d_offset = expected_qs_size;
    printf("  Expected SoA qs section: 0 to %zu\n", expected_qs_size);
    printf("  Expected SoA d offset: %zu\n", expected_d_offset);

    // Read back the weights from GPU
    std::vector<uint8_t> gpu_data(tensor_size);
    ggml_backend_tensor_get(weight, gpu_data.data(), 0, tensor_size);

    // Check if data looks like AoS or SoA
    // In AoS: first 2 bytes are d (0x00 0x3C for 1.0 fp16), then 16 qs bytes
    // In SoA: first bytes are all qs, d values are at offset
    printf("\n  First 20 bytes (hex): ");
    for (int i = 0; i < 20 && i < (int) tensor_size; i++) {
        printf("%02x ", gpu_data[i]);
    }
    printf("\n");

    // Check bytes at expected d_offset (where d values should be in SoA)
    printf("  Bytes at d_offset %zu (should be fp16 d values): ", expected_d_offset);
    for (int i = 0; i < 10 && expected_d_offset + i < tensor_size; i++) {
        printf("%02x ", gpu_data[expected_d_offset + i]);
    }
    printf("\n");

    // Interpret d values at expected offset
    if (expected_d_offset + 2 <= tensor_size) {
        uint16_t d0 = *(uint16_t *) (gpu_data.data() + expected_d_offset);
        uint16_t d1 = *(uint16_t *) (gpu_data.data() + expected_d_offset + 2);
        printf("  d[0] raw = 0x%04x (expected 0x3C00 for 1.0)\n", d0);
        printf("  d[1] raw = 0x%04x (expected 0x3C00 for 1.0)\n", d1);
    }

    // Check AoS interpretation
    printf("\n  AoS interpretation (if NOT reordered):\n");
    block_q4_0_test * aos_check = (block_q4_0_test *) gpu_data.data();
    printf("  block[0].d = 0x%04x, qs[0]=0x%02x\n", aos_check[0].d, aos_check[0].qs[0]);
    printf("  block[1].d = 0x%04x, qs[0]=0x%02x\n", aos_check[1].d, aos_check[1].qs[0]);

    // Manual SoA calculation for row 0
    printf("\n  Manual SoA calculation for row 0:\n");
    float manual_sum = 0.0f;
    for (int block = 0; block < blocks_per_row && block < 5; block++) {
        // In SoA layout:
        // - qs for block N at offset: N * 16
        // - d for block N at offset: d_offset + N * 2
        size_t qs_off = (size_t) block * 16;
        size_t d_off  = expected_d_offset + (size_t) block * 2;

        // Read d value
        uint16_t d_raw = 0;
        if (d_off + 2 <= tensor_size) {
            d_raw = *(uint16_t *) (gpu_data.data() + d_off);
        }

        // Convert fp16 to float
        int   sign    = (d_raw >> 15) & 1;
        int   exp     = (d_raw >> 10) & 0x1F;
        int   mant    = d_raw & 0x3FF;
        float d_float = 0.0f;
        if (exp != 0 && exp != 31) {
            d_float = (sign ? -1.0f : 1.0f) * ldexpf(1.0f + mant / 1024.0f, exp - 15);
        }

        // Calculate sum for this block
        float block_sum = 0.0f;
        for (int j = 0; j < 16 && qs_off + j < tensor_size; j++) {
            uint8_t qs = gpu_data[qs_off + j];
            int     x0 = (qs & 0x0F) - 8;
            int     x1 = (qs >> 4) - 8;
            block_sum += x0 * d_float;
            block_sum += x1 * d_float;
        }
        manual_sum += block_sum;

        if (block < 3) {
            printf("    block[%d]: d_off=%zu d_raw=0x%04x d_float=%.2f block_sum=%.1f\n", block, d_off, d_raw, d_float,
                   block_sum);
        }
    }
    printf("  First 5 blocks sum: %.1f (expected 5 * 224 = 1120)\n", manual_sum);
    printf("  === End Debug ===\n\n");

    // Create input: all 1.0
    std::vector<float> input_data(n_cols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // DEBUG: Verify input was uploaded correctly
    printf("  === Input Upload Verification ===\n");
    printf("  input->data = %p\n", input->data);
    printf("  input->extra = %p\n", input->extra);
    printf("  input_data[0..3] = [%.2f, %.2f, %.2f, %.2f] (expected all 1.0)\n", input_data[0], input_data[1],
           input_data[2], input_data[3]);
    // Read back from GPU to verify
    std::vector<float> input_readback(16);
    ggml_backend_tensor_get(input, input_readback.data(), 0, input_readback.size() * sizeof(float));
    printf("  GPU readback[0..7] = [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", input_readback[0],
           input_readback[1], input_readback[2], input_readback[3], input_readback[4], input_readback[5],
           input_readback[6], input_readback[7]);
    // Check raw hex
    uint32_t * raw = (uint32_t *) input_readback.data();
    printf("  GPU readback[0] hex = 0x%08x (expected 0x3F800000 for 1.0f)\n", raw[0]);
    printf("  === End Input Verification ===\n\n");

    // Compute GPU result
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    // Get GPU output
    std::vector<float> gpu_output(n_rows);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // Compute CPU reference
    // 1. Dequantize weights
    std::vector<float> dequant_weights(n_rows * n_cols);
    dequantize_q4_0_cpu(weight_data.data(), dequant_weights.data(), nblocks);

    // 2. Matrix-vector multiply
    std::vector<float> cpu_output(n_rows);
    mul_mat_vec_cpu(dequant_weights.data(), input_data.data(), cpu_output.data(), n_rows, n_cols);

    // Compare
    int   errors       = 0;
    float max_diff     = 0.0f;
    int   max_diff_idx = 0;
    for (int i = 0; i < n_rows; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) {
            max_diff     = diff;
            max_diff_idx = i;
        }
        // Allow small tolerance for fp16/fp32 differences
        if (diff > 1e-3f * fabsf(cpu_output[i]) + 1e-5f) {
            if (errors < 5) {
                printf("  ERROR row %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", i, gpu_output[i], cpu_output[i], diff);
            }
            errors++;
        }
    }

    printf("  Dimensions: %d x %d, %d blocks\n", n_rows, n_cols, nblocks);
    printf("  Max diff: %.6f at row %d (GPU=%.4f, CPU=%.4f)\n", max_diff, max_diff_idx, gpu_output[max_diff_idx],
           cpu_output[max_diff_idx]);
    printf("  First 5 GPU:  [%.4f, %.4f, %.4f, %.4f, %.4f]\n", gpu_output[0], gpu_output[1], gpu_output[2],
           gpu_output[3], gpu_output[4]);
    printf("  First 5 CPU:  [%.4f, %.4f, %.4f, %.4f, %.4f]\n", cpu_output[0], cpu_output[1], cpu_output[2],
           cpu_output[3], cpu_output[4]);

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (errors > 0) {
        printf("  FAIL: %d/%d rows have incorrect values\n", errors, n_rows);
        return false;
    }

    printf("  PASS: GPU output matches CPU reference\n");
    return true;
}

// Test 9: Minimal dimension test - 64x64 single token decode
// This isolates the SoA bug with minimal data for easy debugging
bool test_minimal_dimensions() {
    printf("Test 9: Minimal dimension DMMV (64x64, batch=1)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Minimal dimensions that still use DMMV path
    const int n_cols   = 64;  // Must be multiple of 32 (QK4_0)
    const int n_rows   = 64;  // Output dimension
    const int n_tokens = 1;   // Single token = DMMV path

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST9_MINIMAL");
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate weight buffer
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    // Allocate compute buffer
    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create simple test pattern:
    // - All d values = 1.0 (fp16 0x3C00)
    // - All qs = 0xFF (nibbles 15,15 -> values +7, +7 after -8 subtraction)
    // - Expected sum per row = n_cols * 7 * 1.0 = 64 * 7 = 448
    const int                    nblocks = n_rows * (n_cols / QK4_0);  // 64 * 2 = 128 blocks
    std::vector<block_q4_0_test> weight_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        weight_data[i].d = 0x3C00;        // 1.0 in fp16
        for (int j = 0; j < QK4_0 / 2; j++) {
            weight_data[i].qs[j] = 0xFF;  // All +7 after dequant
        }
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Input: all 1.0
    std::vector<float> input_data(n_cols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Execute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    // Get GPU output
    std::vector<float> gpu_output(n_rows);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // Expected value per row: sum of n_cols elements each = 7 * 1.0 = 7
    // Total per row = n_cols * 7 = 64 * 7 = 448.0
    const float expected = (float) (n_cols * 7);

    // Check results
    int errors = 0;
    printf("  Expected value per row: %.1f\n", expected);
    printf("  First 10 GPU outputs: ");
    for (int i = 0; i < 10 && i < n_rows; i++) {
        printf("%.1f ", gpu_output[i]);
        if (fabsf(gpu_output[i] - expected) > 0.1f) {
            errors++;
        }
    }
    printf("\n");

    // Check all rows
    for (int i = 10; i < n_rows; i++) {
        if (fabsf(gpu_output[i] - expected) > 0.1f) {
            errors++;
        }
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (errors > 0) {
        printf("  FAIL: %d/%d rows have incorrect values (expected %.1f)\n", errors, n_rows, expected);
        return false;
    }

    printf("  PASS: All %d rows match expected value %.1f\n", n_rows, expected);
    return true;
}

// Test 10: Single block test - exactly one Q4_0 block (32 elements)
// This is the absolute minimal case for debugging
bool test_single_block() {
    printf("Test 10: Single Q4_0 block (32x1, batch=1)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Absolute minimal: 1 row, 32 columns (one Q4_0 block)
    const int n_cols   = 32;  // Exactly one Q4_0 block
    const int n_rows   = 1;   // Single output
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST10_SINGLE_BLOCK");
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create one block with known values:
    // d = 0.5 (fp16 = 0x3800)
    // qs = [0x88, 0x88, ...] -> nibbles (8,8) -> values (0,0) after -8
    // Expected output = 0 (all zeros multiplied by anything)
    block_q4_0_test weight_data;
    weight_data.d = 0x3800;        // 0.5 in fp16
    for (int j = 0; j < QK4_0 / 2; j++) {
        weight_data.qs[j] = 0x88;  // nibbles = 8, values = 0 after -8
    }
    ggml_backend_tensor_set(weight, &weight_data, 0, sizeof(block_q4_0_test));

    // Input: all 1.0
    std::vector<float> input_data(n_cols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Execute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    // Get output
    float gpu_output;
    ggml_backend_tensor_get(output, &gpu_output, 0, sizeof(float));

    printf("  Weight: d=0.5, qs all 0x88 (zero values)\n");
    printf("  Input: all 1.0\n");
    printf("  GPU output: %.6f (expected: 0.0)\n", gpu_output);

    bool pass1 = (fabsf(gpu_output) < 0.05f);

    // Now test with non-zero values:
    // qs = [0xFF, 0xFF, ...] -> nibbles (15,15) -> values (+7,+7) after -8
    // Expected = 32 * 7 * 0.5 = 112.0
    for (int j = 0; j < QK4_0 / 2; j++) {
        weight_data.qs[j] = 0xFF;
    }
    ggml_backend_tensor_set(weight, &weight_data, 0, sizeof(block_q4_0_test));

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(output, &gpu_output, 0, sizeof(float));

    const float expected2 = 32.0f * 7.0f * 0.5f;  // 112.0
    printf("  Weight: d=0.5, qs all 0xFF (+7 values)\n");
    printf("  GPU output: %.6f (expected: %.1f)\n", gpu_output, expected2);

    bool pass2 = (fabsf(gpu_output - expected2) < 0.1f);

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (pass1 && pass2) {
        printf("  PASS: Both test cases correct\n");
        return true;
    } else {
        printf("  FAIL: pass1=%d pass2=%d\n", pass1, pass2);
        return false;
    }
}

// Test 11: Byte-level SoA layout verification
// This test explicitly verifies the SoA transformation is correct
bool test_soa_layout_verification() {
    printf("Test 11: SoA layout byte-level verification\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Use dimensions where we can manually verify:
    // 2 rows x 64 cols = 2 rows x 2 blocks = 4 blocks total
    const int n_cols         = 64;
    const int n_rows         = 2;
    const int blocks_per_row = n_cols / QK4_0;           // 2
    const int nblocks        = n_rows * blocks_per_row;  // 4

    struct ggml_init_params params = {
        .mem_size   = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST11_SOA_VERIFY");

    // Allocate weight buffer
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    // Create distinct values for each block so we can trace them:
    // Block 0: d=1.0, qs=0x11
    // Block 1: d=2.0, qs=0x22
    // Block 2: d=3.0, qs=0x33
    // Block 3: d=4.0, qs=0x44
    std::vector<block_q4_0_test> weight_data(nblocks);
    uint16_t                     d_values[]  = { 0x3C00, 0x4000, 0x4200, 0x4400 };  // 1.0, 2.0, 3.0, 4.0 in fp16
    uint8_t                      qs_values[] = { 0x11, 0x22, 0x33, 0x44 };

    for (int i = 0; i < nblocks; i++) {
        weight_data[i].d = d_values[i];
        for (int j = 0; j < QK4_0 / 2; j++) {
            weight_data[i].qs[j] = qs_values[i];
        }
    }

    printf("  AoS input data:\n");
    for (int i = 0; i < nblocks; i++) {
        printf("    Block %d: d=0x%04x, qs[0]=0x%02x\n", i, weight_data[i].d, weight_data[i].qs[0]);
    }

    // Upload to GPU (triggers SoA reorder if enabled)
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl::test_get_layout_override(&override_layout) || override_layout != GGML_LAYOUT_SOA) {
        printf("\n  NOTE: SoA override not enabled. Skipping layout check.\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    if (!prime_layout_choice(backend, ctx, weight)) {
        printf("\n  FAIL: Could not prime layout choice for SoA verification\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Read back from unified cache layout (SoA)
    size_t               tensor_size = ggml_nbytes(weight);
    std::vector<uint8_t> gpu_data;
    const char *         layout_source = nullptr;
    if (!read_layout_bytes(weight, 0, GGML_LAYOUT_SOA, gpu_data, &layout_source)) {
        printf("\n  FAIL: Could not resolve SoA layout pointer (source=%s)\n",
               layout_source ? layout_source : "null");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    printf("\n  GPU memory layout (first 40 bytes in hex):\n    ");
    for (size_t i = 0; i < 40 && i < tensor_size; i++) {
        printf("%02x ", gpu_data[i]);
        if ((i + 1) % 20 == 0) {
            printf("\n    ");
        }
    }
    printf("\n");

    // Check if SoA is enabled
    bool has_extra = (weight->extra != nullptr);
    printf("\n  tensor->extra: %s (SoA cache source=%s)\n",
           has_extra ? "set" : "NULL",
           layout_source ? layout_source : "unknown");

    // Expected SoA layout for 4 blocks:
    // qs section: 4 blocks * 16 bytes = 64 bytes at offset 0
    // d section: 4 blocks * 2 bytes = 8 bytes at offset 64
    const size_t expected_qs_size  = (size_t) nblocks * 16;  // 64
    const size_t expected_d_offset = expected_qs_size;       // 64

    printf("\n  Interpreting as SoA layout:\n");
    printf("    Expected qs section: 0 to %zu\n", expected_qs_size);
    printf("    Expected d offset: %zu\n", expected_d_offset);

    // Check qs values in SoA layout
    // In SoA, block 0 qs at offset 0, block 1 qs at offset 16, etc.
    bool qs_correct = true;
    for (int blk = 0; blk < nblocks; blk++) {
        size_t  qs_offset = (size_t) blk * 16;
        uint8_t actual_qs = gpu_data[qs_offset];
        printf("    Block %d qs[0] at offset %zu: 0x%02x (expected 0x%02x) %s\n", blk, qs_offset, actual_qs,
               qs_values[blk], actual_qs == qs_values[blk] ? "OK" : "MISMATCH");
        if (actual_qs != qs_values[blk]) {
            qs_correct = false;
        }
    }

    // Check d values in SoA layout
    bool d_correct = true;
    for (int blk = 0; blk < nblocks; blk++) {
        size_t   d_off    = expected_d_offset + (size_t) blk * 2;
        uint16_t actual_d = *(uint16_t *) (gpu_data.data() + d_off);
        printf("    Block %d d at offset %zu: 0x%04x (expected 0x%04x) %s\n", blk, d_off, actual_d, d_values[blk],
               actual_d == d_values[blk] ? "OK" : "MISMATCH");
        if (actual_d != d_values[blk]) {
            d_correct = false;
        }
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (qs_correct && d_correct) {
        printf("\n  PASS: SoA layout is correct\n");
        return true;
    } else {
        printf("\n  FAIL: SoA layout mismatch (qs_correct=%d, d_correct=%d)\n", qs_correct, d_correct);
        return false;
    }
}

// Test 12: Specific nibble extraction test
// Tests that low/high nibble extraction works correctly in SoA mode
bool test_nibble_extraction() {
    printf("Test 12: Nibble extraction verification\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // 1 row, 32 cols = 1 block
    const int n_cols   = 32;
    const int n_rows   = 1;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        .mem_size   = 2 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST12_NIBBLE");
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Test with distinct low/high nibble values:
    // qs[j] = (j & 0x0F) | ((15 - j) << 4)
    // This gives low=j, high=(15-j)
    // After -8: low_val = j-8, high_val = 7-j
    block_q4_0_test weight_data;
    weight_data.d = 0x3C00;  // 1.0 in fp16
    for (int j = 0; j < QK4_0 / 2; j++) {
        int low           = j % 16;
        int high          = (15 - j) % 16;
        weight_data.qs[j] = (uint8_t) (low | (high << 4));
    }
    ggml_backend_tensor_set(weight, &weight_data, 0, sizeof(block_q4_0_test));

    // Input: all 1.0
    std::vector<float> input_data(n_cols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Compute expected sum:
    // Element layout in Q4_0:
    //   Elements 0-15: come from low nibbles of qs[0-15] = 0,1,2,...,15 -> values -8,-7,-6,...,+7
    //   Elements 16-31: come from high nibbles of qs[0-15] = 15,14,13,...,0 -> values +7,+6,+5,...,-8
    // Sum of low nibble values: sum(j-8 for j in 0..15) = -8-7-...-1+0+1+...+7 = -36+28 = -8
    // Sum of high nibble values: sum(7-j for j in 0..15) = 7+6+...+0-1-...-8 = 28-36 = -8
    // Total expected = -8 + -8 = -16
    // (Note: sum(-8 to +7) = -8, not 0, because there are 8 negative values but only 7 positive plus zero)
    const float expected = -16.0f;

    // Execute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    float gpu_output;
    ggml_backend_tensor_get(output, &gpu_output, 0, sizeof(float));

    printf("  Weight pattern: qs[j] = (j & 0xF) | ((15-j) << 4)\n");
    printf("  Low nibbles 0-15: values -8 to +7 (sum=0)\n");
    printf("  High nibbles 15-0: values +7 to -8 (sum=0)\n");
    printf("  GPU output: %.6f (expected: %.1f)\n", gpu_output, expected);

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (fabsf(gpu_output - expected) < 0.05f) {
        printf("  PASS: Nibble extraction correct\n");
        return true;
    } else {
        printf("  FAIL: Expected %.1f, got %.6f (diff=%.6f)\n", expected, gpu_output, gpu_output - expected);
        return false;
    }
}

// Test 13: Row-by-row value verification
// Creates distinct values per row to identify if issue is row-specific
bool test_row_identification() {
    printf("Test 13: Row-by-row value identification\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // 8 rows, 64 cols = 8 rows * 2 blocks/row = 16 blocks
    const int n_cols         = 64;
    const int n_rows         = 8;
    const int n_tokens       = 1;
    const int blocks_per_row = n_cols / QK4_0;  // 2

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_cols, n_rows);
    ggml_set_name(weight, "TEST13_ROWS");
    struct ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    // Allocate buffers
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create distinct pattern per row:
    // Row r: all values = (r + 1) after dequant
    //   d = 1.0, qs = 0x99 for row 0 (nibbles 9,9 -> values +1,+1)
    //   d = 1.0, qs = 0xAA for row 1 (nibbles 10,10 -> values +2,+2)
    //   etc.
    // Expected output row r = n_cols * (r + 1) = 64 * (r + 1)
    const int                    nblocks = n_rows * blocks_per_row;
    std::vector<block_q4_0_test> weight_data(nblocks);

    for (int row = 0; row < n_rows; row++) {
        uint8_t nibble = (uint8_t) (8 + row + 1);  // 9, 10, 11, ... -> values +1, +2, +3, ...
        if (nibble > 15) {
            nibble = 15;                           // Cap at 15
        }
        uint8_t qs_val = (uint8_t) (nibble | (nibble << 4));

        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx            = row * blocks_per_row + blk;
            weight_data[idx].d = 0x3C00;  // 1.0 in fp16
            for (int j = 0; j < QK4_0 / 2; j++) {
                weight_data[idx].qs[j] = qs_val;
            }
        }
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Input: all 1.0
    std::vector<float> input_data(n_cols, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Execute
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_graph_compute(backend, graph);

    // Get output
    std::vector<float> gpu_output(n_rows);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    // Check each row
    int errors = 0;
    printf("  Row-by-row results:\n");
    for (int row = 0; row < n_rows; row++) {
        int val = row + 1;
        if (val > 7) {
            val = 7;  // Max dequant value is +7 (nibble 15 - 8)
        }
        float expected = (float) (n_cols * val);
        float actual   = gpu_output[row];
        float diff     = fabsf(actual - expected);
        bool  ok       = (diff < 0.1f);
        printf("    Row %d: GPU=%.1f, expected=%.1f, diff=%.2f %s\n", row, actual, expected, diff, ok ? "OK" : "ERROR");
        if (!ok) {
            errors++;
        }
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (errors == 0) {
        printf("  PASS: All rows correct\n");
        return true;
    } else {
        printf("  FAIL: %d/%d rows incorrect\n", errors, n_rows);
        return false;
    }
}

// Test 14: Token embedding GET_ROWS on GPU with SoA
// This tests the actual embedding lookup operation using SYCL GET_ROWS kernel
// Critical: tests whether GPU GET_ROWS expects SoA format for embeddings
bool test_token_embd_getrows() {
    printf("Test 14: Token embedding GET_ROWS on GPU (embedding lookup)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Small test dimensions: 100 vocab x 64 embd
    const int n_vocab  = 100;
    const int n_embd   = 64;
    const int n_tokens = 3;

    struct ggml_init_params params = {
        .mem_size   = 8 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create token_embd tensor (Q4_0 quantized embedding matrix) on GPU
    // Name it "token_embd.weight" to match real model behavior
    struct ggml_tensor * token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_vocab);
    ggml_set_name(token_embd, "token_embd.weight");

    // Create token indices tensor
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    // Create GET_ROWS op
    struct ggml_tensor * output = ggml_get_rows(ctx, token_embd, indices);
    ggml_set_name(output, "embd_out");

    // Allocate embedding weight buffer on GPU (triggers SoA reordering)
    size_t                embd_buf_size = ggml_backend_buft_get_alloc_size(buft, token_embd);
    ggml_backend_buffer_t embd_buffer   = ggml_backend_buft_alloc_buffer(buft, embd_buf_size);
    ggml_backend_buffer_set_usage(embd_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(embd_buffer, token_embd, (void *) ggml_backend_buffer_get_base(embd_buffer));

    // Allocate compute buffer
    size_t                indices_size   = ggml_backend_buft_get_alloc_size(buft, indices);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, indices_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, indices, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + indices_size);

    // Create embedding data: row r has all values = r (after dequant)
    const int                    blocks_per_row = n_embd / QK4_0;  // 64/32 = 2 blocks
    const int                    nblocks        = n_vocab * blocks_per_row;
    std::vector<block_q4_0_test> embd_data(nblocks);

    for (int row = 0; row < n_vocab; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx          = row * blocks_per_row + blk;
            embd_data[idx].d = 0x3C00;  // d = 1.0 in fp16
            // nibble value 8+row%8 = values row%8 after -8 subtraction
            uint8_t nibble   = (uint8_t) (8 + (row % 8));
            for (int j = 0; j < QK4_0 / 2; j++) {
                embd_data[idx].qs[j] = (uint8_t) (nibble | (nibble << 4));
            }
        }
    }
    ggml_backend_tensor_set(token_embd, embd_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Set token indices: [0, 50, 99]
    std::vector<int32_t> token_ids = { 0, 50, 99 };
    ggml_backend_tensor_set(indices, token_ids.data(), 0, n_tokens * sizeof(int32_t));

    printf("  token_embd: %dx%d Q4_0\n", n_vocab, n_embd);
    printf("  token_embd->extra: %p\n", token_embd->extra);
    printf("  Token IDs: [%d, %d, %d]\n", token_ids[0], token_ids[1], token_ids[2]);

    // Build and execute graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    bool pass = false;
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
    } else {
        std::vector<float> output_data(n_tokens * n_embd);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        printf("  Results:\n");
        pass = true;
        for (int tok = 0; tok < n_tokens; tok++) {
            int   row_id   = token_ids[tok];
            float expected = (float) (row_id % 8);  // nibble - 8
            float actual   = output_data[tok * n_embd];
            bool  ok       = (fabsf(actual - expected) < 0.01f);
            printf("    Token %d (row %d): expected %.1f, got %.4f %s\n", tok, row_id, expected, actual,
                   ok ? "OK" : "FAIL");
            if (!ok) {
                pass = false;
            }
        }

        int non_zero = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
        }
        printf("  Output: %d/%zu non-zero\n", non_zero, output_data.size());
        if (non_zero == 0) {
            pass = false;
        }
    }

    ggml_backend_buffer_free(embd_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 15: Token embedding GET_ROWS view offset with AoS Q4_0
// Ensures get_rows honors view row offsets with a forced AoS layout.
bool test_token_embd_getrows_view() {
    printf("Test 15: Token embedding GET_ROWS view offset (Q4_0, AoS)\n");

    ggml_layout_mode prev_layout = GGML_LAYOUT_AOS;
    const bool       had_prev    = ggml_sycl::test_get_layout_override(&prev_layout);
    if (had_prev && prev_layout != GGML_LAYOUT_AOS) {
        printf("  SKIP: test layout override requires AoS\n");
        return true;
    }
    ggml_sycl::test_set_layout_override(GGML_LAYOUT_AOS);

    auto restore_override = [&]() {
        if (had_prev) {
            ggml_sycl::test_set_layout_override(prev_layout);
        } else {
            ggml_sycl::test_clear_layout_override();
        }
    };

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        restore_override();
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_vocab         = 16;
    const int n_embd          = 64;
    const int view_row_offset = 5;
    const int view_rows       = 4;
    const int n_tokens        = 2;

    struct ggml_init_params params = {
        .mem_size   = 8 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_vocab);
    ggml_set_name(token_embd, "token_embd.view_base");

    struct ggml_tensor * token_view = ggml_view_2d(ctx, token_embd, n_embd, view_rows, token_embd->nb[1],
                                                   (size_t) view_row_offset * token_embd->nb[1]);
    ggml_set_name(token_view, "token_embd.view");

    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(indices, "token_indices");

    struct ggml_tensor * output = ggml_get_rows(ctx, token_view, indices);
    ggml_set_name(output, "embd_out_view");

    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, token_embd);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, token_embd, (void *) ggml_backend_buffer_get_base(weight_buffer));
    if (ggml_backend_view_init(token_view) != GGML_STATUS_SUCCESS) {
        printf("  FAIL: ggml_backend_view_init failed\n");
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        restore_override();
        return false;
    }

    size_t                indices_size   = ggml_backend_buft_get_alloc_size(buft, indices);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, indices_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, indices, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + indices_size);

    const int                    blocks_per_row = n_embd / QK4_0;
    const int                    nblocks        = n_vocab * blocks_per_row;
    std::vector<block_q4_0_test> embd_data(nblocks);
    for (int row = 0; row < n_vocab; row++) {
        const uint8_t nibble = (uint8_t) (8 + (row % 8));
        for (int blk = 0; blk < blocks_per_row; blk++) {
            const int idx    = row * blocks_per_row + blk;
            embd_data[idx].d = 0x3C00;  // d = 1.0 in fp16
            for (int j = 0; j < QK4_0 / 2; j++) {
                embd_data[idx].qs[j] = (uint8_t) (nibble | (nibble << 4));
            }
        }
    }
    ggml_backend_tensor_set(token_embd, embd_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    std::vector<int32_t> token_ids = { 0, 2 };
    ggml_backend_tensor_set(indices, token_ids.data(), 0, n_tokens * sizeof(int32_t));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    bool pass = false;
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
    } else {
        std::vector<float> output_data(n_tokens * n_embd);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        pass = true;
        for (int tok = 0; tok < n_tokens; tok++) {
            const int   expected_row = view_row_offset + token_ids[tok];
            const float expected     = (float) (expected_row % 8);
            const float actual0      = output_data[tok * n_embd];
            bool        ok           = std::fabs(actual0 - expected) < 0.01f;
            for (int i = 1; i < n_embd; i++) {
                if (std::fabs(output_data[tok * n_embd + i] - expected) > 0.01f) {
                    ok = false;
                    break;
                }
            }
            printf("    Token %d (base row %d): expected %.1f, got %.4f %s\n", tok, expected_row, expected, actual0,
                   ok ? "OK" : "FAIL");
            if (!ok) {
                pass = false;
            }
        }
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    restore_override();

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 16: CPU GET_ROWS correctness (the actual execution path for token_embd)
// In real inference: token_embd stays on CPU, CPU GET_ROWS dequants to F32,
// then F32 data goes to GPU. This test validates the CPU GET_ROWS path.
bool test_cpu_gpu_token_embd() {
    printf("Test 16: CPU GET_ROWS correctness (token embedding lookup)\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("  SKIP: Could not initialize CPU backend\n");
        return true;
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

    const int n_vocab  = 100;
    const int n_embd   = 64;
    const int n_tokens = 3;

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // token_embd on CPU (like mmap'd GGUF)
    struct ggml_tensor * token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_vocab);
    ggml_set_name(token_embd, "token_embd.weight");

    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(indices, "token_indices");

    // GET_ROWS output (F32)
    struct ggml_tensor * embd_f32 = ggml_get_rows(ctx, token_embd, indices);
    ggml_set_name(embd_f32, "embd_f32");

    // Allocate all tensors on CPU buffer
    size_t total_size = ggml_backend_buft_get_alloc_size(cpu_buft, token_embd) +
                        ggml_backend_buft_get_alloc_size(cpu_buft, indices) +
                        ggml_backend_buft_get_alloc_size(cpu_buft, embd_f32) + 4096;
    ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);

    uint8_t * base   = (uint8_t *) ggml_backend_buffer_get_base(cpu_buffer);
    size_t    offset = 0;

    ggml_backend_tensor_alloc(cpu_buffer, token_embd, base + offset);
    offset += ggml_backend_buft_get_alloc_size(cpu_buft, token_embd);

    ggml_backend_tensor_alloc(cpu_buffer, indices, base + offset);
    offset += ggml_backend_buft_get_alloc_size(cpu_buft, indices);

    ggml_backend_tensor_alloc(cpu_buffer, embd_f32, base + offset);

    // Create embedding data: row r has all values = (r % 8) after dequant
    const int                    blocks_per_row = n_embd / QK4_0;
    const int                    nblocks        = n_vocab * blocks_per_row;
    std::vector<block_q4_0_test> embd_data(nblocks);

    for (int row = 0; row < n_vocab; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx          = row * blocks_per_row + blk;
            embd_data[idx].d = 0x3C00;                     // 1.0 in fp16
            uint8_t nibble   = (uint8_t) (8 + (row % 8));  // nibble - 8 = row % 8
            for (int j = 0; j < QK4_0 / 2; j++) {
                embd_data[idx].qs[j] = (uint8_t) (nibble | (nibble << 4));
            }
        }
    }
    ggml_backend_tensor_set(token_embd, embd_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    // Set token indices: [0, 7, 50]
    std::vector<int32_t> token_ids = { 0, 7, 50 };
    ggml_backend_tensor_set(indices, token_ids.data(), 0, n_tokens * sizeof(int32_t));

    printf("  token_embd: %dx%d Q4_0 on CPU\n", n_vocab, n_embd);
    printf("  Token IDs: [%d, %d, %d]\n", token_ids[0], token_ids[1], token_ids[2]);
    printf("  Expected values after dequant: [%.1f, %.1f, %.1f]\n", (float) (token_ids[0] % 8),
           (float) (token_ids[1] % 8), (float) (token_ids[2] % 8));

    // Build and execute graph on CPU
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, embd_f32);
    enum ggml_status status = ggml_backend_graph_compute(cpu_backend, graph);

    bool pass = false;
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
    } else {
        std::vector<float> output_data(n_tokens * n_embd);
        ggml_backend_tensor_get(embd_f32, output_data.data(), 0, output_data.size() * sizeof(float));

        printf("  Results:\n");
        pass = true;
        for (int tok = 0; tok < n_tokens; tok++) {
            int   row_id   = token_ids[tok];
            float expected = (float) (row_id % 8);
            float actual   = output_data[tok * n_embd];
            bool  ok       = (fabsf(actual - expected) < 0.01f);
            printf("    Token %d (row %d): expected %.1f, got %.4f %s\n", tok, row_id, expected, actual,
                   ok ? "OK" : "FAIL");
            if (!ok) {
                pass = false;
            }
        }

        // Check for any garbage values (control characters would show as very wrong values)
        bool has_garbage = false;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (fabsf(output_data[i]) > 100.0f) {  // Expected values are 0-7
                has_garbage = true;
                printf("    WARNING: Garbage value at [%zu]: %.4f\n", i, output_data[i]);
                break;
            }
        }
        if (has_garbage) {
            pass = false;
        }

        int non_zero = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (output_data[i] != 0.0f) {
                non_zero++;
            }
        }
        printf("  Output: %d/%zu non-zero\n", non_zero, output_data.size());
    }

    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 19: SYCL GET_ROWS with host-pinned weights (streaming path)
// Uses host buffer for token_embd and ensures output matches expected rows.
bool test_sycl_host_token_embd_getrows() {
    printf("Test 19: SYCL GET_ROWS with host weights\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    const char * prev_slice = getenv("GGML_SYCL_DMA_SLICE_MB");
    if (!prev_slice || strcmp(prev_slice, "1") != 0) {
        setenv("GGML_SYCL_DMA_SLICE_MB", "1", 1);
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);

    const int n_vocab  = 128;
    const int n_embd   = 64;
    const int n_tokens = 4;

    struct ggml_init_params params = {
        .mem_size   = 8 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, n_embd, n_vocab);
    ggml_set_name(token_embd, "token_embd.weight");

    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(indices, "token_indices");

    struct ggml_tensor * output = ggml_get_rows(ctx, token_embd, indices);
    ggml_set_name(output, "embd_out_host");

    size_t                weight_size = ggml_backend_buft_get_alloc_size(host_buft, token_embd);
    ggml_backend_buffer_t weight_buf  = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, token_embd, (void *) ggml_backend_buffer_get_base(weight_buf));

    size_t                indices_size = ggml_backend_buft_get_alloc_size(dev_buft, indices);
    size_t                output_size  = ggml_backend_buft_get_alloc_size(dev_buft, output);
    ggml_backend_buffer_t compute_buf  = ggml_backend_buft_alloc_buffer(dev_buft, indices_size + output_size + 1024);
    ggml_backend_buffer_set_usage(compute_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buf);
    ggml_backend_tensor_alloc(compute_buf, indices, base);
    ggml_backend_tensor_alloc(compute_buf, output, base + indices_size);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, token_embd);
    }

    const int                    blocks_per_row = n_embd / QK4_0;
    const int                    nblocks        = n_vocab * blocks_per_row;
    std::vector<block_q4_0_test> embd_data(nblocks);

    for (int row = 0; row < n_vocab; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx          = row * blocks_per_row + blk;
            embd_data[idx].d = 0x3C00;  // 1.0 in fp16
            uint8_t nibble   = (uint8_t) (8 + (row % 8));
            for (int j = 0; j < QK4_0 / 2; j++) {
                embd_data[idx].qs[j] = (uint8_t) (nibble | (nibble << 4));
            }
        }
    }

    ggml_backend_tensor_set(token_embd, embd_data.data(), 0, nblocks * sizeof(block_q4_0_test));

    std::vector<int32_t> token_ids = { 0, 5, 42, 127 };
    ggml_backend_tensor_set(indices, token_ids.data(), 0, n_tokens * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool pass = (status == GGML_STATUS_SUCCESS);
    if (!pass) {
        printf("  FAIL: ggml_backend_graph_compute status=%d\n", status);
    }

    std::vector<float> out_vals(n_embd * n_tokens);
    if (pass) {
        ggml_backend_tensor_get(output, out_vals.data(), 0, out_vals.size() * sizeof(float));
    }

    if (pass) {
        int errors = 0;
        for (int t = 0; t < n_tokens; ++t) {
            const int row = token_ids[t];
            const float expected = static_cast<float>(row % 8);
            const float * row_ptr = out_vals.data() + t * n_embd;
            for (int i = 0; i < n_embd && i < 16; ++i) {
                if (fabsf(row_ptr[i] - expected) > 1e-3f) {
                    errors++;
                    break;
                }
            }
        }
        if (errors != 0) {
            printf("  FAIL: %d/%d rows mismatched\n", errors, n_tokens);
            pass = false;
        }
    }

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(compute_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (prev_slice) {
        setenv("GGML_SYCL_DMA_SLICE_MB", prev_slice, 1);
    } else {
        unsetenv("GGML_SYCL_DMA_SLICE_MB");
    }

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =====================================
// Q6_K SoA Tests - output.weight uses this format
// =====================================

// Q6_K block structure for testing (QK_K=256)
#define QK_K_TEST 256

typedef struct {
    uint8_t  ql[QK_K_TEST / 2];       // 128 bytes - lower 4 bits
    uint8_t  qh[QK_K_TEST / 4];       // 64 bytes - upper 2 bits
    int8_t   scales[QK_K_TEST / 16];  // 16 bytes - quantized scales
    uint16_t d;                       // 2 bytes - fp16 super-block scale
} block_q6_K_test;

static_assert(sizeof(block_q6_K_test) == 210, "block_q6_K size mismatch");

// Helper to create Q6_K test data with known values
void create_q6_k_test_data(block_q6_K_test * data, int nrows, int ncols, uint32_t seed) {
    const int nblocks = nrows * (ncols / QK_K_TEST);
    for (int i = 0; i < nblocks; i++) {
        // Use scale = 1.0 in fp16 format (0x3C00)
        data[i].d = 0x3C00;

        // Set all scale factors to 1 (int8_t)
        for (int j = 0; j < QK_K_TEST / 16; j++) {
            data[i].scales[j] = 1;
        }

        // Q6_K stores 6-bit values split between ql (lower 4 bits) and qh (upper 2 bits)
        // Each element: value = ((qh << 4) | ql) - 32
        // For value = 0: ql=0, qh=2 -> (2<<4)|0 = 32 -> 32-32 = 0
        // For value = 7: ql=7, qh=2 -> (2<<4)|7 = 39 -> 39-32 = 7
        for (int j = 0; j < QK_K_TEST / 2; j++) {
            // ql stores 2 nibbles per byte (like Q4)
            // Set all to value 7 (nibble 7, with qh bits making it 39-32=7)
            data[i].ql[j] = 0x77;  // Two nibbles of 7
        }

        for (int j = 0; j < QK_K_TEST / 4; j++) {
            // qh stores upper 2 bits for 4 elements per byte
            // For each element: bits = 2 (so (2<<4)|7 = 39, 39-32=7)
            data[i].qh[j] = 0xAA;  // Each 2-bit field = 2 (binary 10101010)
        }
    }
}

// Test 17: Q6_K MMVQ SoA test (output.weight path)
// This tests the path used for output.weight in Mistral/LLaMA models
bool test_q6_k_mmvq_soa() {
    printf("Test 17: Q6_K MMVQ SoA (output.weight path)\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Test with output.weight dimensions (vocab_size x hidden_dim)
    // Use smaller dimensions for testing but maintain Q6_K alignment
    const int n_cols   = 512;  // Must be multiple of QK_K (256)
    const int n_rows   = 256;  // Smaller vocab for testing
    const int n_tokens = 1;    // Single token = MMVQ path for Q6_K

    printf("  Dimensions: %d x %d (QK_K=%d)\n", n_rows, n_cols, QK_K_TEST);
    printf("  Path: MMVQ (single token decode)\n");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Create Q6_K weight tensor (like output.weight)
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_cols, n_rows);
    ggml_set_name(weight, "TEST16_OUTPUT_WEIGHT");

    // Input tensor (F32, single token)
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_tokens);
    ggml_set_name(input, "TEST16_INPUT");

    // MUL_MAT output
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "TEST16_OUTPUT");

    // Allocate weight buffer with WEIGHTS usage (triggers SoA reordering)
    size_t                weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer   = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        printf("  FAIL: Could not allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *) ggml_backend_buffer_get_base(weight_buffer));

    printf("  Weight buffer allocated, usage=WEIGHTS (SoA enabled)\n");

    // Allocate compute buffer
    size_t                input_size     = ggml_backend_buft_get_alloc_size(buft, input);
    size_t                output_size    = ggml_backend_buft_get_alloc_size(buft, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, output, base + input_size);

    // Create Q6_K weight data
    const int nblocks = n_rows * (n_cols / QK_K_TEST);
    printf("  Creating %d Q6_K blocks...\n", nblocks);

    std::vector<block_q6_K_test> weight_data(nblocks);
    create_q6_k_test_data(weight_data.data(), n_rows, n_cols, 42);

    // Upload weights (triggers SoA transformation)
    printf("  Uploading weights (AoS -> SoA transformation)...\n");
    ggml_backend_tensor_set(weight, weight_data.data(), 0, nblocks * sizeof(block_q6_K_test));

    // Create input: all 1.0s
    std::vector<float> input_data(n_cols * n_tokens, 1.0f);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Build and execute graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("  Executing MUL_MAT graph...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    bool pass = false;
    if (status != GGML_STATUS_SUCCESS) {
        printf("  FAIL: Graph compute failed with status %d\n", status);
    } else {
        // Read output
        std::vector<float> output_data(n_rows * n_tokens);
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

        // Calculate expected output
        // With d=1.0, scales=1, and all values = 7 after dequant
        // Each block sums to: 256 * 7 * 1 * 1 = 1792
        // blocks_per_row = n_cols / QK_K = 512/256 = 2
        // Expected output per row: 2 * 1792 = 3584
        const int blocks_per_row = n_cols / QK_K_TEST;
        float     expected       = (float) (blocks_per_row * QK_K_TEST * 7);  // 2 * 256 * 7 = 3584

        printf("\n  Expected output per row: %.1f\n", expected);
        printf("  (blocks_per_row=%d, QK_K=%d, dequant_val=7)\n", blocks_per_row, QK_K_TEST);

        // Check first few output values
        printf("  Output values:\n");
        pass = true;
        for (int i = 0; i < 5 && i < n_rows; i++) {
            float actual    = output_data[i];
            float rel_error = fabsf(actual - expected) / expected;
            bool  ok        = rel_error < 0.1f;  // 10% tolerance

            printf("    row[%d]: %.2f (expected %.2f, err=%.2f%%) %s\n", i, actual, expected, rel_error * 100.0f,
                   ok ? "OK" : "FAIL");

            if (!ok) {
                pass = false;
            }
        }

        // Check for garbage (NaN, inf, very wrong values)
        int garbage_count = 0;
        for (size_t i = 0; i < output_data.size(); i++) {
            if (std::isnan(output_data[i]) || std::isinf(output_data[i]) || fabsf(output_data[i]) > expected * 10.0f) {
                if (garbage_count < 3) {
                    printf("    GARBAGE at [%zu]: %.4f\n", i, output_data[i]);
                }
                garbage_count++;
            }
        }
        if (garbage_count > 0) {
            printf("  FAIL: Found %d garbage values\n", garbage_count);
            pass = false;
        }

        // Summary statistics
        float sum = 0.0f, min_val = output_data[0], max_val = output_data[0];
        for (float v : output_data) {
            sum += v;
            if (v < min_val) {
                min_val = v;
            }
            if (v > max_val) {
                max_val = v;
            }
        }
        float avg = sum / output_data.size();
        printf("  Stats: min=%.2f, max=%.2f, avg=%.2f\n", min_val, max_val, avg);
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 18: Q6_K SoA layout verification (check actual memory layout)
bool test_q6_k_soa_layout() {
    printf("Test 18: Q6_K SoA layout verification\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n_cols  = 512;
    const int n_rows  = 4;
    const int nblocks = n_rows * (n_cols / QK_K_TEST);  // 4 * 2 = 8 blocks

    printf("  Test tensor: %dx%d Q6_K = %d blocks\n", n_rows, n_cols, nblocks);

    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_cols, n_rows);
    ggml_set_name(weight, "TEST17_Q6K_LAYOUT");

    size_t                buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t buffer   = ggml_backend_buft_alloc_buffer(buft, buf_size);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(buffer, weight, (void *) ggml_backend_buffer_get_base(buffer));

    // Create distinctive test pattern
    std::vector<block_q6_K_test> test_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        test_data[i].d = (uint16_t) (0x3C00 + i);  // 1.0 + small offset per block

        // Fill ql with block index pattern
        for (int j = 0; j < QK_K_TEST / 2; j++) {
            test_data[i].ql[j] = (uint8_t) (i * 16 + (j % 16));
        }

        // Fill qh with complementary pattern
        for (int j = 0; j < QK_K_TEST / 4; j++) {
            test_data[i].qh[j] = (uint8_t) (0xAA + i);
        }

        // Fill scales
        for (int j = 0; j < QK_K_TEST / 16; j++) {
            test_data[i].scales[j] = (int8_t) (i + 1);
        }
    }

    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl::test_get_layout_override(&override_layout) || override_layout != GGML_LAYOUT_SOA) {
        printf("\n  NOTE: SoA override not enabled. Skipping layout check.\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    // Upload (triggers SoA transformation)
    ggml_backend_tensor_set(weight, test_data.data(), 0, nblocks * sizeof(block_q6_K_test));

    if (!prime_layout_choice(backend, ctx, weight)) {
        printf("\n  FAIL: Could not prime layout choice for Q6_K SoA verification\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Read back and verify SoA layout from unified cache
    size_t               tensor_size = ggml_nbytes(weight);
    std::vector<uint8_t> gpu_data;
    const char *         layout_source = nullptr;
    if (!read_layout_bytes(weight, 0, GGML_LAYOUT_SOA, gpu_data, &layout_source)) {
        printf("\n  FAIL: Could not resolve SoA layout pointer (source=%s)\n",
               layout_source ? layout_source : "null");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    printf("\n  === Q6_K SoA Layout Analysis ===\n");

    // SoA layout for Q6_K:
    // - ql: nblocks * 128 bytes at offset 0
    // - qh: nblocks * 64 bytes at offset nblocks*128
    // - scales: nblocks * 16 bytes at offset nblocks*192
    // - d: nblocks * 2 bytes at offset nblocks*208
    size_t ql_offset     = 0;
    size_t qh_offset     = nblocks * (QK_K_TEST / 2);                   // nblocks * 128
    size_t scales_offset = qh_offset + nblocks * (QK_K_TEST / 4);       // + nblocks * 64
    size_t d_offset      = scales_offset + nblocks * (QK_K_TEST / 16);  // + nblocks * 16

    printf("  Expected SoA offsets:\n");
    printf("    ql: 0\n");
    printf("    qh: %zu\n", qh_offset);
    printf("    scales: %zu\n", scales_offset);
    printf("    d: %zu\n", d_offset);

    // Verify layout by checking d values
    printf("\n  Checking d values at SoA offset %zu:\n", d_offset);
    bool pass = true;
    for (int i = 0; i < nblocks && i < 4; i++) {
        uint16_t expected_d = (uint16_t) (0x3C00 + i);
        uint16_t actual_d   = *(uint16_t *) (gpu_data.data() + d_offset + i * 2);
        bool     ok         = (actual_d == expected_d);
        printf("    block[%d].d: expected 0x%04x, got 0x%04x %s\n", i, expected_d, actual_d, ok ? "OK" : "FAIL");
        if (!ok) {
            pass = false;
        }
    }

    // Check ql values
    printf("\n  Checking ql values at SoA offset 0:\n");
    for (int i = 0; i < nblocks && i < 2; i++) {
        uint8_t expected_ql0    = (uint8_t) (i * 16);
        size_t  ql_block_offset = i * (QK_K_TEST / 2);
        uint8_t actual_ql0      = gpu_data[ql_block_offset];
        bool    ok              = (actual_ql0 == expected_ql0);
        printf("    block[%d].ql[0]: expected 0x%02x, got 0x%02x %s\n", i, expected_ql0, actual_ql0,
               ok ? "OK" : "FAIL");
        if (!ok) {
            pass = false;
        }
    }

    // Check if layout is AoS (would mean reordering didn't happen)
    printf("\n  Checking if layout is AoS (should NOT match if SoA is working):\n");
    block_q6_K_test * aos_check = (block_q6_K_test *) gpu_data.data();
    uint16_t          aos_d0    = aos_check[0].d;
    bool              is_aos    = (aos_d0 == 0x3C00);
    printf("    Interpreting as AoS: block[0].d = 0x%04x %s\n", aos_d0,
           is_aos ? "(matches AoS - REORDER NOT WORKING!)" : "(different - SoA working)");
    if (is_aos) {
        printf("  FAIL: Data appears to be in AoS format, SoA reordering not active\n");
        pass = false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char ** argv) {
    printf("=== GGML SYCL SoA Integration Tests ===\n");
    printf("Using ACTUAL ggml backend API functions\n\n");

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
    printf("Layout override: %s\n", layout_override_label());
    printf("\n");

    int passed = 0;
    int failed = 0;

    if (test_buffer_init()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_mul_mat_graph()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_prompt_decode_transition()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_compare_outputs()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_mistral_dimensions()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_q4_vs_q8()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_sequential_ops()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_q4_0_correctness()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    // New focused tests
    if (test_minimal_dimensions()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_single_block()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_soa_layout_verification()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_nibble_extraction()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_row_identification()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_token_embd_getrows()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_token_embd_getrows_view()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_cpu_gpu_token_embd()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_sycl_host_token_embd_getrows()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    // Q6_K tests (output.weight path)
    if (test_q6_k_mmvq_soa()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    if (test_q6_k_soa_layout()) {
        passed++;
    } else {
        failed++;
    }
    printf("\n");

    printf("=================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    if (failed > 0) {
        printf("\nTo compare SoA vs AoS:\n");
        printf("  # SoA mode (default):\n");
        printf("  ./build/bin/test-ggml-sycl-soa\n");
        printf("  # AoS mode:\n");
        printf("  ./build/bin/test-ggml-sycl-soa --layout=aos\n");
    }

    if (has_override) {
        ggml_sycl::test_clear_layout_override();
    }

    return failed > 0 ? 1 : 0;
}
