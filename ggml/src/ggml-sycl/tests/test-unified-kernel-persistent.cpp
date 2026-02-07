//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Integration tests for UnifiedKernel persistent execution
// Tests the plan-build-execute workflow and validates correctness
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sycl/sycl.hpp>

// Note: UNIFIED_KERNEL_TEST_STANDALONE is defined via CMakeLists.txt to provide
// stub implementations for common.cpp symbols needed by unified-kernel.cpp
#include "../unified-kernel.hpp"

#if defined(UNIFIED_KERNEL_TEST_STANDALONE)
static int ggml_sycl_test_extract_layer_index(const char * name) {
    if (!name) return -1;
    const char * blk = std::strstr(name, "blk.");
    if (blk) {
        return std::atoi(blk + 4);
    }
    const char * dash = std::strrchr(name, '-');
    if (!dash || dash[1] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long layer = std::strtol(dash + 1, &end, 10);
    if (end && end != dash + 1 && layer >= 0) {
        return (int) layer;
    }
    return -1;
}
#else
int ggml_sycl_test_extract_layer_index(const char * name);
#endif

static constexpr float TEST_TOLERANCE = 1e-3f;

static void ref_rope_neox(float * q, float * k, const float * cos_cache, const float * sin_cache,
                          int n_heads, int n_kv_heads, int head_dim);

static float max_abs_error(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

static int64_t set_rows_output_bytes(const SetRowsMeta & meta) {
    const int elem_size = (meta.dst_type == 1) ? (int) sizeof(sycl::half) : (int) sizeof(float);
    const int64_t last = (meta.nc   > 0 ? (meta.nc   - 1) * meta.nb0 : 0) +
                         (meta.ne1  > 0 ? (meta.ne1  - 1) * meta.nb1 : 0) +
                         (meta.ne02 > 0 ? (meta.ne02 - 1) * meta.nb2 : 0) +
                         (meta.ne03 > 0 ? (meta.ne03 - 1) * meta.nb3 : 0);
    return last + elem_size;
}

static void print_result(const char * test_name, bool passed, float error = 0.0f) {
    if (passed) {
        printf("  [PASS] %s", test_name);
        if (error > 0) {
            printf(" (max_error=%.2e)", error);
        }
        printf("\n");
    } else {
        printf("  [FAIL] %s (max_error=%.2e, tolerance=%.2e)\n", test_name, error, TEST_TOLERANCE);
    }
}

// =============================================================================
// Extraction Tests
// =============================================================================

static bool test_extract_layer_index() {
    struct Case {
        const char * name;
        int expected;
    };
    const Case cases[] = {
        { "blk.12.attn_q.weight", 12 },
        { "blk.0.ffn_gate.weight", 0 },
        { "norm-0", 0 },
        { "Qcur-31 (reshaped)", 31 },
        { "attn_norm-7", 7 },
        { "no-layer-here", -1 },
    };

    bool passed = true;
    for (const auto & c : cases) {
        const int got = ggml_sycl_test_extract_layer_index(c.name);
        if (got != c.expected) {
            printf("    case '%s' expected=%d got=%d\n", c.name, c.expected, got);
            passed = false;
        }
    }

    print_result("extract_layer_index", passed);
    return passed;
}

// =============================================================================
// CPU Reference Implementations
// =============================================================================

static void ref_rms_norm(const float * input, const float * weights, float * output,
                         int hidden_dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms   = std::sqrt(sum_sq / hidden_dim + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < hidden_dim; i++) {
        output[i] = input[i] * scale * weights[i];
    }
}

static void ref_rms_norm_unweighted(const float * input, float * output, int hidden_dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms   = std::sqrt(sum_sq / hidden_dim + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < hidden_dim; i++) {
        output[i] = input[i] * scale;
    }
}

static void ref_silu_mul(const float * gate, const float * up, float * output, int dim) {
    for (int i = 0; i < dim; i++) {
        float sigmoid_g = 1.0f / (1.0f + std::exp(-gate[i]));
        float silu_g    = gate[i] * sigmoid_g;
        output[i]       = silu_g * up[i];
    }
}

static void ref_dmmv_q4_0(const std::vector<ggml_sycl_unified::block_q4_0_unified> & weights,
                          int N, int K, const float * activations, float * output) {
    const int k_blocks = K / 32;
    for (int n = 0; n < N; n++) {
        float dot = 0.0f;
        for (int b = 0; b < k_blocks; b++) {
            const auto & blk = weights[n * k_blocks + b];
            float d = static_cast<float>(blk.d);
            int k_offset = b * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t qs_byte = blk.qs[i];
                float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;
                dot += w0 * activations[k_offset + i] +
                       w1 * activations[k_offset + i + 16];
            }
        }
        output[n] = dot;
    }
}

static std::vector<uint8_t> pack_q4_0_soa(
    const std::vector<ggml_sycl_unified::block_q4_0_unified> & weights) {
    const size_t total_blocks = weights.size();
    const size_t qs_bytes = total_blocks * (32 / 2);
    const size_t d_bytes  = total_blocks * sizeof(sycl::half);
    std::vector<uint8_t> packed(qs_bytes + d_bytes, 0);
    uint8_t * soa_qs = packed.data();
    uint8_t * soa_d  = packed.data() + qs_bytes;
    for (size_t ib = 0; ib < total_blocks; ib++) {
        const auto & blk = weights[ib];
        memcpy(soa_qs + ib * (32 / 2), blk.qs, 32 / 2);
        memcpy(soa_d + ib * sizeof(sycl::half), &blk.d, sizeof(sycl::half));
    }
    return packed;
}

// =============================================================================
// Test: Persistent RMS Norm (single operation in persistent kernel)
// =============================================================================
static bool test_persistent_rms_norm(sycl::queue & q) {
    const int   hidden_dim = 4096;
    const float eps        = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_output(hidden_dim, 0.0f);
    std::vector<float> h_ref(hidden_dim);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i]   = std::sin(i * 0.01f) * 2.0f;
        h_weights[i] = 1.0f + 0.1f * std::cos(i * 0.05f);
    }

    ref_rms_norm(h_input.data(), h_weights.data(), h_ref.data(), hidden_dim, eps);

    float * d_input   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float * d_output  = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();
    q.memset(d_output, 0, hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Build and execute persistent plan with just one RMS norm
    kernel.begin_persistent(1, 1, hidden_dim, hidden_dim, 32, 8, 128, 0 /*GGML_TYPE_F32*/);
    kernel.add_rms_norm(0, d_weights, d_input, d_output, eps, hidden_dim);
    kernel.execute_persistent();

    q.memcpy(h_output.data(), d_output, hidden_dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    print_result("persistent_rms_norm (dim=4096)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent SiLU Mul (single operation in persistent kernel)
// =============================================================================
static bool test_persistent_silu_mul(sycl::queue & q) {
    const int dim = 11008;  // Mistral intermediate dim

    std::vector<float> h_gate(dim);
    std::vector<float> h_up(dim);
    std::vector<float> h_output(dim, 0.0f);
    std::vector<float> h_ref(dim);

    for (int i = 0; i < dim; i++) {
        h_gate[i] = std::sin(i * 0.005f) * 3.0f;
        h_up[i]   = std::cos(i * 0.005f) * 2.0f;
    }

    ref_silu_mul(h_gate.data(), h_up.data(), h_ref.data(), dim);

    float * d_gate   = sycl::malloc_device<float>(dim, q);
    float * d_up     = sycl::malloc_device<float>(dim, q);
    float * d_output = sycl::malloc_device<float>(dim, q);

    q.memcpy(d_gate, h_gate.data(), dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), dim * sizeof(float)).wait();
    q.memset(d_output, 0, dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Build and execute persistent plan with just one SiLU mul
    kernel.begin_persistent(1, 1, 4096, dim, 32, 8, 128, 0);
    kernel.add_silu_mul(0, d_gate, d_up, d_output);
    kernel.execute_persistent();

    q.memcpy(h_output.data(), d_output, dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    print_result("persistent_silu_mul (dim=11008)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent chain (RMS norm followed by SiLU mul in one kernel)
// =============================================================================
static bool test_persistent_chain(sycl::queue & q) {
    const int   hidden_dim       = 4096;
    const int   intermediate_dim = 11008;
    const float eps              = 1e-5f;

    // Inputs
    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_gate(intermediate_dim);
    std::vector<float> h_up(intermediate_dim);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i]   = std::sin(i * 0.01f);
        h_weights[i] = 1.0f;
    }
    for (int i = 0; i < intermediate_dim; i++) {
        h_gate[i] = std::sin(i * 0.005f);
        h_up[i]   = std::cos(i * 0.005f) + 1.0f;
    }

    // CPU reference for RMS norm
    std::vector<float> h_norm_ref(hidden_dim);
    ref_rms_norm(h_input.data(), h_weights.data(), h_norm_ref.data(), hidden_dim, eps);

    // CPU reference for SiLU mul
    std::vector<float> h_silu_ref(intermediate_dim);
    ref_silu_mul(h_gate.data(), h_up.data(), h_silu_ref.data(), intermediate_dim);

    // Device buffers
    float * d_input       = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights     = sycl::malloc_device<float>(hidden_dim, q);
    float * d_norm_output = sycl::malloc_device<float>(hidden_dim, q);
    float * d_gate        = sycl::malloc_device<float>(intermediate_dim, q);
    float * d_up          = sycl::malloc_device<float>(intermediate_dim, q);
    float * d_silu_output = sycl::malloc_device<float>(intermediate_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_gate, h_gate.data(), intermediate_dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), intermediate_dim * sizeof(float)).wait();
    q.memset(d_norm_output, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_silu_output, 0, intermediate_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Build plan with TWO operations in a chain
    kernel.begin_persistent(1, 1, hidden_dim, intermediate_dim, 32, 8, 128, 0);
    kernel.add_rms_norm(0, d_weights, d_input, d_norm_output, eps, hidden_dim);
    kernel.add_silu_mul(0, d_gate, d_up, d_silu_output);
    kernel.execute_persistent();

    // Check RMS norm output
    std::vector<float> h_norm_out(hidden_dim);
    q.memcpy(h_norm_out.data(), d_norm_output, hidden_dim * sizeof(float)).wait();
    float norm_error = max_abs_error(h_norm_out, h_norm_ref);
    bool  norm_ok    = norm_error < TEST_TOLERANCE;

    // Check SiLU mul output
    std::vector<float> h_silu_out(intermediate_dim);
    q.memcpy(h_silu_out.data(), d_silu_output, intermediate_dim * sizeof(float)).wait();
    float silu_error = max_abs_error(h_silu_out, h_silu_ref);
    bool  silu_ok    = silu_error < TEST_TOLERANCE;

    bool passed = norm_ok && silu_ok;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_norm_output, q);
    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_silu_output, q);

    print_result("persistent_chain: rms_norm", norm_ok, norm_error);
    print_result("persistent_chain: silu_mul", silu_ok, silu_error);
    print_result("persistent_chain: combined", passed);
    return passed;
}

// =============================================================================
// Test: Multi-layer persistent plan
// =============================================================================
static bool test_persistent_multi_layer(sycl::queue & q) {
    const int   n_layers         = 4;
    const int   hidden_dim       = 1024;  // Smaller for faster testing
    const int   intermediate_dim = 2816;
    const float eps              = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim, 1.0f);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.02f);
    }

    // Apply RMS norm n_layers times on CPU for reference
    std::vector<float> h_ref(hidden_dim);
    std::vector<float> h_tmp(hidden_dim);
    std::copy(h_input.begin(), h_input.end(), h_tmp.begin());
    for (int layer = 0; layer < n_layers; layer++) {
        ref_rms_norm(h_tmp.data(), h_weights.data(), h_ref.data(), hidden_dim, eps);
        std::copy(h_ref.begin(), h_ref.end(), h_tmp.begin());
    }

    // Device buffers - use two alternating buffers for in-place chaining
    float * d_buf_a   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_buf_b   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_buf_a, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Build plan: ping-pong between buffers across layers
    kernel.begin_persistent(n_layers, 1, hidden_dim, intermediate_dim, 32, 8, 128, 0);
    for (int layer = 0; layer < n_layers; layer++) {
        float * src = (layer % 2 == 0) ? d_buf_a : d_buf_b;
        float * dst = (layer % 2 == 0) ? d_buf_b : d_buf_a;
        kernel.add_rms_norm(layer, d_weights, src, dst, eps, hidden_dim);
    }
    kernel.execute_persistent();

    // Read final output (depends on n_layers parity)
    float * d_final = (n_layers % 2 == 0) ? d_buf_a : d_buf_b;
    std::vector<float> h_output(hidden_dim);
    q.memcpy(h_output.data(), d_final, hidden_dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    layers=%d, ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_layers, stats.n_operations, stats.total_tiles,
           stats.kernel_time_ms);

    sycl::free(d_buf_a, q);
    sycl::free(d_buf_b, q);
    sycl::free(d_weights, q);

    print_result("persistent_multi_layer (4 layers, dim=1024)", passed, error);
    return passed;
}

// =============================================================================
// Test: Stats and diagnostics
// =============================================================================
static bool test_persistent_stats(sycl::queue & q) {
    const int   hidden_dim = 512;
    const float eps        = 1e-5f;

    float * d_buf     = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights = sycl::malloc_device<float>(hidden_dim, q);

    // Initialize with some data to avoid undefined behavior
    std::vector<float> h_data(hidden_dim, 1.0f);
    q.memcpy(d_buf, h_data.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_data.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Verify supports_persistent
    bool supports = kernel.supports_persistent();
    printf("    supports_persistent: %s\n", supports ? "true" : "false");

    // Build and execute a small plan with two operations across two layers
    kernel.begin_persistent(2, 1, hidden_dim, hidden_dim, 32, 8, 128, 0);
    kernel.add_rms_norm(0, d_weights, d_buf, d_buf, eps, hidden_dim);
    kernel.add_rms_norm(1, d_weights, d_buf, d_buf, eps, hidden_dim);
    kernel.execute_persistent();

    auto stats    = kernel.get_last_stats();
    bool ops_ok   = (stats.n_operations == 2);
    bool layers_ok = (stats.n_layers == 2);
    bool time_ok  = (stats.kernel_time_ms > 0.0);

    bool passed = supports && ops_ok && layers_ok && time_ok;
    printf("    n_operations=%d (expect 2): %s\n", stats.n_operations, ops_ok ? "OK" : "FAIL");
    printf("    n_layers=%d (expect 2): %s\n", stats.n_layers, layers_ok ? "OK" : "FAIL");
    printf("    kernel_time_ms=%.3f (expect >0): %s\n", stats.kernel_time_ms, time_ok ? "OK" : "FAIL");

    sycl::free(d_buf, q);
    sycl::free(d_weights, q);

    print_result("persistent_stats", passed);
    return passed;
}

// =============================================================================
// Test: Persistent DMMV Matmul (Q4_0 dequantizing matrix-vector multiply)
// =============================================================================
static bool test_persistent_dmmv_matmul(sycl::queue & q) {
    // N=256 output columns, K=128 inner dimension
    // K=128 means 4 Q4_0 blocks per row (128 / 32 = 4)
    const int N        = 256;
    const int K        = 128;
    const int k_blocks = K / 32;  // 4 blocks per row

    // Total Q4_0 blocks: N * k_blocks = 256 * 4 = 1024
    const int total_blocks = N * k_blocks;

    // Create deterministic Q4_0 weight blocks on host
    // block_q4_0_unified: 18 bytes = half d + uint8_t qs[16]
    std::vector<ggml_sycl_unified::block_q4_0_unified> h_weights(total_blocks);

    for (int n = 0; n < N; n++) {
        for (int b = 0; b < k_blocks; b++) {
            auto & blk = h_weights[n * k_blocks + b];
            // Scale factor: deterministic, varies by row and block
            float scale = 0.1f + 0.01f * (n % 16) + 0.005f * b;
            blk.d = sycl::half(scale);
            // Fill nibbles with a pattern: low nibble = (i + n) % 16, high nibble = (i + b) % 16
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (i + n) % 16;
                uint8_t hi = (i + b) % 16;
                blk.qs[i] = lo | (hi << 4);
            }
        }
    }

    // Create float activation vector of size K
    std::vector<float> h_activations(K);
    for (int i = 0; i < K; i++) {
        h_activations[i] = std::sin(i * 0.05f) * 0.5f + 0.5f;
    }

    // CPU reference: for each output column n, dot product of dequantized weight row n with activations
    std::vector<float> h_ref(N, 0.0f);
    for (int n = 0; n < N; n++) {
        float dot = 0.0f;
        for (int b = 0; b < k_blocks; b++) {
            const auto & blk = h_weights[n * k_blocks + b];
            float d = static_cast<float>(blk.d);
            int k_offset = b * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t qs_byte = blk.qs[i];
                float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;
                dot += w0 * h_activations[k_offset + i] +
                       w1 * h_activations[k_offset + i + 16];
            }
        }
        h_ref[n] = dot;
    }

    // Allocate device memory
    const size_t weights_bytes = total_blocks * sizeof(ggml_sycl_unified::block_q4_0_unified);
    void *  d_weights     = sycl::malloc_device(weights_bytes, q);
    float * d_activations = sycl::malloc_device<float>(K, q);
    float * d_output      = sycl::malloc_device<float>(N, q);

    q.memcpy(d_weights, h_weights.data(), weights_bytes).wait();
    q.memcpy(d_activations, h_activations.data(), K * sizeof(float)).wait();
    q.memset(d_output, 0, N * sizeof(float)).wait();

    // Configure and run persistent kernel
    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // begin_persistent(n_layers, batch_size, hidden_dim, intermediate_dim, n_heads, n_kv_heads, head_dim, quant_type)
    // quant_type=QUANT_TYPE_Q4_0 for Q4_0 dequantizing matmul
    kernel.begin_persistent(1, 1, K, K, 32, 8, 128, ggml_sycl_unified::QUANT_TYPE_Q4_0);
    // add_matmul(layer, weights, input, output, type, M, N, K)
    // M=1 for DMMV (single vector), N=256 output columns, K=128 inner dim
    kernel.add_matmul(0, d_weights, d_activations, d_output, MatmulType::Q_PROJ, 1, N, K,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_AOS);
    kernel.execute_persistent();

    // Read back results
    std::vector<float> h_output(N, 0.0f);
    q.memcpy(h_output.data(), d_output, N * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    // Print a few values for debugging if failed
    if (!passed) {
        printf("    First 8 values:\n");
        for (int i = 0; i < 8 && i < N; i++) {
            printf("      [%d] got=%.6f ref=%.6f diff=%.2e\n",
                   i, h_output[i], h_ref[i], std::abs(h_output[i] - h_ref[i]));
        }
    }

    sycl::free(d_weights, q);
    sycl::free(d_activations, q);
    sycl::free(d_output, q);

    print_result("persistent_dmmv_matmul (N=256, K=128, Q4_0)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent DMMV Matmul (Q4_0 SoA layout)
// =============================================================================
static bool test_persistent_dmmv_matmul_soa(sycl::queue & q) {
    // N=256 output columns, K=128 inner dimension
    // K=128 means 4 Q4_0 blocks per row (128 / 32 = 4)
    const int N        = 256;
    const int K        = 128;
    const int k_blocks = K / 32;

    const int total_blocks = N * k_blocks;

    std::vector<ggml_sycl_unified::block_q4_0_unified> h_weights(total_blocks);

    for (int n = 0; n < N; n++) {
        for (int b = 0; b < k_blocks; b++) {
            auto & blk = h_weights[n * k_blocks + b];
            float scale = 0.1f + 0.01f * (n % 16) + 0.005f * b;
            blk.d = sycl::half(scale);
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (i + n) % 16;
                uint8_t hi = (i + b) % 16;
                blk.qs[i] = lo | (hi << 4);
            }
        }
    }

    std::vector<float> h_activations(K);
    for (int i = 0; i < K; i++) {
        h_activations[i] = std::sin(i * 0.05f) * 0.5f + 0.5f;
    }

    std::vector<float> h_ref(N, 0.0f);
    for (int n = 0; n < N; n++) {
        float dot = 0.0f;
        for (int b = 0; b < k_blocks; b++) {
            const auto & blk = h_weights[n * k_blocks + b];
            float d = static_cast<float>(blk.d);
            int k_offset = b * 32;
            for (int i = 0; i < 16; i++) {
                uint8_t qs_byte = blk.qs[i];
                float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;
                dot += w0 * h_activations[k_offset + i] +
                       w1 * h_activations[k_offset + i + 16];
            }
        }
        h_ref[n] = dot;
    }

    const size_t qs_bytes = total_blocks * (32 / 2);
    const size_t d_bytes  = total_blocks * sizeof(sycl::half);
    const size_t weights_bytes = qs_bytes + d_bytes;

    std::vector<uint8_t> h_weights_soa(weights_bytes, 0);
    uint8_t * soa_qs = h_weights_soa.data();
    uint8_t * soa_d  = h_weights_soa.data() + qs_bytes;
    for (int ib = 0; ib < total_blocks; ib++) {
        const auto & blk = h_weights[ib];
        memcpy(soa_qs + ib * (32 / 2), blk.qs, 32 / 2);
        memcpy(soa_d + ib * sizeof(sycl::half), &blk.d, sizeof(sycl::half));
    }

    void *  d_weights     = sycl::malloc_device(weights_bytes, q);
    float * d_activations = sycl::malloc_device<float>(K, q);
    float * d_output      = sycl::malloc_device<float>(N, q);

    q.memcpy(d_weights, h_weights_soa.data(), weights_bytes).wait();
    q.memcpy(d_activations, h_activations.data(), K * sizeof(float)).wait();
    q.memset(d_output, 0, N * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, K, K, 32, 8, 128, ggml_sycl_unified::QUANT_TYPE_Q4_0);
    kernel.add_matmul(0, d_weights, d_activations, d_output, MatmulType::Q_PROJ, 1, N, K,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.execute_persistent();

    std::vector<float> h_output(N, 0.0f);
    q.memcpy(h_output.data(), d_output, N * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    if (!passed) {
        printf("    First 8 values:\n");
        for (int i = 0; i < 8 && i < N; i++) {
            printf("      [%d] got=%.6f ref=%.6f diff=%.2e\n",
                   i, h_output[i], h_ref[i], std::abs(h_output[i] - h_ref[i]));
        }
    }

    sycl::free(d_weights, q);
    sycl::free(d_activations, q);
    sycl::free(d_output, q);

    print_result("persistent_dmmv_matmul_soa (N=256, K=128, Q4_0)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent FFN chain (Q4_0 gate/up/down + SiLU)
// =============================================================================
static bool test_persistent_ffn_chain_q4_0(sycl::queue & q) {
    const int hidden_dim       = 128;
    const int intermediate_dim = 256;
    const int K_in             = hidden_dim;
    const int N_gate           = intermediate_dim;
    const int N_down           = hidden_dim;
    const int k_blocks_in      = K_in / 32;
    const int k_blocks_down    = intermediate_dim / 32;

    std::vector<float> h_input(K_in);
    for (int i = 0; i < K_in; i++) {
        h_input[i] = std::sin(i * 0.03f) * 0.7f + 0.2f;
    }

    std::vector<ggml_sycl_unified::block_q4_0_unified> h_gate_w(N_gate * k_blocks_in);
    std::vector<ggml_sycl_unified::block_q4_0_unified> h_up_w(N_gate * k_blocks_in);
    std::vector<ggml_sycl_unified::block_q4_0_unified> h_down_w(N_down * k_blocks_down);

    for (int n = 0; n < N_gate; n++) {
        for (int b = 0; b < k_blocks_in; b++) {
            auto & gate_blk = h_gate_w[n * k_blocks_in + b];
            auto & up_blk   = h_up_w[n * k_blocks_in + b];
            float gate_scale = 0.05f + 0.002f * (n % 32) + 0.001f * b;
            float up_scale   = 0.06f + 0.002f * (n % 16) + 0.001f * b;
            gate_blk.d = sycl::half(gate_scale);
            up_blk.d   = sycl::half(up_scale);
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (i + n) % 16;
                uint8_t hi = (i + b) % 16;
                gate_blk.qs[i] = lo | (hi << 4);
                up_blk.qs[i]   = ((lo + 3) % 16) | (((hi + 5) % 16) << 4);
            }
        }
    }

    for (int n = 0; n < N_down; n++) {
        for (int b = 0; b < k_blocks_down; b++) {
            auto & blk = h_down_w[n * k_blocks_down + b];
            float scale = 0.07f + 0.003f * (n % 24) + 0.001f * b;
            blk.d = sycl::half(scale);
            for (int i = 0; i < 16; i++) {
                uint8_t lo = (i + n + 2) % 16;
                uint8_t hi = (i + b + 1) % 16;
                blk.qs[i] = lo | (hi << 4);
            }
        }
    }

    std::vector<float> h_gate(N_gate);
    std::vector<float> h_up(N_gate);
    std::vector<float> h_silu(N_gate);
    std::vector<float> h_ref(N_down);

    ref_dmmv_q4_0(h_gate_w, N_gate, K_in, h_input.data(), h_gate.data());
    ref_dmmv_q4_0(h_up_w, N_gate, K_in, h_input.data(), h_up.data());
    ref_silu_mul(h_gate.data(), h_up.data(), h_silu.data(), N_gate);
    ref_dmmv_q4_0(h_down_w, N_down, intermediate_dim, h_silu.data(), h_ref.data());

    auto gate_soa = pack_q4_0_soa(h_gate_w);
    auto up_soa   = pack_q4_0_soa(h_up_w);
    auto down_soa = pack_q4_0_soa(h_down_w);

    void * d_gate_w = sycl::malloc_device(gate_soa.size(), q);
    void * d_up_w   = sycl::malloc_device(up_soa.size(), q);
    void * d_down_w = sycl::malloc_device(down_soa.size(), q);
    float * d_input = sycl::malloc_device<float>(K_in, q);
    float * d_gate  = sycl::malloc_device<float>(N_gate, q);
    float * d_up    = sycl::malloc_device<float>(N_gate, q);
    float * d_silu  = sycl::malloc_device<float>(N_gate, q);
    float * d_out   = sycl::malloc_device<float>(N_down, q);

    q.memcpy(d_gate_w, gate_soa.data(), gate_soa.size()).wait();
    q.memcpy(d_up_w, up_soa.data(), up_soa.size()).wait();
    q.memcpy(d_down_w, down_soa.data(), down_soa.size()).wait();
    q.memcpy(d_input, h_input.data(), K_in * sizeof(float)).wait();
    q.memset(d_gate, 0, N_gate * sizeof(float)).wait();
    q.memset(d_up, 0, N_gate * sizeof(float)).wait();
    q.memset(d_silu, 0, N_gate * sizeof(float)).wait();
    q.memset(d_out, 0, N_down * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    const int n_heads = 8;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    kernel.begin_persistent(1, 1, hidden_dim, intermediate_dim,
                            n_heads, n_kv_heads, head_dim,
                            ggml_sycl_unified::QUANT_TYPE_Q4_0);

    kernel.add_matmul(0, d_gate_w, d_input, d_gate, MatmulType::GATE, 1, N_gate, K_in,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_matmul(0, d_up_w, d_input, d_up, MatmulType::UP, 1, N_gate, K_in,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_silu_mul(0, d_gate, d_up, d_silu);
    kernel.add_matmul(0, d_down_w, d_silu, d_out, MatmulType::DOWN, 1, N_down, intermediate_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);

    kernel.execute_persistent();

    std::vector<float> h_out(N_down);
    q.memcpy(h_out.data(), d_out, N_down * sizeof(float)).wait();

    float error  = max_abs_error(h_out, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_gate_w, q);
    sycl::free(d_up_w, q);
    sycl::free(d_down_w, q);
    sycl::free(d_input, q);
    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_silu, q);
    sycl::free(d_out, q);

    print_result("persistent_ffn_chain_q4_0 (hidden=128, inter=256)", passed, error);
    return passed;
}

// =============================================================================
// CPU Reference: Attention (single query, M=1)
// =============================================================================

// Standard attention with GQA support:
// output[h][d] = sum_p softmax(score[p]) * v[kv_h][p][d]
// where score[p] = dot(q[h], k[kv_h][p]) * scale
// and kv_h = h / (n_heads / n_kv_heads) when n_kv_heads < n_heads (GQA)
static void ref_attention(const float * q, const float * k_cache, const float * v_cache,
                          float * output, int n_heads, int n_kv_heads, int head_dim,
                          int seq_len, float scale) {
    for (int h = 0; h < n_heads; h++) {
        const float * q_head = q + h * head_dim;
        // GQA: compute which KV head this query head maps to
        const int kv_head = (n_kv_heads > 0 && n_kv_heads < n_heads)
                            ? h / (n_heads / n_kv_heads)
                            : h;
        const float * k_head = k_cache + kv_head * seq_len * head_dim;
        const float * v_head = v_cache + kv_head * seq_len * head_dim;
        float *       o_head = output + h * head_dim;

        // Compute scores and find max for numerical stability
        std::vector<float> scores(seq_len);
        float max_score = -INFINITY;
        for (int p = 0; p < seq_len; p++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d] * k_head[p * head_dim + d];
            }
            scores[p] = dot * scale;
            max_score = std::max(max_score, scores[p]);
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int p = 0; p < seq_len; p++) {
            scores[p] = std::exp(scores[p] - max_score);
            sum_exp += scores[p];
        }
        for (int p = 0; p < seq_len; p++) {
            scores[p] /= sum_exp;
        }

        // Value aggregation
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int p = 0; p < seq_len; p++) {
                acc += scores[p] * v_head[p * head_dim + d];
            }
            o_head[d] = acc;
        }
    }
}

// =============================================================================
// Test: Persistent Attention (single operation in persistent kernel)
// =============================================================================
static bool test_persistent_attention(sycl::queue & q) {
    const int   n_heads  = 4;
    const int   head_dim = 64;
    const int   seq_len  = 32;
    const float scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int q_size      = n_heads * head_dim;
    const int kv_size     = n_heads * seq_len * head_dim;
    const int output_size = n_heads * head_dim;

    // Create deterministic test data
    std::vector<float> h_q(q_size);
    std::vector<float> h_k(kv_size);
    std::vector<float> h_v(kv_size);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> h_ref(output_size, 0.0f);

    // Initialize with deterministic patterns
    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(i * 0.1f) * 0.5f;
    }
    for (int i = 0; i < kv_size; i++) {
        h_k[i] = std::cos(i * 0.07f) * 0.3f;
        h_v[i] = std::sin(i * 0.03f + 1.0f) * 0.4f;
    }

    // CPU reference (n_kv_heads == n_heads: no GQA)
    ref_attention(h_q.data(), h_k.data(), h_v.data(), h_ref.data(),
                  n_heads, n_heads, head_dim, seq_len, scale);

    // Allocate device memory
    float * d_q      = sycl::malloc_device<float>(q_size, q);
    float * d_k      = sycl::malloc_device<float>(kv_size, q);
    float * d_v      = sycl::malloc_device<float>(kv_size, q);
    float * d_output = sycl::malloc_device<float>(output_size, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_k, h_k.data(), kv_size * sizeof(float)).wait();
    q.memcpy(d_v, h_v.data(), kv_size * sizeof(float)).wait();
    q.memset(d_output, 0, output_size * sizeof(float)).wait();

    // Configure and run persistent kernel
    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // begin_persistent(n_layers, batch_size, hidden_dim, intermediate_dim, n_heads, n_kv_heads, head_dim, quant_type)
    // hidden_dim must be >= head_dim + 32 for SLM (query + reduction space)
    kernel.begin_persistent(1, 1, 4096, 4096, n_heads, n_heads, head_dim, 0);

    AttentionDescriptor desc = {};
    desc.q        = d_q;
    desc.k_cache  = d_k;
    desc.v_cache  = d_v;
    desc.output   = d_output;
    desc.kv_type  = KvCacheType::F32;
    desc.n_heads  = n_heads;
    desc.n_kv_heads = n_heads;  // No GQA for this test
    desc.head_dim = head_dim;
    desc.seq_len  = seq_len;
    desc.scale    = scale;
    const int64_t q_nb0 = sizeof(float);
    const int64_t q_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t q_nb2 = (int64_t) head_dim * sizeof(float);
    desc.q_nb0    = q_nb0;
    desc.q_nb1    = q_nb1;
    desc.q_nb2    = q_nb2;
    desc.q_nb3    = 0;
    const int64_t kv_nb0 = sizeof(float);
    const int64_t kv_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t kv_nb2 = (int64_t) seq_len * head_dim * sizeof(float);
    desc.k_nb0    = kv_nb0;
    desc.k_nb1    = kv_nb1;
    desc.k_nb2    = kv_nb2;
    desc.k_nb3    = 0;
    desc.v_nb0    = kv_nb0;
    desc.v_nb1    = kv_nb1;
    desc.v_nb2    = kv_nb2;
    desc.v_nb3    = 0;
    kernel.add_attention(0, desc);

    kernel.execute_persistent();

    // Read back results
    q.memcpy(h_output.data(), d_output, output_size * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    // Print debug info if failed
    if (!passed) {
        printf("    First 8 output values:\n");
        for (int i = 0; i < 8 && i < output_size; i++) {
            printf("      [%d] got=%.6f ref=%.6f diff=%.2e\n",
                   i, h_output[i], h_ref[i], std::abs(h_output[i] - h_ref[i]));
        }
    }

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_output, q);

    print_result("persistent_attention (heads=4, dim=64, seq=32)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent Attention with longer sequence
// =============================================================================
static bool test_persistent_attention_long_seq(sycl::queue & q) {
    const int   n_heads  = 2;
    const int   head_dim = 128;   // Mistral head_dim
    const int   seq_len  = 512;   // Longer sequence
    const float scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int q_size      = n_heads * head_dim;
    const int kv_size     = n_heads * seq_len * head_dim;
    const int output_size = n_heads * head_dim;

    std::vector<float> h_q(q_size);
    std::vector<float> h_k(kv_size);
    std::vector<float> h_v(kv_size);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> h_ref(output_size, 0.0f);

    // Initialize with varied patterns to exercise softmax
    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(i * 0.05f) * 0.3f;
    }
    for (int i = 0; i < kv_size; i++) {
        h_k[i] = std::cos(i * 0.013f) * 0.2f;
        h_v[i] = std::sin(i * 0.017f + 2.0f) * 0.5f;
    }

    ref_attention(h_q.data(), h_k.data(), h_v.data(), h_ref.data(),
                  n_heads, n_heads, head_dim, seq_len, scale);

    float * d_q      = sycl::malloc_device<float>(q_size, q);
    float * d_k      = sycl::malloc_device<float>(kv_size, q);
    float * d_v      = sycl::malloc_device<float>(kv_size, q);
    float * d_output = sycl::malloc_device<float>(output_size, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_k, h_k.data(), kv_size * sizeof(float)).wait();
    q.memcpy(d_v, h_v.data(), kv_size * sizeof(float)).wait();
    q.memset(d_output, 0, output_size * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, 4096, 4096, n_heads, n_heads, head_dim, 0);

    AttentionDescriptor desc = {};
    desc.q        = d_q;
    desc.k_cache  = d_k;
    desc.v_cache  = d_v;
    desc.output   = d_output;
    desc.kv_type  = KvCacheType::F32;
    desc.n_heads  = n_heads;
    desc.n_kv_heads = n_heads;
    desc.head_dim = head_dim;
    desc.seq_len  = seq_len;
    desc.scale    = scale;
    const int64_t q_nb0 = sizeof(float);
    const int64_t q_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t q_nb2 = (int64_t) head_dim * sizeof(float);
    desc.q_nb0    = q_nb0;
    desc.q_nb1    = q_nb1;
    desc.q_nb2    = q_nb2;
    desc.q_nb3    = 0;
    const int64_t kv_nb0 = sizeof(float);
    const int64_t kv_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t kv_nb2 = (int64_t) seq_len * head_dim * sizeof(float);
    desc.k_nb0    = kv_nb0;
    desc.k_nb1    = kv_nb1;
    desc.k_nb2    = kv_nb2;
    desc.k_nb3    = 0;
    desc.v_nb0    = kv_nb0;
    desc.v_nb1    = kv_nb1;
    desc.v_nb2    = kv_nb2;
    desc.v_nb3    = 0;
    kernel.add_attention(0, desc);

    kernel.execute_persistent();

    q.memcpy(h_output.data(), d_output, output_size * sizeof(float)).wait();

    // Use slightly relaxed tolerance for longer sequences (more FP accumulation)
    const float long_seq_tol = 5e-3f;
    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < long_seq_tol;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    if (!passed) {
        printf("    First 8 output values:\n");
        for (int i = 0; i < 8 && i < output_size; i++) {
            printf("      [%d] got=%.6f ref=%.6f diff=%.2e\n",
                   i, h_output[i], h_ref[i], std::abs(h_output[i] - h_ref[i]));
        }
    }

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_output, q);

    if (passed) {
        printf("  [PASS] persistent_attention_long_seq (heads=2, dim=128, seq=512) (max_error=%.2e)\n", error);
    } else {
        printf("  [FAIL] persistent_attention_long_seq (heads=2, dim=128, seq=512) (max_error=%.2e, tolerance=%.2e)\n",
               error, long_seq_tol);
    }
    return passed;
}

// =============================================================================
// Test: Persistent Attention with GQA (Grouped Query Attention)
// =============================================================================
static bool test_persistent_attention_gqa(sycl::queue & q) {
    // GQA: 4:1 ratio — 8 query heads share 2 KV heads
    // Heads 0-3 share kv_head 0, heads 4-7 share kv_head 1
    const int   n_heads    = 8;
    const int   n_kv_heads = 2;
    const int   head_dim   = 64;
    const int   seq_len    = 32;
    const float scale      = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int q_size      = n_heads * head_dim;
    const int kv_size     = n_kv_heads * seq_len * head_dim;  // KV cache sized by n_kv_heads
    const int output_size = n_heads * head_dim;

    // Create deterministic test data
    std::vector<float> h_q(q_size);
    std::vector<float> h_k(kv_size);
    std::vector<float> h_v(kv_size);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> h_ref(output_size, 0.0f);

    // Initialize with deterministic patterns
    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(i * 0.1f) * 0.5f;
    }
    for (int i = 0; i < kv_size; i++) {
        h_k[i] = std::cos(i * 0.07f) * 0.3f;
        h_v[i] = std::sin(i * 0.03f + 1.0f) * 0.4f;
    }

    // CPU reference with GQA
    ref_attention(h_q.data(), h_k.data(), h_v.data(), h_ref.data(),
                  n_heads, n_kv_heads, head_dim, seq_len, scale);

    // Verify that the CPU reference produces shared outputs:
    // Query heads sharing the same KV head should produce different outputs
    // (because they have different Q vectors) but use the same K/V data.
    // Heads 0 and 1 share kv_head 0, so they use the same K/V but different Q.
    // Quick sanity: heads 0 and 1 outputs should differ (different Q).
    bool q_heads_differ = false;
    for (int d = 0; d < head_dim; d++) {
        if (std::abs(h_ref[0 * head_dim + d] - h_ref[1 * head_dim + d]) > 1e-6f) {
            q_heads_differ = true;
            break;
        }
    }
    if (!q_heads_differ) {
        printf("    WARNING: GQA ref heads 0 and 1 have identical output (unexpected)\n");
    }

    // Allocate device memory
    float * d_q      = sycl::malloc_device<float>(q_size, q);
    float * d_k      = sycl::malloc_device<float>(kv_size, q);
    float * d_v      = sycl::malloc_device<float>(kv_size, q);
    float * d_output = sycl::malloc_device<float>(output_size, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_k, h_k.data(), kv_size * sizeof(float)).wait();
    q.memcpy(d_v, h_v.data(), kv_size * sizeof(float)).wait();
    q.memset(d_output, 0, output_size * sizeof(float)).wait();

    // Configure and run persistent kernel
    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, 4096, 4096, n_heads, n_kv_heads, head_dim, 0);

    AttentionDescriptor desc = {};
    desc.q          = d_q;
    desc.k_cache    = d_k;
    desc.v_cache    = d_v;
    desc.output     = d_output;
    desc.kv_type    = KvCacheType::F32;
    desc.n_heads    = n_heads;
    desc.n_kv_heads = n_kv_heads;  // GQA: 4:1 ratio
    desc.head_dim   = head_dim;
    desc.seq_len    = seq_len;
    desc.scale      = scale;
    const int64_t q_nb0 = sizeof(float);
    const int64_t q_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t q_nb2 = (int64_t) head_dim * sizeof(float);
    desc.q_nb0    = q_nb0;
    desc.q_nb1    = q_nb1;
    desc.q_nb2    = q_nb2;
    desc.q_nb3    = 0;
    const int64_t kv_nb0 = sizeof(float);
    const int64_t kv_nb1 = (int64_t) head_dim * sizeof(float);
    const int64_t kv_nb2 = (int64_t) seq_len * head_dim * sizeof(float);
    desc.k_nb0    = kv_nb0;
    desc.k_nb1    = kv_nb1;
    desc.k_nb2    = kv_nb2;
    desc.k_nb3    = 0;
    desc.v_nb0    = kv_nb0;
    desc.v_nb1    = kv_nb1;
    desc.v_nb2    = kv_nb2;
    desc.v_nb3    = 0;
    kernel.add_attention(0, desc);

    kernel.execute_persistent();

    // Read back results
    q.memcpy(h_output.data(), d_output, output_size * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);
    printf("    n_heads=%d, n_kv_heads=%d, ratio=%d:1\n",
           n_heads, n_kv_heads, n_heads / n_kv_heads);

    // Print debug info if failed
    if (!passed) {
        printf("    First 8 output values per head:\n");
        for (int h = 0; h < n_heads; h++) {
            printf("    Head %d (kv_head=%d):", h, h / (n_heads / n_kv_heads));
            for (int d = 0; d < 4 && d < head_dim; d++) {
                int idx = h * head_dim + d;
                printf(" got=%.4f ref=%.4f", h_output[idx], h_ref[idx]);
            }
            printf("\n");
        }
    }

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_output, q);

    print_result("persistent_attention_gqa (heads=8, kv_heads=2, dim=64, seq=32)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent SET_ROWS + Attention + Residual ADD
// =============================================================================
static bool test_persistent_set_rows_attention_residual(sycl::queue & q) {
    const int n_heads     = 2;
    const int n_kv_heads  = 1;
    const int head_dim    = 8;
    const int seq_len     = 4;
    const int n_tokens    = 1;
    const int position    = 2;
    const int hidden_dim  = n_heads * head_dim;

    const int q_size      = n_heads * head_dim;
    const int kv_row_size = n_kv_heads * head_dim;
    const int kv_cache_size = n_kv_heads * seq_len * head_dim;

    std::vector<float> h_q(q_size);
    std::vector<float> h_residual(q_size);
    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(i * 0.07f) * 0.9f;
        h_residual[i] = std::cos(i * 0.05f) * 0.3f;
    }

    std::vector<float> h_src_k(kv_row_size);
    std::vector<float> h_src_v(kv_row_size);
    for (int i = 0; i < kv_row_size; i++) {
        h_src_k[i] = std::sin(i * 0.11f) * 0.6f;
        h_src_v[i] = std::cos(i * 0.09f) * 0.4f;
    }

    std::vector<int32_t> h_indices(n_tokens, position);

    std::vector<float> h_k_cache(kv_cache_size, 0.0f);
    std::vector<float> h_v_cache(kv_cache_size, 0.0f);
    for (int d = 0; d < head_dim; d++) {
        h_k_cache[position * head_dim + d] = h_src_k[d];
        h_v_cache[position * head_dim + d] = h_src_v[d];
    }

    std::vector<float> h_attn_ref(q_size);
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ref_attention(h_q.data(), h_k_cache.data(), h_v_cache.data(),
                  h_attn_ref.data(), n_heads, n_kv_heads, head_dim, seq_len, scale);

    std::vector<float> h_ref(q_size);
    for (int i = 0; i < q_size; i++) {
        h_ref[i] = h_attn_ref[i] + h_residual[i];
    }

    float * d_q         = sycl::malloc_device<float>(q_size, q);
    float * d_residual  = sycl::malloc_device<float>(q_size, q);
    float * d_src_k     = sycl::malloc_device<float>(kv_row_size, q);
    float * d_src_v     = sycl::malloc_device<float>(kv_row_size, q);
    float * d_k_cache   = sycl::malloc_device<float>(kv_cache_size, q);
    float * d_v_cache   = sycl::malloc_device<float>(kv_cache_size, q);
    int32_t * d_indices = sycl::malloc_device<int32_t>(n_tokens, q);
    float * d_attn_out  = sycl::malloc_device<float>(q_size, q);
    float * d_out       = sycl::malloc_device<float>(q_size, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_residual, h_residual.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_src_k, h_src_k.data(), kv_row_size * sizeof(float)).wait();
    q.memcpy(d_src_v, h_src_v.data(), kv_row_size * sizeof(float)).wait();
    q.memcpy(d_indices, h_indices.data(), n_tokens * sizeof(int32_t)).wait();
    q.memset(d_k_cache, 0, kv_cache_size * sizeof(float)).wait();
    q.memset(d_v_cache, 0, kv_cache_size * sizeof(float)).wait();
    q.memset(d_attn_out, 0, q_size * sizeof(float)).wait();
    q.memset(d_out, 0, q_size * sizeof(float)).wait();

    SetRowsMeta meta = {};
    meta.nc   = head_dim;
    meta.nr   = n_tokens;
    meta.ne1  = seq_len;
    meta.ne02 = n_kv_heads;
    meta.ne03 = 1;
    meta.ne11 = 1;
    meta.ne12 = 1;
    meta.nb00 = sizeof(float);
    meta.nb01 = meta.nb00 * meta.nc;
    meta.nb02 = meta.nb01 * meta.nr;
    meta.nb03 = meta.nb02 * meta.ne02;
    meta.nb0  = sizeof(float);
    meta.nb1  = meta.nb0 * meta.nc;
    meta.nb2  = meta.nb1 * meta.ne1;
    meta.nb3  = meta.nb2 * meta.ne02;
    meta.nb10 = sizeof(int32_t);
    meta.nb11 = meta.nb10 * n_tokens;
    meta.nb12 = meta.nb11;
    meta.src_type = 0;
    meta.dst_type = 0;
    meta.idx_type = 0;
    meta.pad = 0;

    SetRowsMeta * d_meta = sycl::malloc_device<SetRowsMeta>(1, q);
    q.memcpy(d_meta, &meta, sizeof(meta)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, hidden_dim, hidden_dim, n_heads, n_kv_heads, head_dim, 0);

    const int n_elements = head_dim * n_tokens * n_kv_heads;
    const int64_t sr_bytes = set_rows_output_bytes(meta);
    kernel.add_set_rows(0, d_src_k, d_indices, d_k_cache, d_meta, n_elements, nullptr, sr_bytes);
    kernel.add_set_rows(0, d_src_v, d_indices, d_v_cache, d_meta, n_elements, nullptr, sr_bytes);

    AttentionDescriptor desc = {};
    desc.q          = d_q;
    desc.k_cache    = d_k_cache;
    desc.v_cache    = d_v_cache;
    desc.output     = d_attn_out;
    desc.kv_type    = KvCacheType::F32;
    desc.n_heads    = n_heads;
    desc.n_kv_heads = n_kv_heads;
    desc.head_dim   = head_dim;
    desc.seq_len    = seq_len;
    desc.scale      = scale;
    desc.q_nb0      = sizeof(float);
    desc.q_nb1      = (int64_t) head_dim * sizeof(float);
    desc.q_nb2      = (int64_t) head_dim * sizeof(float);
    desc.q_nb3      = 0;
    desc.k_nb0      = sizeof(float);
    desc.k_nb1      = (int64_t) head_dim * sizeof(float);
    desc.k_nb2      = (int64_t) seq_len * head_dim * sizeof(float);
    desc.k_nb3      = 0;
    desc.v_nb0      = sizeof(float);
    desc.v_nb1      = (int64_t) head_dim * sizeof(float);
    desc.v_nb2      = (int64_t) seq_len * head_dim * sizeof(float);
    desc.v_nb3      = 0;
    kernel.add_attention(0, desc);

    kernel.add_add(0, d_attn_out, d_residual, d_out, q_size);

    kernel.execute_persistent();

    std::vector<float> h_out(q_size);
    q.memcpy(h_out.data(), d_out, q_size * sizeof(float)).wait();

    float error  = max_abs_error(h_out, h_ref);
    bool  passed = error < 1e-4f;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_q, q);
    sycl::free(d_residual, q);
    sycl::free(d_src_k, q);
    sycl::free(d_src_v, q);
    sycl::free(d_k_cache, q);
    sycl::free(d_v_cache, q);
    sycl::free(d_indices, q);
    sycl::free(d_attn_out, q);
    sycl::free(d_out, q);
    sycl::free(d_meta, q);

    print_result("persistent_set_rows_attention_residual (heads=2, dim=8, seq=4)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent STRIDED_COPY (transpose view)
// =============================================================================
static bool test_persistent_strided_copy_transpose(sycl::queue & q) {
    const int rows = 3;
    const int cols = 5;
    const int ne0_view = rows;
    const int ne1_view = cols;
    const int n_elements = ne0_view * ne1_view;

    std::vector<float> h_base(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            h_base[r * cols + c] = std::sin((r + 1) * 0.17f + (c + 3) * 0.11f);
        }
    }

    std::vector<float> h_ref(n_elements);
    for (int i1 = 0; i1 < ne1_view; i1++) {
        for (int i0 = 0; i0 < ne0_view; i0++) {
            h_ref[i1 * ne0_view + i0] = h_base[i0 * cols + i1];
        }
    }

    float * d_base = sycl::malloc_device<float>(rows * cols, q);
    float * d_out  = sycl::malloc_device<float>(n_elements, q);

    StridedCopyMeta meta = {};
    meta.ne[0] = ne0_view;
    meta.ne[1] = ne1_view;
    meta.ne[2] = 1;
    meta.ne[3] = 1;
    meta.nb[0] = (int64_t) cols * (int64_t) sizeof(float);  // transpose: stride by rows
    meta.nb[1] = (int64_t) sizeof(float);
    meta.nb[2] = 0;
    meta.nb[3] = 0;
    meta.type_size = (int32_t) sizeof(float);
    meta.pad = 0;

    StridedCopyMeta * d_meta = sycl::malloc_device<StridedCopyMeta>(1, q);
    q.memcpy(d_base, h_base.data(), h_base.size() * sizeof(float)).wait();
    q.memcpy(d_meta, &meta, sizeof(meta)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, cols, cols, 1, 1, ne0_view, 0);
    kernel.add_strided_copy(0, d_base, d_out, d_meta, n_elements,
                            (int64_t) n_elements * (int64_t) sizeof(float));
    kernel.execute_persistent();

    std::vector<float> h_out(n_elements);
    q.memcpy(h_out.data(), d_out, n_elements * sizeof(float)).wait();

    float error  = max_abs_error(h_out, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_base, q);
    sycl::free(d_out, q);
    sycl::free(d_meta, q);

    print_result("persistent_strided_copy_transpose (3x5 -> 5x3)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent Attention with strided Q/K/V buffers
// =============================================================================
static bool test_persistent_attention_strided_kv(sycl::queue & q) {
    const int n_heads    = 4;
    const int n_kv_heads = 2;
    const int head_dim   = 8;
    const int seq_len    = 4;

    const int q_stride = head_dim + 3;
    const int k_pos_stride = head_dim + 2;
    const int v_pos_stride = head_dim + 1;
    const int k_head_stride = k_pos_stride * seq_len + 3;
    const int v_head_stride = v_pos_stride * seq_len + 2;

    std::vector<float> h_q_storage(n_heads * q_stride, 0.0f);
    std::vector<float> h_k_storage(n_kv_heads * k_head_stride, 0.0f);
    std::vector<float> h_v_storage(n_kv_heads * v_head_stride, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            h_q_storage[h * q_stride + d] = std::sin(0.13f * (h * head_dim + d));
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        for (int p = 0; p < seq_len; p++) {
            for (int d = 0; d < head_dim; d++) {
                const int k_idx = h * k_head_stride + p * k_pos_stride + d;
                const int v_idx = h * v_head_stride + p * v_pos_stride + d;
                h_k_storage[k_idx] = std::cos(0.09f * (h * 31 + p * head_dim + d));
                h_v_storage[v_idx] = std::sin(0.07f * (h * 17 + p * head_dim + d));
            }
        }
    }

    std::vector<float> h_q_contig(n_heads * head_dim);
    std::vector<float> h_k_contig(n_kv_heads * seq_len * head_dim);
    std::vector<float> h_v_contig(n_kv_heads * seq_len * head_dim);
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            h_q_contig[h * head_dim + d] = h_q_storage[h * q_stride + d];
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int p = 0; p < seq_len; p++) {
            for (int d = 0; d < head_dim; d++) {
                const int k_idx = h * k_head_stride + p * k_pos_stride + d;
                const int v_idx = h * v_head_stride + p * v_pos_stride + d;
                const int out_idx = h * seq_len * head_dim + p * head_dim + d;
                h_k_contig[out_idx] = h_k_storage[k_idx];
                h_v_contig[out_idx] = h_v_storage[v_idx];
            }
        }
    }

    std::vector<float> h_ref(n_heads * head_dim);
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ref_attention(h_q_contig.data(), h_k_contig.data(), h_v_contig.data(),
                  h_ref.data(), n_heads, n_kv_heads, head_dim, seq_len, scale);

    float * d_q = sycl::malloc_device<float>(h_q_storage.size(), q);
    float * d_k = sycl::malloc_device<float>(h_k_storage.size(), q);
    float * d_v = sycl::malloc_device<float>(h_v_storage.size(), q);
    float * d_out = sycl::malloc_device<float>(n_heads * head_dim, q);

    q.memcpy(d_q, h_q_storage.data(), h_q_storage.size() * sizeof(float)).wait();
    q.memcpy(d_k, h_k_storage.data(), h_k_storage.size() * sizeof(float)).wait();
    q.memcpy(d_v, h_v_storage.data(), h_v_storage.size() * sizeof(float)).wait();
    q.memset(d_out, 0, n_heads * head_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, n_heads * head_dim, n_heads * head_dim,
                            n_heads, n_kv_heads, head_dim, 0);

    AttentionDescriptor desc = {};
    desc.q          = d_q;
    desc.k_cache    = d_k;
    desc.v_cache    = d_v;
    desc.output     = d_out;
    desc.kv_type    = KvCacheType::F32;
    desc.n_heads    = n_heads;
    desc.n_kv_heads = n_kv_heads;
    desc.head_dim   = head_dim;
    desc.seq_len    = seq_len;
    desc.scale      = scale;
    desc.q_nb0      = sizeof(float);
    desc.q_nb1      = (int64_t) head_dim * sizeof(float);
    desc.q_nb2      = (int64_t) q_stride * sizeof(float);
    desc.q_nb3      = 0;
    desc.k_nb0      = sizeof(float);
    desc.k_nb1      = (int64_t) k_pos_stride * sizeof(float);
    desc.k_nb2      = (int64_t) k_head_stride * sizeof(float);
    desc.k_nb3      = 0;
    desc.v_nb0      = sizeof(float);
    desc.v_nb1      = (int64_t) v_pos_stride * sizeof(float);
    desc.v_nb2      = (int64_t) v_head_stride * sizeof(float);
    desc.v_nb3      = 0;

    kernel.add_attention(0, desc);
    kernel.execute_persistent();

    std::vector<float> h_out(n_heads * head_dim);
    q.memcpy(h_out.data(), d_out, h_out.size() * sizeof(float)).wait();

    float error  = max_abs_error(h_out, h_ref);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_out, q);

    print_result("persistent_attention_strided_kv (heads=4, kv=2, dim=8, seq=4)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent Attention with F16 KV cache
// =============================================================================
static bool test_persistent_attention_f16_kv(sycl::queue & q) {
    const int n_heads    = 4;
    const int n_kv_heads = 2;
    const int head_dim   = 16;
    const int seq_len    = 8;

    const int q_size  = n_heads * head_dim;
    const int kv_size = n_kv_heads * seq_len * head_dim;

    std::vector<float> h_q(q_size);
    std::vector<sycl::half> h_k(kv_size);
    std::vector<sycl::half> h_v(kv_size);
    std::vector<float> h_k_ref(kv_size);
    std::vector<float> h_v_ref(kv_size);

    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(0.07f * (i + 1)) * 0.9f;
    }
    for (int i = 0; i < kv_size; i++) {
        const float k_val = std::cos(0.05f * (i + 2)) * 0.8f;
        const float v_val = std::sin(0.06f * (i + 3)) * 0.7f;
        h_k[i] = sycl::half(k_val);
        h_v[i] = sycl::half(v_val);
        h_k_ref[i] = static_cast<float>(h_k[i]);
        h_v_ref[i] = static_cast<float>(h_v[i]);
    }

    std::vector<float> h_ref(q_size);
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ref_attention(h_q.data(), h_k_ref.data(), h_v_ref.data(),
                  h_ref.data(), n_heads, n_kv_heads, head_dim, seq_len, scale);

    float * d_q = sycl::malloc_device<float>(q_size, q);
    sycl::half * d_k = sycl::malloc_device<sycl::half>(kv_size, q);
    sycl::half * d_v = sycl::malloc_device<sycl::half>(kv_size, q);
    float * d_out = sycl::malloc_device<float>(q_size, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_k, h_k.data(), kv_size * sizeof(sycl::half)).wait();
    q.memcpy(d_v, h_v.data(), kv_size * sizeof(sycl::half)).wait();
    q.memset(d_out, 0, q_size * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, q_size, q_size, n_heads, n_kv_heads, head_dim, 0);

    AttentionDescriptor desc = {};
    desc.q          = d_q;
    desc.k_cache    = d_k;
    desc.v_cache    = d_v;
    desc.output     = d_out;
    desc.kv_type    = KvCacheType::F16;
    desc.n_heads    = n_heads;
    desc.n_kv_heads = n_kv_heads;
    desc.head_dim   = head_dim;
    desc.seq_len    = seq_len;
    desc.scale      = scale;
    desc.q_nb0      = sizeof(float);
    desc.q_nb1      = (int64_t) head_dim * sizeof(float);
    desc.q_nb2      = (int64_t) head_dim * sizeof(float);
    desc.q_nb3      = 0;
    desc.k_nb0      = sizeof(sycl::half);
    desc.k_nb1      = (int64_t) head_dim * sizeof(sycl::half);
    desc.k_nb2      = (int64_t) seq_len * head_dim * sizeof(sycl::half);
    desc.k_nb3      = 0;
    desc.v_nb0      = sizeof(sycl::half);
    desc.v_nb1      = (int64_t) head_dim * sizeof(sycl::half);
    desc.v_nb2      = (int64_t) seq_len * head_dim * sizeof(sycl::half);
    desc.v_nb3      = 0;

    kernel.add_attention(0, desc);
    kernel.execute_persistent();

    std::vector<float> h_out(q_size);
    q.memcpy(h_out.data(), d_out, q_size * sizeof(float)).wait();

    float error  = max_abs_error(h_out, h_ref);
    bool  passed = error < 5e-3f;

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_out, q);

    print_result("persistent_attention_f16_kv (heads=4, kv=2, dim=16, seq=8)", passed, error);
    return passed;
}

// =============================================================================
// Test: Mini TG pipeline (RMS + MUL + QKV + ROPE + SET_ROWS + ATTN + OUT + ADD + FFN + ADD)
// =============================================================================
static bool test_persistent_mini_tg_pipeline(sycl::queue & q) {
    const int hidden_dim       = 32;
    const int intermediate_dim = 64;
    const int n_heads          = 4;
    const int n_kv_heads       = 2;
    const int head_dim         = 8;
    const int seq_len          = 4;
    const int position         = 2;
    const float eps            = 1e-5f;

    const int qkv_dim = n_kv_heads * head_dim;

    auto make_weights = [](int N, int K, float scale_base, int seed) {
        const int k_blocks = K / 32;
        std::vector<ggml_sycl_unified::block_q4_0_unified> weights(N * k_blocks);
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < k_blocks; b++) {
                auto & blk = weights[n * k_blocks + b];
                float scale = scale_base + 0.002f * ((n + seed) % 31) + 0.001f * b;
                blk.d = sycl::half(scale);
                for (int i = 0; i < 16; i++) {
                    uint8_t lo = (i + n + seed) % 16;
                    uint8_t hi = (i + b + seed) % 16;
                    blk.qs[i] = lo | (hi << 4);
                }
            }
        }
        return weights;
    };

    const auto w_q    = make_weights(hidden_dim, hidden_dim, 0.05f, 1);
    const auto w_k    = make_weights(qkv_dim, hidden_dim, 0.051f, 2);
    const auto w_v    = make_weights(qkv_dim, hidden_dim, 0.052f, 3);
    const auto w_out  = make_weights(hidden_dim, hidden_dim, 0.053f, 4);
    const auto w_gate = make_weights(intermediate_dim, hidden_dim, 0.054f, 5);
    const auto w_up   = make_weights(intermediate_dim, hidden_dim, 0.055f, 6);
    const auto w_down = make_weights(hidden_dim, intermediate_dim, 0.056f, 7);

    std::vector<float> h_input(hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
        h_input[i] = std::sin(i * 0.07f) * 0.8f + 0.1f;
    }

    std::vector<float> h_norm_wt(hidden_dim);
    std::vector<float> h_ffn_wt(hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
        h_norm_wt[i] = 1.0f + 0.05f * std::cos(i * 0.03f);
        h_ffn_wt[i]  = 1.0f + 0.04f * std::sin(i * 0.02f);
    }

    std::vector<float> h_norm(hidden_dim);
    std::vector<float> h_norm_w(hidden_dim);
    ref_rms_norm_unweighted(h_input.data(), h_norm.data(), hidden_dim, eps);
    for (int i = 0; i < hidden_dim; i++) {
        h_norm_w[i] = h_norm[i] * h_norm_wt[i];
    }

    std::vector<float> h_q(hidden_dim);
    std::vector<float> h_k(qkv_dim);
    std::vector<float> h_v(qkv_dim);
    ref_dmmv_q4_0(w_q, hidden_dim, hidden_dim, h_norm_w.data(), h_q.data());
    ref_dmmv_q4_0(w_k, qkv_dim, hidden_dim, h_norm_w.data(), h_k.data());
    ref_dmmv_q4_0(w_v, qkv_dim, hidden_dim, h_norm_w.data(), h_v.data());

    const int half_dim = head_dim / 2;
    std::vector<float> h_cos(half_dim);
    std::vector<float> h_sin(half_dim);
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / head_dim);
        float angle = position * freq;
        h_cos[i] = std::cos(angle);
        h_sin[i] = std::sin(angle);
    }

    std::vector<float> h_q_rope(h_q);
    std::vector<float> h_k_rope(h_k);
    ref_rope_neox(h_q_rope.data(), h_k_rope.data(), h_cos.data(), h_sin.data(),
                  n_heads, n_kv_heads, head_dim);

    std::vector<float> h_k_cache(qkv_dim * seq_len, 0.0f);
    std::vector<float> h_v_cache(qkv_dim * seq_len, 0.0f);
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            const int src_idx = h * head_dim + d;
            const int dst_idx = h * seq_len * head_dim + position * head_dim + d;
            h_k_cache[dst_idx] = h_k_rope[src_idx];
            h_v_cache[dst_idx] = h_v[src_idx];
        }
    }

    std::vector<float> h_attn(hidden_dim);
    const float scale = 1.0f / std::sqrt((float) head_dim);
    ref_attention(h_q_rope.data(), h_k_cache.data(), h_v_cache.data(),
                  h_attn.data(), n_heads, n_kv_heads, head_dim, seq_len, scale);

    std::vector<float> h_out_proj(hidden_dim);
    ref_dmmv_q4_0(w_out, hidden_dim, hidden_dim, h_attn.data(), h_out_proj.data());

    std::vector<float> h_attn_resid(hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
        h_attn_resid[i] = h_out_proj[i] + h_input[i];
    }

    std::vector<float> h_ffn_norm(hidden_dim);
    std::vector<float> h_ffn_norm_w(hidden_dim);
    ref_rms_norm_unweighted(h_attn_resid.data(), h_ffn_norm.data(), hidden_dim, eps);
    for (int i = 0; i < hidden_dim; i++) {
        h_ffn_norm_w[i] = h_ffn_norm[i] * h_ffn_wt[i];
    }

    std::vector<float> h_gate(intermediate_dim);
    std::vector<float> h_up(intermediate_dim);
    std::vector<float> h_silu(intermediate_dim);
    ref_dmmv_q4_0(w_gate, intermediate_dim, hidden_dim, h_ffn_norm_w.data(), h_gate.data());
    ref_dmmv_q4_0(w_up, intermediate_dim, hidden_dim, h_ffn_norm_w.data(), h_up.data());
    ref_silu_mul(h_gate.data(), h_up.data(), h_silu.data(), intermediate_dim);

    std::vector<float> h_down(hidden_dim);
    ref_dmmv_q4_0(w_down, hidden_dim, intermediate_dim, h_silu.data(), h_down.data());

    std::vector<float> h_final(hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
        h_final[i] = h_down[i] + h_attn_resid[i];
    }

    auto wq_soa   = pack_q4_0_soa(w_q);
    auto wk_soa   = pack_q4_0_soa(w_k);
    auto wv_soa   = pack_q4_0_soa(w_v);
    auto wo_soa   = pack_q4_0_soa(w_out);
    auto wg_soa   = pack_q4_0_soa(w_gate);
    auto wu_soa   = pack_q4_0_soa(w_up);
    auto wd_soa   = pack_q4_0_soa(w_down);

    void * d_wq = sycl::malloc_device(wq_soa.size(), q);
    void * d_wk = sycl::malloc_device(wk_soa.size(), q);
    void * d_wv = sycl::malloc_device(wv_soa.size(), q);
    void * d_wo = sycl::malloc_device(wo_soa.size(), q);
    void * d_wg = sycl::malloc_device(wg_soa.size(), q);
    void * d_wu = sycl::malloc_device(wu_soa.size(), q);
    void * d_wd = sycl::malloc_device(wd_soa.size(), q);

    float * d_input      = sycl::malloc_device<float>(hidden_dim, q);
    float * d_norm       = sycl::malloc_device<float>(hidden_dim, q);
    float * d_norm_wt    = sycl::malloc_device<float>(hidden_dim, q);
    float * d_norm_w     = sycl::malloc_device<float>(hidden_dim, q);
    float * d_q          = sycl::malloc_device<float>(hidden_dim, q);
    float * d_k          = sycl::malloc_device<float>(qkv_dim, q);
    float * d_v          = sycl::malloc_device<float>(qkv_dim, q);
    float * d_q_rope     = sycl::malloc_device<float>(hidden_dim, q);
    float * d_k_rope     = sycl::malloc_device<float>(qkv_dim, q);
    float * d_k_cache    = sycl::malloc_device<float>(qkv_dim * seq_len, q);
    float * d_v_cache    = sycl::malloc_device<float>(qkv_dim * seq_len, q);
    float * d_attn       = sycl::malloc_device<float>(hidden_dim, q);
    float * d_out_proj   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_attn_resid = sycl::malloc_device<float>(hidden_dim, q);
    float * d_ffn_norm   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_ffn_wt     = sycl::malloc_device<float>(hidden_dim, q);
    float * d_ffn_norm_w = sycl::malloc_device<float>(hidden_dim, q);
    float * d_gate       = sycl::malloc_device<float>(intermediate_dim, q);
    float * d_up         = sycl::malloc_device<float>(intermediate_dim, q);
    float * d_silu       = sycl::malloc_device<float>(intermediate_dim, q);
    float * d_down       = sycl::malloc_device<float>(hidden_dim, q);
    float * d_final      = sycl::malloc_device<float>(hidden_dim, q);

    float * d_cos = sycl::malloc_device<float>(half_dim, q);
    float * d_sin = sycl::malloc_device<float>(half_dim, q);

    std::vector<int32_t> h_indices(1, position);
    int32_t * d_indices = sycl::malloc_device<int32_t>(1, q);

    q.memcpy(d_wq, wq_soa.data(), wq_soa.size()).wait();
    q.memcpy(d_wk, wk_soa.data(), wk_soa.size()).wait();
    q.memcpy(d_wv, wv_soa.data(), wv_soa.size()).wait();
    q.memcpy(d_wo, wo_soa.data(), wo_soa.size()).wait();
    q.memcpy(d_wg, wg_soa.data(), wg_soa.size()).wait();
    q.memcpy(d_wu, wu_soa.data(), wu_soa.size()).wait();
    q.memcpy(d_wd, wd_soa.data(), wd_soa.size()).wait();

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_norm_wt, h_norm_wt.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_ffn_wt, h_ffn_wt.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_cos, h_cos.data(), half_dim * sizeof(float)).wait();
    q.memcpy(d_sin, h_sin.data(), half_dim * sizeof(float)).wait();
    q.memcpy(d_indices, h_indices.data(), sizeof(int32_t)).wait();

    q.memset(d_norm, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_norm_w, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_q, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_k, 0, qkv_dim * sizeof(float)).wait();
    q.memset(d_v, 0, qkv_dim * sizeof(float)).wait();
    q.memset(d_q_rope, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_k_rope, 0, qkv_dim * sizeof(float)).wait();
    q.memset(d_k_cache, 0, qkv_dim * seq_len * sizeof(float)).wait();
    q.memset(d_v_cache, 0, qkv_dim * seq_len * sizeof(float)).wait();
    q.memset(d_attn, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_out_proj, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_attn_resid, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_ffn_norm, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_ffn_norm_w, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_gate, 0, intermediate_dim * sizeof(float)).wait();
    q.memset(d_up, 0, intermediate_dim * sizeof(float)).wait();
    q.memset(d_silu, 0, intermediate_dim * sizeof(float)).wait();
    q.memset(d_down, 0, hidden_dim * sizeof(float)).wait();
    q.memset(d_final, 0, hidden_dim * sizeof(float)).wait();

    SetRowsMeta meta = {};
    meta.nc   = head_dim;
    meta.nr   = 1;
    meta.ne1  = seq_len;
    meta.ne02 = n_kv_heads;
    meta.ne03 = 1;
    meta.ne11 = 1;
    meta.ne12 = 1;
    meta.nb00 = sizeof(float);
    meta.nb01 = meta.nb00 * meta.nc;
    meta.nb02 = meta.nb01 * meta.nr;
    meta.nb03 = meta.nb02 * meta.ne02;
    meta.nb0  = sizeof(float);
    meta.nb1  = meta.nb0 * meta.nc;
    meta.nb2  = meta.nb1 * meta.ne1;
    meta.nb3  = meta.nb2 * meta.ne02;
    meta.nb10 = sizeof(int32_t);
    meta.nb11 = meta.nb10;
    meta.nb12 = meta.nb11;
    meta.src_type = 0;
    meta.dst_type = 0;
    meta.idx_type = 0;
    meta.pad = 0;

    SetRowsMeta * d_meta = sycl::malloc_device<SetRowsMeta>(1, q);
    q.memcpy(d_meta, &meta, sizeof(meta)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, hidden_dim, intermediate_dim,
                            n_heads, n_kv_heads, head_dim,
                            ggml_sycl_unified::QUANT_TYPE_Q4_0);

    kernel.add_rms_norm(0, nullptr, d_input, d_norm, eps, hidden_dim);
    kernel.add_mul(0, d_norm, d_norm_wt, d_norm_w, hidden_dim);

    kernel.add_matmul(0, d_wq, d_norm_w, d_q, MatmulType::Q_PROJ, 1, hidden_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_matmul(0, d_wk, d_norm_w, d_k, MatmulType::K_PROJ, 1, qkv_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_matmul(0, d_wv, d_norm_w, d_v, MatmulType::V_PROJ, 1, qkv_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);

    RopeDescriptor q_rope = {};
    q_rope.q         = d_q;
    q_rope.k         = nullptr;
    q_rope.rope_dst  = d_q_rope;
    q_rope.cos_cache = d_cos;
    q_rope.sin_cache = d_sin;
    q_rope.n_heads   = n_heads;
    q_rope.head_dim  = head_dim;
    q_rope.position  = position;
    q_rope.is_neox   = true;
    kernel.add_rope(0, q_rope);

    RopeDescriptor k_rope = {};
    k_rope.q         = d_k;
    k_rope.k         = nullptr;
    k_rope.rope_dst  = d_k_rope;
    k_rope.cos_cache = d_cos;
    k_rope.sin_cache = d_sin;
    k_rope.n_heads   = n_kv_heads;
    k_rope.head_dim  = head_dim;
    k_rope.position  = position;
    k_rope.is_neox   = true;
    kernel.add_rope(0, k_rope);

    const int n_elements = head_dim * 1 * n_kv_heads;
    const int64_t sr_bytes = set_rows_output_bytes(meta);
    kernel.add_set_rows(0, d_k_rope, d_indices, d_k_cache, d_meta, n_elements, nullptr, sr_bytes);
    kernel.add_set_rows(0, d_v, d_indices, d_v_cache, d_meta, n_elements, nullptr, sr_bytes);

    AttentionDescriptor attn_desc = {};
    attn_desc.q          = d_q_rope;
    attn_desc.k_cache    = d_k_cache;
    attn_desc.v_cache    = d_v_cache;
    attn_desc.output     = d_attn;
    attn_desc.kv_type    = KvCacheType::F32;
    attn_desc.n_heads    = n_heads;
    attn_desc.n_kv_heads = n_kv_heads;
    attn_desc.head_dim   = head_dim;
    attn_desc.seq_len    = seq_len;
    attn_desc.scale      = scale;
    attn_desc.q_nb0      = sizeof(float);
    attn_desc.q_nb1      = (int64_t) head_dim * sizeof(float);
    attn_desc.q_nb2      = (int64_t) head_dim * sizeof(float);
    attn_desc.q_nb3      = 0;
    attn_desc.k_nb0      = sizeof(float);
    attn_desc.k_nb1      = (int64_t) head_dim * sizeof(float);
    attn_desc.k_nb2      = (int64_t) seq_len * head_dim * sizeof(float);
    attn_desc.k_nb3      = 0;
    attn_desc.v_nb0      = sizeof(float);
    attn_desc.v_nb1      = (int64_t) head_dim * sizeof(float);
    attn_desc.v_nb2      = (int64_t) seq_len * head_dim * sizeof(float);
    attn_desc.v_nb3      = 0;
    kernel.add_attention(0, attn_desc);

    kernel.add_matmul(0, d_wo, d_attn, d_out_proj, MatmulType::OUT_PROJ, 1, hidden_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_add(0, d_out_proj, d_input, d_attn_resid, hidden_dim);

    kernel.add_rms_norm(0, nullptr, d_attn_resid, d_ffn_norm, eps, hidden_dim);
    kernel.add_mul(0, d_ffn_norm, d_ffn_wt, d_ffn_norm_w, hidden_dim);

    kernel.add_matmul(0, d_wg, d_ffn_norm_w, d_gate, MatmulType::GATE, 1, intermediate_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_matmul(0, d_wu, d_ffn_norm_w, d_up, MatmulType::UP, 1, intermediate_dim, hidden_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_silu_mul(0, d_gate, d_up, d_silu);
    kernel.add_matmul(0, d_wd, d_silu, d_down, MatmulType::DOWN, 1, hidden_dim, intermediate_dim,
                      ggml_sycl_unified::QUANT_TYPE_Q4_0, ggml_sycl_unified::WEIGHT_LAYOUT_SOA);
    kernel.add_add(0, d_down, d_attn_resid, d_final, hidden_dim);

    kernel.execute_persistent();

    std::vector<float> h_norm_out(hidden_dim);
    std::vector<float> h_norm_w_out(hidden_dim);
    std::vector<float> h_q_out(hidden_dim);
    std::vector<float> h_k_out(qkv_dim);
    std::vector<float> h_v_out(qkv_dim);
    std::vector<float> h_q_rope_out(hidden_dim);
    std::vector<float> h_k_rope_out(qkv_dim);
    std::vector<float> h_attn_out(hidden_dim);
    std::vector<float> h_out_proj_out(hidden_dim);
    std::vector<float> h_attn_resid_out(hidden_dim);
    std::vector<float> h_ffn_norm_out(hidden_dim);
    std::vector<float> h_ffn_norm_w_out(hidden_dim);
    std::vector<float> h_gate_out(intermediate_dim);
    std::vector<float> h_up_out(intermediate_dim);
    std::vector<float> h_silu_out(intermediate_dim);
    std::vector<float> h_down_out(hidden_dim);
    std::vector<float> h_final_out(hidden_dim);
    std::vector<float> h_k_cache_out(qkv_dim * seq_len);
    std::vector<float> h_v_cache_out(qkv_dim * seq_len);

    q.memcpy(h_norm_out.data(), d_norm, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_norm_w_out.data(), d_norm_w, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_q_out.data(), d_q, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_k_out.data(), d_k, qkv_dim * sizeof(float)).wait();
    q.memcpy(h_v_out.data(), d_v, qkv_dim * sizeof(float)).wait();
    q.memcpy(h_q_rope_out.data(), d_q_rope, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_k_rope_out.data(), d_k_rope, qkv_dim * sizeof(float)).wait();
    q.memcpy(h_k_cache_out.data(), d_k_cache, qkv_dim * seq_len * sizeof(float)).wait();
    q.memcpy(h_v_cache_out.data(), d_v_cache, qkv_dim * seq_len * sizeof(float)).wait();
    q.memcpy(h_attn_out.data(), d_attn, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_out_proj_out.data(), d_out_proj, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_attn_resid_out.data(), d_attn_resid, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_ffn_norm_out.data(), d_ffn_norm, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_ffn_norm_w_out.data(), d_ffn_norm_w, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_gate_out.data(), d_gate, intermediate_dim * sizeof(float)).wait();
    q.memcpy(h_up_out.data(), d_up, intermediate_dim * sizeof(float)).wait();
    q.memcpy(h_silu_out.data(), d_silu, intermediate_dim * sizeof(float)).wait();
    q.memcpy(h_down_out.data(), d_down, hidden_dim * sizeof(float)).wait();
    q.memcpy(h_final_out.data(), d_final, hidden_dim * sizeof(float)).wait();

    float tol = 1e-3f;
    bool passed = true;
    std::string first_fail;
    float first_err = 0.0f;

    auto check = [&](const char * name, const std::vector<float> & got, const std::vector<float> & ref) {
        float err = max_abs_error(got, ref);
        if (err > tol && passed) {
            first_fail = name;
            first_err = err;
            passed = false;
        }
        return err;
    };

    const float err_norm       = check("rms_norm", h_norm_out, h_norm);
    const float err_norm_w     = check("norm_mul", h_norm_w_out, h_norm_w);
    const float err_q          = check("q_proj", h_q_out, h_q);
    const float err_k          = check("k_proj", h_k_out, h_k);
    const float err_v          = check("v_proj", h_v_out, h_v);
    const float err_q_rope     = check("q_rope", h_q_rope_out, h_q_rope);
    const float err_k_rope     = check("k_rope", h_k_rope_out, h_k_rope);
    const float err_k_cache    = check("k_cache", h_k_cache_out, h_k_cache);
    const float err_v_cache    = check("v_cache", h_v_cache_out, h_v_cache);
    const float err_attn       = check("attention", h_attn_out, h_attn);
    const float err_out_proj   = check("out_proj", h_out_proj_out, h_out_proj);
    const float err_attn_resid = check("attn_resid", h_attn_resid_out, h_attn_resid);
    const float err_ffn_norm   = check("ffn_norm", h_ffn_norm_out, h_ffn_norm);
    const float err_ffn_norm_w = check("ffn_norm_mul", h_ffn_norm_w_out, h_ffn_norm_w);
    const float err_gate       = check("ffn_gate", h_gate_out, h_gate);
    const float err_up         = check("ffn_up", h_up_out, h_up);
    const float err_silu       = check("ffn_silu", h_silu_out, h_silu);
    const float err_down       = check("ffn_down", h_down_out, h_down);
    const float err_final      = check("final", h_final_out, h_final);

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);
    printf("    max_err: norm=%.2e norm_mul=%.2e q=%.2e k=%.2e v=%.2e q_rope=%.2e k_rope=%.2e\n",
           err_norm, err_norm_w, err_q, err_k, err_v, err_q_rope, err_k_rope);
    printf("    max_err: k_cache=%.2e v_cache=%.2e attn=%.2e out=%.2e resid=%.2e ffn_norm=%.2e ffn_mul=%.2e\n",
           err_k_cache, err_v_cache, err_attn, err_out_proj, err_attn_resid, err_ffn_norm, err_ffn_norm_w);
    printf("    max_err: gate=%.2e up=%.2e silu=%.2e down=%.2e final=%.2e\n",
           err_gate, err_up, err_silu, err_down, err_final);

    if (!passed) {
        printf("    First mismatch: %s (max_error=%.2e, tol=%.2e)\n",
               first_fail.c_str(), first_err, tol);
    }

    sycl::free(d_wq, q);
    sycl::free(d_wk, q);
    sycl::free(d_wv, q);
    sycl::free(d_wo, q);
    sycl::free(d_wg, q);
    sycl::free(d_wu, q);
    sycl::free(d_wd, q);
    sycl::free(d_input, q);
    sycl::free(d_norm, q);
    sycl::free(d_norm_wt, q);
    sycl::free(d_norm_w, q);
    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_q_rope, q);
    sycl::free(d_k_rope, q);
    sycl::free(d_k_cache, q);
    sycl::free(d_v_cache, q);
    sycl::free(d_attn, q);
    sycl::free(d_out_proj, q);
    sycl::free(d_attn_resid, q);
    sycl::free(d_ffn_norm, q);
    sycl::free(d_ffn_wt, q);
    sycl::free(d_ffn_norm_w, q);
    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_silu, q);
    sycl::free(d_down, q);
    sycl::free(d_final, q);
    sycl::free(d_cos, q);
    sycl::free(d_sin, q);
    sycl::free(d_indices, q);
    sycl::free(d_meta, q);

    print_result("persistent_mini_tg_pipeline (hidden=32, heads=4, inter=64)", passed, first_err);
    return passed;
}

// =============================================================================
// CPU Reference: Neox-style RoPE
// =============================================================================

static void ref_rope_neox(float * q, float * k, const float * cos_cache, const float * sin_cache,
                          int n_heads, int n_kv_heads, int head_dim) {
    const int half_dim = head_dim / 2;

    // Apply to Q heads
    for (int h = 0; h < n_heads; h++) {
        float * data = q + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float x0 = data[i];
            float x1 = data[i + half_dim];
            data[i]            = x0 * cos_cache[i] - x1 * sin_cache[i];
            data[i + half_dim] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }

    // Apply to K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float * data = k + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float x0 = data[i];
            float x1 = data[i + half_dim];
            data[i]            = x0 * cos_cache[i] - x1 * sin_cache[i];
            data[i + half_dim] = x0 * sin_cache[i] + x1 * cos_cache[i];
        }
    }
}

// =============================================================================
// Test: Persistent RoPE (neox-style rotary position embeddings)
// =============================================================================
static bool test_persistent_rope(sycl::queue & q) {
    // Mistral-like config: 32 query heads, 8 KV heads (GQA 4:1), 128-dim heads
    const int n_heads    = 32;
    const int n_kv_heads = 8;
    const int head_dim   = 128;
    const int half_dim   = head_dim / 2;
    const int position   = 42;

    const int q_size = n_heads * head_dim;
    const int k_size = n_kv_heads * head_dim;

    // Initialize Q and K with deterministic patterns
    std::vector<float> h_q(q_size);
    std::vector<float> h_k(k_size);
    for (int i = 0; i < q_size; i++) {
        h_q[i] = std::sin(i * 0.01f) * 2.0f;
    }
    for (int i = 0; i < k_size; i++) {
        h_k[i] = std::cos(i * 0.02f) * 1.5f;
    }

    // Pre-compute cos/sin caches for the given position
    std::vector<float> h_cos(half_dim);
    std::vector<float> h_sin(half_dim);
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / head_dim);
        float angle = position * freq;
        h_cos[i] = std::cos(angle);
        h_sin[i] = std::sin(angle);
    }

    // CPU reference
    std::vector<float> h_q_ref(h_q);
    std::vector<float> h_k_ref(h_k);
    ref_rope_neox(h_q_ref.data(), h_k_ref.data(), h_cos.data(), h_sin.data(),
                  n_heads, n_kv_heads, head_dim);

    // Allocate device memory
    float * d_q   = sycl::malloc_device<float>(q_size, q);
    float * d_k   = sycl::malloc_device<float>(k_size, q);
    float * d_cos = sycl::malloc_device<float>(half_dim, q);
    float * d_sin = sycl::malloc_device<float>(half_dim, q);

    q.memcpy(d_q, h_q.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_k, h_k.data(), k_size * sizeof(float)).wait();
    q.memcpy(d_cos, h_cos.data(), half_dim * sizeof(float)).wait();
    q.memcpy(d_sin, h_sin.data(), half_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, 4096, 11008, n_heads, n_kv_heads, head_dim, 0);

    RopeDescriptor desc = {};
    desc.q         = d_q;
    desc.k         = d_k;
    desc.cos_cache = d_cos;
    desc.sin_cache = d_sin;
    desc.n_heads   = n_heads;
    desc.head_dim  = head_dim;
    desc.position  = position;
    desc.is_neox   = true;  // Test uses NEOX-style split-pair layout
    kernel.add_rope(0, desc);

    kernel.execute_persistent();

    // Read back results (RoPE is in-place)
    std::vector<float> h_q_out(q_size);
    std::vector<float> h_k_out(k_size);
    q.memcpy(h_q_out.data(), d_q, q_size * sizeof(float)).wait();
    q.memcpy(h_k_out.data(), d_k, k_size * sizeof(float)).wait();

    float q_error = max_abs_error(h_q_out, h_q_ref);
    float k_error = max_abs_error(h_k_out, h_k_ref);
    float error   = std::max(q_error, k_error);

    // RoPE uses exact same FP ops, so tolerance can be tight
    const float rope_tol = 1e-5f;
    bool passed = error < rope_tol;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);
    printf("    q_error=%.2e, k_error=%.2e\n", q_error, k_error);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_cos, q);
    sycl::free(d_sin, q);

    print_result("persistent_rope (n_heads=32, n_kv_heads=8, head_dim=128)", passed, error);
    return passed;
}

// =============================================================================
// Test: Persistent RoPE single-tensor mode (NEOX)
// =============================================================================
static bool test_persistent_rope_single(sycl::queue & q) {
    const int n_heads  = 16;
    const int head_dim = 64;
    const int half_dim = head_dim / 2;
    const int position = 17;

    const int q_size = n_heads * head_dim;

    std::vector<float> h_src(q_size);
    for (int i = 0; i < q_size; i++) {
        h_src[i] = std::sin(i * 0.013f) * 1.3f;
    }

    std::vector<float> h_cos(half_dim);
    std::vector<float> h_sin(half_dim);
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / head_dim);
        float angle = position * freq;
        h_cos[i] = std::cos(angle);
        h_sin[i] = std::sin(angle);
    }

    std::vector<float> h_ref(h_src);
    std::vector<float> h_dummy(1, 0.0f);
    ref_rope_neox(h_ref.data(), h_dummy.data(), h_cos.data(), h_sin.data(),
                  n_heads, 0, head_dim);

    float * d_src = sycl::malloc_device<float>(q_size, q);
    float * d_dst = sycl::malloc_device<float>(q_size, q);
    float * d_cos = sycl::malloc_device<float>(half_dim, q);
    float * d_sin = sycl::malloc_device<float>(half_dim, q);

    q.memcpy(d_src, h_src.data(), q_size * sizeof(float)).wait();
    q.memcpy(d_cos, h_cos.data(), half_dim * sizeof(float)).wait();
    q.memcpy(d_sin, h_sin.data(), half_dim * sizeof(float)).wait();
    q.memset(d_dst, 0, q_size * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.begin_persistent(1, 1, 4096, 11008, n_heads, 0, head_dim, 0);

    RopeDescriptor desc = {};
    desc.q         = d_src;
    desc.k         = nullptr;
    desc.rope_dst  = d_dst;
    desc.cos_cache = d_cos;
    desc.sin_cache = d_sin;
    desc.n_heads   = n_heads;
    desc.head_dim  = head_dim;
    desc.position  = position;
    desc.is_neox   = true;
    kernel.add_rope(0, desc);

    kernel.execute_persistent();

    std::vector<float> h_out(q_size);
    q.memcpy(h_out.data(), d_dst, q_size * sizeof(float)).wait();

    float error = max_abs_error(h_out, h_ref);
    bool passed = error < 1e-5f;

    auto stats = kernel.get_last_stats();
    printf("    ops=%d, tiles=%d, time=%.2f ms\n",
           stats.n_operations, stats.total_tiles, stats.kernel_time_ms);

    sycl::free(d_src, q);
    sycl::free(d_dst, q);
    sycl::free(d_cos, q);
    sycl::free(d_sin, q);

    print_result("persistent_rope_single (n_heads=16, head_dim=64)", passed, error);
    return passed;
}

// =============================================================================
// Test: Plan cancellation
// =============================================================================
static bool test_persistent_cancel(sycl::queue & q) {
    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    // Start building a plan
    kernel.begin_persistent(1, 1, 128, 256, 32, 8, 128, 0);
    bool building_before = kernel.is_building_plan();

    // Cancel the plan
    kernel.cancel_persistent();
    bool building_after = kernel.is_building_plan();

    bool passed = building_before && !building_after;
    printf("    building_before_cancel=%s, building_after_cancel=%s\n",
           building_before ? "true" : "false",
           building_after ? "true" : "false");

    print_result("persistent_cancel", passed);
    return passed;
}

int main() {
    printf("UnifiedKernel Persistent Execution Tests\n");
    printf("=========================================\n\n");

    try {
        sycl::queue q(sycl::gpu_selector_v);
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        int passed = 0;
        int failed = 0;

        printf("Extraction Tests:\n");
        if (test_extract_layer_index()) { passed++; } else { failed++; }

        printf("Single Operation Tests:\n");
        if (test_persistent_rms_norm(q)) { passed++; } else { failed++; }
        if (test_persistent_silu_mul(q)) { passed++; } else { failed++; }

        printf("\nChained Operation Tests:\n");
        if (test_persistent_chain(q))    { passed++; } else { failed++; }

        printf("\nMulti-Layer Tests:\n");
        if (test_persistent_multi_layer(q)) { passed++; } else { failed++; }

        printf("\nDMMV Matmul Tests:\n");
        if (test_persistent_dmmv_matmul(q)) { passed++; } else { failed++; }
        if (test_persistent_dmmv_matmul_soa(q)) { passed++; } else { failed++; }
        if (test_persistent_ffn_chain_q4_0(q)) { passed++; } else { failed++; }

        printf("\nAttention Tests:\n");
        if (test_persistent_attention(q))          { passed++; } else { failed++; }
        if (test_persistent_attention_long_seq(q))  { passed++; } else { failed++; }
        if (test_persistent_attention_gqa(q))       { passed++; } else { failed++; }
        if (test_persistent_set_rows_attention_residual(q)) { passed++; } else { failed++; }
        if (test_persistent_strided_copy_transpose(q)) { passed++; } else { failed++; }
        if (test_persistent_attention_strided_kv(q))   { passed++; } else { failed++; }
        if (test_persistent_attention_f16_kv(q))       { passed++; } else { failed++; }
        if (test_persistent_mini_tg_pipeline(q))    { passed++; } else { failed++; }

        printf("\nRoPE Tests:\n");
        if (test_persistent_rope(q))                 { passed++; } else { failed++; }
        if (test_persistent_rope_single(q))          { passed++; } else { failed++; }

        printf("\nDiagnostics Tests:\n");
        if (test_persistent_stats(q))    { passed++; } else { failed++; }
        if (test_persistent_cancel(q))   { passed++; } else { failed++; }

        printf("\n=========================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);

        return failed > 0 ? 1 : 0;

    } catch (const sycl::exception & e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }
}
