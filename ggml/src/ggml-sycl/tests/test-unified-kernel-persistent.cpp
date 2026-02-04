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
#include <sycl/sycl.hpp>

// Note: UNIFIED_KERNEL_TEST_STANDALONE is defined via CMakeLists.txt to provide
// stub implementations for common.cpp symbols needed by unified-kernel.cpp
#include "../unified-kernel.hpp"

static constexpr float TEST_TOLERANCE = 1e-3f;

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

static void ref_silu_mul(const float * gate, const float * up, float * output, int dim) {
    for (int i = 0; i < dim; i++) {
        float sigmoid_g = 1.0f / (1.0f + std::exp(-gate[i]));
        float silu_g    = gate[i] * sigmoid_g;
        output[i]       = silu_g * up[i];
    }
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
    kernel.add_rms_norm(0, d_weights, d_input, d_output);
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
    kernel.add_rms_norm(0, d_weights, d_input, d_norm_output);
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
        kernel.add_rms_norm(layer, d_weights, src, dst);
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
    const int hidden_dim = 512;

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
    kernel.add_rms_norm(0, d_weights, d_buf, d_buf);
    kernel.add_rms_norm(1, d_weights, d_buf, d_buf);
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
    kernel.add_matmul(0, d_weights, d_activations, d_output, MatmulType::Q_PROJ, 1, N, K);
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
    desc.n_heads  = n_heads;
    desc.n_kv_heads = n_heads;  // No GQA for this test
    desc.head_dim = head_dim;
    desc.seq_len  = seq_len;
    desc.scale    = scale;
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
    desc.n_heads  = n_heads;
    desc.n_kv_heads = n_heads;
    desc.head_dim = head_dim;
    desc.seq_len  = seq_len;
    desc.scale    = scale;
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
    desc.n_heads    = n_heads;
    desc.n_kv_heads = n_kv_heads;  // GQA: 4:1 ratio
    desc.head_dim   = head_dim;
    desc.seq_len    = seq_len;
    desc.scale      = scale;
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

        printf("Single Operation Tests:\n");
        if (test_persistent_rms_norm(q)) { passed++; } else { failed++; }
        if (test_persistent_silu_mul(q)) { passed++; } else { failed++; }

        printf("\nChained Operation Tests:\n");
        if (test_persistent_chain(q))    { passed++; } else { failed++; }

        printf("\nMulti-Layer Tests:\n");
        if (test_persistent_multi_layer(q)) { passed++; } else { failed++; }

        printf("\nDMMV Matmul Tests:\n");
        if (test_persistent_dmmv_matmul(q)) { passed++; } else { failed++; }

        printf("\nAttention Tests:\n");
        if (test_persistent_attention(q))          { passed++; } else { failed++; }
        if (test_persistent_attention_long_seq(q))  { passed++; } else { failed++; }
        if (test_persistent_attention_gqa(q))       { passed++; } else { failed++; }

        printf("\nRoPE Tests:\n");
        if (test_persistent_rope(q))                 { passed++; } else { failed++; }

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
