//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unit tests for UnifiedKernel core operations
// TDD: These tests are written BEFORE the implementations
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

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

static void ref_rms_norm(const float * input, const float * weights, float * output, int hidden_dim, float eps) {
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
// Test Cases
// =============================================================================

static bool test_rms_norm_basic(sycl::queue & q) {
    const int   hidden_dim = 128;
    const float eps        = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_output(hidden_dim);
    std::vector<float> h_ref_output(hidden_dim);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i]   = std::sin(i * 0.1f);
        h_weights[i] = 1.0f + 0.1f * std::cos(i * 0.05f);
    }

    ref_rms_norm(h_input.data(), h_weights.data(), h_ref_output.data(), hidden_dim, eps);

    float * d_input   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float * d_output  = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    RmsNormDescriptor desc = {};
    desc.input             = d_input;
    desc.weights           = d_weights;
    desc.output            = d_output;
    desc.hidden_dim        = hidden_dim;
    desc.eps               = eps;

    kernel.rms_norm(desc);
    q.wait();

    q.memcpy(h_output.data(), d_output, hidden_dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref_output);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    print_result("rms_norm_basic (dim=128)", passed, error);
    return passed;
}

static bool test_rms_norm_large(sycl::queue & q) {
    const int   hidden_dim = 4096;  // Mistral 7B hidden dim
    const float eps        = 1e-5f;

    std::vector<float> h_input(hidden_dim);
    std::vector<float> h_weights(hidden_dim);
    std::vector<float> h_output(hidden_dim);
    std::vector<float> h_ref_output(hidden_dim);

    for (int i = 0; i < hidden_dim; i++) {
        h_input[i]   = std::sin(i * 0.01f) * 0.5f;
        h_weights[i] = 1.0f + 0.05f * std::cos(i * 0.02f);
    }

    ref_rms_norm(h_input.data(), h_weights.data(), h_ref_output.data(), hidden_dim, eps);

    float * d_input   = sycl::malloc_device<float>(hidden_dim, q);
    float * d_weights = sycl::malloc_device<float>(hidden_dim, q);
    float * d_output  = sycl::malloc_device<float>(hidden_dim, q);

    q.memcpy(d_input, h_input.data(), hidden_dim * sizeof(float)).wait();
    q.memcpy(d_weights, h_weights.data(), hidden_dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    RmsNormDescriptor desc = {};
    desc.input             = d_input;
    desc.weights           = d_weights;
    desc.output            = d_output;
    desc.hidden_dim        = hidden_dim;
    desc.eps               = eps;

    kernel.rms_norm(desc);
    q.wait();

    q.memcpy(h_output.data(), d_output, hidden_dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref_output);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_input, q);
    sycl::free(d_weights, q);
    sycl::free(d_output, q);

    print_result("rms_norm_large (dim=4096)", passed, error);
    return passed;
}

static bool test_silu_mul_basic(sycl::queue & q) {
    const int dim = 256;

    std::vector<float> h_gate(dim);
    std::vector<float> h_up(dim);
    std::vector<float> h_output(dim);
    std::vector<float> h_ref_output(dim);

    for (int i = 0; i < dim; i++) {
        h_gate[i] = std::sin(i * 0.1f) * 2.0f;
        h_up[i]   = std::cos(i * 0.15f) * 1.5f;
    }

    ref_silu_mul(h_gate.data(), h_up.data(), h_ref_output.data(), dim);

    float * d_gate   = sycl::malloc_device<float>(dim, q);
    float * d_up     = sycl::malloc_device<float>(dim, q);
    float * d_output = sycl::malloc_device<float>(dim, q);

    q.memcpy(d_gate, h_gate.data(), dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.silu_mul(d_gate, d_up, d_output, dim);
    q.wait();

    q.memcpy(h_output.data(), d_output, dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref_output);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    print_result("silu_mul_basic (dim=256)", passed, error);
    return passed;
}

static bool test_silu_mul_large(sycl::queue & q) {
    const int dim = 11008;  // Mistral 7B intermediate dim

    std::vector<float> h_gate(dim);
    std::vector<float> h_up(dim);
    std::vector<float> h_output(dim);
    std::vector<float> h_ref_output(dim);

    for (int i = 0; i < dim; i++) {
        h_gate[i] = std::sin(i * 0.005f) * 3.0f;
        h_up[i]   = std::cos(i * 0.007f) * 2.0f;
    }

    ref_silu_mul(h_gate.data(), h_up.data(), h_ref_output.data(), dim);

    float * d_gate   = sycl::malloc_device<float>(dim, q);
    float * d_up     = sycl::malloc_device<float>(dim, q);
    float * d_output = sycl::malloc_device<float>(dim, q);

    q.memcpy(d_gate, h_gate.data(), dim * sizeof(float)).wait();
    q.memcpy(d_up, h_up.data(), dim * sizeof(float)).wait();

    ggml_sycl::UnifiedKernel     kernel(q);
    ggml_sycl_unified::XMXConfig config = {};
    config.supported                    = true;
    config.slm_size                     = 64 * 1024;
    kernel.configure(config);

    kernel.silu_mul(d_gate, d_up, d_output, dim);
    q.wait();

    q.memcpy(h_output.data(), d_output, dim * sizeof(float)).wait();

    float error  = max_abs_error(h_output, h_ref_output);
    bool  passed = error < TEST_TOLERANCE;

    sycl::free(d_gate, q);
    sycl::free(d_up, q);
    sycl::free(d_output, q);

    print_result("silu_mul_large (dim=11008)", passed, error);
    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== UnifiedKernel Operation Tests ===\n\n");

    sycl::queue q(sycl::gpu_selector_v);
    printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int n_pass = 0;
    int n_fail = 0;

    printf("RMS Norm Tests:\n");
    if (test_rms_norm_basic(q)) {
        n_pass++;
    } else {
        n_fail++;
    }
    if (test_rms_norm_large(q)) {
        n_pass++;
    } else {
        n_fail++;
    }

    printf("\nSiLU Mul Tests:\n");
    if (test_silu_mul_basic(q)) {
        n_pass++;
    } else {
        n_fail++;
    }
    if (test_silu_mul_large(q)) {
        n_pass++;
    } else {
        n_fail++;
    }

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    return n_fail > 0 ? 1 : 0;
}
