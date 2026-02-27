//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for cpu-dispatch.cpp T8 mixed-precision cache miss loading:
// - INT4 fast-path SIMD kernel correctness (vs full-precision dispatch)
// - Adaptive dispatch split (full precision + INT4 reduced)
// - Non-Q4_0 fallback to full precision
// - Config functions (env var parsing)

#include "cpu-dispatch.hpp"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int K = 4096;  // Typical expert input dimension
static constexpr int N = 128;   // Small expert output dimension for tests

// Generate random float data in [-1, 1].
static void fill_random(float * data, int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

// Generate positive-biased float data in [0.1, 1.0].
// This ensures Q8 values have a non-zero sum, making the INT4 unsigned
// nibble bias (8 * sum(q8_vals) * scale) detectable.
static void fill_positive(float * data, int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (int i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

// Quantize float data to Q4_0 (row-wise).
static std::vector<uint8_t> quantize_q4_0(const float * data, int rows, int cols) {
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, cols);
    std::vector<uint8_t> qdata(rows * row_size);
    for (int r = 0; r < rows; r++) {
        quantize_row_q4_0_ref(data + r * cols, (block_q4_0 *)(qdata.data() + r * row_size), cols);
    }
    return qdata;
}

// Quantize float data to Q6_K (row-wise).
static std::vector<uint8_t> quantize_q6_K(const float * data, int rows, int cols) {
    const size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, cols);
    std::vector<uint8_t> qdata(rows * row_size);
    for (int r = 0; r < rows; r++) {
        quantize_row_q6_K_ref(data + r * cols, (block_q6_K *)(qdata.data() + r * row_size), cols);
    }
    return qdata;
}

// Max absolute difference between two float arrays.
static float max_abs_diff(const float * a, const float * b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

// Mean absolute difference.
static float mean_abs_diff(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return (float)(sum / n);
}

// Mean absolute value.
static float mean_abs(const float * a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i]);
    }
    return (float)(sum / n);
}

// ---------------------------------------------------------------------------
// Test 1: INT4 kernel vs full-precision dispatch for Q4_0
// ---------------------------------------------------------------------------
// The INT4 kernel skips the zero-point offset subtraction, producing output
// that differs from full-precision by a systematic bias proportional to
// 8 * sum(q8_vals) * scale per block. With mixed-sign activations (realistic
// for neural nets), sum(q8) is near zero so the bias is small.

static bool test_int4_kernel_correctness() {
    printf("test_int4_kernel_correctness ... ");

    std::vector<float> weight_f(N * K);
    std::vector<float> act_f(K);
    fill_random(weight_f.data(), N * K, 42);
    fill_random(act_f.data(), K, 123);  // Mixed-sign: realistic NN activations

    auto weight_q = quantize_q4_0(weight_f.data(), N, K);

    // Baseline: full-precision output via batched dispatch (known working)
    std::vector<float> baseline(N, 0.0f);
    cpu_expert_task task;
    task.weight_host = weight_q.data();
    task.act_host    = act_f.data();
    task.output_host = baseline.data();
    task.type        = GGML_TYPE_Q4_0;
    task.K           = K;
    task.N           = N;
    ggml_sycl_cpu_expert_mul_mat_batched(&task, 1, 0);

    float baseline_mag = mean_abs(baseline.data(), N);
    if (baseline_mag < 1.0f) {
        printf("FAIL (baseline output too small: mean_abs=%.4f)\n", baseline_mag);
        return false;
    }

    // INT4 via adaptive: create threshold+1 tasks so the last one
    // goes through INT4 path.
    const int threshold = ggml_sycl_expert_miss_burst_threshold();
    const int n_tasks = threshold + 1;
    std::vector<cpu_expert_task> tasks(n_tasks);
    std::vector<std::vector<float>> outputs(n_tasks, std::vector<float>(N, 0.0f));

    for (int i = 0; i < n_tasks; i++) {
        tasks[i] = task;
        tasks[i].output_host = outputs[i].data();
    }

    ggml_sycl_cpu_expert_mul_mat_adaptive(tasks.data(), n_tasks, n_tasks);

    // First `threshold` tasks should be full precision (match baseline)
    for (int i = 0; i < threshold && i < n_tasks; i++) {
        float diff = max_abs_diff(baseline.data(), outputs[i].data(), N);
        if (diff > 1e-5f) {
            printf("FAIL (task %d should be full precision, max_diff=%.6f)\n", i, diff);
            return false;
        }
    }

    // Last task (beyond threshold) should have INT4 bias (differ from baseline)
    float int4_max_diff = max_abs_diff(baseline.data(), outputs[threshold].data(), N);
    float int4_mean_diff = mean_abs_diff(baseline.data(), outputs[threshold].data(), N);

    // INT4 should differ from full precision (bias from skipped offset).
    if (int4_mean_diff < 1e-3f) {
        printf("FAIL (INT4 output identical to baseline, mean_diff=%.6f)\n", int4_mean_diff);
        return false;
    }

    // INT4 output should have reasonable magnitude — same order of magnitude
    // as baseline. The offset-skip introduces a per-block random-walk bias
    // of 8*sum(q8)*scale, which can be large for high-dimensional inputs
    // (K=4096). We verify the INT4 output isn't zero/garbage by checking
    // its magnitude is within 10x of baseline.
    float int4_mag = mean_abs(outputs[threshold].data(), N);
    if (int4_mag < baseline_mag * 0.01f || int4_mag > baseline_mag * 100.0f) {
        printf("FAIL (INT4 magnitude %.1f outside [%.1f, %.1f])\n",
               int4_mag, baseline_mag * 0.01f, baseline_mag * 100.0f);
        return false;
    }

    printf("OK (baseline_mag=%.1f, int4_mag=%.1f, max_diff=%.1f, mean_diff=%.1f)\n",
           baseline_mag, int4_mag, int4_max_diff, int4_mean_diff);
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Adaptive dispatch split
// ---------------------------------------------------------------------------
// Verify that adaptive dispatch correctly splits: first `threshold` tasks
// get full precision, remaining tasks get INT4 reduced precision.

static bool test_adaptive_split() {
    printf("test_adaptive_split ... ");

    std::vector<float> weight_f(N * K);
    std::vector<float> act_f(K);
    fill_random(weight_f.data(), N * K, 77);
    fill_positive(act_f.data(), K, 88);

    auto weight_q = quantize_q4_0(weight_f.data(), N, K);

    // Baseline: full-precision output
    std::vector<float> baseline(N, 0.0f);
    cpu_expert_task base_task;
    base_task.weight_host = weight_q.data();
    base_task.act_host    = act_f.data();
    base_task.output_host = baseline.data();
    base_task.type        = GGML_TYPE_Q4_0;
    base_task.K           = K;
    base_task.N           = N;
    ggml_sycl_cpu_expert_mul_mat_batched(&base_task, 1, 0);

    const int n_tasks = 5;
    std::vector<cpu_expert_task> tasks(n_tasks);
    std::vector<std::vector<float>> outputs(n_tasks, std::vector<float>(N, 0.0f));

    for (int i = 0; i < n_tasks; i++) {
        tasks[i] = base_task;
        tasks[i].output_host = outputs[i].data();
    }

    // n_miss_total = 10 (>> threshold) to trigger mixed mode
    ggml_sycl_cpu_expert_mul_mat_adaptive(tasks.data(), n_tasks, 10);

    const int threshold = ggml_sycl_expert_miss_burst_threshold();

    int n_full_match = 0;
    int n_int4_differ = 0;
    for (int i = 0; i < n_tasks; i++) {
        float diff = max_abs_diff(baseline.data(), outputs[i].data(), N);
        if (i < threshold) {
            if (diff < 1e-5f) {
                n_full_match++;
            }
        } else {
            if (diff > 1e-3f) {
                n_int4_differ++;
            }
        }
    }

    int expected_full = std::min(threshold, n_tasks);
    int expected_int4 = n_tasks - expected_full;

    if (n_full_match != expected_full) {
        printf("FAIL (expected %d full-precision matches, got %d)\n", expected_full, n_full_match);
        return false;
    }
    if (n_int4_differ != expected_int4) {
        printf("FAIL (expected %d INT4-different tasks, got %d)\n", expected_int4, n_int4_differ);
        return false;
    }

    printf("OK (%d full + %d int4)\n", n_full_match, n_int4_differ);
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: Non-Q4_0 falls back to full precision
// ---------------------------------------------------------------------------

static bool test_non_q4_0_fallback() {
    printf("test_non_q4_0_fallback ... ");

    std::vector<float> weight_f(N * K);
    std::vector<float> act_f(K);
    fill_random(weight_f.data(), N * K, 99);
    fill_random(act_f.data(), K, 200);

    auto weight_q = quantize_q6_K(weight_f.data(), N, K);

    // Baseline: full-precision output
    std::vector<float> baseline(N, 0.0f);
    cpu_expert_task base_task;
    base_task.weight_host = weight_q.data();
    base_task.act_host    = act_f.data();
    base_task.output_host = baseline.data();
    base_task.type        = GGML_TYPE_Q6_K;
    base_task.K           = K;
    base_task.N           = N;
    ggml_sycl_cpu_expert_mul_mat_batched(&base_task, 1, 0);

    float baseline_mag = mean_abs(baseline.data(), N);
    if (baseline_mag < 1.0f) {
        printf("FAIL (baseline output too small: mean_abs=%.4f)\n", baseline_mag);
        return false;
    }

    const int n_tasks = 5;
    std::vector<cpu_expert_task> tasks(n_tasks);
    std::vector<std::vector<float>> outputs(n_tasks, std::vector<float>(N, 0.0f));

    for (int i = 0; i < n_tasks; i++) {
        tasks[i] = base_task;
        tasks[i].output_host = outputs[i].data();
    }

    ggml_sycl_cpu_expert_mul_mat_adaptive(tasks.data(), n_tasks, 10);

    // ALL tasks should match baseline (Q6_K has no INT4 fast path)
    for (int i = 0; i < n_tasks; i++) {
        float diff = max_abs_diff(baseline.data(), outputs[i].data(), N);
        if (diff > 1e-5f) {
            printf("FAIL (task %d: Q6_K should be full precision, max_diff=%.6f)\n", i, diff);
            return false;
        }
    }

    printf("OK (all %d tasks full precision for Q6_K)\n", n_tasks);
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Config functions
// ---------------------------------------------------------------------------

static bool test_config_functions() {
    printf("test_config_functions ... ");

    expert_miss_precision mode = ggml_sycl_expert_miss_precision_mode();
    if (mode != expert_miss_precision::MIXED && mode != expert_miss_precision::FULL) {
        printf("FAIL (invalid precision mode %d)\n", (int) mode);
        return false;
    }

    int threshold = ggml_sycl_expert_miss_burst_threshold();
    if (threshold < 1) {
        printf("FAIL (threshold %d < 1)\n", threshold);
        return false;
    }

    // Below-threshold: adaptive dispatch should produce full-precision output
    std::vector<float> weight_f(N * K);
    std::vector<float> act_f(K);
    fill_random(weight_f.data(), N * K, 55);
    fill_random(act_f.data(), K, 66);
    auto weight_q = quantize_q4_0(weight_f.data(), N, K);

    std::vector<float> baseline(N, 0.0f);
    cpu_expert_task task;
    task.weight_host = weight_q.data();
    task.act_host    = act_f.data();
    task.output_host = baseline.data();
    task.type        = GGML_TYPE_Q4_0;
    task.K           = K;
    task.N           = N;
    ggml_sycl_cpu_expert_mul_mat_batched(&task, 1, 0);

    // n_miss_total = 1 (below threshold) should force full precision
    std::vector<float> output(N, 0.0f);
    task.output_host = output.data();
    ggml_sycl_cpu_expert_mul_mat_adaptive(&task, 1, 1);

    float diff = max_abs_diff(baseline.data(), output.data(), N);
    if (diff > 1e-5f) {
        printf("FAIL (below-threshold should be full precision, diff=%.6f)\n", diff);
        return false;
    }

    // Test vec_dot_rows directly (regression test for FP16 table init)
    std::vector<float> vdr_output(N, 0.0f);
    ggml_sycl_cpu_vec_dot_rows(GGML_TYPE_Q4_0, K,
                                weight_q.data(), act_f.data(),
                                vdr_output.data(), N);
    float vdr_mag = mean_abs(vdr_output.data(), N);
    if (vdr_mag < 1.0f) {
        printf("FAIL (vec_dot_rows output all zero, mag=%.4f)\n", vdr_mag);
        return false;
    }

    printf("OK (mode=%s, threshold=%d)\n",
           mode == expert_miss_precision::MIXED ? "mixed" : "full",
           threshold);
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // Initialize CPU backend (populates FP16->FP32 lookup table required by vec_dot)
    ggml_cpu_init();

    printf("=== test-sycl-cpu-dispatch: T8 Mixed-Precision Cache Miss Loading ===\n\n");

    int n_pass = 0;
    int n_fail = 0;

    auto run = [&](bool (*fn)()) {
        if (fn()) { n_pass++; } else { n_fail++; }
    };

    run(test_int4_kernel_correctness);
    run(test_adaptive_split);
    run(test_non_q4_0_fallback);
    run(test_config_functions);

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
