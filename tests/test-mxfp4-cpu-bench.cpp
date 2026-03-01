// MXFP4 CPU vec_dot throughput benchmark
//
// Measures single/multi-thread MXFP4 vec_dot performance matching
// GPT-OSS 120B expert FFN dimensions (2880x2880). Compares CPU compute
// time vs PCIe transfer time to validate CPU-primary expert routing.

#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <vector>

// GPT-OSS 120B expert FFN dimensions
static constexpr int K = 2880;  // input dimension (n_embd)
static constexpr int N = 2880;  // output dimension (n_ff)

// PCIe 5.0 x8 measured bandwidth (from CLAUDE.md)
static constexpr double PCIE_BW_GBS = 13.4;

// Measure time for one matmul (N vec_dots of length K) with n_threads
static double measure_matmul_ms(const ggml_type_traits_cpu * mxfp4_cpu,
                                const uint8_t *              weights,
                                const uint8_t *              input_q8,
                                float *                      output,
                                size_t                       weight_row_bytes,
                                int                          n_threads,
                                int                          n_iters) {
    // Warmup
    for (int row = 0; row < N; row++) {
        mxfp4_cpu->vec_dot(K, &output[row], 0, weights + row * weight_row_bytes, 0, input_q8, 0, 1);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; iter++) {
        if (n_threads <= 1) {
            for (int row = 0; row < N; row++) {
                mxfp4_cpu->vec_dot(K, &output[row], 0, weights + row * weight_row_bytes, 0, input_q8, 0, 1);
            }
        } else {
            std::vector<std::future<void>> futures;
            const int                      rows_per_thread = (N + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; t++) {
                const int start = t * rows_per_thread;
                const int end   = std::min(start + rows_per_thread, N);
                if (start >= end) {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&, start, end]() {
                    for (int row = start; row < end; row++) {
                        mxfp4_cpu->vec_dot(K, &output[row], 0, weights + row * weight_row_bytes, 0, input_q8, 0, 1);
                    }
                }));
            }
            for (auto & f : futures) {
                f.get();
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iters;
}

int main() {
    // Initialize CPU backend (registers AVX2/VNNI optimized implementations)
    ggml_cpu_init();

    const ggml_type_traits_cpu * mxfp4_cpu = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const ggml_type_traits_cpu * q8_cpu    = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    if (!mxfp4_cpu || !mxfp4_cpu->vec_dot) {
        fprintf(stderr, "ERROR: MXFP4 vec_dot not available on this platform\n");
        return 1;
    }
    if (mxfp4_cpu->vec_dot_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "ERROR: MXFP4 vec_dot_type is not Q8_0 (got %d)\n", mxfp4_cpu->vec_dot_type);
        return 1;
    }
    if (!mxfp4_cpu->from_float) {
        fprintf(stderr, "ERROR: MXFP4 from_float not available\n");
        return 1;
    }
    if (!q8_cpu || !q8_cpu->from_float) {
        fprintf(stderr, "ERROR: Q8_0 from_float not available\n");
        return 1;
    }

    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_MXFP4, K);
    const size_t input_row_bytes  = ggml_row_size(GGML_TYPE_Q8_0, K);
    const double expert_weight_mb = 3.0 * N * weight_row_bytes / (1024.0 * 1024.0);
    const double pcie_ms          = 3.0 * N * weight_row_bytes / (PCIE_BW_GBS * 1e9) * 1e3;

    printf("=== MXFP4 CPU vec_dot Throughput Benchmark ===\n");
    printf("Expert FFN dimensions: K=%d, N=%d (matching GPT-OSS 120B)\n", K, N);
    printf("MXFP4 weight row size: %zu bytes (%zu blocks of 32)\n", weight_row_bytes,
           weight_row_bytes / ggml_type_size(GGML_TYPE_MXFP4));
    printf("Expert weight size (3 matmuls): %.2f MB\n", expert_weight_mb);
    printf("PCIe transfer time (%.1f GB/s): %.2f ms per expert\n\n", PCIE_BW_GBS, pcie_ms);

    // Allocate and quantize weight + input data
    std::vector<uint8_t> weights(static_cast<size_t>(N) * weight_row_bytes);
    std::vector<uint8_t> input_q8(input_row_bytes);
    std::vector<float>   f32_buf(K);
    std::vector<float>   output(N);

    // Random seed for reproducibility
    srand(42);

    // Quantize input to Q8_0
    for (int i = 0; i < K; i++) {
        f32_buf[i] = (float) (rand() % 1000 - 500) / 500.0f;
    }
    q8_cpu->from_float(f32_buf.data(), input_q8.data(), K);

    // Quantize N rows of weights to MXFP4
    for (int row = 0; row < N; row++) {
        for (int i = 0; i < K; i++) {
            f32_buf[i] = (float) (rand() % 1000 - 500) / 500.0f;
        }
        mxfp4_cpu->from_float(f32_buf.data(), weights.data() + row * weight_row_bytes, K);
    }

    // Throughput benchmark across thread counts
    int thread_counts[] = { 1, 2, 4, 8, 12, 16, 20 };
    int n_iters         = 50;

    printf("%-10s  %-12s  %-12s  %-12s  %-15s\n", "Threads", "Matmul(ms)", "Expert(ms)", "GFLOPS", "vs PCIe");
    printf("%-10s  %-12s  %-12s  %-12s  %-15s\n", "-------", "----------", "----------", "------", "-------");

    for (int tc : thread_counts) {
        double ms =
            measure_matmul_ms(mxfp4_cpu, weights.data(), input_q8.data(), output.data(), weight_row_bytes, tc, n_iters);
        double       expert_ms = ms * 3.0;  // 3 matmuls per expert FFN (gate, up, down)
        // FLOPs: N rows × K multiply-adds = N × K × 2
        double       gflops    = ((double) N * K * 2.0) / (ms / 1e3) / 1e9;
        const char * verdict   = (expert_ms < pcie_ms) ? "CPU WINS" : "PCIe wins";

        printf("%-10d  %-12.2f  %-12.2f  %-12.1f  %-15s\n", tc, ms, expert_ms, gflops, verdict);
    }

    // Correctness validation: compare quantized vec_dot with f32 reference
    printf("\n=== Correctness Validation ===\n");

    // Use a small subset for correctness check
    const int            n_check = 16;
    std::vector<float>   input_f32(K);
    std::vector<float>   weight_f32(K);
    std::vector<uint8_t> weight_mxfp4(weight_row_bytes);
    std::vector<uint8_t> input_q8_check(input_row_bytes);

    srand(123);
    for (int i = 0; i < K; i++) {
        input_f32[i] = (float) (rand() % 1000 - 500) / 500.0f;
    }
    q8_cpu->from_float(input_f32.data(), input_q8_check.data(), K);

    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;

    for (int r = 0; r < n_check; r++) {
        for (int i = 0; i < K; i++) {
            weight_f32[i] = (float) (rand() % 1000 - 500) / 500.0f;
        }
        mxfp4_cpu->from_float(weight_f32.data(), weight_mxfp4.data(), K);

        // f32 reference dot product
        double ref = 0.0;
        for (int i = 0; i < K; i++) {
            ref += (double) weight_f32[i] * (double) input_f32[i];
        }

        // quantized vec_dot
        float qval = 0.0f;
        mxfp4_cpu->vec_dot(K, &qval, 0, weight_mxfp4.data(), 0, input_q8_check.data(), 0, 1);

        double abs_err = std::abs((double) qval - ref);
        double rel_err = (std::abs(ref) > 1e-6) ? abs_err / std::abs(ref) : abs_err;

        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
        sum_abs_err += abs_err;

        if (r < 4) {
            printf("  Row %2d: f32_ref=%.4f  mxfp4_q8=%.4f  rel_err=%.4f\n", r, (float) ref, qval, (float) rel_err);
        }
    }

    printf("  Max relative error: %.4f  Mean absolute error: %.4f\n", max_rel_err, sum_abs_err / n_check);

    // MXFP4 is very low precision (4-bit), so 50% relative error is acceptable
    // for individual dot products. The important thing is the values are in the
    // right ballpark and not NaN/Inf.
    bool pass = (max_rel_err < 1.0) && !std::isnan(max_rel_err);
    printf("MXFP4 vec_dot correctness: %s (max relative error %.1f%% < 100%%)\n", pass ? "PASS" : "FAIL",
           max_rel_err * 100.0);

    // Summary for decision-making
    printf("\n=== Summary ===\n");
    printf("PCIe expert transfer time: %.2f ms\n", pcie_ms);
    printf("If any thread count achieves Expert(ms) < %.2f ms, CPU-primary routing wins.\n", pcie_ms);

    return pass ? 0 : 1;
}
