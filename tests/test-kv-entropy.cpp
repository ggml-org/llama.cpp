// test-kv-entropy.cpp
// Measure quantization MSE and estimate bits/scalar under Gaussian model
// for each KV cache format (with and without blue-noise dithering)
//
// Reference: Fergus Finn "Speculative KV coding" (May 2026)
// bits/scalar = 0.5 * log2(2*pi*e * MSE)
//
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <string>
#include <map>

// Realistic KV cache distribution: Gaussian with std ~10
static std::vector<float> generate_kv_data(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 10.0f);
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) {
        float v = dist(rng);
        if (rng() % 100 < 5) {
            v *= (rng() % 10 + 1);
        }
        data[i] = v;
    }
    return data;
}
static double compute_mse(const float* orig, const float* dequant, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = orig[i] - dequant[i];
        sum += diff * diff;
    }
    return sum / n;
}

static double bits_per_scalar(double variance) {
    if (variance <= 0) return 0.0;
    return 0.5 * log2(2.0 * M_PI * M_E * variance);
}

struct EntropyResult {
    std::string name;
    double mse;
    double bits_scalar;
    double compression_ratio;
};

struct TestConfig {
    std::string name;
    void (*quant_ref)(const float*, void*, int64_t);
    void (*dequant)(const void*, float*, int64_t);
    int block_size;
    int type_size;
};

int main() {
    const int QK = 256;
    const int N_BLOCKS = 64;
    const int N = N_BLOCKS * QK;

    printf("=== KV Cache Quantization Entropy Analysis ===\n");
    printf("Data: %d elements, Gaussian(0,10) with 5%% heavy-tail outliers\n", N);
    printf("Model: Gaussian residual (blog: 0.5*log2(2*pi*e*MSE))\n\n");

    auto data = generate_kv_data(N, 42);

    // All-positive data (like K after RoPE)
    std::vector<float> data_pos(N);
    {
        std::mt19937 rng(43);
        std::normal_distribution<float> dist(5.0f, 5.0f);
        for (int i = 0; i < N; i++) data_pos[i] = std::max(0.0f, dist(rng));
    }

    // Bimodal data (like V cache)
    std::vector<float> data_bimodal(N);
    {
        std::mt19937 rng(44);
        for (int i = 0; i < N; i++)
            data_bimodal[i] = (rng() % 2) ?
                std::normal_distribution<float>(-3.0f, 3.0f)(rng) :
                std::normal_distribution<float>(3.0f, 3.0f)(rng);
    }

    std::vector<TestConfig> tests = {
        {"q4_0",    [](const float* x, void* y, int64_t k) { quantize_row_q4_0_ref(x, (block_q4_0*)y, k); },
                    [](const void* x, float* y, int64_t k) { dequantize_row_q4_0((const block_q4_0*)x, y, k); },
                    QK4_0, sizeof(block_q4_0)},
        {"q4_0_blue", [](const float* x, void* y, int64_t k) { quantize_row_q4_0_blue_ref(x, (block_q4_0*)y, k); },
                      [](const void* x, float* y, int64_t k) { dequantize_row_q4_0((const block_q4_0*)x, y, k); },
                      QK4_0, sizeof(block_q4_0)},
        {"q4_1",    [](const float* x, void* y, int64_t k) { quantize_row_q4_1_ref(x, (block_q4_1*)y, k); },
                    [](const void* x, float* y, int64_t k) { dequantize_row_q4_1((const block_q4_1*)x, y, k); },
                    QK4_1, sizeof(block_q4_1)},
        {"q4_1_blue", [](const float* x, void* y, int64_t k) { quantize_row_q4_1_blue_ref(x, (block_q4_1*)y, k); },
                      [](const void* x, float* y, int64_t k) { dequantize_row_q4_1((const block_q4_1*)x, y, k); },
                      QK4_1, sizeof(block_q4_1)},
        {"q2_K",    [](const float* x, void* y, int64_t k) { quantize_row_q2_K_ref(x, (block_q2_K*)y, k); },
                    [](const void* x, float* y, int64_t k) { dequantize_row_q2_K((const block_q2_K*)x, y, k); },
                    QK_K, sizeof(block_q2_K)},
        {"q2_K_blue", [](const float* x, void* y, int64_t k) { quantize_row_q2_K_blue_ref(x, (block_q2_K*)y, k); },
                      [](const void* x, float* y, int64_t k) { dequantize_row_q2_K((const block_q2_K*)x, y, k); },
                      QK_K, sizeof(block_q2_K)},
        {"q3_K",    [](const float* x, void* y, int64_t k) { quantize_row_q3_K_ref(x, (block_q3_K*)y, k); },
                    [](const void* x, float* y, int64_t k) { dequantize_row_q3_K((const block_q3_K*)x, y, k); },
                    QK_K, sizeof(block_q3_K)},
        {"q3_K_blue", [](const float* x, void* y, int64_t k) { quantize_row_q3_K_blue_ref(x, (block_q3_K*)y, k); },
                      [](const void* x, float* y, int64_t k) { dequantize_row_q3_K((const block_q3_K*)x, y, k); },
                      QK_K, sizeof(block_q3_K)},
    };

    const char* dataset_names[] = {"Gaussian(0,10)+outliers", "All-positive (post-RoPE K)", "Bimodal (V-like)"};
    std::vector<float>* datasets[] = {&data, &data_pos, &data_bimodal};

    for (int di = 0; di < 3; di++) {
        printf("\n========================================\n");
        printf("Dataset: %s\n", dataset_names[di]);
        printf("========================================\n");
        printf("  %-20s  %-16s  %s\n", "Format", "MSE", "bits/scalar");
        printf("  %-20s  %-16s  %s\n", "------", "---", "-----------");

        std::vector<EntropyResult> results;
        for (const auto& t : tests) {
            int nb = N / t.block_size;
            std::vector<uint8_t> q(nb * t.type_size);
            std::vector<float> dq(N);
            t.quant_ref(datasets[di]->data(), q.data(), N);
            t.dequant(q.data(), dq.data(), N);

            double mse = compute_mse(datasets[di]->data(), dq.data(), N);
            double bits = bits_per_scalar(mse);
            printf("  %-20s  MSE=%12.6f  bits/scalar=%6.4f\n",
                   t.name.c_str(), mse, bits);
            results.push_back({t.name, mse, bits, 16.0/bits});
        }

        printf("\n  Blue-noise delta:\n");
        for (size_t i = 1; i < results.size(); i += 2) {
            auto& blue = results[i];
            auto& norm = results[i-1];
            double delta = blue.bits_scalar - norm.bits_scalar;
            double pct = 100.0 * delta / norm.bits_scalar;
            printf("  %s vs %s: Δbits = %+.4f (%+.2f%%)\n",
                   blue.name.c_str(), norm.name.c_str(), delta, pct);
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}