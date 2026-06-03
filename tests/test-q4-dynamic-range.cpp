//
// Test: Q4_0 vs Q4_1 on skewed distributions
// Q4_0 = symmetric, Q4_1 = asymmetric (min/max)
// If K values are skewed, Q4_1 should win.
//
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

void test_distribution(const char *label, const std::vector<float> &x) {
    int n = x.size();
    std::vector<block_q4_0> b40(n/32);
    std::vector<block_q4_1> b41(n/32);
    
    quantize_row_q4_0_ref(x.data(), b40.data(), n);
    quantize_row_q4_1_ref(x.data(), b41.data(), n);
    
    std::vector<float> d40(n), d41(n);
    dequantize_row_q4_0(b40.data(), d40.data(), n);
    dequantize_row_q4_1(b41.data(), d41.data(), n);
    
    auto stats = [&](const float *d, const char *name) {
        double max_e=0, mean_e=0, max_pct=0, sq_e=0, sq_o=0;
        for (int i = 0; i < n; i++) {
            float e = fabsf(x[i] - d[i]);
            max_e = fmax(max_e, e);
            mean_e += e;
            sq_e += e*e; sq_o += x[i]*x[i];
            max_pct = fmax(max_pct, e / (fabsf(x[i]) + 1e-10f));
        }
        mean_e /= n;
        printf("  %-6s: max_abs=%.4f  mean_abs=%.4f  rel_l2=%.4f  max_pct_err=%.1f%%\n",
               name, max_e, mean_e, sqrt(sq_e/(sq_o+1e-30)), max_pct*100);
    };
    
    printf("%s (n=%d):\n", label, n);
    stats(d40.data(), "Q4_0");
    stats(d41.data(), "Q4_1");
    
    // Also check: what fraction of values are in [first half, second half]?
    double sum_first=0, sum_second=0;
    for (int i = 0; i < n/2; i++) sum_first += fabsf(x[i]);
    for (int i = n/2; i < n; i++) sum_second += fabsf(x[i]);
    printf("  skew_ratio=%.2f (first_half_mag/second_half_mag)\n", sum_first/(sum_second+1e-30));
    printf("\n");
}

int main() {
    const int n = 1024;
    std::mt19937 rng(42);
    
    // Distribution 1: centered at 0 (gaussian)
    std::vector<float> gauss(n);
    std::normal_distribution<float> norm(0, 1);
    for (auto &v : gauss) v = norm(rng);
    test_distribution("Gaussian(0,1) symmetric", gauss);
    
    // Distribution 2: all positive (e.g., after ReLU / RoPE)
    std::vector<float> positive(n);
    std::normal_distribution<float> pos_dist(5, 2);
    for (auto &v : positive) v = fabsf(pos_dist(rng));
    test_distribution("All-positive (skewed)", positive);
    
    // Distribution 3: shifted (mean != 0)
    std::vector<float> shifted(n);
    for (auto &v : shifted) v = norm(rng) + 0.5f;
    test_distribution("Shifted gauss (+0.5)", shifted);
    
    // Distribution 4: heavy outliers (simulates attention K outliers)
    std::vector<float> outliers(n);
    for (int i = 0; i < n; i++) {
        outliers[i] = (i % 32 == 0) ? norm(rng) * 10.0f : norm(rng) * 0.5f;
    }
    test_distribution("Heavy outliers (1/32)", outliers);
    
    // Distribution 5: bimodal (positive cluster + negative cluster)
    std::vector<float> bimodal(n);
    for (int i = 0; i < n; i++) {
        bimodal[i] = (i < n/2) ? norm(rng) + 2.0f : norm(rng) - 2.0f;
    }
    test_distribution("Bimodal (+2 / -2)", bimodal);
    
    printf("Done.\n");
    return 0;
}
