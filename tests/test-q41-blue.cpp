// Quick test: Q4_1 vs Q4_1_BLUE on skewed distributions (simulating K values)
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

int main() {
    const int n = 1024;
    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0, 1);
    
    // Simulate K values after RoPE: all-positive with some outliers
    std::vector<float> K(n);
    for (int i = 0; i < n; i++) {
        K[i] = fabsf(norm(rng)) * (1.0f + ((i % 32 == 0) ? 3.0f : 0.0f));
    }
    
    std::vector<block_q4_1> b41(n/32), b41b(n/32);
    quantize_row_q4_1_ref(K.data(), b41.data(), n);
    quantize_row_q4_1_blue_ref(K.data(), b41b.data(), n);
    
    std::vector<float> d41(n), d41b(n);
    dequantize_row_q4_1(b41.data(), d41.data(), n);
    dequantize_row_q4_1(b41b.data(), d41b.data(), n);
    
    // Error metrics
    auto stats = [&](const float *d, const char *name) {
        double max_e=0, mean_e=0, sq_e=0, sq_o=0;
        for (int i = 0; i < n; i++) {
            float e = fabsf(K[i] - d[i]);
            max_e = fmax(max_e, e);
            mean_e += e; sq_e += e*e; sq_o += K[i]*K[i];
        }
        mean_e /= n;
        double cos_sim = 0, nK=0, nd=0;
        for (int i = 0; i < n; i++) { cos_sim += K[i]*d[i]; nK += K[i]*K[i]; nd += d[i]*d[i]; }
        cos_sim /= sqrt(nK)*sqrt(nd);
        printf("  %s: max_err=%.4f mean_err=%.4f rel_l2=%.4f cos=%.8f\n",
               name, max_e, mean_e, sqrt(sq_e/(sq_o+1e-30)), cos_sim);
    };
    stats(d41.data(), "Q4_1");
    stats(d41b.data(), "Q4_1_BLUE");
    
    // Attention drift: use first K as query
    auto attn = [&](float *Kd, std::vector<float> &scores) {
        const int n_ctx = 32, d_qk = 32;
        for (int t = 0; t < n_ctx; t++) {
            scores[t] = 0;
            for (int d = 0; d < d_qk; d++)
                scores[t] += K[d] * Kd[t*d_qk + d];
        }
        float mx = -INFINITY;
        for (auto v : scores) if (v > mx) mx = v;
        double sum = 0;
        for (auto &v : scores) { v = expf(v - mx); sum += v; }
        for (auto &v : scores) v /= sum;
    };
    
    const int n_ctx = 32, d_qk = 32;
    std::vector<float> so(n_ctx), s41(n_ctx), s41b(n_ctx);
    attn(K.data(), so);
    attn(d41.data(), s41);
    attn(d41b.data(), s41b);
    
    double kl41=0, kl41b=0;
    for (int t = 0; t < n_ctx; t++) {
        if (so[t] > 0) {
            kl41 += so[t] * log(so[t]/(s41[t]+1e-30));
            kl41b += so[t] * log(so[t]/(s41b[t]+1e-30));
        }
    }
    printf("\nAttention drift (KL divergence from original):\n");
    printf("  KL(orig||Q4_1)       = %.8f\n", kl41);
    printf("  KL(orig||Q4_1_BLUE)  = %.8f\n", kl41b);
    printf("  Blue improves: %s\n", kl41b < kl41 ? "YES" : "NO (or equal)");
    
    printf("\nDone.\n");
    return 0;
}
