// Round-trip test: quantize -> dequantize K/V and measure error
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

// Test 1: Packing
void test_packing() {
    printf("=== Test 1: Packing ===\n");
    const int qk = 32;
    float test_vals[qk];
    float known[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    for (int i = 0; i < qk; i++) test_vals[i] = known[i % 5];
    block_q4_0 block;
    quantize_row_q4_0_ref(test_vals, &block, qk);
    
    float max_val = 0.0f, amax = 0.0f;
    for (int j = 0; j < qk; j++) {
        if (fabsf(test_vals[j]) > amax) { amax = fabsf(test_vals[j]); max_val = test_vals[j]; }
    }
    float d = max_val / -8.0f;
    float id = d ? 1.0f/d : 0.0f;
    printf("  max=%f, d=%f, stored_d=%f\n", max_val, d, ggml_fp16_to_fp32(block.d));
    
    int errors = 0;
    for (int j = 0; j < qk/2; j++) {
        float x0 = test_vals[j] * id;
        float x1 = test_vals[qk/2 + j] * id;
        uint8_t e0 = (uint8_t)((int8_t)(x0 + 8.5f)); if (e0 > 15) e0 = 15;
        uint8_t e1 = (uint8_t)((int8_t)(x1 + 8.5f)); if (e1 > 15) e1 = 15;
        uint8_t a0 = block.qs[j] & 0x0F;
        uint8_t a1 = (block.qs[j] >> 4) & 0x0F;
        if (e0 != a0 || e1 != a1) { errors++; if(errors<=3) printf("  MISMATCH j=%d: (%d,%d) vs (%d,%d)\n",j,e0,e1,a0,a1); }
    }
    printf("  Packing: %s (%d errors)\n", errors==0?"PASS":"FAIL", errors);
    
    float deq[qk];
    dequantize_row_q4_0(&block, deq, qk);
    printf("  First 5 orig -> deq:\n");
    for (int j = 0; j < 5; j++) printf("    [%d] %.3f -> %.3f (err=%.4f)\n", j, test_vals[j], deq[j], test_vals[j]-deq[j]);
    printf("\n");
}

// Test 2: Round-trip error statistics
void test_roundtrip_error() {
    printf("=== Test 2: Round-trip error ===\n");
    const int n = 1024;
    std::vector<float> x(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    std::vector<block_q4_0> bq4(n/32), bblue(n/32);
    quantize_row_q4_0_ref(x.data(), bq4.data(), n);
    quantize_row_q4_0_blue_ref(x.data(), bblue.data(), n);

    std::vector<float> dq4(n), dblue(n);
    dequantize_row_q4_0(bq4.data(), dq4.data(), n);
    dequantize_row_q4_0(bblue.data(), dblue.data(), n);

    auto metrics = [&](const float *d, const char *label) {
        double max_abs=0, sum_abs=0, sq_err=0, sq_orig=0, dot=0, n_orig=0, n_deq=0;
        int nan_c=0, inf_c=0;
        for (int i = 0; i < n; i++) {
            float e = fabsf(x[i] - d[i]);
            if (std::isnan(x[i])||std::isnan(d[i])) nan_c++;
            if (std::isinf(x[i])||std::isinf(d[i])) inf_c++;
            max_abs = fmax(max_abs, e);
            sum_abs += e; sq_err += e*e;
            sq_orig += (double)x[i]*x[i];
            dot += (double)x[i]*d[i];
            n_orig += (double)x[i]*x[i];
            n_deq += (double)d[i]*d[i];
        }
        printf("  %s:\n", label);
        printf("    max_abs_err=%.6f  mean_abs_err=%.6f\n", max_abs, sum_abs/n);
        printf("    rel_l2=%.6f  cos_sim=%.8f\n", sqrt(sq_err/(sq_orig+1e-30)), dot/(sqrt(n_orig)*sqrt(n_deq)+1e-30));
        printf("    nan=%d  inf=%d\n", nan_c, inf_c);
    };
    metrics(dq4.data(), "Q4_0");
    metrics(dblue.data(), "Q4_0_BLUE");
    printf("\n");
}

// Test 3: Attention drift
void test_attention_drift() {
    printf("=== Test 3: Attention drift ===\n");
    const int n_ctx = 64, d_qk = 128;
    std::vector<float> K(n_ctx * d_qk);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : K) v = dist(rng);

    int nb = (n_ctx * d_qk) / 32;
    std::vector<block_q4_0> Kq4(nb), Kblue(nb);
    quantize_row_q4_0_ref(K.data(), Kq4.data(), n_ctx*d_qk);
    quantize_row_q4_0_blue_ref(K.data(), Kblue.data(), n_ctx*d_qk);

    std::vector<float> Kd_q4(n_ctx*d_qk), Kd_blue(n_ctx*d_qk);
    dequantize_row_q4_0(Kq4.data(), Kd_q4.data(), n_ctx*d_qk);
    dequantize_row_q4_0(Kblue.data(), Kd_blue.data(), n_ctx*d_qk);

    auto attn = [&](float *Kd, std::vector<float> &scores) {
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

    std::vector<float> so(n_ctx), sq(n_ctx), sb(n_ctx);
    float *Kp = K.data();
    attn(Kp, so); attn(Kd_q4.data(), sq); attn(Kd_blue.data(), sb);

    double kl_q4=0, kl_blue=0, cos_q4=0, cos_blue=0;
    int top5_orig[5]={0}, top5_q4[5]={0}, top5_blue[5]={0};
    for (int t = 0; t < n_ctx; t++) {
        if (so[t] > 0) {
            kl_q4 += so[t] * log(so[t]/(sq[t]+1e-30));
            kl_blue += so[t] * log(so[t]/(sb[t]+1e-30));
        }
        cos_q4 += so[t]*sq[t]; cos_blue += so[t]*sb[t];
        for (int k = 0; k < 5; k++) {
            if (so[t] > so[top5_orig[k]]) { for (int k2=4; k2>k; k2--) top5_orig[k2]=top5_orig[k2-1]; top5_orig[k]=t; break; }
        }
    }
    for (int t = 0; t < n_ctx; t++) {
        for (int k = 0; k < 5; k++) {
            if (sq[t] > sq[top5_q4[k]]) { for (int k2=4; k2>k; k2--) top5_q4[k2]=top5_q4[k2-1]; top5_q4[k]=t; break; }
        }
    }
    for (int t = 0; t < n_ctx; t++) {
        for (int k = 0; k < 5; k++) {
            if (sb[t] > sb[top5_blue[k]]) { for (int k2=4; k2>k; k2--) top5_blue[k2]=top5_blue[k2-1]; top5_blue[k]=t; break; }
        }
    }

    int ov_q4=0, ov_blue=0;
    for (int i=0;i<5;i++) for (int j=0;j<5;j++) {
        if (top5_orig[i]==top5_q4[j]) ov_q4++;
        if (top5_orig[i]==top5_blue[j]) ov_blue++;
    }
    printf("  KL(orig||q4)=%.8f  KL(orig||blue)=%.8f\n", kl_q4, kl_blue);
    printf("  top5_overlap_q4=%d/5  top5_overlap_blue=%d/5\n", ov_q4, ov_blue);
    printf("  attn_cos_q4=%.6f  attn_cos_blue=%.6f\n", cos_q4, cos_blue);
    printf("\n");
}

int main() {
    test_packing();
    test_roundtrip_error();
    test_attention_drift();
    printf("Tests complete.\n");
    return 0;
}
