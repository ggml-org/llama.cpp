// Unit test: AVX-512 Q6_K 8x8 GEMV/GEMM vs the generic reference kernels on random data.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "repack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// same interleave as repack.cpp's make_block_q6_Kx8 (blck_size_interleave = 8)
static block_q6_Kx8 make_block_q6_Kx8_ref(const block_q6_K * in) {
    block_q6_Kx8 out;
    const unsigned bl = 8;
    for (int i = 0; i < 8; i++) out.d[i] = in[i].d;
    const int end_ls = QK_K * 4 / bl;
    for (int i = 0; i < end_ls; ++i) {
        int src_id = i % 8, src_offset = (i / 8) * bl, dst_offset = i * bl;
        memcpy(&out.ql[dst_offset], &in[src_id].ql[src_offset], bl);
    }
    const int end_hs = end_ls / 2;
    for (int i = 0; i < end_hs; ++i) {
        int src_id = i % 8, src_offset = (i / 8) * bl, dst_offset = i * bl;
        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], bl);
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < QK_K / 16; j++) out.scales[j * 8 + i] = in[i].scales[j];
    return out;
}

static double max_rel_err(const std::vector<float> & a, const std::vector<float> & b, double & max_abs, double & ref_mag) {
    double m = 0; max_abs = 0; ref_mag = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = fabs((double) a[i] - (double) b[i]);
        double r = fabs((double) b[i]);
        if (d > max_abs) max_abs = d;
        if (r > ref_mag) ref_mag = r;
    }
    m = ref_mag > 0 ? max_abs / ref_mag : max_abs;
    return m;
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    const int Ks[] = { 256, 512, 2048 };
    const int Ns[] = { 8, 16, 64, 512 };
    const int Ms[] = { 1, 4, 8, 16, 20 };
    for (int K : Ks) for (int N : Ns) {
        // random weights -> Q6_K rows -> interleaved
        std::vector<float> w((size_t) K * N);
        for (auto & v : w) v = nd(rng);
        const int nb = K / QK_K;
        std::vector<block_q6_K> q((size_t) N * nb);
        for (int r = 0; r < N; r++) quantize_row_q6_K_ref(w.data() + (size_t) r * K, q.data() + (size_t) r * nb, K);
        std::vector<block_q6_Kx8> qi((size_t) (N / 8) * nb);
        for (int x = 0; x < N / 8; x++) for (int b = 0; b < nb; b++) {
            block_q6_K tmp[8];
            for (int c = 0; c < 8; c++) tmp[c] = q[(size_t) (x * 8 + c) * nb + b];
            qi[(size_t) x * nb + b] = make_block_q6_Kx8_ref(tmp);
        }
        for (int M : Ms) {
            std::vector<float> a((size_t) M * K);
            for (auto & v : a) v = nd(rng) * 3.0f;
            std::vector<float> out_new((size_t) M * N, 0.f), out_ref((size_t) M * N, 0.f), out_f32((size_t) M * N, 0.f);
            // f32 reference from dequantized weights
            std::vector<float> wd((size_t) K * N);
            for (int r = 0; r < N; r++) dequantize_row_q6_K(q.data() + (size_t) r * nb, wd.data() + (size_t) r * K, K);
            for (int m = 0; m < M; m++) for (int c = 0; c < N; c++) {
                double s = 0; for (int k = 0; k < K; k++) s += (double) a[(size_t) m * K + k] * wd[(size_t) c * K + k];
                out_f32[(size_t) m * N + c] = (float) s;
            }
            if (M == 1) {
                std::vector<block_q8_K> aq(nb);
                quantize_row_q8_K_ref(a.data(), aq.data(), K);
                ggml_gemv_q6_K_8x8_q8_K(K, out_new.data(), N, qi.data(), aq.data(), 1, N);
                ggml_gemv_q6_K_8x8_q8_K_generic(K, out_ref.data(), N, qi.data(), aq.data(), 1, N);
            } else {
                const int M4 = M - M % 4;
                std::vector<block_q8_Kx4> aq4((size_t) (M4 / 4) * nb);
                for (int g = 0; g < M4 / 4; g++) ggml_quantize_mat_q8_K_4x8(a.data() + (size_t) g * 4 * K, aq4.data() + (size_t) g * nb, K);
                ggml_gemm_q6_K_8x8_q8_K(K, out_new.data(), N, qi.data(), aq4.data(), M4, N);
                ggml_gemm_q6_K_8x8_q8_K_generic(K, out_ref.data(), N, qi.data(), aq4.data(), M4, N);
                for (int m = M4; m < M; m++) {
                    std::vector<block_q8_K> aq(nb);
                    quantize_row_q8_K_ref(a.data() + (size_t) m * K, aq.data(), K);
                    ggml_gemv_q6_K_8x8_q8_K(K, out_new.data() + (size_t) m * N, N, qi.data(), aq.data(), 1, N);
                    ggml_gemv_q6_K_8x8_q8_K_generic(K, out_ref.data() + (size_t) m * N, N, qi.data(), aq.data(), 1, N);
                }
            }
            double mabs, mag;
            double rel_vs_generic = max_rel_err(out_new, out_ref, mabs, mag);
            double mabs2, mag2;
            double rel_vs_f32 = max_rel_err(out_new, out_f32, mabs2, mag2);
            const bool ok = rel_vs_generic < 1e-5 && rel_vs_f32 < 2e-2;
            printf("K=%5d N=%4d M=%3d  vs generic: max|d|=%.3e (rel %.2e)   vs f32: rel %.2e  %s\n", K, N, M, mabs, rel_vs_generic, rel_vs_f32, ok ? "ok" : "FAIL");
            if (!ok) fails++;
        }
    }
    printf("%s (%d failures)\n", fails ? "FAILED" : "ALL OK", fails);
    return fails ? 1 : 0;
}
