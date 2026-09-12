
#include "ggml-cpu/repack.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
extern "C" void q4kp_original_gemv(int n, float *s, size_t bs, const void *x,
                              const void *y, int nr, int nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, x, y, nr, nc);
}
extern "C" void q4kp_scalar_gemv(int n, float *s, const void *x, const void *y, int nc) {
    const auto *packed = static_cast<const uint8_t *>(x);
    const auto *activation = static_cast<const uint8_t *>(y);
    const int nb = n / 256;
    for (int row = 0; row < nc; ++row) {
        float acc = 0.0f, acc_min = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t *w = packed + ((row / 8) * nb + b) * 1152;
            const uint8_t *a = activation + b * sizeof(block_q8_K);
            int32_t dot = 0, correction = 0;
            const int lane = row % 8;
            for (int g = 0; g < 8; ++g) {
                const uint8_t *codes = w + 32 + g * 12;
                const int sc = lane < 4 ? codes[lane] & 63 :
                    (codes[8 + lane - 4] & 15) | ((codes[lane - 4] >> 6) << 4);
                const int mn = lane < 4 ? codes[4 + lane] & 63 :
                    (codes[8 + lane - 4] >> 4) | ((codes[lane] >> 6) << 4);
                int32_t local = 0, sum = 0;
                for (int k = 0; k < 32; ++k) {
                    const int qs_col = (g / 2) * 32 + k;
                    const uint8_t qbyte = w[128 + (qs_col / 8) * 64 + lane * 8 + qs_col % 8];
                    const int quant = (qbyte >> ((g % 2) * 4)) & 15;
                    const int q8 = static_cast<int8_t>(a[4 + g * 32 + k]);
                    local += quant * q8;
                    sum += q8;
                }
                dot += local * sc;
                correction += sum * mn;
            }
            uint16_t half_d, half_min;
            float input_d;
            std::memcpy(&half_d, w + lane * 2, 2);
            std::memcpy(&half_min, w + 16 + lane * 2, 2);
            std::memcpy(&input_d, a, sizeof(input_d));
            const float scale = _cvtsh_ss(half_d) * input_d;
            const float min_scale = _cvtsh_ss(half_min) * input_d;
            acc = std::fma(static_cast<float>(dot), scale, acc);
            acc_min = std::fma(static_cast<float>(correction), min_scale, acc_min);
        }
        s[row] = acc - acc_min;
    }
}
extern "C" void q4kp_original_gemm(int n, float *s, size_t bs, const void *x,
                                   const void *y, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, x, y, nr, nc);
}
extern "C" void q4kp_scalar_gemm(int n, float *s, size_t bs, const void *x,
                                 const void *y, int nr, int nc) {
    const auto *packed = static_cast<const uint8_t *>(x);
    const auto *activation = static_cast<const uint8_t *>(y);
    const int nb = n / 256;
    for (int ar = 0; ar < nr; ++ar) {
        for (int row = 0; row < nc; ++row) {
            float acc = 0.0f, acc_min = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *w = packed + ((row / 8) * nb + b) * 1152;
                const uint8_t *a = activation + ((ar / 4) * nb + b) * sizeof(block_q8_Kx4);
                uint16_t half_d, half_min;
                float input_d;
                const int lane = row % 8;
                std::memcpy(&half_d, w + lane * 2, 2);
                std::memcpy(&half_min, w + 16 + lane * 2, 2);
                std::memcpy(&input_d, a + (ar % 4) * 4, sizeof(input_d));
                const float scale = _cvtsh_ss(half_d) * input_d;
                const float min_scale = _cvtsh_ss(half_min) * input_d;
                // AVX2 GEMM accumulates one 64-column pair per FMA, unlike GEMV.
                for (int pair = 0; pair < 4; ++pair) {
                    int32_t dot = 0, correction = 0;
                    for (int g = 2 * pair; g < 2 * pair + 2; ++g) {
                        const uint8_t *codes = w + 32 + g * 12;
                        const int sc = lane < 4 ? codes[lane] & 63 :
                            (codes[8 + lane - 4] & 15) | ((codes[lane - 4] >> 6) << 4);
                        const int mn = lane < 4 ? codes[4 + lane] & 63 :
                            (codes[8 + lane - 4] >> 4) | ((codes[lane] >> 6) << 4);
                        int32_t local = 0, sum = 0;
                        for (int k = 0; k < 32; ++k) {
                            const int qs_col = (g / 2) * 32 + k;
                            const uint8_t qbyte = w[128 + (qs_col / 8) * 64 + lane * 8 + qs_col % 8];
                            const int quant = (qbyte >> ((g % 2) * 4)) & 15;
                            const int col = g * 32 + k;
                            const int q8 = static_cast<int8_t>(a[16 + (col / 8) * 32 + (ar % 4) * 8 + col % 8]);
                            local += quant * q8;
                            sum += q8;
                        }
                        dot += local * sc;
                        correction += sum * mn;
                    }
                    acc = std::fma(static_cast<float>(dot), scale, acc);
                    acc_min = std::fma(static_cast<float>(correction), min_scale, acc_min);
                }
            }
            s[ar * bs + row] = acc - acc_min;
        }
    }
}


extern "C" int q4kp_test_cpu_supported() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}
