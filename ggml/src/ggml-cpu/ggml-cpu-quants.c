#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-cpu-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    quantize_row_q8_0_native(x, y, k);
#else
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    quantize_row_q8_1_native(x, y, k);
#else
    quantize_row_q8_1_ref(x, y, k);
#endif
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

//===================================== Q8_K ==============================================

void quantize_row_q8_K_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
#if defined(__wasm__)
    quantize_row_q8_K_native(x, y, k);
#else
    quantize_row_q8_K_ref(x, y, k);
#endif
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q4_0_q8_0_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q4_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    // TODO: add WASM SIMD
#if defined(__aarch64__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q4_1_q8_1_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q5_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__)
    ggml_vec_dot_q5_0_q8_0_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q5_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__)
    ggml_vec_dot_q5_1_q8_1_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q8_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q8_0_q8_0_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_tq1_0_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__x86_64__)
    ggml_vec_dot_tq1_0_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_tq2_0_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__x86_64__)
    ggml_vec_dot_tq2_0_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q2_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__)
    ggml_vec_dot_q2_K_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

void ggml_vec_dot_q3_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q3_K_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}

void ggml_vec_dot_q4_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q4_K_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
#ifdef __ARM_FEATURE_MATMUL_INT8
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_FEATURE_SVE
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, K_SCALE_SIZE);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const int vector_length = ggml_cpu_get_sve_cnt()*8;
        const svuint8_t m4b = svdup_n_u8(0xf);
        const svint32_t mzero = svdup_n_s32(0);
        svint32_t sumi1 = svdup_n_s32(0);
        svint32_t sumi1_1 = svdup_n_s32(0);
        svint32_t sumi1_2 = svdup_n_s32(0);
        svint32_t sumi2 = svdup_n_s32(0);
        svint32_t sumi2_1 = svdup_n_s32(0);
        svint32_t sumi2_2 = svdup_n_s32(0);
        switch (vector_length) {
            case 128:
                {
                    for (int j = 0; j < QK_K/64; ++j) {
                        svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), m4b));
                        svint8_t q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi1_1 = svmla_n_s32_x(svptrue_b32(), sumi1_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);
                        q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4+16), m4b));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi1_2 = svmla_n_s32_x(svptrue_b32(), sumi1_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);

                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), 4));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi2_1 = svmla_n_s32_x(svptrue_b32(), sumi2_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4+16), 4));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi2_2 = svmla_n_s32_x(svptrue_b32(), sumi2_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                        q4 += 32;
                    }
                    sumi1 = svadd_s32_x(svptrue_b32(), sumi1_1, sumi1_2);
                    sumi2 = svadd_s32_x(svptrue_b32(), sumi2_1, sumi2_2);
                    sumf += d * (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), sumi1, sumi2)));
                } break;
            case 256:
            case 512:
                {
                    for (int j = 0; j < QK_K/64; ++j) {
                        const svuint8_t q4bits  = svld1_u8(svptrue_pat_b8(SV_VL32), q4); q4 += 32;
                        svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_pat_b8(SV_VL32), q4bits, m4b));
                        svint8_t q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8); q8 += 32;
                        sumi1 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);

                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q4bits, 4));
                        q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8); q8 += 32;
                        sumi2 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                    }
                    sumf += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), sumi1, sumi2)));
                } break;
            default:
                assert(false && "Unsupported vector length");
                break;
        }
    }
    *s = sumf;
#elif defined __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;

#elif defined __wasm_simd128__
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin); // Corrected sign

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process scales and mins
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Sum mins * q8sums
        int32_t sumi = 0;
        const int16_t * GGML_RESTRICT q8sums = y[i].bsums;
        const uint8_t * m = (const uint8_t *)&utmp[2];
        for (int j = 0; j < 16; j += 2) {
            sumi += (q8sums[j] + q8sums[j+1]) * m[j/2];
        }
        sumf -= dmin * sumi;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            // Load 64 4-bit weights (32 bytes)
            const v128_t q4x0 = wasm_v128_load(q4);
            const v128_t q4x1 = wasm_v128_load(q4 + 16);
            q4 += 32;

            // Split into low/high nibbles
            const v128_t q4l0 = wasm_v128_and(q4x0, wasm_i8x16_splat(0x0F));
            const v128_t q4h0 = wasm_u8x16_shr(q4x0, 4);
            const v128_t q4l1 = wasm_v128_and(q4x1, wasm_i8x16_splat(0x0F));
            const v128_t q4h1 = wasm_u8x16_shr(q4x1, 4);

            // Load 64 8-bit values (64 bytes)
            const v128_t q8x0 = wasm_v128_load(q8);
            const v128_t q8x1 = wasm_v128_load(q8 + 16);
            const v128_t q8x2 = wasm_v128_load(q8 + 32);
            const v128_t q8x3 = wasm_v128_load(q8 + 48);
            q8 += 64;

            // Low nibble products
            v128_t vacc1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(q4l0),
                wasm_i16x8_extend_low_i8x16(q8x0)
            );
            vacc1 = wasm_i32x4_add(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(q4l0),
                wasm_i16x8_extend_high_i8x16(q8x0)
            ));
            vacc1 = wasm_i32x4_add(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(q4l1),
                wasm_i16x8_extend_low_i8x16(q8x1)
            ));
            vacc1 = wasm_i32x4_add(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(q4l1),
                wasm_i16x8_extend_high_i8x16(q8x1)
            ));

            // High nibble products
            v128_t vacc2 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(q4h0),
                wasm_i16x8_extend_low_i8x16(q8x2)
            );
            vacc2 = wasm_i32x4_add(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(q4h0),
                wasm_i16x8_extend_high_i8x16(q8x2)
            ));
            vacc2 = wasm_i32x4_add(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_low_i8x16(q4h1),
                wasm_i16x8_extend_low_i8x16(q8x3)
            ));
            vacc2 = wasm_i32x4_add(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_high_i8x16(q4h1),
                wasm_i16x8_extend_high_i8x16(q8x3)
            ));

            // Accumulate scaled results
            int32_t vacc1_sum = wasm_i32x4_extract_lane(vacc1, 0) + wasm_i32x4_extract_lane(vacc1, 1) +
                                wasm_i32x4_extract_lane(vacc1, 2) + wasm_i32x4_extract_lane(vacc1, 3);
            sumi1 += vacc1_sum * scales[2*j];

            int32_t vacc2_sum = wasm_i32x4_extract_lane(vacc2, 0) + wasm_i32x4_extract_lane(vacc2, 1) +
                                wasm_i32x4_extract_lane(vacc2, 2) + wasm_i32x4_extract_lane(vacc2, 3);
            sumi2 += vacc2_sum * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __riscv_xtheadvector

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        int tmp, tmp2, sumi;
        __asm__ __volatile__(
            "li %[t1], 12\n\t"
            "th.vsetvli zero, %[t1], e8, m1\n\t"
            "th.vlb.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
            "li %[t1], 4\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vslidedown.vi v2, v1, 2\n\t"
            "th.vmv.v.v v3, v2\n\t"
            "th.vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
            "li %[t1], 2\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vmv.v.i v4, 4\n\t"
            "th.vand.vx v8, v1, %[kmask1]\n\t"
            "th.vslide1up.vx v5, v4, zero\n\t" // {0, 4}
            "th.vsrl.vi v6, v1, 6\n\t"
            "th.vsrl.vv v7, v2, v5\n\t"
            "th.vand.vx v0, v6, %[kmask3]\n\t"
            "th.vand.vx v2, v7, %[kmask2]\n\t"
            "th.vsll.vi v6, v0, 4\n\t"
            "li %[t2], 8\n\t"
            "addi %[t1], %[utmp], 4\n\t"
            "th.vor.vv v1, v6, v2\n\t"
            "th.vssw.v v8, (%[utmp]), %[t2]\n\t"
            "th.vssw.v v1, (%[t1]), %[t2]\n\t"
            "th.vsetvli zero, zero, e32, m2\n\t" // vl == 8
            "th.vlw.v v2, (%[bsums])\n\t"
            "th.vsetvli zero, %[t2], e16, m1\n\t"
            "th.vnsrl.vi v0, v2, 0\n\t"
            "th.vnsrl.vi v1, v2, 16\n\t"
            "th.vadd.vv v2, v0, v1\n\t"
            "th.vlbu.v v4, (%[mins])\n\t"
            "th.vwmul.vv v6, v4, v2\n\t"
            "th.vmv.v.x v0, zero\n\t"
            "th.vsetvli zero, %[t2], e32, m2\n\t"
            "th.vredsum.vs v0, v6, v0\n\t"
            "th.vmv.x.s %[sumi], v0"
            : [t1] "=&r" (tmp), [t2] "=&r" (tmp2), [sumi] "=&r" (sumi)
            : [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
            , [s6b] "r" (x[i].scales), [kmask1] "r" (kmask1)
            , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf -= dmin * sumi;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        sumi = 0;
        const uint8_t * scale = scales;

        for (int j = 0; j < QK_K/128; ++j) {
            int vl128 = 128, vl64 = 64, vl32 = 32;
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vlb.v v0, (%[q4])\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vand.vi v0, v0, 0xF\n\t"
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vwmul.vv v28, v6, v14\n\t"
                "th.vwmul.vv v20, v4, v10\n\t"
                "th.vwmul.vv v24, v2, v12\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vlbu.v v1, (%[scale])\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vsetvli zero, %[vl32], e16, m4\n\t"
                "th.vwredsum.vs v6, v24, v0\n\t"
                "th.vwredsum.vs v7, v28, v0\n\t"
                "th.vwredsum.vs v4, v16, v0\n\t"
                "th.vwredsum.vs v5, v20, v0\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v6, v7, 1\n\t"
                "th.vslideup.vi v4, v5, 1\n\t"
                "th.vslideup.vi v4, v6, 2\n\t"
                "th.vmul.vv v8, v4, v1\n\t"
                "th.vredsum.vs v0, v8, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[sumi], %[sumi], %[tmp]"
                : [tmp] "=&r" (tmp), [sumi] "+&r" (sumi)
                : [vl128] "r" (vl128), [vl64] "r" (vl64), [vl32] "r" (vl32)
                , [q4] "r" (q4), [q8] "r" (q8), [scale] "r" (scale)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );

            q4 += 64;    q8 += 128;    scale += 4;
        }

        sumf += d * sumi;

    }

    *s = sumf;

#elif defined __riscv_v

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            size_t vl = 8;

            const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

            vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
            vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
            vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
            vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
            vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

            vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
            sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

            const uint8_t * GGML_RESTRICT q4 = x[i].qs;
            const int8_t  * GGML_RESTRICT q8 = y[i].qs;

            vl = 32;

            int32_t sum_1 = 0;
            int32_t sum_2 = 0;

            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

            for (int j = 0; j < QK_K/64; ++j) {
                // load Q4
                vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

                // load Q8 and multiply it with lower Q4 nibble
                vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
                vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
                vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

                sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

                // load Q8 and multiply it with upper Q4 nibble
                vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
                vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
                vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
                vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

                sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

                q4 += 32;    q8 += 64;

            }

            sumf += d*(sum_1 + sum_2);

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

            int tmp, tmp2, sumi;
            __asm__ __volatile__(
                "vsetivli zero, 12, e8, m1\n\t"
                "vle8.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
                "vsetivli zero, 4, e32, m1\n\t"
                "vslidedown.vi v2, v1, 2\n\t"
                "vmv1r.v v3, v2\n\t"
                "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                "vsetivli zero, 2, e32, m1\n\t"
                "vmv.v.i v4, 4\n\t"
                "vand.vx v8, v1, %[kmask1]\n\t"
                "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                "vsrl.vi v6, v1, 6\n\t"
                "vsrl.vv v7, v2, v5\n\t"
                "vand.vx v0, v6, %[kmask3]\n\t"
                "vand.vx v2, v7, %[kmask2]\n\t"
                "vsll.vi v6, v0, 4\n\t"
                "li %[t2], 8\n\t"
                "addi %[t1], %[utmp], 4\n\t"
                "vor.vv v1, v6, v2\n\t"
                "vsse32.v v8, (%[utmp]), %[t2]\n\t"
                "vsse32.v v1, (%[t1]), %[t2]\n\t"
                "vsetivli zero, 8, e16, m1\n\t"
                "vle32.v v2, (%[bsums])\n\t"
                "vnsrl.wi v0, v2, 0\n\t"
                "vnsrl.wi v1, v2, 16\n\t"
                "vadd.vv v2, v0, v1\n\t"
                "vle8.v v3, (%[mins])\n\t"
                "vzext.vf2 v4, v3\n\t"
                "vwmul.vv v6, v4, v2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vsetivli zero, 8, e32, m2\n\t"
                "vredsum.vs v0, v6, v0\n\t"
                "vmv.x.s %[sumi], v0"
                : [t1] "=&r" (tmp), [t2] "=&r" (tmp2), [sumi] "=&r" (sumi)
                : [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
                , [s6b] "r" (x[i].scales), [kmask1] "r" (kmask1)
                , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            sumf -= dmin * sumi;

            const uint8_t * restrict q4 = x[i].qs;
            const int8_t  * restrict q8 = y[i].qs;

            sumi = 0;
            const uint8_t * scale = scales;

            for (int j = 0; j < QK_K/128; ++j) {
                int vl128 = 128, vl64 = 64, vl32 = 32;
                __asm__ __volatile__(
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v8, (%[q8])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vle8.v v0, (%[q4])\n\t"
                    "vsrl.vi v4, v0, 4\n\t"
                    "vand.vi v0, v0, 0xF\n\t"
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vwmul.vv v28, v6, v14\n\t"
                    "vwmul.vv v20, v4, v10\n\t"
                    "vwmul.vv v24, v2, v12\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vle8.v v2, (%[scale])\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vzext.vf4 v1, v2\n\t"
                    "vsetvli zero, %[vl32], e16, m4\n\t"
                    "vwredsum.vs v6, v24, v0\n\t"
                    "vwredsum.vs v7, v28, v0\n\t"
                    "vwredsum.vs v4, v16, v0\n\t"
                    "vwredsum.vs v5, v20, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v6, v7, 1\n\t"
                    "vslideup.vi v4, v5, 1\n\t"
                    "vslideup.vi v4, v6, 2\n\t"
                    "vmul.vv v8, v4, v1\n\t"
                    "vredsum.vs v0, v8, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[sumi], %[sumi], %[tmp]"
                    : [tmp] "=&r" (tmp), [sumi] "+&r" (sumi)
                    : [vl128] "r" (vl128), [vl64] "r" (vl64), [vl32] "r" (vl32)
                    , [q4] "r" (q4), [q8] "r" (q8), [scale] "r" (scale)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );

                q4 += 64;    q8 += 128;    scale += 4;
            }

            sumf += d * sumi;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed char lowMask1 = vec_splats((int8_t)0x3f);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((uint8_t)2);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[i].dmin));
        vector float vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
        vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
        vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
        vector signed char u3 = vec_sr(u2, v4);

        vector signed char u30 = u1;
        vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);

        u1 = vec_and(u0, lowMask1);
        u2 = vec_or(u30, u31);

        vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);

        vector signed short vscales = vec_unpackh(utmps);
        vector signed short q4xmins = vec_unpackl(utmps);
        vector signed short q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        vector signed short q4xmins1 = vec_mergel(q4xmins, q4xmins);

        vector signed int prod0 = vec_mule(q4xmins0, q8ysums0);
        vector signed int prod1 = vec_mule(q4xmins1, q8ysums1);
        vector signed int prod2 = vec_mulo(q4xmins0, q8ysums0);
        vector signed int prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        for (int j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q4);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q4);
            vector signed char qxs2 = (vector signed char)vec_xl(32, q4);
            vector signed char qxs3 = (vector signed char)vec_xl(48, q4);
            q4 += 64;

            vector unsigned char q4x00 = (vector unsigned char)vec_and(qxs0, lowMask);
            vector unsigned char q4x01 = (vector unsigned char)vec_sr(qxs0, v4);
            vector unsigned char q4x10 = (vector unsigned char)vec_and(qxs1, lowMask);
            vector unsigned char q4x11 = (vector unsigned char)vec_sr(qxs1, v4);
            vector unsigned char q4x20 = (vector unsigned char)vec_and(qxs2, lowMask);
            vector unsigned char q4x21 = (vector unsigned char)vec_sr(qxs2, v4);
            vector unsigned char q4x30 = (vector unsigned char)vec_and(qxs3, lowMask);
            vector unsigned char q4x31 = (vector unsigned char)vec_sr(qxs3, v4);

            vector signed char q8y00 = vec_xl(  0, q8);
            vector signed char q8y10 = vec_xl( 16, q8);
            vector signed char q8y01 = vec_xl( 32, q8);
            vector signed char q8y11 = vec_xl( 48, q8);
            vector signed char q8y20 = vec_xl( 64, q8);
            vector signed char q8y30 = vec_xl( 80, q8);
            vector signed char q8y21 = vec_xl( 96, q8);
            vector signed char q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector signed int qv00 = vec_msum(q8y00, q4x00, v0);
            vector signed int qv01 = vec_msum(q8y01, q4x01, v0);
            vector signed int qv10 = vec_msum(q8y10, q4x10, v0);
            vector signed int qv11 = vec_msum(q8y11, q4x11, v0);
            vector signed int qv20 = vec_msum(q8y20, q4x20, v0);
            vector signed int qv21 = vec_msum(q8y21, q4x21, v0);
            vector signed int qv30 = vec_msum(q8y30, q4x30, v0);
            vector signed int qv31 = vec_msum(q8y31, q4x31, v0);

            vector signed int vscales_h = vec_unpackh(vscales);
            vector signed int vs0 = vec_splat(vscales_h, 0);
            vector signed int vs1 = vec_splat(vscales_h, 1);
            vector signed int vs2 = vec_splat(vscales_h, 2);
            vector signed int vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_add(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

    __m256 acc = (__m256)__lasx_xvldi(0);
    __m128 acc_m = (__m128)__lsx_vldi(0);

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales128 = lsx_set_w(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i mins128 = __lsx_vexth_h_b(mins_and_scales128);
        const __m128i scales128 = __lsx_vsllwil_h_b(mins_and_scales128, 0);

        const __m256i q8sums = __lasx_xvld((const __m256i*)y[i].bsums, 0);
        const __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        const __m128i prod = lsx_madd_h(mins128, q8s);
        acc_m = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(dmin), __lsx_vffint_s_w(prod), acc_m);

        const __m256i scales = lasx_insertf128(scales128, scales128);

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = lasx_xvrepl128vei_h(scales, 2 * j + 0);
            const __m256i scale_h = lasx_xvrepl128vei_h(scales, 2 * j + 1);

            const __m256i q4bits = __lasx_xvld((const __m256i*)q4, 0); q4 += 32;
            const __m256i q4l = __lasx_xvandi_b(q4bits, 0xf);
            const __m256i q4h = __lasx_xvsrli_b(q4bits, 4);

            const __m256i q8l = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16l = lasx_madd_h_b(q4l, q8l);
            p16l = lasx_madd_h(scale_l, p16l);

            const __m256i q8h = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16h = lasx_madd_h_b(q4h, q8h);
            p16h = lasx_madd_h(scale_h, p16h);
            const __m256i sumj = __lasx_xvadd_w(p16l, p16h);

            sumi = __lasx_xvadd_w(sumi, sumj);
        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vpermi_w((__m128i)acc_m, (__m128i)acc_m, 0xee));
    __m128i tmp1 = __lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w((__m128i)acc_m, 1), 0);
    acc_m = __lsx_vfadd_s(acc_m, (__m128)tmp1);


    *s = hsum_float_8(acc) + ((v4f32)acc_m)[0];
#elif defined(__VXE__) || defined(__VXE2__)
    const uint8x16_t v_lm = vec_splat_u8(0x0F);
    const int32x4_t v_z = vec_splat_s32(0);

    uint8x16_t v_x[2];
    int8x16_t  v_xl[2];
    int8x16_t  v_y[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        const int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        memcpy(utmp, x[i].scales, 12);

        uint32x4_t v_mins8 = { 0 };
        v_mins8 = vec_insert(utmp[1] & kmask1, v_mins8, 0);
        v_mins8 = vec_insert(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), v_mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t v_minsh = (int16x8_t)vec_unpackh((uint8x16_t)v_mins8);

        const int32x4_t v_minso = vec_mulo(v_ysums, v_minsh);
        const int32x4_t v_minse = vec_mule(v_ysums, v_minsh);
        const int32x4_t v_mins = v_minso + v_minse;
        sumf -= dmin * (v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3]);

        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * GGML_RESTRICT x0 = x[i].qs;
        const int8_t  * GGML_RESTRICT y0 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            v_x[0] = vec_xl(0 , x0);
            v_x[1] = vec_xl(16, x0);
            x0 += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_and(v_x[0], v_lm);
            v_xl[1] = (int8x16_t)vec_and(v_x[1], v_lm);

            const int32x4_t p1 = ggml_vec_dot(ggml_vec_dot(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi1 += (p1[0] + p1[1] + p1[2] + p1[3]) * scales[2*j+0];

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_sr(v_x[0], 4);
            v_xl[1] = (int8x16_t)vec_sr(v_x[1], 4);

            const int32x4_t p2 = ggml_vec_dot(ggml_vec_dot(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi2 += (p2[0] + p2[1] + p2[2] + p2[3]) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;
#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q5_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q5_K_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q6_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__wasm__) \
 || defined(__x86_64__) \
 || defined(__riscv) \
 || defined(__powerpc__) \
 || defined(__loongarch__) \
 || defined(__s390__)
    ggml_vec_dot_q6_K_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_iq2_xxs_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__x86_64__) \
 || defined(__powerpc__) \
 || defined(__loongarch__)
    ggml_vec_dot_iq2_xxs_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}

void ggml_vec_dot_iq2_xs_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__aarch64__) \
 || defined(__x86_64__) \
 || defined(__powerpc__) \
 || defined(__loongarch__)
    ggml_vec_dot_iq2_xs_q8_K_native(n, s, bs, vx, bx, vy, by, nrc);
#else
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);
    const uint8x16_t m1 = vdupq_n_u8(1);
    const int32x4_t vzero = vdupq_n_s32(0);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2s.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[0] | ((qh[ib32+0] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[1] | ((qh[ib32+0] << 6) & 0x300)))));
            q2s.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[2] | ((qh[ib32+0] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[3] | ((qh[ib32+0] << 2) & 0x300)))));
            q2s.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[4] | ((qh[ib32+1] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[5] | ((qh[ib32+1] << 6) & 0x300)))));
            q2s.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[6] | ((qh[ib32+1] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[7] | ((qh[ib32+1] << 2) & 0x300)))));
            qs += 8;

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            q2s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[0]);
            q2s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[1]);

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            signs += 4;

            q2s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[2]);
            q2s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[3]);

            const int32x4_t p1 = ggml_vdotq_s32(vzero, q2s.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vzero, q2s.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vzero, q2s.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vzero, q2s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * (1 + 2*(x[i].scales[ib32+0] & 0xf));
            sumi2 += vaddvq_s32(p2) * (1 + 2*(x[i].scales[ib32+0] >>  4));
            sumi1 += vaddvq_s32(p3) * (1 + 2*(x[i].scales[ib32+1] & 0xf));
            sumi2 += vaddvq_s32(p4) * (1 + 2*(x[i].scales[ib32+1] >>  4));
        }
        sumf += d*(sumi1 + sumi2);
    }

    *s = 0.125f * sumf;

#elif defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m128i scales16_0 = _mm_cvtepi8_epi16(scales8);
        const __m128i scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales8, 8));

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                  iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                  iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                  iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                  iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)]);
            qs += 8;

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 0)));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 1)));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 0)));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 1)));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector unsigned char mask0 = vec_xl( 0, k_mask1);
    const vector unsigned char mask1 = vec_xl(16, k_mask1);
    const vector signed char mask2 = (vector signed char)vec_xl( 0, k_mask2);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        const uint8_t *  GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const uint8_t *  GGML_RESTRICT sc = x[i].scales;
        const int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            vector signed char vsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&signs[0]);
            vector signed char vsigns23 = (vector signed char)vec_splats(*(const uint32_t *)&signs[2]);
            signs += 4;

            vector signed char vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed char vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed char vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            vector signed char vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (vector signed char)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed char)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed char)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed char)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed char q2x0 = vec_sub(vec_xor(vsigns0, (vector signed char)aux64x2_0), vsigns0);
            vector signed char q2x1 = vec_sub(vec_xor(vsigns1, (vector signed char)aux64x2_1), vsigns1);
            vector signed char q2x2 = vec_sub(vec_xor(vsigns2, (vector signed char)aux64x2_2), vsigns2);
            vector signed char q2x3 = vec_sub(vec_xor(vsigns3, (vector signed char)aux64x2_3), vsigns3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            const uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            const uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            const uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            const uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector signed short vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector signed short vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector signed short vscales3 = vec_splats((int16_t)(2*ls3+1));

            vsumi0 = vec_msum(qv0, vscales0, vsumi0);
            vsumi1 = vec_msum(qv1, vscales1, vsumi1);
            vsumi2 = vec_msum(qv2, vscales2, vsumi2);
            vsumi3 = vec_msum(qv3, vscales3, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };


    const __m128i m4 = __lsx_vreplgr2vr_b(0xf);
    const __m128i m1 = __lsx_vreplgr2vr_b(1);

    const __m256i mask1 = __lasx_xvld((const __m256i*)k_mask1, 0);
    const __m256i mask2 = __lasx_xvld((const __m256i*)k_mask2, 0);
    uint64_t aux64;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        __m128i tmp1;
        memcpy(&aux64, x[i].scales, 8);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64, 0);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64 >> 4, 1);
        const __m128i scales8 = __lsx_vadd_b(__lsx_vslli_h(__lsx_vand_v(tmp1, m4), 1), m1);
        const __m256i scales16 = lasx_ext8_16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q2_1 = lasx_set_d(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = lasx_set_d(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = lasx_madd_h(dot1, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = lasx_madd_h(dot2, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#else

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;

#endif

}

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, gas, 2*sizeof(uint32_t)); gas += 2*sizeof(uint32_t);
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]);
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]);
            q3 += 16;
            q3s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >>  7) & 127))));
            q3s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >> 21) & 127))));
            q3s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q3s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q3s.val[0] = vmulq_s8(q3s.val[0], vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(q3s.val[1], vreinterpretq_s8_u32(aux32x4_1));
            q3s.val[2] = vmulq_s8(q3s.val[2], vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(q3s.val[3], vreinterpretq_s8_u32(aux32x4_3));
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.5f * sumf;

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            const __m128i q2_2_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint32_t * GGML_RESTRICT signs = (const uint32_t *)(x[i].qs + QK_K/4);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

#pragma GCC unroll 1
        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned int aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            vector unsigned int aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            vector unsigned int aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            vector unsigned int aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            vector unsigned long long aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            vector unsigned long long aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            vector unsigned long long aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            vector unsigned long long aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            vector signed char q3x0 = vec_mul((vector signed char)aux64x2_0, (vector signed char)aux32x4_0);
            vector signed char q3x1 = vec_mul((vector signed char)aux64x2_1, (vector signed char)aux32x4_1);
            vector signed char q3x2 = vec_mul((vector signed char)aux64x2_2, (vector signed char)aux32x4_2);
            vector signed char q3x3 = vec_mul((vector signed char)aux64x2_3, (vector signed char)aux32x4_3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            const uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            const uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q2_1 = lasx_set_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = lasx_set_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;

            const __m256i s2_1 = lasx_set_d(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = lasx_set_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            const __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;

            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);

#else

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
#endif
}

void ggml_vec_dot_iq3_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    typedef union {
        uint16x8_t vec_index;
        uint16_t   index[8];
    } vec_index_t;

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);

    const int16x8_t  hshift = vld1q_s16(k_shift);
    const uint16x8_t m256   = vdupq_n_u16(256);
    const uint8x16_t m1     = vdupq_n_u8(1);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;
    vec_index_t idx;

    uint32_t scales32[2];
    const uint8_t * scales8 = (const uint8_t *)scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(scales32, x[i].scales, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const uint8x16_t idx_l = vld1q_u8(qs); qs += 16;
            idx.vec_index = vorrq_u16(vmovl_u8(vget_low_u8 (idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+0]), hshift), m256));
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);
            idx.vec_index = vorrq_u16(vmovl_u8(vget_high_u8(idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+1]), hshift), m256));
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);


            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            q3s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_1));

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            signs += 4;

            q3s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_3));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * scales8[ib32/2+0];
            sumi2 += vaddvq_s32(p2) * scales8[ib32/2+4];
        }
        sumf += d*(sumi1 + sumi2);
    }
    *s = sumf;

#elif defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    const __m128i idx_mul_0 = _mm_set_epi32(32, 64, 128, 256);
    const __m128i idx_mul_1 = _mm_set_epi32(2, 4, 8, 16);
    const __m128i idx_mask  = _mm_set1_epi32(256);

    typedef union {
        __m128i  vec[4];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i qs_tmp = _mm_loadu_si128((const __m128i *)qs);
            const __m128i idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
            const __m128i idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8)); qs += 16;
            idx.vec[0] = _mm_set1_epi32(qh[ib32+0]);
            idx.vec[1] = idx.vec[0];
            idx.vec[2] = _mm_set1_epi32(qh[ib32+1]);
            idx.vec[3] = idx.vec[2];

            idx.vec[0] = _mm_and_si128(_mm_mullo_epi32(idx.vec[0], idx_mul_0), idx_mask);
            idx.vec[1] = _mm_and_si128(_mm_mullo_epi32(idx.vec[1], idx_mul_1), idx_mask);
            idx.vec[2] = _mm_and_si128(_mm_mullo_epi32(idx.vec[2], idx_mul_0), idx_mask);
            idx.vec[3] = _mm_and_si128(_mm_mullo_epi32(idx.vec[3], idx_mul_1), idx_mask);

            idx.vec[0] = _mm_or_si128(idx.vec[0], _mm_cvtepi16_epi32(idx_l_0));
            idx.vec[1] = _mm_or_si128(idx.vec[1], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
            idx.vec[2] = _mm_or_si128(idx.vec[2], _mm_cvtepi16_epi32(idx_l_1));
            idx.vec[3] = _mm_or_si128(idx.vec[3], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));

            const __m128i q2_1_0 = _mm_set_epi32(iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]]);
            const __m128i q2_2_0 = _mm_set_epi32(iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]]);

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | (signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | (signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector unsigned char mask0 = vec_xl( 0, k_mask1);
    const vector unsigned char mask1 = vec_xl(16, k_mask1);
    const vector signed char mask2 = (vector signed char)vec_xl( 0, k_mask2);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        const uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        const uint8_t *  GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].signs);
        const uint8_t *  GGML_RESTRICT sc = x[i].scales;
        const int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned int aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            vector unsigned int aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            vector unsigned int aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            vector unsigned int aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            vector signed char vsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&signs[0]);
            vector signed char vsigns02 = (vector signed char)vec_splats(*(const uint32_t *)&signs[2]);
            signs += 4;

            vector signed char vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed char vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed char vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            vector signed char vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (vector signed char)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed char)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed char)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed char)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed char q3x0 = vec_sub(vec_xor(vsigns0, (vector signed char)aux32x4_0), vsigns0);
            vector signed char q3x1 = vec_sub(vec_xor(vsigns1, (vector signed char)aux32x4_1), vsigns1);
            vector signed char q3x2 = vec_sub(vec_xor(vsigns2, (vector signed char)aux32x4_2), vsigns2);
            vector signed char q3x3 = vec_sub(vec_xor(vsigns3, (vector signed char)aux32x4_3), vsigns3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            const uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            const uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = __lasx_xvld((const __m256i*)k_mask1, 0);
    const __m256i mask2 = __lasx_xvld((const __m256i*)k_mask2, 0);

    __m256i idx_shift = lasx_set_w(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = __lasx_xvreplgr2vr_w(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i idx_l = lasx_extu8_16(__lsx_vld(qs, 0)); qs += 16;
            idx.vec[0] = __lasx_xvreplgr2vr_w(qh[ib32+0]);
            idx.vec[1] = __lasx_xvreplgr2vr_w(qh[ib32+1]);
            idx.vec[0] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = __lasx_xvor_v(idx.vec[0], lasx_ext16_32(lasx_extracti128(idx_l, 0)));
            idx.vec[1] = __lasx_xvor_v(idx.vec[1], lasx_ext16_32(lasx_extracti128(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = lasx_set_w(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = lasx_set_w(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | (signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | (signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = hsum_float_8(accumf);

#else

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
#endif
}

#if defined(__AVX2__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}
#elif defined(__loongarch_asx)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i a = __lasx_xvmulwev_h_b(x, y);
    const __m256i b = __lasx_xvmulwod_h_b(x, y);
    return __lasx_xvadd_h(a, b);
}
#endif

void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __ARM_NEON

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi1 = 0, sumi2 = 0, sumi3 = 0;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[ib+0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[ib+0] << 5) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[ib+0] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[ib+0] >> 1) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[ib+1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[ib+1] << 5) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[ib+1] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[ib+1] >> 1) & 0x700)))));
            qs += 8;

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[0], q8b.val[0]), q1b.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[2], q8b.val[2]), q1b.val[3], q8b.val[3]);

            const int ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            sumi1 += vaddvq_s32(p1) * ls1;
            sumi2 += vaddvq_s32(p2) * ls2;
            sumi3 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * ls1 * (qh[ib+0] & 0x8000 ? -1 : 1)
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * ls2 * (qh[ib+1] & 0x8000 ? -1 : 1);

        }

        sumf += y[i].d * GGML_FP16_TO_FP32(x[i].d) * (sumi1 + sumi2 + IQ1S_DELTA * sumi3);
    }

    *s = sumf;

#elif defined __AVX2__

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib], 0x700070007000700ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib + 1], 0x700070007000700ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);
#else
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
#endif
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#elif defined __AVX__
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)]);
            qs += 8;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(ls1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(ls1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(ls2));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(ls2));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#elif defined(__POWER9_VECTOR__)
    const vector unsigned char v0 = vec_splats((unsigned char)0x0);
    const vector unsigned short vsign = vec_splats((unsigned short)0x8000);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = vec_splats((int32_t)0);
        vector signed int vsumi1 = vec_splats((int32_t)0);
        vector signed int vsumi2 = vec_splats((int32_t)0);
        vector signed int vsumi3 = vec_splats((int32_t)0);
        vector signed int vsumi8 = vec_splats((int32_t)0);

        const uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        const uint16_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        const int16_t  * GGML_RESTRICT qs = y[i].bsums;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            vector signed char q1x0 = (vector signed char)aux64x2_0;
            vector signed char q1x1 = (vector signed char)aux64x2_1;
            vector signed char q1x2 = (vector signed char)aux64x2_2;
            vector signed char q1x3 = (vector signed char)aux64x2_3;

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            const uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            const uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));
            vector signed short vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            vector signed short q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (vector signed short)v0);

            vector signed short qxh = (vector signed short)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            vector __bool short vsel = vec_cmpge(qxh, (vector signed short)v0);

            vector signed short q8ysum = vec_sel((vector signed short)vec_xor((vector unsigned short)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_add(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    __m256 accum = (__m256)__lasx_xvldi(0);
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = __lasx_xvldi(0);
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            __m256i q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)], 0);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], 1);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)], 2);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], 3);

            __m256i q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)], 0);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], 1);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)], 2);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], 3);

            qs += 8;
            const __m256i q8b_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8b_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;

            __m256i tmp1, tmp5, tmp6;
            tmp1 = __lasx_xvreplgr2vr_h(ls1);
            tmp5 = __lasx_xvmulwev_w_h(dot1, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot1, tmp1);
            const __m256i p1 = __lasx_xvadd_w(tmp5, tmp6);

            tmp1 = __lasx_xvreplgr2vr_h(ls2);
            tmp5 = __lasx_xvmulwev_w_h(dot2, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot2, tmp1);
            const __m256i p2 = __lasx_xvadd_w(tmp5, tmp6);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#else

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;

#endif
}

void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

#if defined __ARM_NEON
    const int32x4_t mask  = vdupq_n_s32(0x7);
    const int32x4_t mone  = vdupq_n_s32(1);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x4_t deltas;
    deltas.val[0] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(+1));
    deltas.val[1] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(+1));
    deltas.val[2] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(-1));
    deltas.val[3] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1));

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int32x4_t sumi1 = mzero;
        int32x4_t sumi2 = mzero;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[0] << 4) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[1] << 4) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[2] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[2] << 4) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[3] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[3] << 4) & 0x700)))));

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[0], q8b.val[0]), ggml_vdotq_s32(mzero, q1b.val[1], q8b.val[1]));
            const int32x4_t p2 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[2], q8b.val[2]), ggml_vdotq_s32(mzero, q1b.val[3], q8b.val[3]));
            const int32x4_t p12 = vpaddq_s32(p1, p2);

            const uint32_t * qh32 = (const uint32_t *)qh; // we are 4-byte aligned, so we can do that
            aux32 = ((qh32[0] >> 3) & 0x01010101) | ((qh32[0] >> 6) & 0x02020202);

            const int32x4_t p3 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[0]], q8b.val[0]), ggml_vdotq_s32(mzero, deltas.val[aux8[1]], q8b.val[1]));
            const int32x4_t p4 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[2]], q8b.val[2]), ggml_vdotq_s32(mzero, deltas.val[aux8[3]], q8b.val[3]));
            const int32x4_t p34 = vpaddq_s32(p3, p4);

            int32x4_t scales_4 = ggml_vld1q_u32(sc[ib/2] >> 0, sc[ib/2] >> 3, sc[ib/2] >> 6, sc[ib/2] >> 9);

            scales_4 = vaddq_s32(vshlq_n_s32(vandq_s32(scales_4, mask), 1), mone);

            sumi1 = vmlaq_s32(sumi1, scales_4, p12);
            sumi2 = vmlaq_s32(sumi2, scales_4, p34);

            qs += 8; qh += 4;

        }

        sumf += y[i].d * GGML_FP16_TO_FP32(scale.f16) * (vaddvq_s32(sumi1) + IQ1M_DELTA * vaddvq_s32(sumi2));
    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mone8 = _mm256_set1_epi8(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        // Extract 3-bit scales (16 values)
        __m256i scales = _mm256_set1_epi64x(*(const uint64_t*)sc);
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);

        // Indices to repeat each scale 8 times.
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
        __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh) & 0x7777, 0xf000f000f000f00ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh + 2) & 0x7777, 0xf000f000f000f00ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);

            // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
            const uint64_t delta_sign = _pdep_u64(*(const uint32_t*)(qh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
            const __m256i delta1 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign)));
            const __m256i delta2 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign >> 32)));
#else
            const __m256i q1b_1 = _mm256_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            const __m256i q1b_2 = _mm256_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );

            const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
#endif
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
            const __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));

            __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);

            const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            const __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#elif defined __AVX__
    const __m128i mask = _mm_set1_epi16(0x7);
    const __m128i mone = _mm_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)]);
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);

            const __m128i delta1_0 = _mm_set_epi64x(qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta1_1 = _mm_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_0 = _mm_set_epi64x(qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_1 = _mm_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            const __m128i dot3_0 = mul_add_epi8_sse(delta1_0, q8b_1_0);
            const __m128i dot3_1 = mul_add_epi8_sse(delta1_1, q8b_1_1);
            const __m128i dot4_0 = mul_add_epi8_sse(delta2_0, q8b_2_0);
            const __m128i dot4_1 = mul_add_epi8_sse(delta2_1, q8b_2_1);

            __m128i scale1_0 = _mm_set1_epi16(sc[ib/2] >> 0);
            __m128i scale1_1 = _mm_set1_epi16(sc[ib/2] >> 3);
            __m128i scale2_0 = _mm_set1_epi16(sc[ib/2] >> 6);
            __m128i scale2_1 = _mm_set1_epi16(sc[ib/2] >> 9);

            scale1_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_0, mask), 1), mone);
            scale1_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_1, mask), 1), mone);
            scale2_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_0, mask), 1), mone);
            scale2_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_1, mask), 1), mone);
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, scale1_0);
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, scale1_1);
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, scale2_0);
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, scale2_1);
            const __m128i p3_0 = _mm_madd_epi16(dot3_0, scale1_0);
            const __m128i p3_1 = _mm_madd_epi16(dot3_1, scale1_1);
            const __m128i p4_0 = _mm_madd_epi16(dot4_0, scale2_0);
            const __m128i p4_1 = _mm_madd_epi16(dot4_1, scale2_1);

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_add_epi32(p3_0, p4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_add_epi32(p3_1, p4_1));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum1);
        accum2 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi2_1, sumi2_0))), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#else

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;

#endif
}

void ggml_vec_dot_iq4_nl_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint8x16x2_t q4bits;
    int8x16x4_t q4b;
    int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {

        q4bits.val[0] = vld1q_u8(x[ib + 0].qs);
        q4bits.val[1] = vld1q_u8(x[ib + 1].qs);
        q8b.val[0]    = vld1q_s8(y[ib + 0].qs);
        q8b.val[1]    = vld1q_s8(y[ib + 0].qs + 16);
        q8b.val[2]    = vld1q_s8(y[ib + 1].qs);
        q8b.val[3]    = vld1q_s8(y[ib + 1].qs + 16);

        q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
        q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
        q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
        q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

        sumf +=
            GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib + 0].d) * vaddvq_s32(prod_1) +
            GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) * vaddvq_s32(prod_2);
    }

#elif defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[ib + 0].d)*GGML_FP16_TO_FP32(x[ib + 0].d)),
                _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[ib + 1].d)*GGML_FP16_TO_FP32(x[ib + 1].d)),
                _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);

    const vector signed char values = vec_xl( 0, kvalues_iq4nl);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(values, values, (vector unsigned char)q4x0);
        q4x1 = vec_perm(values, values, (vector unsigned char)q4x1);

        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined (__loongarch_asx)

    const __m128i values128 = __lsx_vld((const __m128i*)kvalues_iq4nl, 0);
    const __m128i m4b  = __lsx_vreplgr2vr_b(0x0f);
    const __m256i mone = __lasx_xvreplgr2vr_h(1);

    __m256 accum1 = (__m256)__lasx_xvldi(0);
    __m256 accum2 = (__m256)__lasx_xvldi(0);
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = __lsx_vld((const __m128i*)x[ib + 0].qs, 0);
        const __m128i q4bits_2 = __lsx_vld((const __m128i*)x[ib + 1].qs, 0);
        const __m256i q8b_1 = __lasx_xvld((const __m256i *)y[ib + 0].qs, 0);
        const __m256i q8b_2 = __lasx_xvld((const __m256i *)y[ib + 1].qs, 0);
        const __m256i q4b_1 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_1, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_1, m4b)));
        const __m256i q4b_2 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_2, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = lasx_madd_h(p16_1, mone);
        const __m256i p_2 = lasx_madd_h(p16_2, mone);
        accum1 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib + 0].d)*GGML_FP16_TO_FP32(x[ib + 0].d)),
                __lasx_xvffint_s_w(p_1), accum1);
        accum2 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib + 1].d)*GGML_FP16_TO_FP32(x[ib + 1].d)),
                __lasx_xvffint_s_w(p_2), accum2);
    }

    sumf = hsum_float_8(__lasx_xvfadd_s(accum1, accum2));

#elif defined(__VXE__) || defined(__VXE2__)
    const int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    const uint8x16_t v_m = vec_splat_u8(0x0F);

    for (; ib < nb; ++ib) {
        const block_iq4_nl * GGML_RESTRICT x0 = &x[ib];
        const block_q8_0   * GGML_RESTRICT y0 = &y[ib];

        const uint8x16_t v_x = vec_xl(0, x0->qs);
        int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        v_xl = vec_perm(v_k, v_k, (uchar8x16_t)v_xl);
        v_xh = vec_perm(v_k, v_k, (uchar8x16_t)v_xh);

        const int8x16_t v_yl = vec_xl(0      , y0->qs);
        const int8x16_t v_yh = vec_xl(QK8_0/2, y0->qs);
        const int32x4_t v_xy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);

        sumf += GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d) * (v_xy[0] + v_xy[1] + v_xy[2] + v_xy[3]);
    }
#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_FP16_TO_FP32(y[ib].d)*GGML_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    ggml_uint8x16x2_t q4bits;
    ggml_int8x16x4_t q4b;
    ggml_int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    float sumf = 0;

    for (int ibl = 0; ibl < nb; ++ibl) {

        const int8_t  * q8 = y[ibl].qs;
        const uint8_t * q4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/64; ++ib) {

            q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;
            q8b    = ggml_vld1q_s8_x4(q8); q8 += 64;

            q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
            q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
            q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
            q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

            prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
            prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

            int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            sumi1 += vaddvq_s32(prod_1) * ls1;
            sumi2 += vaddvq_s32(prod_2) * ls2;

        }

        sumf += GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;

#elif defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            sumi1 = _mm256_add_epi32(p_1, sumi1);
            sumi2 = _mm256_add_epi32(p_2, sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1));
            const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1));
            const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2));
            const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2));
            sumi1_0 = _mm_add_epi32(p_1_0, sumi1_0);
            sumi1_1 = _mm_add_epi32(p_1_1, sumi1_1);
            sumi2_0 = _mm_add_epi32(p_2_0, sumi2_0);
            sumi2_1 = _mm_add_epi32(p_2_1, sumi2_1);
        }
        __m128i sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
        __m128i sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(MM256_SET_M128I(sumi12_1, sumi12_0))), accum);
    }

    *s = hsum_float_8(accum);

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector signed char values = vec_xl( 0, kvalues_iq4nl);

    for (int ibl = 0; ibl < nb; ++ibl) {

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ibl].d));
        vector float vyd = vec_splats(y[ibl].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        const uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        const uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        const int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        for (int ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q4);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q4);
            q4 += 32;

            vector signed char q4x00 = (vector signed char)vec_and(qxs0, lowMask);
            vector signed char q4x01 = (vector signed char)vec_sr(qxs0, v4);
            vector signed char q4x10 = (vector signed char)vec_and(qxs1, lowMask);
            vector signed char q4x11 = (vector signed char)vec_sr(qxs1, v4);

            q4x00 = vec_perm(values, values, (vector unsigned char)q4x00);
            q4x01 = vec_perm(values, values, (vector unsigned char)q4x01);
            q4x10 = vec_perm(values, values, (vector unsigned char)q4x10);
            q4x11 = vec_perm(values, values, (vector unsigned char)q4x11);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            const uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            const uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            vector signed short vscales01 = vec_splats((int16_t)ls0);
            vector signed short vscales23 = vec_splats((int16_t)ls1);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const __m128i values128 = __lsx_vld((const __m128i*)kvalues_iq4nl, 0);

    __m256 accum = (__m256)__lasx_xvldi(0);

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = __lsx_vld((const __m128i*)qs, 0); qs += 16;
            const __m128i q4bits_2 = __lsx_vld((const __m128i*)qs, 0); qs += 16;
            const __m256i q8b_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8b_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q4b_1 = lasx_insertf128(__lsx_vshuf_b(values128, values128, __lsx_vsrli_b(q4bits_1, 4)),
                                                  __lsx_vshuf_b(values128, values128, __lsx_vandi_b(q4bits_1, 0xf)));
            const __m256i q4b_2 = lasx_insertf128(__lsx_vshuf_b(values128, values128, __lsx_vsrli_b(q4bits_2, 4)),
                                                  __lsx_vshuf_b(values128, values128, __lsx_vandi_b(q4bits_2, 0xf)));
            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i p_1 = lasx_madd_h(p16_1, __lasx_xvreplgr2vr_h(ls1));
            const __m256i p_2 = lasx_madd_h(p16_2, __lasx_xvreplgr2vr_h(ls2));
            sumi1 = __lasx_xvadd_w(p_1, sumi1);
            sumi2 = __lasx_xvadd_w(p_2, sumi2);
        }
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);
#elif defined(__VXE__) || defined(__VXE2__)
    const int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    const uint8x16_t v_m = vec_splat_u8(0x0F);

    float sumf = 0;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        const int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        uint16_t h = x[ibl].scales_h;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/64; ++ib) {
            const uint8x16_t v_x0 = vec_xl(0       , q4);
            const uint8x16_t v_x1 = vec_xl(QK4_NL/2, q4);
            q4 += 32;

            int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
            int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
            int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
            int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

            v_x0l = vec_perm(v_k, v_k, (uchar8x16_t)v_x0l);
            v_x0h = vec_perm(v_k, v_k, (uchar8x16_t)v_x0h);
            v_x1l = vec_perm(v_k, v_k, (uchar8x16_t)v_x1l);
            v_x1h = vec_perm(v_k, v_k, (uchar8x16_t)v_x1h);

            const int8x16_t v_y0 = vec_xl( 0, q8);
            const int8x16_t v_y1 = vec_xl(16, q8);
            const int8x16_t v_y2 = vec_xl(32, q8);
            const int8x16_t v_y3 = vec_xl(48, q8);
            q8 += 64;

            int32x4_t vsumi0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_x0l, v_y0), v_x0h, v_y1);
            int32x4_t vsumi1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_x1l, v_y2), v_x1h, v_y3);

            int ls1 = ((x[ibl].scales_l[ib] & 0xF) | ((h << 4) & 0x30)) - 32;
            int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;

            h >>= 4;

            sumi1 += (vsumi0[0] + vsumi0[1] + vsumi0[2] + vsumi0[3]) * ls1;
            sumi2 += (vsumi1[0] + vsumi1[1] + vsumi1[2] + vsumi1[3]) * ls2;
        }

        sumf += GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;

#else
    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
#endif
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}
