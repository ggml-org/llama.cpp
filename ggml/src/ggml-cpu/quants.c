#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"

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

void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}

void quantize_row_q2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_0_ref(x, y, k);
}

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

void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

// b-posit8 W8A8 (Anomly): the reference encoder is already exact + reproducible
// (round-to-nearest over the fixed lattice, power-of-two block scale), so the
// runtime from_float is the reference path.
void quantize_row_bposit8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_bposit8_ref(x, y, k);
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

//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            const int8_t  * GGML_RESTRICT qy   = yb->qs;

            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK2_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        for (int k = 0; k < 2; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT qs = &x[i].qs[k * 8];
            const int8_t  * GGML_RESTRICT qy = yb->qs;

            for (int b = 0; b < 8; ++b) {
                const uint8_t byte = qs[b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((int)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((int)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((int)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

// ============================================================================
// b-posit8 W8A8 exact-quire dot product (Anomly)
// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
//
// Every product is accumulated WITHOUT intermediate rounding into a 256-bit
// two's-complement Kulisch quire (8x32-bit limbs, radix point at bit 96), with
// a single rounding at readout. This is bit-exact and reproducible on any
// hardware (GPU / x86 / RISC-V), verified bit-for-bit against the open-bposit
// rational (Fraction) reference: single-block 64/64, multi-block streaming,
// and catastrophic-cancellation (exact 0 where fp32 drifts).
// ============================================================================
#define GGML_BP8_ES    2
#define GGML_BP8_QFRAC 96

// exact integer form of a b-posit8 code: value = M * 2^E. zero/NaR -> M=0.
static void ggml_bp8_code_to_ME(uint8_t p, int64_t * M, int * E) {
    if (p == 0x00 /*ZERO*/ || p == 0x80 /*NaR*/) { *M = 0; *E = 0; return; }
    int s = (p >> 7) & 1;
    int rest = p & 0x7F;
    if (s) rest = ((~rest) + 1) & 0x7F;              // two's complement of trailing 7 bits
    int leading = (rest >> 6) & 1;
    int rs = 0;
    while (rs < 7 && ((rest >> (6 - rs)) & 1) == leading) rs++;
    int k_reg, e = 0, fb = 0, fw = 0;
    if (rs == 7) {
        k_reg = leading ? 6 : -7;
    } else {
        k_reg = leading ? (rs - 1) : -rs;
        int rem = 7 - (rs + 1);
        int r2  = rest & ((1 << rem) - 1);
        int ew  = GGML_BP8_ES < rem ? GGML_BP8_ES : rem;
        if (ew > 0) { e = (r2 >> (rem - ew)) & ((1 << ew) - 1); e <<= (GGML_BP8_ES - ew); }
        rem -= ew; fw = rem; fb = fw > 0 ? (r2 & ((1 << fw) - 1)) : 0;
    }
    int64_t m = (1 << fw) + fb;                       // integer mantissa >= 1 (<= 31 for bp8)
    *M = s ? -m : m;
    *E = 4 * k_reg + e - fw;                           // useed = 16 = 2^4
}

// add P * 2^shift into a 256-bit two's-complement accumulator (8x32 limbs).
static inline void ggml_q256_add_shifted(uint32_t q[8], int64_t P, int shift) {
    if (P == 0) return;
    uint32_t t[8];
    uint32_t sx = (P < 0) ? 0xFFFFFFFFu : 0u;
    uint64_t up = (uint64_t) P;
    t[0] = (uint32_t) up; t[1] = (uint32_t)(up >> 32);
    for (int i = 2; i < 8; i++) t[i] = sx;
    if (shift > 0) {
        int words = shift >> 5, bits = shift & 31;
        if (bits) {
            uint32_t prev = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t cur = t[i];
                t[i] = (cur << bits) | prev;
                prev = (uint32_t)((uint64_t) cur >> (32 - bits));
            }
        }
        if (words) for (int i = 7; i >= 0; i--) t[i] = (i - words >= 0) ? t[i - words] : 0u;
    } else if (shift < 0) {                             // tiny term below the radix: arithmetic right shift
        int sh = -shift, words = sh >> 5, bits = sh & 31;
        if (words) for (int i = 0; i < 8; i++) t[i] = (i + words < 8) ? t[i + words] : sx;
        if (bits) {
            uint32_t next = sx;
            for (int i = 7; i >= 0; i--) { uint32_t cur = t[i]; t[i] = (cur >> bits) | (next << (32 - bits)); next = cur; }
        }
    }
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t) q[i] + t[i] + carry; q[i] = (uint32_t) v; carry = v >> 32; }
}

// final readout: signed 256-bit quire (radix at QFRAC) -> double, one rounding.
static double ggml_q256_to_double(const uint32_t q[8]) {
    uint32_t m[8]; for (int i = 0; i < 8; i++) m[i] = q[i];
    int neg = (m[7] >> 31) & 1;
    if (neg) { uint64_t c = 1; for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t)(~m[i]) + c; m[i] = (uint32_t) v; c = v >> 32; } }
    double v = 0.0;
    for (int i = 7; i >= 0; i--) v = v * 4294967296.0 + (double) m[i];
    v = ldexp(v, -GGML_BP8_QFRAC);
    return neg ? -v : v;
}

// precomputed (M,E) lattice: identical values to ggml_bp8_code_to_ME, hoisted
// out of the dot inner loop. Idempotent init (all threads fill the same values).
static int64_t g_bp8_lut_M[256];
static int     g_bp8_lut_E[256];
static int32_t g_bp8_lut_M32[256];   // same lattice as 32-bit lanes, for SIMD gathers
static int32_t g_bp8_lut_E32[256];
static volatile int g_bp8_lut_ready = 0;
static void ggml_bp8_lut_init(void) {
    if (g_bp8_lut_ready) return;
    for (int c = 0; c < 256; c++) {
        ggml_bp8_code_to_ME((uint8_t) c, &g_bp8_lut_M[c], &g_bp8_lut_E[c]);
        g_bp8_lut_M32[c] = (int32_t) g_bp8_lut_M[c];
        g_bp8_lut_E32[c] = (int32_t) g_bp8_lut_E[c];
    }
    g_bp8_lut_ready = 1;
}

// Binned accumulation (2026-09-05): products that share a shift are summed in an int64
// first and placed into the quire ONCE per distinct shift. Exact by construction for
// shift >= 0 — placement is a pure left shift with no truncation and 256-bit two's-
// complement addition is associative — and |M_x*M_y| < 2^10 leaves 2^53 terms of headroom
// in the int64. Terms with shift < 0 (below the radix point; only with absurd block
// scales) are truncated per term exactly as before, because sum-of-truncations differs
// from truncation-of-sum. Bit-identical to the per-term kernel on the rational golden set,
// 4,000 random rows incl. sub-radix rows, and under K-permutation (openevolve workspace
// bp8_vecdot_speed, evaluator gate); 6.9x median throughput on x86.
#define GGML_BP8_SHIFT_MAX 512   // shift = Ex+Ey+se+96 with |E| <= 31, |se| <= 254 -> < 512

void ggml_vec_dot_bposit8_bposit8_scalar(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK_BPOSIT8;
    const int nb = n / qk;
    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    ggml_bp8_lut_init();
    const block_bposit8 * GGML_RESTRICT x = vx;
    const block_bposit8 * GGML_RESTRICT y = vy;

    uint32_t quire[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int64_t bins[GGML_BP8_SHIFT_MAX] = { 0 };
    for (int ib = 0; ib < nb; ++ib) {
        const int se = (int) x[ib].scale_exp + (int) y[ib].scale_exp + GGML_BP8_QFRAC;
        const uint8_t * GGML_RESTRICT xq = x[ib].qs;
        const uint8_t * GGML_RESTRICT yq = y[ib].qs;
        // 8-way unrolled lanes (OpenEvolve bp8_vecdot_speed round 3, +5-9% on x86); per-lane
        // semantics identical to the scalar loop: bin for shift >= 0, per-term placement otherwise.
#define GGML_BP8_LANE(k) do { \
            const int64_t P_ = g_bp8_lut_M[xq[j + (k)]] * g_bp8_lut_M[yq[j + (k)]]; \
            if (P_ != 0) { \
                const int sh_ = g_bp8_lut_E[xq[j + (k)]] + g_bp8_lut_E[yq[j + (k)]] + se; \
                if (sh_ >= 0 && sh_ < GGML_BP8_SHIFT_MAX) bins[sh_] += P_; \
                else ggml_q256_add_shifted(quire, P_, sh_); \
            } } while (0)
        // 16-way (OpenEvolve round 4b, +6-8% at the kernel under the bit-exact gate)
        for (int j = 0; j < qk; j += 16) {
            GGML_BP8_LANE(0);  GGML_BP8_LANE(1);  GGML_BP8_LANE(2);  GGML_BP8_LANE(3);
            GGML_BP8_LANE(4);  GGML_BP8_LANE(5);  GGML_BP8_LANE(6);  GGML_BP8_LANE(7);
            GGML_BP8_LANE(8);  GGML_BP8_LANE(9);  GGML_BP8_LANE(10); GGML_BP8_LANE(11);
            GGML_BP8_LANE(12); GGML_BP8_LANE(13); GGML_BP8_LANE(14); GGML_BP8_LANE(15);
        }
#undef GGML_BP8_LANE
    }
    // Flush the bins. OpenEvolve (workspace bp8_vecdot_speed, 2026-09-05) found that a plain
    // scan of the 512 bins beats tracking touched bins (no hit[] bookkeeping in the hot loop);
    // 1.42x over the tracked version, bit-identical under the same gate.
    int i = 0;
    while (i < GGML_BP8_SHIFT_MAX - 7) {                 // skip leading zero bins 8 at a time (round 4b)
        if (bins[i] | bins[i+1] | bins[i+2] | bins[i+3] | bins[i+4] | bins[i+5] | bins[i+6] | bins[i+7]) break;
        i += 8;
    }
    for (; i < GGML_BP8_SHIFT_MAX - 7; i += 8) {
        if (bins[i]     != 0) ggml_q256_add_shifted(quire, bins[i],     i);
        if (bins[i + 1] != 0) ggml_q256_add_shifted(quire, bins[i + 1], i + 1);
        if (bins[i + 2] != 0) ggml_q256_add_shifted(quire, bins[i + 2], i + 2);
        if (bins[i + 3] != 0) ggml_q256_add_shifted(quire, bins[i + 3], i + 3);
        if (bins[i + 4] != 0) ggml_q256_add_shifted(quire, bins[i + 4], i + 4);
        if (bins[i + 5] != 0) ggml_q256_add_shifted(quire, bins[i + 5], i + 5);
        if (bins[i + 6] != 0) ggml_q256_add_shifted(quire, bins[i + 6], i + 6);
        if (bins[i + 7] != 0) ggml_q256_add_shifted(quire, bins[i + 7], i + 7);
    }
    for (; i < GGML_BP8_SHIFT_MAX; i++) {
        if (bins[i] != 0) ggml_q256_add_shifted(quire, bins[i], i);
    }
    *s = (float) ggml_q256_to_double(quire);
}

// ---------------------------------------------------------------------------
// Fast dot — the type's DEFAULT vec_dot: BPOSIT8 weights x BF16 activations (vec_dot_type = BF16,
// so the activations are a plain fp32->bf16 conversion, no quantiser). Each weight code is
// M * 2^E (|M| < 2^5, |E| <= 31) = exactly a bfloat16, so the products are exact in fp32 and the
// terms are summed in fp32 — the ordinary llama.cpp contract (as for Q8_0, the rounding depends on
// the summation order and the machine). The exact 256-bit quire path above (W8A8, posit-coded
// activations) is the deterministic profile's vec_dot (GGML_DETERMINISTIC); tests/test-bposit8-fast.c
// gates this path against a double-precision dot of the same operands (within 4e-5 of the
// absolute-term sum) and checks the shared lattice table.
static float g_bp8_lut_F[256];   // M * 2^E as float (exact: |M| < 2^5, |E| <= 31)
static volatile int g_bp8_lutf_ready = 0;
static void ggml_bp8_lutf_init(void) {
    if (g_bp8_lutf_ready) return;
    ggml_bp8_lut_init();
    for (int c = 0; c < 256; c++) g_bp8_lut_F[c] = ldexpf((float) g_bp8_lut_M[c], g_bp8_lut_E[c]);
    g_bp8_lutf_ready = 1;
}
// bf16 image of the lattice, as high/low byte tables (for the AVX-512 BF16 and NEON BF16 kernels):
// exact, because every value has at most 5 significant bits and |E| <= 31 is inside bf16's range
static uint8_t g_bp8_lut_BFhi[256], g_bp8_lut_BFlo[256];
static volatile int g_bp8_lutbf_ready = 0;
static void ggml_bp8_lutbf_init(void) {
    if (g_bp8_lutbf_ready) return;
    ggml_bp8_lutf_init();
    for (int c = 0; c < 256; c++) {
        union { float f; uint32_t u; } v; v.f = g_bp8_lut_F[c];
        const uint16_t b = (uint16_t) (v.u >> 16);
        g_bp8_lut_BFhi[c] = (uint8_t) (b >> 8); g_bp8_lut_BFlo[c] = (uint8_t) b;
    }
    g_bp8_lutbf_ready = 1;
}
static inline float ggml_bp8_pow2f(int e) {   // 2^e as float; normal range by exponent bits, else ldexpf
    if (e >= -126 && e <= 127) { union { uint32_t u; float f; } v; v.u = (uint32_t) (e + 127) << 23; return v.f; }
    return ldexpf(1.0f, e);
}
static void ggml_vec_dot_bposit8_bf16_scalar(int n, float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lutf_init();
    const block_bposit8 * GGML_RESTRICT x = vx;
    const ggml_bf16_t   * GGML_RESTRICT y = vy;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t    * GGML_RESTRICT xq = x[ib].qs;
        const ggml_bf16_t * GGML_RESTRICT yb = y + ib * QK_BPOSIT8;
        float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;
        for (int j = 0; j < QK_BPOSIT8; j += 4) {
            sb0 += g_bp8_lut_F[xq[j    ]] * GGML_BF16_TO_FP32(yb[j    ]);
            sb1 += g_bp8_lut_F[xq[j + 1]] * GGML_BF16_TO_FP32(yb[j + 1]);
            sb2 += g_bp8_lut_F[xq[j + 2]] * GGML_BF16_TO_FP32(yb[j + 2]);
            sb3 += g_bp8_lut_F[xq[j + 3]] * GGML_BF16_TO_FP32(yb[j + 3]);
        }
        sumf += ((sb0 + sb1) + (sb2 + sb3)) * ggml_bp8_pow2f((int) x[ib].scale_exp);
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// Vectorised exact-quire dot (AVX-512 VBMI, 2026-09-09). Per 32-block: every code's (M, E)
// comes from 256-entry byte tables held in registers (vpermi2b, two per table + one blend for
// 64 codes), P = M_x*M_y (|P| < 2^10) is a 16-bit lane product, sh = E_x+E_y+se a 16-bit lane
// sum. The block is anchored at its smallest shift over the non-zero products and summed
// EXACTLY in int64 lanes as P << (sh - smin) (vpsllvq); the block sum goes into a 128-bit bin
// keyed by smin, and the bins are placed into the quire once at the end. Every step is a pure
// left shift or an integer addition, so the integer that reaches the quire is the same one the
// scalar kernel builds term by term: bit-identical by construction. Headroom: rel <= 48 keeps
// |P << rel| < 2^58, 32 terms < 2^63, and a 128-bit bin holds any number of blocks. Blocks
// that exceed that range, or contain a sub-radix term (smin < 0, where the scalar kernel
// truncates PER TERM), take the scalar per-term path for that block. Gate (random, extreme,
// forced-fallback, sub-radix, huge-shift and permuted rows, vector == scalar bit for bit):
// tests/bposit8-quire-ref/bp8_vec_gate.c.
#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))   // the vector paths use __int128
#define GGML_BP8_HAVE_VEC 1
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__))   // the vector paths use __int128
#define GGML_BP8_HAVE_VEC 1
#include <arm_neon.h>
#endif

#if defined(GGML_BP8_HAVE_VEC)
#define GGML_BP8_VEC_REL_MAX 48
__extension__ typedef          __int128 ggml_bp8_i128;   // the exact bins (ISO C has no 128-bit type; __extension__ keeps -Wpedantic quiet)
__extension__ typedef unsigned __int128 ggml_bp8_u128;

static int8_t g_bp8_lut_M8[256];
static int8_t g_bp8_lut_E8[256];
static volatile int g_bp8_lut8_ready = 0;
static void ggml_bp8_lut8_init(void) {
    if (g_bp8_lut8_ready) return;
    ggml_bp8_lut_init();
    for (int c = 0; c < 256; c++) { g_bp8_lut_M8[c] = (int8_t) g_bp8_lut_M[c]; g_bp8_lut_E8[c] = (int8_t) g_bp8_lut_E[c]; }
    g_bp8_lut8_ready = 1;
}

// v = hi*2^64 + lo (lo unsigned) placed at bit `shift` of the quire.
static inline void ggml_q256_add_shifted_128(uint32_t q[8], ggml_bp8_u128 v, int shift) {
    const uint64_t lo = (uint64_t) v, hi = (uint64_t)(v >> 64);
    if (lo) {   // unsigned low half: place as two 32-bit pieces to keep add_shifted's int64 contract
        ggml_q256_add_shifted(q, (int64_t)(lo & 0xFFFFFFFFull), shift);
        ggml_q256_add_shifted(q, (int64_t)(lo >> 32), shift + 32);
    }
    if (hi) ggml_q256_add_shifted(q, (int64_t) hi, shift + 64);   // signed high half
}
#endif // GGML_BP8_HAVE_VEC

#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__)
// 256-entry byte table lookup for 64 indices: bits [6:0] select within two 64-byte halves,
// bit 7 selects the half.
static inline __m512i ggml_bp8_lut256(__m512i idx, __m512i t0, __m512i t1, __m512i t2, __m512i t3) {
    const __m512i lo = _mm512_permutex2var_epi8(t0, idx, t1);
    const __m512i hi = _mm512_permutex2var_epi8(t2, idx, t3);
    return _mm512_mask_blend_epi8(_mm512_movepi8_mask(idx), lo, hi);
}


static inline int ggml_bp8_hmin_epi16(__m512i v) {
    __m256i a = _mm256_min_epi16(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
    __m128i b = _mm_min_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    b = _mm_min_epi16(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
    b = _mm_min_epi16(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
    b = _mm_min_epi16(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(2, 3, 0, 1)));
    return (int16_t) _mm_cvtsi128_si32(b);
}
static inline int ggml_bp8_hmax_epi16(__m512i v) {
    __m256i a = _mm256_max_epi16(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
    __m128i b = _mm_max_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    b = _mm_max_epi16(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
    b = _mm_max_epi16(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
    b = _mm_max_epi16(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(2, 3, 0, 1)));
    return (int16_t) _mm_cvtsi128_si32(b);
}

// One (weight row, activation column) pair's 32-block: exact anchored sum into its bin set, or
// the per-term path. Shared by the single and the 2x2 kernels.
typedef struct { uint32_t quire[8]; ggml_bp8_i128 bins[GGML_BP8_SHIFT_MAX]; int lo, hi; } ggml_bp8_acc;
static inline void ggml_bp8_acc_init(ggml_bp8_acc * a) { memset(a, 0, sizeof *a); a->lo = GGML_BP8_SHIFT_MAX; a->hi = -1; }
static inline float ggml_bp8_acc_readout(ggml_bp8_acc * a) {
    for (int i = a->lo; i <= a->hi; i++) if (a->bins[i] != 0) ggml_q256_add_shifted_128(a->quire, (ggml_bp8_u128) a->bins[i], i);
    return (float) ggml_q256_to_double(a->quire);
}
static inline void ggml_bp8_block_avx512(ggml_bp8_acc * a, __m512i mx16, __m512i my16, __m512i e16, int se) {
    const __m512i P16  = _mm512_mullo_epi16(mx16, my16);
    const __m512i SH16 = _mm512_add_epi16(e16, _mm512_set1_epi16((short) se));
    const __mmask32 nz = _mm512_cmpneq_epi16_mask(P16, _mm512_setzero_si512());
    if (nz == 0) return;
    const int smin = ggml_bp8_hmin_epi16(_mm512_mask_blend_epi16(nz, _mm512_set1_epi16(0x3FFF),  SH16));
    const int smax = ggml_bp8_hmax_epi16(_mm512_mask_blend_epi16(nz, _mm512_set1_epi16(-0x3FFF), SH16));
    if (smin < 0 || smax - smin > GGML_BP8_VEC_REL_MAX || smin >= GGML_BP8_SHIFT_MAX) {
        int16_t pv[32], sv[32];
        _mm512_storeu_si512(pv, P16); _mm512_storeu_si512(sv, SH16);
        for (int k = 0; k < 32; k++) if (pv[k] != 0) ggml_q256_add_shifted(a->quire, (int64_t) pv[k], sv[k]);
        return;
    }
    const __m512i rel16 = _mm512_sub_epi16(SH16, _mm512_set1_epi16((short) smin));
    const __m128i p0 = _mm512_castsi512_si128(P16),  p1 = _mm512_extracti32x4_epi32(P16, 1),  p2 = _mm512_extracti32x4_epi32(P16, 2),  p3 = _mm512_extracti32x4_epi32(P16, 3);
    const __m128i r0 = _mm512_castsi512_si128(rel16), r1 = _mm512_extracti32x4_epi32(rel16, 1), r2 = _mm512_extracti32x4_epi32(rel16, 2), r3 = _mm512_extracti32x4_epi32(rel16, 3);
    __m512i acc = _mm512_sllv_epi64(_mm512_cvtepi16_epi64(p0), _mm512_cvtepi16_epi64(r0));
    acc = _mm512_add_epi64(acc, _mm512_sllv_epi64(_mm512_cvtepi16_epi64(p1), _mm512_cvtepi16_epi64(r1)));
    acc = _mm512_add_epi64(acc, _mm512_sllv_epi64(_mm512_cvtepi16_epi64(p2), _mm512_cvtepi16_epi64(r2)));
    acc = _mm512_add_epi64(acc, _mm512_sllv_epi64(_mm512_cvtepi16_epi64(p3), _mm512_cvtepi16_epi64(r3)));
    a->bins[smin] += (ggml_bp8_i128) _mm512_reduce_add_epi64(acc);
    if (smin < a->lo) a->lo = smin;
    if (smin > a->hi) a->hi = smin;
}
// 64 codes -> (M, E) as two 512-bit int8 vectors
#define GGML_BP8_DECODE64(idx, M, E) do { \
        M = ggml_bp8_lut256((idx), M0, M1, M2, M3); E = ggml_bp8_lut256((idx), E0, E1, E2, E3); } while (0)
#define GGML_BP8_TABLES \
    const __m512i M0 = _mm512_loadu_si512(g_bp8_lut_M8), M1 = _mm512_loadu_si512(g_bp8_lut_M8 + 64); \
    const __m512i M2 = _mm512_loadu_si512(g_bp8_lut_M8 + 128), M3 = _mm512_loadu_si512(g_bp8_lut_M8 + 192); \
    const __m512i E0 = _mm512_loadu_si512(g_bp8_lut_E8), E1 = _mm512_loadu_si512(g_bp8_lut_E8 + 64); \
    const __m512i E2 = _mm512_loadu_si512(g_bp8_lut_E8 + 128), E3 = _mm512_loadu_si512(g_bp8_lut_E8 + 192);
static inline __m512i ggml_bp8_load2(const block_bposit8 * x, int ib, int nb) {   // codes of blocks ib, ib+1 (zero-padded)
    __m512i v = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x[ib].qs));
    return _mm512_inserti64x4(v, ib + 1 < nb ? _mm256_loadu_si256((const __m256i *) x[ib + 1].qs) : _mm256_setzero_si256(), 1);
}
#define GGML_BP8_HALF(v, h) ((h) ? _mm512_extracti64x4_epi64((v), 1) : _mm512_castsi512_si256(v))

static void ggml_vec_dot_bposit8_bposit8_avx512(int n, float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lut8_init();
    const block_bposit8 * GGML_RESTRICT x = vx;
    const block_bposit8 * GGML_RESTRICT y = vy;
    ggml_bp8_acc a; ggml_bp8_acc_init(&a);
    GGML_BP8_TABLES
    for (int ib = 0; ib < nb; ib += 2) {
        __m512i mx8, ex8, my8, ey8;
        GGML_BP8_DECODE64(ggml_bp8_load2(x, ib, nb), mx8, ex8);
        GGML_BP8_DECODE64(ggml_bp8_load2(y, ib, nb), my8, ey8);
        for (int h = 0; h < 2 && ib + h < nb; h++) {
            const int se = (int) x[ib + h].scale_exp + (int) y[ib + h].scale_exp + GGML_BP8_QFRAC;
            ggml_bp8_block_avx512(&a, _mm512_cvtepi8_epi16(GGML_BP8_HALF(mx8, h)), _mm512_cvtepi8_epi16(GGML_BP8_HALF(my8, h)),
                _mm512_add_epi16(_mm512_cvtepi8_epi16(GGML_BP8_HALF(ex8, h)), _mm512_cvtepi8_epi16(GGML_BP8_HALF(ey8, h))), se);
        }
    }
    *s = ggml_bp8_acc_readout(&a);
}

// 2x2 micro-tile for the matmul: weight rows x0, x1 (stride bx) against activation columns
// y0, y1 (stride by); each code is decoded once and used by two pairs. s[r + bs*c].
static void ggml_vec_dot_bposit8_bposit8_avx512_2x2(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lut8_init();
    const block_bposit8 * GGML_RESTRICT x0 = vx, * GGML_RESTRICT x1 = (const block_bposit8 *)((const uint8_t *) vx + bx);
    const block_bposit8 * GGML_RESTRICT y0 = vy, * GGML_RESTRICT y1 = (const block_bposit8 *)((const uint8_t *) vy + by);
    ggml_bp8_acc a00, a10, a01, a11;
    ggml_bp8_acc_init(&a00); ggml_bp8_acc_init(&a10); ggml_bp8_acc_init(&a01); ggml_bp8_acc_init(&a11);
    GGML_BP8_TABLES
    for (int ib = 0; ib < nb; ib++) {
        // pack (x0 block, x1 block) and (y0 block, y1 block) into one register each: one decode per operand
        __m512i mx, ex, my, ey;
        GGML_BP8_DECODE64(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x0[ib].qs)), _mm256_loadu_si256((const __m256i *) x1[ib].qs), 1), mx, ex);
        GGML_BP8_DECODE64(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) y0[ib].qs)), _mm256_loadu_si256((const __m256i *) y1[ib].qs), 1), my, ey);
        const __m512i mx0 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(mx, 0)), mx1 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(mx, 1));
        const __m512i my0 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(my, 0)), my1 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(my, 1));
        const __m512i ex0 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(ex, 0)), ex1 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(ex, 1));
        const __m512i ey0 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(ey, 0)), ey1 = _mm512_cvtepi8_epi16(GGML_BP8_HALF(ey, 1));
        const int sx0 = x0[ib].scale_exp, sx1 = x1[ib].scale_exp, sy0 = y0[ib].scale_exp, sy1 = y1[ib].scale_exp;
        ggml_bp8_block_avx512(&a00, mx0, my0, _mm512_add_epi16(ex0, ey0), sx0 + sy0 + GGML_BP8_QFRAC);
        ggml_bp8_block_avx512(&a10, mx1, my0, _mm512_add_epi16(ex1, ey0), sx1 + sy0 + GGML_BP8_QFRAC);
        ggml_bp8_block_avx512(&a01, mx0, my1, _mm512_add_epi16(ex0, ey1), sx0 + sy1 + GGML_BP8_QFRAC);
        ggml_bp8_block_avx512(&a11, mx1, my1, _mm512_add_epi16(ex1, ey1), sx1 + sy1 + GGML_BP8_QFRAC);
    }
    s[0] = ggml_bp8_acc_readout(&a00); s[1] = ggml_bp8_acc_readout(&a10);
    s[bs] = ggml_bp8_acc_readout(&a01); s[bs + 1] = ggml_bp8_acc_readout(&a11);
}

// Fast dot, AVX-512 (VBMI): the weight codes decode through two byte tables straight into bf16
// lanes (exact); the activations are bf16 already. With AVX-512 BF16, vdpbf16ps forms the 32
// products and pair sums per block in one instruction; without it the lanes are widened to fp32
// (a shift) and FMA'd. The block scale 2^scale_exp is applied once per block with vscalefps.
// permutation so that unpacklo_epi8 yields block A (bytes 0..31 of the load) and unpackhi block B,
// both with their 32 lanes in element order (so activations are loaded straight)
static inline __m512i ggml_bp8_bf16_perm(void) {
    uint8_t p[64];
    for (int l = 0; l < 4; l++) for (int k = 0; k < 8; k++) { p[16 * l + k] = (uint8_t) (8 * l + k); p[16 * l + 8 + k] = (uint8_t) (32 + 8 * l + k); }
    return _mm512_loadu_si512(p);
}
#define GGML_BP8_BFTABLES \
    const __m512i H0 = _mm512_loadu_si512(g_bp8_lut_BFhi), H1 = _mm512_loadu_si512(g_bp8_lut_BFhi + 64); \
    const __m512i H2 = _mm512_loadu_si512(g_bp8_lut_BFhi + 128), H3 = _mm512_loadu_si512(g_bp8_lut_BFhi + 192); \
    const __m512i L0 = _mm512_loadu_si512(g_bp8_lut_BFlo), L1 = _mm512_loadu_si512(g_bp8_lut_BFlo + 64); \
    const __m512i L2 = _mm512_loadu_si512(g_bp8_lut_BFlo + 128), L3 = _mm512_loadu_si512(g_bp8_lut_BFlo + 192); \
    const __m512i BFPERM = ggml_bp8_bf16_perm();
// 64 codes (block A in bytes 0..31, block B in 32..63) -> two vectors of 32 bf16 lanes
#define GGML_BP8_BF16_DECODE(codes, va, vb) do { \
        const __m512i idx_ = _mm512_permutexvar_epi8(BFPERM, (codes)); \
        const __m512i hi_ = ggml_bp8_lut256(idx_, H0, H1, H2, H3), lo_ = ggml_bp8_lut256(idx_, L0, L1, L2, L3); \
        va = _mm512_unpacklo_epi8(lo_, hi_); vb = _mm512_unpackhi_epi8(lo_, hi_); } while (0)
// one block: 32 bf16 weight lanes x 32 bf16 activation lanes, scaled by 2^se, added to acc
static inline __m512 ggml_bp8_block_bf16(__m512 acc, __m512i xa, __m512i ya, int se) {
#if defined(__AVX512BF16__)
    const __m512 pb = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh) xa, (__m512bh) ya);
#else
    const __m512 x0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(xa)), 16));
    const __m512 x1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(xa, 1)), 16));
    const __m512 y0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(ya)), 16));
    const __m512 y1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(ya, 1)), 16));
    const __m512 pb = _mm512_fmadd_ps(x1, y1, _mm512_mul_ps(x0, y0));
#endif
    return _mm512_add_ps(acc, _mm512_scalef_ps(pb, _mm512_set1_ps((float) se)));
}
static void ggml_vec_dot_bposit8_bf16_avx512(int n, float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lutbf_init();
    const block_bposit8 * GGML_RESTRICT x = vx;
    const ggml_bf16_t   * GGML_RESTRICT y = vy;
    GGML_BP8_BFTABLES
    __m512 acc = _mm512_setzero_ps();
    for (int ib = 0; ib < nb; ib += 2) {
        __m512i xa, xb;
        GGML_BP8_BF16_DECODE(ggml_bp8_load2(x, ib, nb), xa, xb);
        acc = ggml_bp8_block_bf16(acc, xa, _mm512_loadu_si512(y + ib * QK_BPOSIT8), (int) x[ib].scale_exp);
        if (ib + 1 < nb) acc = ggml_bp8_block_bf16(acc, xb, _mm512_loadu_si512(y + (ib + 1) * QK_BPOSIT8), (int) x[ib + 1].scale_exp);
    }
    *s = _mm512_reduce_add_ps(acc);
}
// 2x2 tile: weight rows x0, x1 decoded together (one 64-code decode per block), activation rows y0, y1
static void ggml_vec_dot_bposit8_bf16_avx512_2x2(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lutbf_init();
    const block_bposit8 * GGML_RESTRICT x0 = vx, * GGML_RESTRICT x1 = (const block_bposit8 *)((const uint8_t *) vx + bx);
    const ggml_bf16_t   * GGML_RESTRICT y0 = vy, * GGML_RESTRICT y1 = (const ggml_bf16_t *)((const uint8_t *) vy + by);
    GGML_BP8_BFTABLES
    __m512 a00 = _mm512_setzero_ps(), a10 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps();
    for (int ib = 0; ib < nb; ib++) {
        __m512i vx0, vx1;
        GGML_BP8_BF16_DECODE(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) x0[ib].qs)), _mm256_loadu_si256((const __m256i *) x1[ib].qs), 1), vx0, vx1);
        const __m512i vy0 = _mm512_loadu_si512(y0 + ib * QK_BPOSIT8), vy1 = _mm512_loadu_si512(y1 + ib * QK_BPOSIT8);
        const int sx0 = x0[ib].scale_exp, sx1 = x1[ib].scale_exp;
        a00 = ggml_bp8_block_bf16(a00, vx0, vy0, sx0);
        a10 = ggml_bp8_block_bf16(a10, vx1, vy0, sx1);
        a01 = ggml_bp8_block_bf16(a01, vx0, vy1, sx0);
        a11 = ggml_bp8_block_bf16(a11, vx1, vy1, sx1);
    }
    s[0] = _mm512_reduce_add_ps(a00); s[1] = _mm512_reduce_add_ps(a10);
    s[bs] = _mm512_reduce_add_ps(a01); s[bs + 1] = _mm512_reduce_add_ps(a11);
}
#endif // AVX-512 VBMI

#if defined(__aarch64__) && defined(__ARM_NEON)
// NEON (AArch64) form of the same anchored-block kernel: (M, E) byte tables in registers
// (four vqtbl4q lookups per 16 codes per table, out-of-range indices read as 0 so the four
// 64-entry quarters OR together), int16 products via vmull_s8, per-block anchor from vminvq /
// vmaxvq over the non-zero lanes, exact int64 lane sums via vshlq_s64 with per-lane shifts,
// 128-bit bins by anchor. Same integer as the scalar kernel; same gate.
typedef struct { uint8x16x4_t q[4]; } ggml_bp8_tbl256;
static inline void ggml_bp8_tbl256_load(ggml_bp8_tbl256 * t, const int8_t * tab) {
    for (int k = 0; k < 4; k++) t->q[k] = vld1q_u8_x4((const uint8_t *) tab + 64 * k);
}
static inline int8x16_t ggml_bp8_lut256_neon(const ggml_bp8_tbl256 * t, uint8x16_t idx) {
    const uint8x16_t c64 = vdupq_n_u8(64);
    uint8x16_t r = vqtbl4q_u8(t->q[0], idx);
    idx = vsubq_u8(idx, c64); r = vorrq_u8(r, vqtbl4q_u8(t->q[1], idx));
    idx = vsubq_u8(idx, c64); r = vorrq_u8(r, vqtbl4q_u8(t->q[2], idx));
    idx = vsubq_u8(idx, c64); r = vorrq_u8(r, vqtbl4q_u8(t->q[3], idx));
    return vreinterpretq_s8_u8(r);
}
// P << rel for 8 int16 lanes into an int64x2 accumulator pair (exact: |P| < 2^10, rel <= 48)
static inline void ggml_bp8_acc8_neon(int64x2_t * a0, int64x2_t * a1, int16x8_t p16, int16x8_t rel16) {
    const int32x4_t pl = vmovl_s16(vget_low_s16(p16)),  ph = vmovl_high_s16(p16);
    const int32x4_t rl = vmovl_s16(vget_low_s16(rel16)), rh = vmovl_high_s16(rel16);
    *a0 = vaddq_s64(*a0, vshlq_s64(vmovl_s32(vget_low_s32(pl)), vmovl_s32(vget_low_s32(rl))));
    *a1 = vaddq_s64(*a1, vshlq_s64(vmovl_high_s32(pl),          vmovl_high_s32(rl)));
    *a0 = vaddq_s64(*a0, vshlq_s64(vmovl_s32(vget_low_s32(ph)), vmovl_s32(vget_low_s32(rh))));
    *a1 = vaddq_s64(*a1, vshlq_s64(vmovl_high_s32(ph),          vmovl_high_s32(rh)));
}

static void ggml_vec_dot_bposit8_bposit8_neon(int n, float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int qk = QK_BPOSIT8;
    const int nb = n / qk;
    ggml_bp8_lut8_init();
    const block_bposit8 * GGML_RESTRICT x = vx;
    const block_bposit8 * GGML_RESTRICT y = vy;
    uint32_t quire[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    ggml_bp8_i128 bins[GGML_BP8_SHIFT_MAX];
    memset(bins, 0, sizeof bins);
    int bin_lo = GGML_BP8_SHIFT_MAX, bin_hi = -1;
    ggml_bp8_tbl256 TM, TE;
    ggml_bp8_tbl256_load(&TM, g_bp8_lut_M8);
    ggml_bp8_tbl256_load(&TE, g_bp8_lut_E8);
    const int16x8_t big16 = vdupq_n_s16(0x3FFF), nbig16 = vdupq_n_s16(-0x3FFF);
    for (int ib = 0; ib < nb; ib++) {
        const int se = (int) x[ib].scale_exp + (int) y[ib].scale_exp + GGML_BP8_QFRAC;
        const int16x8_t vse = vdupq_n_s16((int16_t) se);
        int16x8_t P[4], SH[4];
        int16x8_t vmin = big16, vmax = nbig16;
        for (int h = 0; h < 2; h++) {                      // two 16-code halves
            const uint8x16_t ix = vld1q_u8(x[ib].qs + 16 * h), iy = vld1q_u8(y[ib].qs + 16 * h);
            const int8x16_t mx = ggml_bp8_lut256_neon(&TM, ix), my = ggml_bp8_lut256_neon(&TM, iy);
            const int8x16_t ex = ggml_bp8_lut256_neon(&TE, ix), ey = ggml_bp8_lut256_neon(&TE, iy);
            const int8x16_t e8 = vaddq_s8(ex, ey);           // E_x+E_y in [-64, 54]: fits int8
            P[2 * h]      = vmull_s8(vget_low_s8(mx), vget_low_s8(my));
            P[2 * h + 1]  = vmull_high_s8(mx, my);
            SH[2 * h]     = vaddq_s16(vmovl_s8(vget_low_s8(e8)), vse);
            SH[2 * h + 1] = vaddq_s16(vmovl_high_s8(e8), vse);
            for (int q = 2 * h; q < 2 * h + 2; q++) {
                const uint16x8_t nz = vmvnq_u16(vceqzq_s16(P[q]));
                vmin = vminq_s16(vmin, vbslq_s16(nz, SH[q], big16));
                vmax = vmaxq_s16(vmax, vbslq_s16(nz, SH[q], nbig16));
            }
        }
        const int smin = vminvq_s16(vmin), smax = vmaxvq_s16(vmax);
        if (smin == 0x3FFF) continue;                          // all-zero block
        if (smin < 0 || smax - smin > GGML_BP8_VEC_REL_MAX || smin >= GGML_BP8_SHIFT_MAX) {
            int16_t pv[8], sv[8];
            for (int q = 0; q < 4; q++) {
                vst1q_s16(pv, P[q]); vst1q_s16(sv, SH[q]);
                for (int k = 0; k < 8; k++) if (pv[k] != 0) ggml_q256_add_shifted(quire, (int64_t) pv[k], sv[k]);
            }
            continue;
        }
        const int16x8_t vsmin = vdupq_n_s16((int16_t) smin);
        int64x2_t a0 = vdupq_n_s64(0), a1 = vdupq_n_s64(0);
        for (int q = 0; q < 4; q++) ggml_bp8_acc8_neon(&a0, &a1, P[q], vsubq_s16(SH[q], vsmin));
        bins[smin] += (ggml_bp8_i128) vaddvq_s64(vaddq_s64(a0, a1));
        if (smin < bin_lo) bin_lo = smin;
        if (smin > bin_hi) bin_hi = smin;
    }
    for (int i = bin_lo; i <= bin_hi; i++) {
        if (bins[i] != 0) ggml_q256_add_shifted_128(quire, (ggml_bp8_u128) bins[i], i);
    }
    *s = (float) ggml_q256_to_double(quire);
}

#if defined(__ARM_FEATURE_BF16)
// Fast dot, NEON BF16 (Apple M-series, Armv8.6+): weight codes -> bf16 lanes through the hi/lo
// byte tables (vqtbl4q, four per 64-entry quarter; lanes in element order after the zip),
// activations are bf16 already, vbfdotq_f32 forms the products and pair sums in fp32, the block
// scale is one fmla by 2^scale_exp.
typedef struct { bfloat16x8_t v[4]; } ggml_bp8_bf32;   // one 32-code block as bf16 lanes
static inline void ggml_bp8_bf16_block_neon(const ggml_bp8_tbl256 * th, const ggml_bp8_tbl256 * tl, const uint8_t * qs, ggml_bp8_bf32 * o) {
    for (int h = 0; h < 2; h++) {
        const uint8x16_t codes = vld1q_u8(qs + 16 * h);
        const uint8x16_t hi = vreinterpretq_u8_s8(ggml_bp8_lut256_neon(th, codes));
        const uint8x16_t lo = vreinterpretq_u8_s8(ggml_bp8_lut256_neon(tl, codes));
        o->v[2 * h]     = vreinterpretq_bf16_u8(vzip1q_u8(lo, hi));   // little-endian: low byte first
        o->v[2 * h + 1] = vreinterpretq_bf16_u8(vzip2q_u8(lo, hi));
    }
}
static inline float32x4_t ggml_bp8_dot_block_neon(float32x4_t acc, const ggml_bp8_bf32 * x, const ggml_bf16_t * y, int se) {
    const bfloat16_t * yb = (const bfloat16_t *) y;
    float32x4_t p = vbfdotq_f32(vdupq_n_f32(0.0f), x->v[0], vld1q_bf16(yb));
    p = vbfdotq_f32(p, x->v[1], vld1q_bf16(yb + 8));
    p = vbfdotq_f32(p, x->v[2], vld1q_bf16(yb + 16));
    p = vbfdotq_f32(p, x->v[3], vld1q_bf16(yb + 24));
    return vfmaq_f32(acc, p, vdupq_n_f32(ggml_bp8_pow2f(se)));
}
static void ggml_vec_dot_bposit8_bf16_neon(int n, float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lutbf_init();
    ggml_bp8_tbl256 th, tl;
    ggml_bp8_tbl256_load(&th, (const int8_t *) g_bp8_lut_BFhi);
    ggml_bp8_tbl256_load(&tl, (const int8_t *) g_bp8_lut_BFlo);
    const block_bposit8 * GGML_RESTRICT x = vx;
    const ggml_bf16_t   * GGML_RESTRICT y = vy;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int ib = 0; ib < nb; ib++) {
        ggml_bp8_bf32 xb;
        ggml_bp8_bf16_block_neon(&th, &tl, x[ib].qs, &xb);
        acc = ggml_bp8_dot_block_neon(acc, &xb, y + ib * QK_BPOSIT8, (int) x[ib].scale_exp);
    }
    *s = vaddvq_f32(acc);
}
static void ggml_vec_dot_bposit8_bf16_neon_2x2(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by) {
    const int nb = n / QK_BPOSIT8;
    ggml_bp8_lutbf_init();
    ggml_bp8_tbl256 th, tl;
    ggml_bp8_tbl256_load(&th, (const int8_t *) g_bp8_lut_BFhi);
    ggml_bp8_tbl256_load(&tl, (const int8_t *) g_bp8_lut_BFlo);
    const block_bposit8 * GGML_RESTRICT x0 = vx, * GGML_RESTRICT x1 = (const block_bposit8 *)((const uint8_t *) vx + bx);
    const ggml_bf16_t   * GGML_RESTRICT y0 = vy, * GGML_RESTRICT y1 = (const ggml_bf16_t *)((const uint8_t *) vy + by);
    float32x4_t a00 = vdupq_n_f32(0.0f), a10 = a00, a01 = a00, a11 = a00;
    for (int ib = 0; ib < nb; ib++) {
        ggml_bp8_bf32 bx0, bx1;
        ggml_bp8_bf16_block_neon(&th, &tl, x0[ib].qs, &bx0);
        ggml_bp8_bf16_block_neon(&th, &tl, x1[ib].qs, &bx1);
        const int sx0 = x0[ib].scale_exp, sx1 = x1[ib].scale_exp;
        a00 = ggml_bp8_dot_block_neon(a00, &bx0, y0 + ib * QK_BPOSIT8, sx0);
        a10 = ggml_bp8_dot_block_neon(a10, &bx1, y0 + ib * QK_BPOSIT8, sx1);
        a01 = ggml_bp8_dot_block_neon(a01, &bx0, y1 + ib * QK_BPOSIT8, sx0);
        a11 = ggml_bp8_dot_block_neon(a11, &bx1, y1 + ib * QK_BPOSIT8, sx1);
    }
    s[0] = vaddvq_f32(a00); s[1] = vaddvq_f32(a10);
    s[bs] = vaddvq_f32(a01); s[bs + 1] = vaddvq_f32(a11);
}
#endif // NEON BF16
#endif // NEON

// Dispatcher: the vector path where the build has it, the scalar reference otherwise or
// when GGML_BP8_SCALAR is set in the environment (A/B and gate runs).
void ggml_vec_dot_bposit8_bposit8(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1 || nrc == 2);
#if defined(GGML_BP8_HAVE_VEC)
    static int use_scalar = -1;
    if (use_scalar < 0) use_scalar = getenv("GGML_BP8_SCALAR") != NULL;
    if (!use_scalar) {
        assert(n % QK_BPOSIT8 == 0);
#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__)
        if (nrc == 2) { ggml_vec_dot_bposit8_bposit8_avx512_2x2(n, s, bs, vx, bx, vy, by); return; }
        ggml_vec_dot_bposit8_bposit8_avx512(n, s, vx, vy);
        return;
#else
        if (nrc == 1) { ggml_vec_dot_bposit8_bposit8_neon(n, s, vx, vy); return; }
        // NEON: no 2x2 tile yet; four single dots, same s layout
        ggml_vec_dot_bposit8_bposit8_neon(n, s,          vx, vy);
        ggml_vec_dot_bposit8_bposit8_neon(n, s + 1,      (const uint8_t *) vx + bx, vy);
        ggml_vec_dot_bposit8_bposit8_neon(n, s + bs,     vx, (const uint8_t *) vy + by);
        ggml_vec_dot_bposit8_bposit8_neon(n, s + bs + 1, (const uint8_t *) vx + bx, (const uint8_t *) vy + by);
        return;
#endif
    }
#endif
    if (nrc == 1) { ggml_vec_dot_bposit8_bposit8_scalar(n, s, 0, vx, 0, vy, 0, 1); return; }
    ggml_vec_dot_bposit8_bposit8_scalar(n, s,          0, vx, 0, vy, 0, 1);
    ggml_vec_dot_bposit8_bposit8_scalar(n, s + 1,      0, (const uint8_t *) vx + bx, 0, vy, 0, 1);
    ggml_vec_dot_bposit8_bposit8_scalar(n, s + bs,     0, vx, 0, (const uint8_t *) vy + by, 0, 1);
    ggml_vec_dot_bposit8_bposit8_scalar(n, s + bs + 1, 0, (const uint8_t *) vx + bx, 0, (const uint8_t *) vy + by, 0, 1);
}

// Fast dispatcher (the type's default vec_dot): BPOSIT8 weights x BF16 activations.
void ggml_vec_dot_bposit8_bf16(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1 || nrc == 2);
    assert(n % QK_BPOSIT8 == 0);
#if defined(__AVX512VBMI__) && defined(__AVX512BW__) && defined(__AVX512F__) && defined(GGML_BP8_HAVE_VEC)
    if (nrc == 2) { ggml_vec_dot_bposit8_bf16_avx512_2x2(n, s, bs, vx, bx, vy, by); return; }
    ggml_vec_dot_bposit8_bf16_avx512(n, s, vx, vy);
    return;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16) && defined(GGML_BP8_HAVE_VEC)
    if (nrc == 2) { ggml_vec_dot_bposit8_bf16_neon_2x2(n, s, bs, vx, bx, vy, by); return; }
    ggml_vec_dot_bposit8_bf16_neon(n, s, vx, vy);
    return;
#else
    if (nrc == 1) { ggml_vec_dot_bposit8_bf16_scalar(n, s, vx, vy); return; }
    ggml_vec_dot_bposit8_bf16_scalar(n, s,          vx, vy);
    ggml_vec_dot_bposit8_bf16_scalar(n, s + 1,      (const uint8_t *) vx + bx, vy);
    ggml_vec_dot_bposit8_bf16_scalar(n, s + bs,     vx, (const uint8_t *) vy + by);
    ggml_vec_dot_bposit8_bf16_scalar(n, s + bs + 1, (const uint8_t *) vx + bx, (const uint8_t *) vy + by);
#endif
}

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

        sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

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
}

void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
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
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
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
}

void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
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
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
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
}

void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
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
}

void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

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

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
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
