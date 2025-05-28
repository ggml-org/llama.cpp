#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "../../ggml-cpu-quants.h"
#include "../../ggml-cpu-impl.h"

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

void quantize_row_q8_0_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__VXE__) || defined(__VXE2__)
    for (int i = 0; i < nb; i++) {
        __vector float srcv [8];
        __vector float asrcv[8];
        __vector float amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const __vector float v = vec_mul(srcv[j], vec_splats(id));
            const __vector int32_t vi = vec_signed(v);

            y[i].qs[4*j + 0] = vec_extract(vi, 0);
            y[i].qs[4*j + 1] = vec_extract(vi, 1);
            y[i].qs[4*j + 2] = vec_extract(vi, 2);
            y[i].qs[4*j + 3] = vec_extract(vi, 3);
        }
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__VXE__) || defined(__VXE2__)
    for (int i = 0; i < nb; i++) {
        __vector float srcv [8];
        __vector float asrcv[8];
        __vector float amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        __vector int32_t acc = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const __vector float v = vec_mul(srcv[j], vec_splats(id));
            const __vector int32_t vi = vec_signed(v);

            y[i].qs[4*j + 0] = vec_extract(vi, 0);
            y[i].qs[4*j + 1] = vec_extract(vi, 1);
            y[i].qs[4*j + 2] = vec_extract(vi, 2);
            y[i].qs[4*j + 3] = vec_extract(vi, 3);

            acc = vec_add(acc, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(d * (acc[0] + acc[1] + acc[2] + acc[3]));
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};


//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
    __vector float acc = vec_splats(0.0f);

    const __vector uint8_t v_m = vec_splats((const uint8_t)0x0F);
    const __vector int8_t  v_s = vec_splats( (const int8_t)0x08);

    for (; ib < nb; ++ib) {
        const __vector uint8_t v_x = vec_xl(0, x[ib].qs);
        const __vector int8_t v_xl = (const __vector int8_t)(v_x & v_m);
        const __vector int8_t v_xh = (const __vector int8_t)(v_x >> 4);

        const __vector int8_t v_xls = vec_sub(v_xl, v_s);
        const __vector int8_t v_xhs = vec_sub(v_xh, v_s);

        const __vector int8_t v_yl = vec_xl(0      , y[ib].qs);
        const __vector int8_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        const __vector int16_t v_xylso = vec_mulo(v_xls, v_yl);
        const __vector int16_t v_xylse = vec_mule(v_xls, v_yl);
        const __vector int16_t v_xyhso = vec_mulo(v_xhs, v_yh);
        const __vector int16_t v_xyhse = vec_mule(v_xhs, v_yh);

        __vector int16_t v_xy_ = v_xylso + v_xylse + v_xyhso + v_xyhse; v_xy_ += vec_reve(v_xy_);

        const __vector float v_xy = vec_float(vec_unpackh(v_xy_));
        const __vector float v_d = vec_splats(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3];

#endif
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
}

void ggml_vec_dot_q4_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
    float summs = 0;
    float32x4_t acc = vec_splats(0.0f);

    const uint8x16_t v_m = vec_splat_u8(0x0F);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        const uint8x16_t v_x = vec_xl(0, x[ib].qs);
        const int8x16_t v_xl = (const int8x16_t)(v_x & v_m);
        const int8x16_t v_xh = (const int8x16_t)(v_x >> 4);

        const int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        const int8x16_t v_yh = vec_xl(QK8_1/2, y[ib].qs);

        const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        const float32x4_t v_xy = vec_float(v_xy_);

        const float32x4_t v_d = vec_splats(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3] + summs;
    
#endif
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
}

void ggml_vec_dot_q8_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
    __vector float acc = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        const int8x16_t v_xl = vec_xl(0      , x[ib].qs);
        const int8x16_t v_xh = vec_xl(QK8_0/2, x[ib].qs);
        const int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        const int8x16_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        const float32x4_t v_xy = vec_float(v_xy_);
        const float32x4_t v_d = vec_splats(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3];

#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_q3_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
    uint32_t aux[3];
    uint32_t utmp[4];

    const int32x4_t v_z = vec_splat_s32(0);
    const uint8x16_t v_3m = vec_splat_u8(0x03);

    const uint8x16_t v_0c = vec_splat_u8(1);
    const uint8x16_t v_1c = vec_sl(v_0c, 1);
    const uint8x16_t v_2c = vec_sl(v_0c, 2);
    const uint8x16_t v_3c = vec_sl(v_0c, 3);

    uint8x16_t q3h[4];
    uint8x16_t q3b[2];
    int8x16_t q3bytes[4];
    int8x16_t q8bytes[4];
    uint8x16_t qhbits[2];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict x0l = x[i].qs;
        const uint8_t * restrict x0h = x[i].hmask;
        const int8_t  * restrict y0  = y[i].qs;

        qhbits[0] = vec_xl(0 , x0h);
        qhbits[1] = vec_xl(16, x0h);

        int32_t isum = 0;

        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= 32;

        for (int j = 0; j < QK_K/128; ++j) {
            int32x4_t isum0, isum1, isum2, isum3;

            q3b[0] = vec_xl(0 , x0l);
            q3b[1] = vec_xl(16, x0l);
            x0l += 32;

            q8bytes[0] = vec_xl(0  , y0);
            q8bytes[1] = vec_xl(16 , y0);
            q8bytes[2] = vec_xl(32 , y0);
            q8bytes[3] = vec_xl(48 , y0);
            q8bytes[4] = vec_xl(64 , y0);
            q8bytes[5] = vec_xl(80 , y0);
            q8bytes[6] = vec_xl(96 , y0);
            q8bytes[7] = vec_xl(112, y0);
            y0 += 128;

            q3h[0] = vec_sl(vec_andc(v_0c, qhbits[0]), 2);
            q3h[1] = vec_sl(vec_andc(v_0c, qhbits[1]), 2);
            q3h[2] = vec_sl(vec_andc(v_1c, qhbits[0]), 1);
            q3h[3] = vec_sl(vec_andc(v_1c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(q3b[0], v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(q3b[1], v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 2), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 2), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot(v_z, q3bytes[0], q8bytes[0]);
            isum1 = ggml_vec_dot(v_z, q3bytes[1], q8bytes[1]);
            isum2 = ggml_vec_dot(v_z, q3bytes[2], q8bytes[2]);
            isum3 = ggml_vec_dot(v_z, q3bytes[3], q8bytes[3]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            q3h[0] = vec_andc(v_2c, qhbits[0]);
            q3h[1] = vec_andc(v_2c, qhbits[1]);
            q3h[2] = vec_sr(vec_andc(v_3c, qhbits[0]), 1);
            q3h[3] = vec_sr(vec_andc(v_3c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 4), v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 4), v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 6), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 6), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot(v_z, q3bytes[0], q8bytes[4]);
            isum1 = ggml_vec_dot(v_z, q3bytes[1], q8bytes[5]);
            isum2 = ggml_vec_dot(v_z, q3bytes[2], q8bytes[6]);
            isum3 = ggml_vec_dot(v_z, q3bytes[3], q8bytes[7]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits[0] = vec_sr(qhbits[0], 4);
                qhbits[1] = vec_sr(qhbits[1], 4);
            }
        }

        sum += d * isum;
    }

    *s = sum;

#else
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

void ggml_vec_dot_q4_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
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

void ggml_vec_dot_q5_K_q8_K_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
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

#if defined(__VXE__) || defined(__VXE2__)
    const uint8x16_t v_lm = vec_splat_u8(0x0F);
    const uint8x16_t v_1m = vec_splat_u8(0x01);
    const uint8x16_t v_2m = vec_splat_u8(0x02);

    const int32x4_t v_z = vec_splat_s32(0);

    const uchar8x16_t v_minsm = {
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
    };

    int8x16_t  q5b[4];
    uint8x16_t q5h[4];

    uint8x16_t v_xl[2];
    uint8x16_t v_xh[2];
    int8x16_t  v_y[4];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        const int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x16_t v_mins16 = vec_xl(0, (const uint8_t *)utmp);
        const uint8x16_t v_mins8 = vec_perm(v_mins16, v_mins16, v_minsm);
        const int16x8_t v_minsh = (int16x8_t)vec_unpackh(v_mins8);

        const int32x4_t v_minsho = vec_mulo(v_ysums, v_minsh);
        const int32x4_t v_minshe = vec_mule(v_ysums, v_minsh);
        const int32x4_t v_mins = vec_add(v_minsho, v_minshe);
        const int32_t mins = v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3];

        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * GGML_RESTRICT x0l = x[i].qs;
        const uint8_t * GGML_RESTRICT x0h = x[i].qh;
        const int8_t  * GGML_RESTRICT y0 = y[i].qs;

        v_xh[0] = vec_xl(0 , x0h);
        v_xh[1] = vec_xl(16, x0h);

        int32_t sumi = 0;
        for (int j = 0; j < QK_K/64; ++j) {
            v_xl[0] = vec_xl(0 , x0l);
            v_xl[1] = vec_xl(16, x0l);
            x0l += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            q5h[0] = vec_sl(vec_and(v_1m, v_xh[0]), 4);
            q5h[1] = vec_sl(vec_and(v_1m, v_xh[1]), 4);
            q5h[2] = vec_sl(vec_and(v_2m, v_xh[0]), 3);
            q5h[3] = vec_sl(vec_and(v_2m, v_xh[1]), 3);
            v_xh[0] = vec_sr(v_xh[0], 2);
            v_xh[1] = vec_sr(v_xh[1], 2);

            q5b[0] = (int8x16_t)vec_or(vec_and(v_xl[0], v_lm), q5h[0]);
            q5b[1] = (int8x16_t)vec_or(vec_and(v_xl[1], v_lm), q5h[1]);
            q5b[2] = (int8x16_t)vec_or(vec_sr(v_xl[0], 4), q5h[2]);
            q5b[3] = (int8x16_t)vec_or(vec_sr(v_xl[1], 4), q5h[3]);

            int32x4_t sumi0 = ggml_vec_dot(ggml_vec_dot(v_z, q5b[0], v_y[0]), q5b[1], v_y[1]);
            int32x4_t sumi1 = ggml_vec_dot(ggml_vec_dot(v_z, q5b[2], v_y[2]), q5b[3], v_y[3]);

            sumi += (sumi0[0] + sumi0[1] + sumi0[2] + sumi0[3]) * *scales++;
            sumi += (sumi1[0] + sumi1[1] + sumi1[2] + sumi1[3]) * *scales++;
        }

        sumf += d * sumi - dmin * mins;
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

