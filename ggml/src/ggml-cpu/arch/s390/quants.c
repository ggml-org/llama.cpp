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