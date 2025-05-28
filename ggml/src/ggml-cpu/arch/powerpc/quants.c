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

#if defined(__POWER9_VECTOR__)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

void quantize_row_q8_0_native(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
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

#if defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        vector int accv = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_FP32_TO_FP16(d * vec_extract(accv, 0));
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

#if defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector signed char v8 = vec_splats((signed char)0x8);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

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

#if defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[ib].m));
        vector float vys = {GGML_FP16_TO_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector unsigned char q4x0 = (vector unsigned char)vec_and(qxs, lowMask);
        vector unsigned char q4x1 = (vector unsigned char)vec_sr(qxs, v4);

        vector signed int vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);
    
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