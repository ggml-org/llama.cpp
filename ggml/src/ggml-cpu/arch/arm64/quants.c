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

#if defined(__ARM_NEON)
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

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
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
#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};


//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * GGML_RESTRICT vx0 = vx;
        const block_q4_0 * GGML_RESTRICT vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * GGML_RESTRICT vy0 = vy;
        const block_q8_0 * GGML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * GGML_RESTRICT b_x0 = &vx0[i];
            const block_q4_0 * GGML_RESTRICT b_x1 = &vx1[i];
            const block_q8_0 * GGML_RESTRICT b_y0 = &vy0[i];
            const block_q8_0 * GGML_RESTRICT b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt()*8;

    // VLA Implementation using switch case
    switch (vector_length) {
        case 128:
            {
                // predicate for activating higher lanes for 4 float32 elements
                const svbool_t ph4 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx0r, 0x0F));
                    const svint8_t qx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx0r, 0x04));
                    const svint8_t qx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx1r, 0x0F));
                    const svint8_t qx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx1r, 0x04));

                    // sub 8
                    const svint8_t qx0ls = svsub_n_s8_x(svptrue_b8(), qx0h, 8);
                    const svint8_t qx0hs = svsub_n_s8_x(svptrue_b8(), qx0l, 8);
                    const svint8_t qx1ls = svsub_n_s8_x(svptrue_b8(), qx1h, 8);
                    const svint8_t qx1hs = svsub_n_s8_x(svptrue_b8(), qx1l, 8);

                    // load y
                    const svint8_t qy0h = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy0l = svld1_s8(svptrue_b8(), y0->qs + 16);
                    const svint8_t qy1h = svld1_s8(svptrue_b8(), y1->qs);
                    const svint8_t qy1l = svld1_s8(svptrue_b8(), y1->qs + 16);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph4, sumv0, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx0ls, qy0l),
                                    svdot_s32(svdup_n_s32(0), qx0hs, qy0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph4, sumv1, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx1ls, qy1l),
                                    svdot_s32(svdup_n_s32(0), qx1hs, qy1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 256:
            {
                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for  16 int8 elements
                const svbool_t pl16 = svnot_b_z(svptrue_b8(), ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating higher lanes for 32 int8 elements
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);

                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
                const svbool_t pl16 = svnot_b_z(ph32, ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(ph32, x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(ph32, x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(ph32, qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(ph32, qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(ph32, y0->qs);
                    const svint8_t qy1 = svld1_s8(ph32, y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph32, sumv0, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph32, sumv1, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(ph32, svadd_f32_x(ph32, sumv0, sumv1));
            } break;
        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
        const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
        const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
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
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_1 * GGML_RESTRICT vx0 = vx;
        const block_q4_1 * GGML_RESTRICT vx1 = (const block_q4_1 *) ((const uint8_t*)vx + bx);
        const block_q8_1 * GGML_RESTRICT vy0 = vy;
        const block_q8_1 * GGML_RESTRICT vy1 = (const block_q8_1 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t summs0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_1 * GGML_RESTRICT b_x0 = &vx0[i];
            const block_q4_1 * GGML_RESTRICT b_x1 = &vx1[i];
            const block_q8_1 * GGML_RESTRICT b_y0 = &vy0[i];
            const block_q8_1 * GGML_RESTRICT b_y1 = &vy1[i];

            float32_t summs_t[4] = {
                GGML_FP16_TO_FP32(b_x0->m) * GGML_FP16_TO_FP32(b_y0->s),
                GGML_FP16_TO_FP32(b_x1->m) * GGML_FP16_TO_FP32(b_y0->s),
                GGML_FP16_TO_FP32(b_x0->m) * GGML_FP16_TO_FP32(b_y1->s),
                GGML_FP16_TO_FP32(b_x1->m) * GGML_FP16_TO_FP32(b_y1->s)
            };
            summs0 = vaddq_f32(summs0, vld1q_f32(summs_t));

            const uint8x16_t m4b = vdupq_n_u8(0x0F);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t x0_l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t x0_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t x1_l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t x1_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            // mmla into int32x4_t
            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        sumv2 = vaddq_f32(sumv2, summs0);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_1 * GGML_RESTRICT x0 = &x[ib + 0];
        const block_q4_1 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_1 * GGML_RESTRICT y0 = &y[ib + 0];
        const block_q8_1 * GGML_RESTRICT y1 = &y[ib + 1];

        summs += GGML_FP16_TO_FP32(x0->m) * GGML_FP16_TO_FP32(y0->s) + GGML_FP16_TO_FP32(x1->m) * GGML_FP16_TO_FP32(y1->s);

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

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

void ggml_vec_dot_q5_0_q8_0_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_0 * GGML_RESTRICT x0 = &x[ib];
        const block_q5_0 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * GGML_RESTRICT y0 = &y[ib];
        const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        // extract the 5th bit via lookup table ((!b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_1[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

#endif
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
}

void ggml_vec_dot_q5_1_q8_1_native(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs0 = 0.0f;
    float summs1 = 0.0f;

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_1 * GGML_RESTRICT x0 = &x[ib];
        const block_q5_1 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_1 * GGML_RESTRICT y0 = &y[ib];
        const block_q8_1 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        summs0 += GGML_FP16_TO_FP32(x0->m) * GGML_FP16_TO_FP32(y0->s);
        summs1 += GGML_FP16_TO_FP32(x1->m) * GGML_FP16_TO_FP32(y1->s);

        // extract the 5th bit via lookup table ((b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_0[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_0[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_0[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_0[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_0[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_0[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_0[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_0[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit
        const int8x16_t v0_0lf = vorrq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vorrq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vorrq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vorrq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;

#endif
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
}