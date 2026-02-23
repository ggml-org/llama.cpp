#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
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

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)

    size_t vl = QK8_0;

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)

    size_t vl = QK8_1;

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_1, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m2_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_CPU_FP32_TO_FP16(sum*d);
    }

#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__riscv_v)
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

    size_t vl = qk / 2;

    for (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        // subtract offset
        vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
        vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
#else
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__riscv_v)
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

    size_t vl = qk / 2;

    for (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
#else
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__riscv_v)
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

    size_t vl;
    size_t vlenb = __riscv_vlenb();

    for (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        if (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } else {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        qh = __riscv_vmnand_mm_b4(qh, qh, vl);
        vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
#else
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__riscv_v)
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

    size_t vl;
    size_t vlenb = __riscv_vlenb();

    for (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        if (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } else {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
#else
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined(__riscv_v)
    size_t vl = qk;

    for (; ib < nb; ++ib) {
        // load elements
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);

        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
#else

    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);

    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __riscv_xtheadvector

    float sumf = 0;
    uint8_t atmp[16];

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
        uint8_t *patmp = atmp;
        int vsums;
        int tmp;
        __asm__ __volatile__(
            "th.vsetvli zero, %[vl16], e8, m1\n\t"
            "th.vmv.v.x v8, zero\n\t"
            "th.vlb.v v1, (%[sc])\n\t"
            "th.vand.vi v0, v1, 0xF\n\t"
            "th.vsrl.vi v1, v1, 4\n\t"
            "th.vsb.v v0, (%[scale])\n\t"
            "th.vwaddu.vx v16, v1, zero\n\t"
            "th.vsetvli zero, %[vl16], e16, m2\n\t"
            "th.vlh.v v2, (%[bsums])\n\t"
            "th.vwmul.vv v4, v16, v2\n\t"
            "th.vsetvli zero, %[vl16], e32, m4\n\t"
            "th.vredsum.vs v8, v4, v8\n\t"
            "th.vmv.x.s %[vsums], v8"
            : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
            : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
            , [vl16] "r" (16)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += dmin * vsums;
        int isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v0, (%[q2])\n\t"
                "th.vsrl.vi v2, v0, 2\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vsrl.vi v6, v0, 6\n\t"
                "th.vand.vi v0, v0, 0x3\n\t"
                "th.vand.vi v2, v2, 0x3\n\t"
                "th.vand.vi v4, v4, 0x3\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "th.vsetvli zero, %[vl16], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlbu.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [isum] "+&r" (isum)
                : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                , [vl16] "r" (16), [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q2 += 32; q8 += 128; patmp += 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;

#elif defined __riscv_v

    float sumf = 0;
    uint8_t atmp[16];

    const int vector_length = __riscv_vlenb() * 8;
    uint8_t temp_01[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * q2 = x[i].qs;
            const int8_t *  q8 = y[i].qs;
            const uint8_t * sc = x[i].scales;

            const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            size_t vl = 16;

            vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
            vuint8m1_t aux    = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

            vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

            vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
            vuint8mf2_t mins8    = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
            vint16m1_t  mins     = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
            vint32m2_t  prod     = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
            vint32m1_t  vsums    = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

            sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

            vl = 32;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t v_b   = __riscv_vle8_v_u8m1(temp_01, vl);

            uint8_t is   = 0;
            int     isum = 0;

            for (int j = 0; j < QK_K / 128; ++j) {
                // load Q2
                vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

                vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
                vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
                vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
                vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);

                // duplicate scale elements for product
                vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
                vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
                vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
                vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);

                vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
                vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
                vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
                vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

                // load Q8
                vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
                vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
                vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

                vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
                vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
                vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
                vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

                isum += __riscv_vmv_x_s_i32m1_i32(isum1);

                q2 += 32;
                q8 += 128;
                is = 8;
            }

            sumf += dall * isum;
        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * q2 = x[i].qs;
            const  int8_t * q8 = y[i].qs;
            const uint8_t * sc = x[i].scales;
            const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
            uint8_t *patmp = atmp;
            int vsums;
            int tmp, t1, t2, t3, t4, t5, t6, t7;
            __asm__ __volatile__(
                "vsetivli zero, 16, e8, m1\n\t"
                "vmv.v.x v8, zero\n\t"
                "lb zero, 15(%[sc])\n\t"
                "vle8.v v1, (%[sc])\n\t"
                "vle8.v v2, (%[bsums])\n\t"
                "addi %[tmp], %[bsums], 16\n\t"
                "vand.vi v0, v1, 0xF\n\t"
                "vsrl.vi v1, v1, 4\n\t"
                "vle8.v v3, (%[tmp])\n\t"
                "vse8.v v0, (%[scale])\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vzext.vf2 v0, v1\n\t"
                "vwmul.vv v4, v0, v2\n\t"
                "vsetivli zero, 16, e32, m4\n\t"
                "vredsum.vs v8, v4, v8\n\t"
                "vmv.x.s %[vsums], v8"
                : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
                : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            sumf += dmin * vsums;
            int isum = 0;

            for (int j = 0; j < QK_K/128; ++j) {
                __asm__ __volatile__(
                    "lb zero, 31(%[q2])\n\t"
                    "addi %[tmp], %[q2], 16\n\t"
                    "addi %[t1], %[q8], 16\n\t"
                    "vsetivli zero, 16, e8, m1\n\t"
                    "vle8.v v0, (%[q2])\n\t"
                    "vle8.v v1, (%[tmp])\n\t"
                    "vsrl.vi v2, v0, 2\n\t"
                    "vsrl.vi v3, v1, 2\n\t"
                    "vsrl.vi v4, v0, 4\n\t"
                    "addi %[tmp], %[q8], 32\n\t"
                    "vle8.v v8, (%[q8])\n\t"
                    "vle8.v v9, (%[t1])\n\t"
                    "addi %[t1], %[t1], 32\n\t"
                    "vsrl.vi v5, v1, 4\n\t"
                    "vsrl.vi v6, v0, 6\n\t"
                    "vsrl.vi v7, v1, 6\n\t"
                    "vle8.v v10, (%[tmp])\n\t"
                    "vle8.v v11, (%[t1])\n\t"
                    "addi %[tmp], %[tmp], 32\n\t"
                    "addi %[t1], %[t1], 32\n\t"
                    "vand.vi v0, v0, 0x3\n\t"
                    "vand.vi v1, v1, 0x3\n\t"
                    "vand.vi v2, v2, 0x3\n\t"
                    "vle8.v v12, (%[tmp])\n\t"
                    "vle8.v v13, (%[t1])\n\t"
                    "addi %[tmp], %[tmp], 32\n\t"
                    "addi %[t1], %[t1], 32\n\t"
                    "vand.vi v3, v3, 0x3\n\t"
                    "vand.vi v4, v4, 0x3\n\t"
                    "vand.vi v5, v5, 0x3\n\t"
                    "vle8.v v14, (%[tmp])\n\t"
                    "vle8.v v15, (%[t1])\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v18, v1, v9\n\t"
                    "vwmul.vv v20, v2, v10\n\t"
                    "vwmul.vv v22, v3, v11\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vwmul.vv v26, v5, v13\n\t"
                    "vwmul.vv v28, v6, v14\n\t"
                    "vwmul.vv v30, v7, v15\n\t"
                    "vsetivli zero, 8, e16, m1\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "lbu %[tmp], 0(%[scale])\n\t"
                    "vwredsum.vs v8, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "lbu %[t1], 1(%[scale])\n\t"
                    "vwredsum.vs v10, v20, v0\n\t"
                    "vwredsum.vs v11, v22, v0\n\t"
                    "lbu %[t2], 2(%[scale])\n\t"
                    "vwredsum.vs v12, v24, v0\n\t"
                    "vwredsum.vs v13, v26, v0\n\t"
                    "lbu %[t3], 3(%[scale])\n\t"
                    "vwredsum.vs v14, v28, v0\n\t"
                    "vwredsum.vs v15, v30, v0\n\t"
                    "lbu %[t4], 4(%[scale])\n\t"
                    "vwredsum.vs v8, v17, v8\n\t"
                    "vwredsum.vs v9, v19, v9\n\t"
                    "lbu %[t5], 5(%[scale])\n\t"
                    "vwredsum.vs v10, v21, v10\n\t"
                    "vwredsum.vs v11, v23, v11\n\t"
                    "lbu %[t6], 6(%[scale])\n\t"
                    "vwredsum.vs v12, v25, v12\n\t"
                    "vwredsum.vs v13, v27, v13\n\t"
                    "lbu %[t7], 7(%[scale])\n\t"
                    "vwredsum.vs v14, v29, v14\n\t"
                    "vwredsum.vs v15, v31, v15\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vmul.vx v0, v8, %[tmp]\n\t"
                    "vmul.vx v1, v9, %[t1]\n\t"
                    "vmacc.vx v0, %[t2], v10\n\t"
                    "vmacc.vx v1, %[t3], v11\n\t"
                    "vmacc.vx v0, %[t4], v12\n\t"
                    "vmacc.vx v1, %[t5], v13\n\t"
                    "vmacc.vx v0, %[t6], v14\n\t"
                    "vmacc.vx v1, %[t7], v15\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "vmv.x.s %[t1], v1\n\t"
                    "add %[isum], %[isum], %[tmp]\n\t"
                    "add %[isum], %[isum], %[t1]"
                    : [tmp] "=&r" (tmp), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3)
                    , [t4] "=&r" (t4), [t5] "=&r" (t5), [t6] "=&r" (t6), [t7] "=&r" (t7)
                    , [isum] "+&r" (isum)
                    : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                q2 += 32; q8 += 128; patmp += 8;
            }

            sumf += dall * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);

    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined __riscv_xtheadvector

    uint32_t utmp[4];
    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;

        int8_t * scale = (int8_t *)utmp;
        int tmp;
        __asm__ __volatile__(
            "li %[tmp], 12\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vlb.v v0, (%[s6b])\n\t"
            "th.vmv.v.v v2, v0\n\t"
            "li %[tmp], 2\n\t"
            "th.vsetvli zero, %[tmp], e64, m1\n\t"
            "th.vmv.v.x v9, %[sh]\n\t"\
            "th.vslidedown.vi v1, v0, 1\n\t"
            "th.vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
            "th.vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
            "li %[tmp], 4\n\t"
            "th.vsetvli zero, %[tmp], e32, m1\n\t"
            "th.vid.v v9\n\t"
            "th.vmv.x.s %[tmp], v1\n\t"
            "th.vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
            "th.vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
            "th.vsrl.vv v4, v1, v9\n\t"
            "th.vsrl.vv v2, v0, v8\n\t"
            "th.vand.vx v5, v4, %[kmask1]\n\t"
            "th.vand.vx v3, v2, %[kmask2]\n\t"
            "th.vsll.vi v6, v5, 4\n\t"
            "th.vor.vv v7, v6, v3\n\t"
            "li %[tmp], 16\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vsub.vx v0, v7, %[c]\n\t"
            "th.vsb.v v0, (%[scale])"
            : [tmp] "=&r" (tmp)
            : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
            , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        uint8_t m = 1;
        int isum = 0;
        for (int j = 0; j < QK_K; j += 128) {
            __asm__ __volatile__(
                // fixme: use v0p7 mask layout directly
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v8, (%[q3])\n\t"
                "th.vsrl.vi v10, v8, 2\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vsrl.vi v14, v8, 6\n\t"
                "th.vand.vi v8, v8, 3\n\t"
                "th.vand.vi v10, v10, 3\n\t"
                "th.vand.vi v12, v12, 3\n\t"
                "th.vlb.v v2, (%[qh])\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v8, v8, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v10, v10, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v12, v12, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v14, v14, -4, v0.t\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[tmp], 16\n\t"
                "th.vsetvli zero, %[tmp], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlb.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [m] "+&r" (m), [isum] "+&r" (isum)
                : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q3 += 32;    q8 += 128;   scale += 8;
        }

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
    }

    *s = sumf;

#elif defined __riscv_v

    uint32_t utmp[4];
    float sumf = 0;
    uint32_t aux[3];
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            const uint8_t * GGML_RESTRICT q3 = x[i].qs;
            const uint8_t * GGML_RESTRICT qh = x[i].hmask;
            const  int8_t * GGML_RESTRICT q8 = y[i].qs;

            memcpy(aux, x[i].scales, 12);
            utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
            utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
            utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
            utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

            int8_t * scale = (int8_t *)utmp;
            for (int j = 0; j < 16; ++j) scale[j] -= 32;


            size_t vl = 32;
            uint8_t m =  1;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

            int sum_t = 0;

            for (int j = 0; j < QK_K; j += 128) {

                vl = 32;

                // load Q3
                vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

                vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
                vint8m1_t q3_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03 , vl));
                vint8m1_t q3_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03 , vl));
                vint8m1_t q3_3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03 , vl));

                // compute mask for subtraction
                vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
                vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
                vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
                vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
                vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
                m <<= 1;

                // load Q8 and take product with Q3
                vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
                vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
                vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
                vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

                vl = 16;

                // retrieve lane to multiply with scale
                vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
                vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
                vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
                vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
                vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
                vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
                vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
                vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
                vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
                vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

                sum_t +=  __riscv_vmv_x_s_i32m1_i32(isum3);

                q3 += 32;    q8 += 128;   scale += 8;

            }

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            sumf += d*sum_t;

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * restrict q3 = x[i].qs;
            const uint8_t * restrict qh = x[i].hmask;
            const  int8_t * restrict q8 = y[i].qs;

            int8_t * scale = (int8_t *)utmp;
            int tmp, t1, t2, t3, t4, t5, t6, t7;
            __asm__ __volatile__(
                "vsetivli zero, 12, e8, m1\n\t"
                "vle8.v v0, (%[s6b])\n\t"
                "vmv1r.v v2, v0\n\t"
                "vsetivli zero, 2, e64, m1\n\t"
                "vmv.v.x v9, %[sh]\n\t"\
                "vslidedown.vi v1, v0, 1\n\t"
                "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
                "vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
                "vsetivli zero, 4, e32, m1\n\t"
                "vid.v v9\n\t"
                "vmv.x.s %[tmp], v1\n\t"
                "vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
                "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
                "vsrl.vv v4, v1, v9\n\t"
                "vsrl.vv v2, v0, v8\n\t"
                "vand.vx v5, v4, %[kmask1]\n\t"
                "vand.vx v3, v2, %[kmask2]\n\t"
                "vsll.vi v6, v5, 4\n\t"
                "vor.vv v7, v6, v3\n\t"
                "vsetivli zero, 16, e8, m1\n\t"
                "vsub.vx v0, v7, %[c]\n\t"
                "vse8.v v0, (%[scale])"
                : [tmp] "=&r" (tmp)
                : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
                , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );

            uint8_t m = 1;
            int isum = 0;
            for (int j = 0; j < QK_K; j += 128) {
                __asm__ __volatile__(
                    "lb zero, 31(%[q3])\n\t"
                    "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                    "vle8.v v8, (%[q3])\n\t"
                    "vsrl.vi v10, v8, 2\n\t"
                    "vsrl.vi v12, v8, 4\n\t"
                    "vsrl.vi v14, v8, 6\n\t"
                    "lb zero, 64(%[q8])\n\t"
                    "vand.vi v8, v8, 3\n\t"
                    "vand.vi v10, v10, 3\n\t"
                    "vand.vi v12, v12, 3\n\t"
                    "vle8.v v2, (%[qh])\n\t"
                    "lb zero, 127(%[q8])\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v8, v8, -4, v0.t\n\t"
                    "lb zero, 0(%[q8])\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v10, v10, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v12, v12, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v14, v14, -4, v0.t\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v0, (%[q8])\n\t"
                    "lb %[tmp], 0(%[scale])\n\t"
                    "lb %[t1], 1(%[scale])\n\t"
                    "lb %[t2], 2(%[scale])\n\t"
                    "lb %[t3], 3(%[scale])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v8, v16, v0\n\t"
                    "lb %[t4], 4(%[scale])\n\t"
                    "lb %[t5], 5(%[scale])\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v10, v20, v0\n\t"
                    "vwredsum.vs v11, v22, v0\n\t"
                    "vwredsum.vs v12, v24, v0\n\t"
                    "lb %[t6], 6(%[scale])\n\t"
                    "lb %[t7], 7(%[scale])\n\t"
                    "vwredsum.vs v13, v26, v0\n\t"
                    "vwredsum.vs v14, v28, v0\n\t"
                    "vwredsum.vs v15, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vmul.vx v0, v8, %[tmp]\n\t"
                    "vmul.vx v1, v9, %[t1]\n\t"
                    "vmacc.vx v0, %[t2], v10\n\t"
                    "vmacc.vx v1, %[t3], v11\n\t"
                    "vmacc.vx v0, %[t4], v12\n\t"
                    "vmacc.vx v1, %[t5], v13\n\t"
                    "vmacc.vx v0, %[t6], v14\n\t"
                    "vmacc.vx v1, %[t7], v15\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "vmv.x.s %[t1], v1\n\t"
                    "add %[isum], %[isum], %[tmp]\n\t"
                    "add %[isum], %[isum], %[t1]"
                    : [tmp] "=&r" (tmp), [t1] "=&r" (t1), [t2] "=&r" (t2), [t3] "=&r" (t3)
                    , [t4] "=&r" (t4), [t5] "=&r" (t5), [t6] "=&r" (t6), [t7] "=&r" (t7)
                    , [m] "+&r" (m), [isum] "+&r" (isum)
                    : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                    , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                q3 += 32;    q8 += 128;   scale += 8;
            }

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
            sumf += d * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);

    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif

}

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#if defined __riscv_xtheadvector

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

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

            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

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
            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            float ftmp, ft2;
            const uint8_t * restrict q40;
            const uint8_t * restrict q41;
            const uint8_t * restrict q42;
            const uint8_t * restrict q43;
            const int8_t  * restrict q80;
            const int8_t  * restrict q81;
            const int8_t  * restrict q82;
            const int8_t  * restrict q83;
            int s0, s1, s2, s3;

            __asm__ __volatile__(
                "li %[s1], 8\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vle32.v v1, (%[s6b])\n\t"
                "vslide1down.vx v1, v1, zero\n\t"
                "vmv.v.x v16, zero\n\t"
                "vslidedown.vi v2, v1, 2\n\t"
                "vmv1r.v v3, v2\n\t"
                "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                "vsetivli zero, 2, e32, m1, ta, ma\n\t"
                "vmv.v.i v4, 4\n\t"
                "vand.vx v8, v1, %[kmask1]\n\t"
                "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                "vsrl.vi v6, v1, 6\n\t"
                "vsrl.vv v7, v2, v5\n\t"
                "vsse32.v v8, (%[utmp]), %[s1]\n\t"
                "vand.vx v0, v6, %[kmask3]\n\t"
                "vand.vx v2, v7, %[kmask2]\n\t"
                "vsll.vi v6, v0, 4\n\t"
                "addi %[s0], %[utmp], 4\n\t"
                "vor.vv v1, v6, v2\n\t"
                "vsse32.v v1, (%[s0]), %[s1]\n\t"
                "vsetivli zero, 8, e16, m1, ta, ma\n\t"
                "vle32.v v2, (%[bsums])\n\t"
                "vnsrl.wi v0, v2, 0\n\t"
                "vnsrl.wi v1, v2, 16\n\t"
                "vadd.vv v2, v0, v1\n\t"
                "vle8.v v3, (%[mins])\n\t"
                "vzext.vf2 v4, v3\n\t"
                "vwmul.vv v6, v4, v2\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vredsum.vs v0, v6, v16\n\t"
                "vredsum.vs v0, v7, v0\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfmv.f.s %[ftmp], v0\n\t"
                "vsetivli zero, 16, e8, m1, ta, ma\n\t"
                "vle8.v v0, (%[xs])\n\t"
                "fnmsub.s %[sumf], %[dmin], %[ftmp], %[sumf]\n\t"
                "addi %[q40], %[xs], 64\n\t"
                "addi %[q41], %[xs], 16\n\t"
                "addi %[q42], %[xs], 32\n\t"
                "addi %[q43], %[xs], 48\n\t"
                "addi %[q80], %[ys], 64\n\t"
                "vle8.v v1, (%[q41])\n\t"
                "vle8.v v2, (%[q42])\n\t"
                "addi %[q81], %[ys], 16\n\t"
                "addi %[q41], %[q41], 64\n\t"
                "addi %[q82], %[ys], 32\n\t"
                "vle8.v v3, (%[q43])\n\t"
                "vle8.v v8, (%[ys])\n\t"
                "addi %[q42], %[q42], 64\n\t"
                "addi %[q83], %[ys], 48\n\t"
                "addi %[q43], %[q43], 64\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "vle8.v v9, (%[q81])\n\t"
                "vle8.v v10, (%[q82])\n\t"
                "vand.vi v0, v0, 0xF\n\t"
                "addi %[q81], %[q81], 64\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "addi %[q82], %[q82], 64\n\t"
                "vle8.v v11, (%[q83])\n\t"
                "vle8.v v12, (%[q80])\n\t"
                "vand.vi v1, v1, 0xF\n\t"
                "addi %[q83], %[q83], 64\n\t"
                "vsrl.vi v6, v2, 4\n\t"
                "addi %[q80], %[q80], 64\n\t"
                "vle8.v v13, (%[q81])\n\t"
                "vle8.v v14, (%[q82])\n\t"
                "vand.vi v2, v2, 0xF\n\t"
                "addi %[q81], %[q81], 64\n\t"
                "vsrl.vi v7, v3, 4\n\t"
                "addi %[q82], %[q82], 64\n\t"
                "vwmul.vv v16, v0, v8\n\t"
                "vle8.v v15, (%[q83])\n\t"
                "vle8.v v0, (%[q40])\n\t"
                "vand.vi v3, v3, 0xF\n\t"
                "addi %[q83], %[q83], 64\n\t"
                "vwmul.vv v24, v2, v12\n\t"
                "vwmul.vv v20, v4, v10\n\t"
                "vwmul.vv v28, v6, v14\n\t"
                "vwmacc.vv v16, v1, v9\n\t"
                "vle8.v v1, (%[q41])\n\t"
                "vle8.v v2, (%[q42])\n\t"
                "vwmacc.vv v24, v3, v13\n\t"
                "vwmacc.vv v20, v5, v11\n\t"
                "vwmacc.vv v28, v7, v15\n\t"
                "addi %[q40], %[q80], 64\n\t"
                "addi %[q41], %[q81], 64\n\t"
                "vle8.v v3, (%[q43])\n\t"
                "vle8.v v8, (%[q80])\n\t"
                "addi %[q42], %[q82], 64\n\t"
                "addi %[q43], %[q83], 64\n\t"
                "vsrl.vi v4, v0, 4\n\t"
                "vle8.v v9, (%[q81])\n\t"
                "vle8.v v10, (%[q82])\n\t"
                "vand.vi v0, v0, 0xF\n\t"
                "vsrl.vi v5, v1, 4\n\t"
                "vsrl.vi v7, v3, 4\n\t"
                "vand.vi v3, v3, 0xF\n\t"
                "vle8.v v11, (%[q83])\n\t"
                "vle8.v v12, (%[q40])\n\t"
                "vand.vi v1, v1, 0xF\n\t"
                "vsrl.vi v6, v2, 4\n\t"
                "vand.vi v2, v2, 0xF\n\t"
                "vwmul.vv v18, v0, v8\n\t"
                "vle8.v v13, (%[q41])\n\t"
                "vle8.v v14, (%[q42])\n\t"
                "vwmul.vv v26, v2, v12\n\t"
                "vwmul.vv v22, v4, v10\n\t"
                "vwmul.vv v30, v6, v14\n\t"
                "vwmacc.vv v18, v1, v9\n\t"
                "vle8.v v15, (%[q43])\n\t"
                "vwmacc.vv v26, v3, v13\n\t"
                "vwmacc.vv v22, v5, v11\n\t"
                "vwmacc.vv v30, v7, v15\n\t"
                "vmv.v.x v0, zero\n\t"
                "vsetivli zero, 16, e16, m2, ta, ma\n\t"
                "vwredsum.vs v4, v16, v0\n\t"
                "lbu %[s0], 0(%[scale])\n\t"
                "vwredsum.vs v5, v20, v0\n\t"
                "lbu %[s1], 1(%[scale])\n\t"
                "vwredsum.vs v6, v24, v0\n\t"
                "lbu %[s2], 2(%[scale])\n\t"
                "vwredsum.vs v7, v28, v0\n\t"
                "lbu %[s3], 3(%[scale])\n\t"
                "vwredsum.vs v8, v18, v0\n\t"
                "lbu %[q40], 4(%[scale])\n\t"
                "vwredsum.vs v9, v22, v0\n\t"
                "lbu %[q41], 5(%[scale])\n\t"
                "vwredsum.vs v10, v26, v0\n\t"
                "lbu %[q42], 6(%[scale])\n\t"
                "vwredsum.vs v11, v30, v0\n\t"
                "lbu %[q43], 7(%[scale])\n\t"
                "vsetivli zero, 4, e32, m1, ta, ma\n\t"
                "vmul.vx v0, v4, %[s0]\n\t"
                "vmul.vx v1, v8, %[q40]\n\t"
                "vmacc.vx v0, %[s1], v5\n\t"
                "vmacc.vx v1, %[q41], v9\n\t"
                "vmacc.vx v0, %[s2], v6\n\t"
                "vmacc.vx v1, %[q42], v10\n\t"
                "vmacc.vx v0, %[s3], v7\n\t"
                "vmacc.vx v1, %[q43], v11\n\t"
                "vfcvt.f.x.v v0, v0\n\t"
                "vfcvt.f.x.v v1, v1\n\t"
                "vfmv.f.s %[ft2], v0\n\t"
                "vfmv.f.s %[ftmp], v1\n\t"
                "fadd.s %[ft2], %[ft2], %[ftmp]\n\t"
                "fmadd.s %[sumf], %[d], %[ft2], %[sumf]"
                : [ftmp] "=&f" (ftmp), [sumf] "+&f" (sumf), [ft2] "=&f" (ft2)
                , [s0] "=&r" (s0), [s1] "=&r" (s1), [s2] "=&r" (s2), [s3] "=&r" (s3)
                , [q40] "=&r" (q40), [q41] "=&r" (q41), [q42] "=&r" (q42), [q43] "=&r" (q43)
                , [q80] "=&r" (q80), [q81] "=&r" (q81), [q82] "=&r" (q82), [q83] "=&r" (q83)
                : [d] "f" (d), [ys] "r" (y[i].qs), [xs] "r" (x[i].qs), [scale] "r" (scales)
                , [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
                , [s6b] "r" (&x[i]), [kmask1] "r" (kmask1), [dmin] "f" (dmin)
                , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    UNUSED(x);
    UNUSED(y);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(nb);
    UNUSED(utmp);

    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
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

#if defined __riscv_v

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    float sums = 0.0;

    size_t vl;

    for (int i = 0; i < nb; ++i) {

        vl = 8;

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;

        vint16m1_t q8sums_0 = __riscv_vlse16_v_i16m1(y[i].bsums, 4, vl);
        vint16m1_t q8sums_1 = __riscv_vlse16_v_i16m1(y[i].bsums+1, 4, vl);
        vint16m1_t q8sums = __riscv_vadd_vv_i16m1(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf2_t mins8 = __riscv_vle8_v_u8mf2(mins, vl);
        vint16m1_t v_mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        vl = 32;
        int32_t aux32 = 0;
        int is = 0;

        uint8_t m = 1;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m2_t vqh = __riscv_vle8_v_u8m2(hm, vl);

        for (int j = 0; j < QK_K/64; ++j) {
            // load Q5 and Q8
            vuint8m2_t q5_x = __riscv_vle8_v_u8m2(q5, vl);
            vint8m2_t  q8_y1 = __riscv_vle8_v_i8m2(q8, vl);
            vint8m2_t  q8_y2 = __riscv_vle8_v_i8m2(q8+32, vl);

            // compute mask for addition
            vint8m2_t q5_a = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(q5_x, 0x0F, vl));
            vuint8m2_t qh_m1 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_1 = __riscv_vmsne_vx_u8m2_b4(qh_m1, 0, vl);
            vint8m2_t q5_m1 = __riscv_vadd_vx_i8m2_mu(vmask_1, q5_a, q5_a, 16, vl);
            m <<= 1;

            vint8m2_t q5_l = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(q5_x, 0x04, vl));
            vuint8m2_t qh_m2 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_2 = __riscv_vmsne_vx_u8m2_b4(qh_m2, 0, vl);
            vint8m2_t q5_m2 = __riscv_vadd_vx_i8m2_mu(vmask_2, q5_l, q5_l, 16, vl);
            m <<= 1;

            vint16m4_t v0 = __riscv_vwmul_vv_i16m4(q5_m1, q8_y1, vl);
            vint16m4_t v1 = __riscv_vwmul_vv_i16m4(q5_m2, q8_y2, vl);

            vint32m8_t vs1 = __riscv_vwmul_vx_i32m8(v0, scales[is++], vl);
            vint32m8_t vs2 = __riscv_vwmul_vx_i32m8(v1, scales[is++], vl);

            vint32m1_t vacc1 = __riscv_vredsum_vs_i32m8_i32m1(vs1, vzero, vl);
            vint32m1_t vacc2 = __riscv_vredsum_vs_i32m8_i32m1(vs2, vacc1, vl);

            aux32 += __riscv_vmv_x_s_i32m1_i32(vacc2);
            q5 += 32;    q8 += 64;

        }

        sums += aux32 * d;

    }

    *s = sumf+sums;

#else

    UNUSED(x);
    UNUSED(y);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(nb);
    UNUSED(utmp);

    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __riscv_xtheadvector

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int sum_t = 0;
        int t0;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t" // vl == 32
                "th.vlb.v v4, (%[qh])\n\t"
                "th.vsll.vi v0, v4, 4\n\t"
                "th.vsll.vi v2, v4, 2\n\t"
                "th.vsrl.vi v6, v4, 2\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vlb.v v8, (%[q6])\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vand.vi v8, v8, 0xF\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t" // vl == 128
                "th.vand.vx v0, v0, %[mask]\n\t"
                "th.vor.vv v8, v8, v0\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsub.vx v8, v8, %[vl32]\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[t0], 16\n\t"
                "th.vsetvli zero, %[t0], e16, m2\n\t" // vl == 16
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[t0], 4\n\t"
                "th.vsetvli zero, %[t0], e32, m1\n\t" // vl == 4
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[t0], 8\n\t"
                "th.vsetvli zero, %[t0], e32, m2\n\t" // vl == 8
                "th.vlb.v v4, (%[scale])\n\t"
                "th.vmul.vv v2, v4, v10\n\t"
                "th.vredsum.vs v0, v2, v0\n\t"
                "th.vmv.x.s %[t0], v0\n\t"
                "add %[sumi], %[sumi], %[t0]"
                : [sumi] "+&r" (sum_t), [t0] "=&r" (t0)
                : [qh] "r" (qh), [q6] "r" (q6), [q8] "r" (q8), [scale] "r" (scale)
                , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                , [mask] "r" (0x30)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q6 += 64;   qh += 32;   q8 += 128;   scale += 8;
        }

        sumf += d * sum_t;

    }

    *s = sumf;

#elif defined __riscv_v

    float sumf = 0;
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            const uint8_t * GGML_RESTRICT q6 = x[i].ql;
            const uint8_t * GGML_RESTRICT qh = x[i].qh;
            const  int8_t * GGML_RESTRICT q8 = y[i].qs;

            const int8_t * GGML_RESTRICT scale = x[i].scales;

            size_t vl;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

            int sum_t = 0;
            int is = 0;

            for (int j = 0; j < QK_K/128; ++j) {

                vl = 32;

                // load qh
                vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

                // load Q6
                vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
                vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6+32, vl);

                vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
                vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
                vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
                vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

                vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
                vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03 , vl);
                vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03 , vl);
                vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03 , vl);

                vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
                vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
                vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
                vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

                vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
                vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
                vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
                vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

                // load Q8 and take product
                vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
                vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
                vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
                vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

                vl = 16;

                vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is+0], vl);
                vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is+1], vl);
                vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is+2], vl);
                vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is+3], vl);
                vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is+4], vl);
                vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is+5], vl);
                vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is+6], vl);
                vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is+7], vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
                vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
                vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

                sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

                q6 += 64;   qh += 32;   q8 += 128;   is=8;

            }

            sumf += d * sum_t;

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {

            __builtin_prefetch(&x[i + 1].d, 0, 1);

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            const uint8_t * restrict q6 = x[i].ql;
            const uint8_t * restrict qh = x[i].qh;
            const  int8_t * restrict q8 = y[i].qs;

            const int8_t * restrict scale = x[i].scales;

            int q6h;
            float ftmp;

            for (int j = 0; j < QK_K/128; ++j) {
                __asm__ __volatile__(
                    "addi %[q6h], %[q6], 32\n\t"
                    "ld t0, 0(%[scale])\n\t"
                    "addi %[scale], %[scale], 8\n\t"
                    "slli t6, t0, 1 * 8\n\t"
                    "lb zero, 0(%[q6])\n\t"
                    "slli t5, t0, 2 * 8\n\t"
                    "slli t4, t0, 3 * 8\n\t"
                    "lb zero, 0(%[q6h])\n\t"
                    "slli t3, t0, 4 * 8\n\t"
                    "slli t2, t0, 5 * 8\n\t"
                    "lb zero, 0(%[qh])\n\t"
                    "lb zero, 31(%[q6h])\n\t"
                    "slli t1, t0, 6 * 8\n\t"
                    "srai a7, t0, 56\n\t"
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vle8.v v8, (%[q6])\n\t"
                    "srai t6, t6, 56\n\t"
                    "srai t5, t5, 56\n\t"
                    "srai t4, t4, 56\n\t"
                    "srai t3, t3, 56\n\t"
                    "vle8.v v10, (%[q6h])\n\t"
                    "addi %[q6], %[q6], 64\n\t"
                    "slli t0, t0, 7 * 8\n\t"
                    "srai t2, t2, 56\n\t"
                    "srai t1, t1, 56\n\t"
                    "srai t0, t0, 56\n\t"
                    "vle8.v v4, (%[qh])\n\t"
                    "vsrl.vi v12, v8, 4\n\t"
                    "vsrl.vi v14, v10, 4\n\t"
                    "lb zero, 0(%[q8])\n\t"
                    "vand.vi v8, v8, 0xF\n\t"
                    "vand.vi v10, v10, 0xF\n\t"
                    "lb zero, 32(%[q8])\n\t"
                    "vsll.vi v0, v4, 4\n\t"
                    "vsll.vi v2, v4, 2\n\t"
                    "lb zero, 64(%[q8])\n\t"
                    "vsrl.vi v6, v4, 2\n\t"
                    "vand.vx v0, v0, %[mask]\n\t"
                    "lb zero, 96(%[q8])\n\t"
                    "vand.vx v2, v2, %[mask]\n\t"
                    "vand.vx v4, v4, %[mask]\n\t"
                    "vand.vx v6, v6, %[mask]\n\t"
                    "vor.vv v8, v8, v0\n\t"
                    "lb zero, 127(%[q8])\n\t"
                    "vor.vv v10, v10, v2\n\t"
                    "vor.vv v12, v12, v4\n\t"
                    "vor.vv v14, v14, v6\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v0, (%[q8])\n\t"
                    "vsub.vx v8, v8, %[vl32]\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v10, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v8, v20, v0\n\t"
                    "vwredsum.vs v7, v22, v0\n\t"
                    "vwredsum.vs v11, v24, v0\n\t"
                    "vwredsum.vs v12, v26, v0\n\t"
                    "vwredsum.vs v13, v28, v0\n\t"
                    "vwredsum.vs v14, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vmul.vx v0, v10, t0\n\t"
                    "vmul.vx v1, v9, t1\n\t"
                    "vmacc.vx v0, t2, v8\n\t"
                    "vmacc.vx v1, t3, v7\n\t"
                    "vmacc.vx v0, t4, v11\n\t"
                    "vmacc.vx v1, t5, v12\n\t"
                    "vmacc.vx v0, t6, v13\n\t"
                    "vmacc.vx v1, a7, v14\n\t"
                    "vadd.vv v0, v0, v1\n\t"
                    "vfcvt.f.x.v v0, v0\n\t"
                    "vfmv.f.s %[ftmp], v0\n\t"
                    "fmadd.s %[sumf], %[d], %[ftmp], %[sumf]"
                    : [q6] "+&r" (q6), [q6h] "=&r" (q6h)
                    , [scale] "+&r" (scale)
                    , [sumf] "+&f" (sumf), [ftmp] "=&f" (ftmp)
                    : [qh] "r" (qh), [q8] "r" (q8)
                    , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                    , [mask] "r" (0x30), [d] "f" (d)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                    , "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7"
                    , "a6", "a5", "a4", "a3"
                );
                qh += 32;   q8 += 128;
            }
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);

    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static const uint8_t sign_gather_indices_arr[64] = {
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
};

static const uint8_t sign_bit_masks_arr[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
};

static void ggml_vec_dot_iq2_s_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    const uint64_t * grid64 = (const uint64_t *)iq2s_grid;

    // --- Pre-load Constants ---
    uint16_t gather_qh_arr[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 8);
    uint16_t shift_qh_arr[8] = {11, 9, 7, 5, 11, 9, 7, 5};
    vuint16mf2_t v_shift_qh = __riscv_vle16_v_u16mf2(shift_qh_arr, 8);

    // Constants for sign extraction
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float combined_scale = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT scales = x[i].scales;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const uint8_t * signs_ptr = qs + 32;

        float sum_block = 0.0f;

        for (int ib = 0; ib < 4; ++ib) {
            // Combine low + high bits
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 8);
            qs += 8;
            uint16_t qh_val;
            memcpy(&qh_val, qh, 2);
            qh += 2;
            vuint8mf8_t v_qh_raw = __riscv_vle8_v_u8mf8((const uint8_t*)&qh_val, 2);
            vuint16mf4_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf4(v_qh_raw, 2);
            vuint16mf2_t v_qh_u16_ext = __riscv_vlmul_ext_v_u16mf4_u16mf2(v_qh_u16);
            vuint16mf2_t v_qh_expanded = __riscv_vrgather_vv_u16mf2(v_qh_u16_ext, v_gather_qh, 8);
            v_qh_expanded = __riscv_vsll_vv_u16mf2(v_qh_expanded, v_shift_qh, 8);

            // Mask: We want bits 11-12. 0x1800 = 0001 1000 0000 0000
            v_qh_expanded = __riscv_vand_vx_u16mf2(v_qh_expanded, 0x1800, 8);
            vuint16mf2_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 8);

            // Multiply by 8 to get byte offset, instead of element offset
            v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16, 3, 8);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_expanded, 8);

            // Lookup Grid using Byte Offsets
            vuint64m2_t v_grid_vals = __riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 8);

            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u64m2_u8m2(v_grid_vals);
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_grid_u8);

            // Load signs and generate sign mask
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 8);
            signs_ptr += 8;

            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 64);

            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 64);

            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 0), v_zero, 16));
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 1), v_zero, 16));
            int32_t s2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 2), v_zero, 16));
            int32_t s3 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m1_i32m1(
                __riscv_vget_v_i16m4_i16m1(v_dot, 3), v_zero, 16));

            uint8_t sc0 = scales[0];
            uint8_t sc1 = scales[1];
            scales += 2;

            sum_block += s0 * (2 * (sc0 & 0xF) + 1);
            sum_block += s1 * (2 * (sc0 >> 4)  + 1);
            sum_block += s2 * (2 * (sc1 & 0xF) + 1);
            sum_block += s3 * (2 * (sc1 >> 4)  + 1);
        }
        sumf += sum_block * combined_scale;
    }
    *s = 0.125f * sumf;
}

static void ggml_vec_dot_iq2_s_q8_K_vl128(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    const uint64_t * grid64 = (const uint64_t *)iq2s_grid;

    // Pre-load Constants
    vuint8m2_t v_ids = __riscv_vid_v_u8m2(32);
    vuint8m2_t v_sign_gather_indices = __riscv_vsrl_vx_u8m2(v_ids, 3, 32);
    vuint8m2_t v_ones = __riscv_vmv_v_x_u8m2(1, 32);
    vuint8m2_t v_shift_amts = __riscv_vand_vx_u8m2(v_ids, 7, 32);
    vuint8m2_t v_sign_masks = __riscv_vsll_vv_u8m2(v_ones, v_shift_amts, 32);
    uint16_t shift_qh_arr[4] = {11, 9, 7, 5};
    vuint16mf2_t v_shift_qh = __riscv_vle16_v_u16mf2(shift_qh_arr, 4);

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float combined_scale = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT scales = x[i].scales;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const uint8_t * signs_ptr = qs + 32;
        float sum_block = 0.0f;

        for (int ib = 0; ib < 8; ++ib) {

            // Load Low Bits [4 bytes]
            vuint8mf4_t v_qs_u8 = __riscv_vle8_v_u8mf4(qs, 4);
            qs += 4;

            // Load 1 byte. It contains bits for 4 mini-blocks.
            uint8_t qh_val = *qh++;

            // Combine Low + High bits of 10bit indices
            vuint8mf4_t v_qh_raw = __riscv_vmv_v_x_u8mf4(qh_val, 4);
            vuint16mf2_t v_qh_u16 = __riscv_vwcvtu_x_x_v_u16mf2(v_qh_raw, 4);
            vuint16mf2_t v_qh_mf2 = __riscv_vsll_vv_u16mf2(v_qh_u16, v_shift_qh, 4);
            v_qh_mf2 = __riscv_vand_vx_u16mf2(v_qh_mf2, 0x1800, 4);
            vuint16mf2_t v_qs_u16_mf2 = __riscv_vwcvtu_x_x_v_u16mf2(v_qs_u8, 4);
            vuint16mf2_t v_qs_u16 = __riscv_vsll_vx_u16mf2(v_qs_u16_mf2, 3, 4);
            vuint16mf2_t v_grid_offsets = __riscv_vor_vv_u16mf2(v_qs_u16, v_qh_mf2, 4);

            // Lookup Grid
            vint8m2_t v_grid_i8 = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u64m2_u8m2(__riscv_vluxei16_v_u64m2(grid64, v_grid_offsets, 4)));

            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs_ptr, 4);
            signs_ptr += 4;
            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 32);

            // generating sign mask
            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 32);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 32);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 32);
            q8 += 32;

            // apply signs
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative,v_q8, v_q8, 0, 32);
            vint16m4_t v_dot = __riscv_vwmul_vv_i16m4(v_grid_i8, v_q8_signed, 32);

            // Reduction
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            // Reduce 0-15 (First Half)
            int32_t s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 0), v_zero, 16));

            // Reduce 16-31 (Second Half)
            int32_t s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(
                __riscv_vget_v_i16m4_i16m2(v_dot, 1), v_zero, 16));

            // Apply sub Scales
            uint8_t sc = *scales++;

            sum_block += s0 * (2 * (sc & 0xF) + 1);
            sum_block += s1 * (2 * (sc >> 4)  + 1);
        }
        sumf += sum_block * combined_scale;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 128:
            ggml_vec_dot_iq2_s_q8_K_vl128(n, s, bs, vx, bx, vy, by, nrc);
            break;
        case 256:
            ggml_vec_dot_iq2_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static void ggml_vec_dot_iq3_s_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint64_t * grid64 = (const uint64_t *)iq3s_grid;

    // --- Pre-load Constants ---
    const uint16_t qh_bit_shifts_arr[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    vuint8m2_t v_sign_gather_indices = __riscv_vle8_v_u8m2(sign_gather_indices_arr, 64);
    vuint8m2_t v_sign_masks = __riscv_vle8_v_u8m2(sign_bit_masks_arr, 64);
    vuint16m1_t v_qh_shifts = __riscv_vle16_v_u16m1(qh_bit_shifts_arr, 16);

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float combined_scale = d * y[i].d;

        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT scales = x[i].scales;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        float sum_block = 0.0f;

        // Loop: Process 64 weights (16 mini-blocks of 4) per iteration
        for (int ib = 0; ib < 4; ++ib) {

            vuint8mf2_t v_qs_u8 = __riscv_vle8_v_u8mf2(qs, 16);
            qs += 16;

            uint16_t qh_val;
            memcpy(&qh_val, qh, 2);
            qh += 2;

            vuint16m1_t v_qh_val = __riscv_vmv_v_x_u16m1(qh_val, 16);
            // Extract bits: (qh >> i) & 1
            v_qh_val = __riscv_vsrl_vv_u16m1(v_qh_val, v_qh_shifts, 16);
            v_qh_val = __riscv_vand_vx_u16m1(v_qh_val, 1, 16);

            vuint16m1_t v_qs_u16 = __riscv_vwcvtu_x_x_v_u16m1(v_qs_u8, 16);
            v_qs_u16 = __riscv_vsll_vx_u16m1(v_qs_u16, 2, 16);
            v_qh_val = __riscv_vsll_vx_u16m1(v_qh_val, 10, 16);
            vuint16m1_t v_grid_offsets = __riscv_vor_vv_u16m1(v_qs_u16, v_qh_val, 16);

            // Grid value is 4xuint8
            vuint32m2_t v_grid_packed = __riscv_vluxei16_v_u32m2((const uint32_t *)grid64, v_grid_offsets, 16);
            vuint8m2_t v_grid_u8 = __riscv_vreinterpret_v_u32m2_u8m2(v_grid_packed);
            vuint8mf4_t v_signs_raw = __riscv_vle8_v_u8mf4(signs, 8);
            signs += 8;

            // Generate sign mask
            vuint8m2_t v_signs_source = __riscv_vlmul_ext_v_u8mf4_u8m2(v_signs_raw);
            vuint8m2_t v_signs_bcast = __riscv_vrgather_vv_u8m2(v_signs_source, v_sign_gather_indices, 64);
            vuint8m2_t v_sign_bits = __riscv_vand_vv_u8m2(v_signs_bcast, v_sign_masks, 64);
            vbool4_t m_negative = __riscv_vmsne_vx_u8m2_b4(v_sign_bits, 0, 64);

            vint8m2_t v_q8 = __riscv_vle8_v_i8m2(q8, 64);
            q8 += 64;

            // Apply Signs
            vint8m2_t v_q8_signed = __riscv_vrsub_vx_i8m2_mu(m_negative, v_q8, v_q8, 0, 64);
            vint16m4_t v_dot = __riscv_vwmulsu_vv_i16m4(v_q8_signed, v_grid_u8, 64);

            // Reduction
            vint16m2_t v_dot_lo = __riscv_vget_v_i16m4_i16m2(v_dot, 0);
            vint16m2_t v_dot_hi = __riscv_vget_v_i16m4_i16m2(v_dot, 1);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

            int32_t s_lo = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_lo, v_zero, 32));
            int32_t s_hi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(v_dot_hi, v_zero, 32));

            // Apply sub-scales
            uint8_t sc_byte = *scales++;
            int sc_lo = (sc_byte & 0xF) * 2 + 1;
            int sc_hi = (sc_byte >> 4)  * 2 + 1;

            sum_block += s_lo * sc_lo + s_hi * sc_hi;
        }
        sumf += sum_block * combined_scale;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 256:
            ggml_vec_dot_iq3_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static void ggml_vec_dot_tq1_0_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.0f;
    uint8_t pow[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    for (int i = 0; i < nb; i++) {
        // First loop.
        vint32m4_t suml1;
        {
            const int vl = 32;
            vuint8m1_t tq = __riscv_vle8_v_u8m1(x[i].qs, vl);

            vuint16m2_t tq0 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(tq, 3, vl), 8, vl);
            vuint16m2_t tq1 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 3, vl), 3, vl), 8, vl);
            vuint16m2_t tq2 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 9, vl), 3, vl), 8, vl);
            vuint16m2_t tq3 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 27, vl), 3, vl), 8, vl);
            vuint16m2_t tq4 = __riscv_vsrl_vx_u16m2(__riscv_vwmulu_vx_u16m2(__riscv_vmul_vx_u8m1(tq, 81, vl), 3, vl), 8, vl);

            vint16m2_t q80 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 0, vl), vl);
            vint16m2_t q81 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 32, vl), vl);
            vint16m2_t q82 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 64, vl), vl);
            vint16m2_t q83 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 96, vl), vl);
            vint16m2_t q84 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(y[i].qs + 128, vl), vl);

            vint16m2_t sum0 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq0, 1, vl)), q80, vl);
            vint16m2_t sum1 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq1, 1, vl)), q81, vl);
            vint16m2_t sum2 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq2, 1, vl)), q82, vl);
            vint16m2_t sum3 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq3, 1, vl)), q83, vl);
            vint16m2_t sum4 = __riscv_vmul_vv_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsub_vx_u16m2(tq4, 1, vl)), q84, vl);

            vint32m4_t sumi0 = __riscv_vwadd_vv_i32m4(sum0, sum1, vl);
            vint32m4_t sumi1 = __riscv_vwadd_vv_i32m4(sum2, sum3, vl);
            suml1 = __riscv_vadd_vv_i32m4(__riscv_vwcvt_x_x_v_i32m4(sum4, vl), __riscv_vadd_vv_i32m4(sumi0, sumi1, vl), vl);
        }

        // Second loop.
        vint32m2_t suml2;
        {
            const int vl = 16;
            vuint8mf2_t tq = __riscv_vle8_v_u8mf2(x[i].qs + 32, vl);

            vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(tq, 3 * 1, vl), 8, vl);
            vuint16m1_t tq1 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 3, vl), 3, vl), 8, vl);
            vuint16m1_t tq2 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 9, vl), 3, vl), 8, vl);
            vuint16m1_t tq3 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 27, vl), 3, vl), 8, vl);
            vuint16m1_t tq4 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vx_u8mf2(tq, 81, vl), 3, vl), 8, vl);

            vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 160, vl), vl);
            vint16m1_t q81 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 176, vl), vl);
            vint16m1_t q82 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 192, vl), vl);
            vint16m1_t q83 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 208, vl), vl);
            vint16m1_t q84 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 224, vl), vl);

            vint16m1_t sum0 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
            vint16m1_t sum1 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq1, 1, vl)), q81, vl);
            vint16m1_t sum2 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq2, 1, vl)), q82, vl);
            vint16m1_t sum3 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq3, 1, vl)), q83, vl);
            vint16m1_t sum4 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq4, 1, vl)), q84, vl);

            vint32m2_t sumi0 = __riscv_vwadd_vv_i32m2(sum0, sum1, vl);
            vint32m2_t sumi1 = __riscv_vwadd_vv_i32m2(sum2, sum3, vl);
            suml2 = __riscv_vadd_vv_i32m2(__riscv_vwcvt_x_x_v_i32m2(sum4, vl), __riscv_vadd_vv_i32m2(sumi0, sumi1, vl), vl);
        }

        // Third loop.
        vint32m2_t suml3;
        {
            const int vl = 16;

            uint32_t qh;
            memcpy(&qh, &x[i].qh[0], 4);
            // Prevent fusion with vmv.
            __asm__ __volatile__("" : "+r"(qh));
            vuint8mf2_t tq = __riscv_vreinterpret_v_u32mf2_u8mf2(__riscv_vmv_v_x_u32mf2(qh, vl / 4));

            vuint8mf2_t p = __riscv_vle8_v_u8mf2(pow, vl);

            vuint16m1_t tq0 = __riscv_vsrl_vx_u16m1(__riscv_vwmulu_vx_u16m1(__riscv_vmul_vv_u8mf2(tq, p, vl), 3, vl), 8, vl);

            vint16m1_t q80 = __riscv_vwcvt_x_x_v_i16m1(__riscv_vle8_v_i8mf2(y[i].qs + 240, vl), vl);

            vint16m1_t sum0 = __riscv_vmul_vv_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsub_vx_u16m1(tq0, 1, vl)), q80, vl);
            suml3 = __riscv_vwcvt_x_x_v_i32m2(sum0, vl);
        }

        vint32m2_t sumb = __riscv_vadd_vv_i32m2(__riscv_vget_v_i32m4_i32m2(suml1, 0), __riscv_vget_v_i32m4_i32m2(suml1, 1), 16);
        sumb = __riscv_vadd_vv_i32m2(sumb, suml2, 16);
        sumb = __riscv_vadd_vv_i32m2(sumb, suml3, 16);

        vint32m1_t sum = __riscv_vredsum_vs_i32m2_i32m1(sumb, __riscv_vmv_v_x_i32m1(0, 1), 16);
        sumf += __riscv_vmv_x_s_i32m1_i32(sum) * y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 256:
            ggml_vec_dot_tq1_0_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static void ggml_vec_dot_tq2_0_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
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

        for (size_t j = 0; j < sizeof(x[0].qs); j += 32) {
            const int8_t * py0 = &y[i].qs[j * 4 + 0 * 32];
            const int8_t * py1 = &y[i].qs[j * 4 + 1 * 32];
            const int8_t * py2 = &y[i].qs[j * 4 + 2 * 32];
            const int8_t * py3 = &y[i].qs[j * 4 + 3 * 32];
            const uint8_t* px  = &x[i].qs[j];

            size_t vlmax_16m2 = __riscv_vsetvl_e16m2(32);
            vint16m2_t vacc16 = __riscv_vmv_v_x_i16m2(0, vlmax_16m2);

            size_t vl = __riscv_vsetvl_e8m1(32);

            vuint8m1_t vx_u8 = __riscv_vle8_v_u8m1(px, vl);

            vint8m1_t vy0 = __riscv_vle8_v_i8m1(py0 , vl);
            vint8m1_t vy1 = __riscv_vle8_v_i8m1(py1, vl);
            vint8m1_t vy2 = __riscv_vle8_v_i8m1(py2, vl);
            vint8m1_t vy3 = __riscv_vle8_v_i8m1(py3, vl);

            // l=0 (bits 1:0)
            vuint8m1_t t0 = __riscv_vand_vx_u8m1(vx_u8, 0x03, vl);
            vint8m1_t vq0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t0), 1, vl);

            // l=1 (bits 3:2)
            vuint8m1_t t1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 2, vl), 0x03, vl);
            vint8m1_t vq1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t1), 1, vl);

            // l=2 (bits 5:4)
            vuint8m1_t t2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vx_u8, 4, vl), 0x03, vl);
            vint8m1_t vq2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t2), 1, vl);

            // l=3 (bits 7:6)
            vuint8m1_t t3 = __riscv_vsrl_vx_u8m1(vx_u8, 6, vl); // No final AND needed as vsrl shifts in zeros
            vint8m1_t vq3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(t3), 1, vl);

            // 4. Multiply and accumulate
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq0, vy0, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq1, vy1, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq2, vy2, vl);
            vacc16 = __riscv_vwmacc_vv_i16m2(vacc16, vq3, vy3, vl);

            vlmax_16m2 = __riscv_vsetvl_e16m2(32);
            vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vred32 = __riscv_vwredsum_vs_i16m2_i32m1(vacc16, vzero32, vlmax_16m2);

            sumi += __riscv_vmv_x_s_i32m1_i32(vred32);
        }
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        sumf += (float)sumi * d;
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 256:
            ggml_vec_dot_tq2_0_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static void ggml_vec_dot_iq1_s_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
    for (int i = 0; i < nb; ++i) {
        // Load qh once for the entire superblock.
        vuint16mf2_t qh = __riscv_vle16_v_u16mf2(x[i].qh, 8);

        // Calculate ls.
        vuint16mf2_t temp = __riscv_vsrl_vx_u16mf2(qh, 12, 8);
        temp = __riscv_vand_vx_u16mf2(temp, 7, 8);
        vint32m1_t ls = __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vwmulu_vx_u32m1(temp, 2, 8));
        ls = __riscv_vadd_vx_i32m1(ls, 1, 8);

        // Calculate delta.
        vbool32_t mask = __riscv_vmseq_vx_u16mf2_b32(__riscv_vand_vx_u16mf2(qh, 0x8000, 8), 0, 8);
        vint32m1_t delta_neg = __riscv_vmv_v_x_i32m1(-1, 8);
        vint32m1_t delta_pos = __riscv_vmv_v_x_i32m1(1, 8);
        vint32m1_t delta = __riscv_vmerge_vvm_i32m1(delta_neg, delta_pos, mask, 8);

        // Load qs.
        vuint8m1_t qs = __riscv_vle8_v_u8m1(x[i].qs, 32);

        // Prepare the indices.
        const uint64_t shift = 0x0009000600030000;
        vuint16m2_t qh_shift = __riscv_vreinterpret_v_u64m2_u16m2(__riscv_vmv_v_x_u64m2(shift, 8));
        vuint16m2_t qh_gather_index = __riscv_vreinterpret_v_i16m2_u16m2(
            __riscv_vdiv_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vid_v_u16m2(32)), 4, 32));
        vuint16m2_t qh_ext = __riscv_vlmul_ext_v_u16m1_u16m2(__riscv_vlmul_ext_v_u16mf2_u16m1(qh));
        vuint16m2_t qh_index = __riscv_vrgather_vv_u16m2(qh_ext, qh_gather_index, 32);
        qh_index = __riscv_vsrl_vv_u16m2(qh_index, qh_shift, 32);
        qh_index = __riscv_vand_vx_u16m2(qh_index, 7, 32);
        qh_index = __riscv_vsll_vx_u16m2(qh_index, 8, 32);
        qh_index = __riscv_vor_vv_u16m2(qh_index, __riscv_vzext_vf2_u16m2(qs, 32), 32);
        vuint16m2_t index = __riscv_vsll_vx_u16m2(qh_index, 3, 32);

        // Final lsums.
        int32_t lsums_s[8];
        vint32m1_t one_scalar = __riscv_vmv_v_x_i32m1(0, 1);

        // Sub-blocks 1-4
        {
            vuint16m1_t grid_index0 = __riscv_vget_v_u16m2_u16m1(index, 0);
            vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, grid_index0, 16));
            vint8m4_t q80 = __riscv_vle8_v_i8m4(y[i].qs, 128);
            vint16m8_t lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);
            lsums_s[0] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 0), one_scalar, 32));
            lsums_s[1] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 1), one_scalar, 32));
            lsums_s[2] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 2), one_scalar, 32));
            lsums_s[3] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum0, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        // Sub-blocks 5-8
        {
            vuint16m1_t grid_index1 = __riscv_vget_v_u16m2_u16m1(index, 1);
            vint8m4_t grid1 = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, grid_index1, 16));
            vint8m4_t q81 = __riscv_vle8_v_i8m4(&y[i].qs[128], 128);
            vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(grid1, q81, 128);
            lsums_s[4] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 0), one_scalar, 32));
            lsums_s[5] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 1), one_scalar, 32));
            lsums_s[6] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 2), one_scalar, 32));
            lsums_s[7] = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(__riscv_vget_v_i16m8_i16m2(lsum1, 3), one_scalar, 32));
        }
        __asm__ __volatile__("" ::: "memory");
        vint32m1_t lsums = __riscv_vle32_v_i32m1(&lsums_s[0], 8);

        // Calculate the bsums.
        vint16m1_t bsums_0 = __riscv_vle16_v_i16m1(y[i].bsums, 16);
        const vuint32m1_t bsums_i32 = __riscv_vreinterpret_v_u16m1_u32m1(__riscv_vreinterpret_v_i16m1_u16m1(bsums_0));
        const vint16mf2_t bsums_i32_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 0, 8));
        const vint16mf2_t bsums_i32_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(bsums_i32, 16, 8));
        const vint32m1_t bsums = __riscv_vwadd_vv_i32m1(bsums_i32_0, bsums_i32_1, 8);

        // Accumulation.
        vint32m1_t sumi_v = __riscv_vmul_vv_i32m1(ls, lsums, 8);
        vint32m1_t sumi1_v = __riscv_vmul_vv_i32m1(__riscv_vmul_vv_i32m1(ls, delta, 8), bsums, 8);

        // Update sumf.
        int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        int sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(sumi1_v, __riscv_vmv_v_x_i32m1(0.0f, 1), 8));
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 256:
            ggml_vec_dot_iq1_s_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

static void ggml_vec_dot_iq1_m_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // Accumulators.
        vint32m2_t acc1 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, 16);

        // We process 4 sub-blocks together.
        for (int ib = 0; ib < QK_K/128; ib++) {
            // Load qh for 4 sub-blocks.
            const vuint8mf4_t qh_8 = __riscv_vle8_v_u8mf4(qh, 8);
            const vuint16mf2_t qh_16_lo = __riscv_vzext_vf2_u16mf2(qh_8, 8);
            const vuint16mf2_t qh_16_hi = __riscv_vsll_vx_u16mf2(qh_16_lo, 8, 8);
            const vuint16m1_t qhb = __riscv_vzext_vf2_u16m1(
                __riscv_vreinterpret_v_u16mf2_u8mf2(__riscv_vor_vv_u16mf2(qh_16_lo, qh_16_hi, 8)), 16);
            qh += 8;

            // Prepare grid indices.
            const vuint16m1_t qsb = __riscv_vzext_vf2_u16m1(__riscv_vle8_v_u8mf2(&qs[0], 16), 16);
            const vuint16m1_t shift = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00040008, 8));
            vuint16m1_t index = __riscv_vor_vv_u16m1(qsb, __riscv_vand_vx_u16m1(__riscv_vsll_vv_u16m1(qhb, shift, 16), 0x700, 16), 16);
            index = __riscv_vsll_vx_u16m1(index, 3, 16);
            qs += 16;

            // Load the grid.
            const vint8m4_t iq1b = __riscv_vreinterpret_v_i64m4_i8m4(__riscv_vreinterpret_v_u64m4_i64m4(
                __riscv_vluxei16_v_u64m4(iq1s_grid, index, 16)));

            // Prepare the deltas.
            const vbool16_t mask = __riscv_vmsgtu_vx_u16m1_b16(
                __riscv_vand_vv_u16m1(qhb, __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vmv_v_x_u32m1(0x00800008, 8)), 16), 0, 16);
            const vint64m4_t delta_pos = __riscv_vmv_v_x_i64m4(0x0101010101010101, 16);
            const vint64m4_t delta_neg = __riscv_vmv_v_x_i64m4(0xffffffffffffffff, 16);
            const vint8m4_t delta = __riscv_vreinterpret_v_i64m4_i8m4(
                __riscv_vmerge_vvm_i64m4(delta_pos, delta_neg, mask, 16));

            // Load q8 for sub-blocks.
            const vint8m4_t q8b = __riscv_vle8_v_i8m4(q8, 128);
            q8 += 128;

            // Calculate the lsums.
            const vint16m8_t lsum1 = __riscv_vwmul_vv_i16m8(iq1b, q8b, 128);
            const vint16m8_t lsum2 = __riscv_vwmul_vv_i16m8(delta, q8b, 128);

            // Prepare the scales.
            const int16_t ls_0_0 = 2*((sc[0] >> 0) & 0x7) + 1;
            const int16_t ls_0_1 = 2*((sc[0] >> 3) & 0x7) + 1;
            const int16_t ls_1_0 = 2*((sc[0] >> 6) & 0x7) + 1;
            const int16_t ls_1_1 = 2*((sc[0] >> 9) & 0x7) + 1;
            const int16_t ls_2_0 = 2*((sc[1] >> 0) & 0x7) + 1;
            const int16_t ls_2_1 = 2*((sc[1] >> 3) & 0x7) + 1;
            const int16_t ls_3_0 = 2*((sc[1] >> 6) & 0x7) + 1;
            const int16_t ls_3_1 = 2*((sc[1] >> 9) & 0x7) + 1;
            sc += 2;

            // Accumulate in acc0 and acc1 for each sub-block.
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_0_0, __riscv_vget_v_i16m8_i16m1(lsum1, 0), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_0_1, __riscv_vget_v_i16m8_i16m1(lsum1, 1), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_0_0, __riscv_vget_v_i16m8_i16m1(lsum2, 0), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_0_1, __riscv_vget_v_i16m8_i16m1(lsum2, 1), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_1_0, __riscv_vget_v_i16m8_i16m1(lsum1, 2), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_1_1, __riscv_vget_v_i16m8_i16m1(lsum1, 3), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_1_0, __riscv_vget_v_i16m8_i16m1(lsum2, 2), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_1_1, __riscv_vget_v_i16m8_i16m1(lsum2, 3), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_2_0, __riscv_vget_v_i16m8_i16m1(lsum1, 4), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_2_1, __riscv_vget_v_i16m8_i16m1(lsum1, 5), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_2_0, __riscv_vget_v_i16m8_i16m1(lsum2, 4), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_2_1, __riscv_vget_v_i16m8_i16m1(lsum2, 5), 16);
            //
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_3_0, __riscv_vget_v_i16m8_i16m1(lsum1, 6), 16);
            acc1 = __riscv_vwmacc_vx_i32m2(acc1, ls_3_1, __riscv_vget_v_i16m8_i16m1(lsum1, 7), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_3_0, __riscv_vget_v_i16m8_i16m1(lsum2, 6), 16);
            acc2 = __riscv_vwmacc_vx_i32m2(acc2, ls_3_1, __riscv_vget_v_i16m8_i16m1(lsum2, 7), 16);
        }

        // Reduce and accumulate in `sumf`.
        vint32m1_t one = __riscv_vmv_v_x_i32m1(0, 1);
        int sumi1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc1, one, 16));
        int sumi2 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(acc2, one, 16));
        sumf += y[i].d * GGML_CPU_FP16_TO_FP32(scale.f16) * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined __riscv_v_intrinsic
    switch (__riscv_vlenb() * 8) {
        case 256:
            ggml_vec_dot_iq1_m_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
            break;
        default:
            ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
            break;
    }
#else
    ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
