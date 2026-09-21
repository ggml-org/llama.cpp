#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-exp.h"
#include "dma-queue.h"
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "gated-delta-net-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_gdn_context {
    struct htp_ops_context * octx;
    const struct htp_gdn_kernel_params * kparams;
    struct htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_base;
    uint32_t row_start;
    uint32_t nrows;
};

static inline HVX_Vector gdn_mul_dot_f32(float * restrict dst, const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vm);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot_f32(float * restrict dst, float mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vmul);
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_add_scaled_dot_f32(float * restrict dst, const HVX_Vector * restrict src,
        HVX_Vector vscale, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vmemu(dst + i * epv);
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];
        HVX_Vector out  = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out = hvx_vec_add_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, vdot), zero));
    }

    return hvx_vec_reduce_sum_f32(acc);
}

static inline HVX_Vector gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static inline HVX_Vector gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm   = mul[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = mul[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_mul_scalar_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vmul);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline HVX_Vector gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict src, const float * restrict scale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);
    const HVX_Vector scale4 = hvx_vec_splat_f32(scale[4]);
    const HVX_Vector scale5 = hvx_vec_splat_f32(scale[5]);
    const HVX_Vector scale6 = hvx_vec_splat_f32(scale[6]);
    const HVX_Vector scale7 = hvx_vec_splat_f32(scale[7]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs   = src[nvec];
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static inline void gdn_step_kda_f32(
    float * restrict s_work,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];
    HVX_Vector vg[4];

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.7228f;
    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32_guard(hvx_vmemu(g_t + i * epv), max_exp, inf);
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32_guard(hvx_vmemu(g_t + nvec * epv), max_exp, inf);
    }

    const HVX_Vector vbeta  = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;
        float * row4 = s_work + (uint64_t) (j + 4) * S_v;
        float * row5 = s_work + (uint64_t) (j + 5) * S_v;
        float * row6 = s_work + (uint64_t) (j + 6) * S_v;
        float * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_Vector vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_Vector vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + (uint64_t) j * S_v;
        HVX_Vector vsum = gdn_mul_dot_f32(row, vg, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

static inline void gdn_step_scalar_f32(
    float * restrict s_work,
    float * restrict attn_out,
    const float * restrict q_t,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    float beta_val,
    float scale,
    uint32_t S_v
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector vq[4];
    HVX_Vector vk[4];

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
    }

    const float gate       = expf(g_t[0]);
    const HVX_Vector vgate = hvx_vec_splat_f32(gate);
    const HVX_Vector vbeta = hvx_vec_splat_f32(beta_val);
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    float delta[8] __attribute__((aligned(128)));

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;
        float * row4 = s_work + (uint64_t) (j + 4) * S_v;
        float * row5 = s_work + (uint64_t) (j + 5) * S_v;
        float * row6 = s_work + (uint64_t) (j + 6) * S_v;
        float * row7 = s_work + (uint64_t) (j + 7) * S_v;

        HVX_Vector vsums = gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 8 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 8 * sizeof(float), res_attn);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (uint64_t) (j + 0) * S_v;
        float * row1 = s_work + (uint64_t) (j + 1) * S_v;
        float * row2 = s_work + (uint64_t) (j + 2) * S_v;
        float * row3 = s_work + (uint64_t) (j + 3) * S_v;

        HVX_Vector vsums = gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_t + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);
        hvx_vec_store_u(delta, 4 * sizeof(float), vdelta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, delta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(attn_out + j, 4 * sizeof(float), res_attn);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + (uint64_t) j * S_v;
        HVX_Vector vsum = gdn_mul_scalar_dot_f32(row, gate, vk, S_v);
        HVX_Vector vv_t = hvx_vec_splat_f32(v_t[j]);
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        attn_out[j] = hvx_vec_get_f32(hvx_vec_mul_f32_f32(vres, vscale));
    }
}

static void gated_delta_net_f32_pp_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_tokens = kparams->n_tokens;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t K        = kparams->K;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_cache ? (float *) (uintptr_t) dst_cache->data : (dst_base + S_v * H * n_tokens * n_seqs);

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const uint64_t state_size_per_snap = (uint64_t) kparams->state_size_per_snap;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_tokens * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + ((uint64_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                    (uint64_t) iq3 * q->nb[3] + (uint64_t) t * q->nb[2] + (uint64_t) iq1 * q->nb[1]);
            const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                    (uint64_t) ik3 * k->nb[3] + (uint64_t) t * k->nb[2] + (uint64_t) ik1 * k->nb[1]);
            const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                    (uint64_t) iv3 * v->nb[3] + (uint64_t) t * v->nb[2] + (uint64_t) iv1 * v->nb[1]);
            const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                    (uint64_t) iv3 * g->nb[3] + (uint64_t) t * g->nb[2] + (uint64_t) iv1 * g->nb[1]);
            const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                    (uint64_t) iv3 * beta->nb[3] + (uint64_t) t * beta->nb[2] + (uint64_t) iv1 * beta->nb[1]);

            if (kparams->kda) {
                gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            } else {
                gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
            }

            if (K > 1) {
                const int64_t target_slot = (int64_t) n_tokens - 1 - (int64_t) t;
                if (target_slot > 0 && target_slot < (int64_t) K) {
                    float * curr_state_o = state_out_base + (uint64_t) target_slot * state_size_per_snap + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
                    hvx_copy_f32_uu((uint8_t *) curr_state_o, (const uint8_t *) s_work_curr, S_v * S_v);
                }
            }

            attn_data += (uint64_t) S_v * H;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

static void gated_delta_net_f32_tg_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;
    const struct htp_gdn_kernel_params * kparams = gctx->kparams;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = kparams->S_v;
    const uint32_t H        = kparams->H;
    const uint32_t n_seqs   = kparams->n_seqs;
    const uint32_t row_end  = gctx->row_start + gctx->nrows;

    if (ith >= gctx->nrows) {
        return;
    }

    const struct htp_tensor * dst_cache = octx->dsts[1];
    const float scale = kparams->scale;
    float * dst_base  = (float *) (uintptr_t) dst->data;

    dma_queue * dma_q = octx->ctx->dma[ith];
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    float * s_work[2];
    s_work[0] = (float *) (gctx->vtcm_base + layout->bytes_per_thread * ith);
    s_work[1] = s_work[0] + layout->state_aligned / sizeof(float);

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + S_v * H * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        dma_addr_t ps_in  = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);
        dma_addr_t ps_out = state_out_dma_base + ((uint64_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);

        // Push dummy write-back
        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);

        // Push fetch
        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        float * s_work_curr = s_work[curr_spad_idx];

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);

        dma_addr_t s_out  = state_out_dma_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        float * attn_data = dst_base + ((uint64_t) iv3 * H + iv1) * S_v;

        const float * q_t = (const float *) ((const uint8_t *) (uintptr_t) q->data +
                (uint64_t) iq3 * q->nb[3] + (uint64_t) iq1 * q->nb[1]);
        const float * k_t = (const float *) ((const uint8_t *) (uintptr_t) k->data +
                (uint64_t) ik3 * k->nb[3] + (uint64_t) ik1 * k->nb[1]);
        const float * v_t = (const float *) ((const uint8_t *) (uintptr_t) v->data +
                (uint64_t) iv3 * v->nb[3] + (uint64_t) iv1 * v->nb[1]);
        const float * g_t = (const float *) ((const uint8_t *) (uintptr_t) g->data +
                (uint64_t) iv3 * g->nb[3] + (uint64_t) iv1 * g->nb[1]);
        const float beta_val = *(const float *) ((const uint8_t *) (uintptr_t) beta->data +
                (uint64_t) iv3 * beta->nb[3] + (uint64_t) iv1 * beta->nb[1]);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        if (kparams->kda) {
            gdn_step_kda_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        } else {
            gdn_step_scalar_f32(s_work_curr, attn_data, q_t, k_t, v_t, g_t, beta_val, scale, S_v);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        // Push real write-back
        dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);

        // Prefetch next block (if any)
        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            dma_addr_t ps_in = state->data + ((uint64_t) piv3 * state_seq_stride + (uint64_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];
    const uint32_t K        = octx->op_params[0];

    if (S_v == 0 || S_v > HTP_GDN_MAX_SV || H == 0 || n_tokens == 0 || n_seqs == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] == 0 || k->ne[1] == 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] == 0 || k->ne[3] == 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }
    // state holds s0 only: [S_v, S_v, H, n_seqs]
    if (state->ne[0] != S_v || state->ne[1] != S_v || state->ne[2] != H || state->ne[3] != n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return HTP_STATUS_NO_SUPPORT;
    }

    for (int i = 0; i < 5; i++) {
        if (htp_tensor_is_extended(octx->src[i])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (htp_tensor_is_extended(octx->dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->dsts[1]) {
        const struct htp_tensor * dst_cache = octx->dsts[1];
        if (dst_cache->type != HTP_TYPE_F32 || htp_tensor_is_extended(dst_cache)) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    const struct htp_gdn_kernel_params * kparams = (const struct htp_gdn_kernel_params *) octx->kernel_params;
    struct htp_gdn_kernel_params kparams_local;
    if (!kparams || kparams->S_v == 0) {
        const uint32_t rq3 = n_seqs / q->ne[3];
        const uint32_t rk3 = n_seqs / k->ne[3];
        const uint32_t total_rows = H * n_seqs;
        uint32_t n_threads = (total_rows < octx->n_threads) ? total_rows : octx->n_threads;
        if (n_threads == 0) {
            n_threads = 1;
        }

        memset(&kparams_local, 0, sizeof(kparams_local));
        kparams_local.n_threads           = n_threads;
        kparams_local.S_v                 = S_v;
        kparams_local.H                   = H;
        kparams_local.n_tokens            = n_tokens;
        kparams_local.n_seqs              = n_seqs;
        kparams_local.K                   = K;
        kparams_local.total_rows          = total_rows;
        kparams_local.rows_per_thread     = (total_rows + n_threads - 1) / n_threads;
        struct htp_gdn_vtcm_layout layout_local;
        htp_gdn_vtcm_layout_build(&layout_local, S_v, n_threads);
        kparams_local.state_aligned       = (uint32_t) layout_local.state_aligned;
        kparams_local.vtcm_per_thread     = (uint32_t) layout_local.bytes_per_thread;
        kparams_local.vtcm_size           = (uint32_t) layout_local.total_bytes;
        kparams_local.kda                 = (g->ne[0] == S_v) ? 1 : 0;
        kparams_local.scale               = 1.0f / sqrtf((float) S_v);
        kparams_local.state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
        kparams_local.state_size_per_snap = S_v * S_v * H * n_seqs;

        kparams_local.div_H         = init_fastdiv_values(H);
        kparams_local.div_q1        = init_fastdiv_values(q->ne[1]);
        kparams_local.div_k1        = init_fastdiv_values(k->ne[1]);
        kparams_local.div_rq3       = init_fastdiv_values(rq3);
        kparams_local.div_rk3       = init_fastdiv_values(rk3);
        kparams_local.div_n_threads = init_fastdiv_values(n_threads);

        kparams = &kparams_local;
    }

    const uint32_t total_rows = kparams->total_rows;
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->op_params[1] != 0) {
        row_start = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = (nrows < kparams->n_threads) ? nrows : kparams->n_threads;

    struct htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.row_start = row_start;
    gctx.nrows     = nrows;
    gctx.vtcm_base = octx->ctx->vtcm_base;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_threads);

    if (gctx.layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_threads %u\n",
         q->ne[0], q->ne[1], q->ne[2], q->ne[3],
         k->ne[0], k->ne[1], k->ne[2], k->ne[3],
         v->ne[0], v->ne[1], v->ne[2], v->ne[3],
         state->ne[0], state->ne[1], state->ne[2], state->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.total_bytes, n_threads);

    if (n_tokens == 1) {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_tg_thread, &gctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, gated_delta_net_f32_pp_thread, &gctx, n_threads);
    }

    return HTP_STATUS_OK;
}
