#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <HAP_farf.h>

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-exp.h"
#include "dma-queue.h"
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hmx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"
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

static inline HVX_Vector gdn_mul_scalar_dot_f32(float * restrict dst, HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
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

    for (uint32_t i = 0; i < nvec; ++i) {
        vq[i] = hvx_vmemu(q_t + i * epv);
        vk[i] = hvx_vmemu(k_t + i * epv);
        vg[i] = hvx_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    if (nloe) {
        vq[nvec] = hvx_vmemu(q_t + nvec * epv);
        vk[nvec] = hvx_vmemu(k_t + nvec * epv);
        vg[nvec] = hvx_vec_exp_f32(hvx_vmemu(g_t + nvec * epv));
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

    const HVX_Vector vgate  = hvx_vec_exp_f32(hvx_vec_splat_f32(g_t[0]));
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
        HVX_Vector vsum = gdn_mul_scalar_dot_f32(row, vgate, vk, S_v);
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

struct htp_gdn_hmx_gemm_task {
    const __fp16 * row_tiles;
    const __fp16 * col_tiles;
    __fp16 *       out_tiles;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
    uint32_t       dot_stride;
    uint8_t *      hmx_scales;
};

static void htp_gdn_hmx_gemm_worker(void * data) {
    struct htp_gdn_hmx_gemm_task * task = (struct htp_gdn_hmx_gemm_task *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)task->hmx_scales));

    const size_t dot_stride = task->dot_stride;
    for (uint32_t r = 0; r < task->n_row_tiles; ++r) {
        const __fp16 * r_tiles = task->row_tiles + r * dot_stride;
        const __fp16 * c_tiles = task->col_tiles;
        __fp16 *       o_tile  = task->out_tiles + r * task->n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < task->n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, task->n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

static inline void htp_gdn_push_hmx_gemm_task(
    hmx_queue_t q,
    struct htp_gdn_hmx_gemm_task * task,
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 * out_tiles,
    uint32_t n_row_tiles,
    uint32_t n_col_tiles,
    uint32_t n_dot_tiles,
    uint8_t * scales
) {
    task->row_tiles   = row_tiles;
    task->col_tiles   = col_tiles;
    task->out_tiles   = out_tiles;
    task->n_row_tiles = n_row_tiles;
    task->n_col_tiles = n_col_tiles;
    task->n_dot_tiles = n_dot_tiles;
    task->dot_stride  = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    task->hmx_scales  = scales;

    hmx_queue_push(q, hmx_queue_make_desc(htp_gdn_hmx_gemm_worker, task));
}

static inline void gdn_unpack_64x64_tiles_to_vectors(
    HVX_Vector * restrict rows,
    const __fp16 * restrict tiles
) {
    const HVX_Vector * t00 = (const HVX_Vector *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t01 = (const HVX_Vector *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t10 = (const HVX_Vector *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    const HVX_Vector * t11 = (const HVX_Vector *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp0 = Q6_W_vdeal_VVR(t01[r], t00[r], -2);
        rows[2 * r + 0] = Q6_V_lo_W(vp0);
        rows[2 * r + 1] = Q6_V_hi_W(vp0);

        HVX_VectorPair vp1 = Q6_W_vdeal_VVR(t11[r], t10[r], -2);
        rows[32 + 2 * r + 0] = Q6_V_lo_W(vp1);
        rows[32 + 2 * r + 1] = Q6_V_hi_W(vp1);
    }
}

static inline void gdn_pack_64x64_vectors_to_tiles(
    __fp16 * restrict tiles,
    const HVX_Vector * restrict rows
) {
    HVX_Vector * t00 = (HVX_Vector *) (tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t01 = (HVX_Vector *) (tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t10 = (HVX_Vector *) (tiles + 2 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * t11 = (HVX_Vector *) (tiles + 3 * HMX_FP16_TILE_N_ELMS);

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp0 = Q6_W_vshuff_VVR(rows[2 * r + 1], rows[2 * r + 0], -2);
        t00[r] = Q6_V_lo_W(vp0);
        t01[r] = Q6_V_hi_W(vp0);

        HVX_VectorPair vp1 = Q6_W_vshuff_VVR(rows[32 + 2 * r + 1], rows[32 + 2 * r + 0], -2);
        t10[r] = Q6_V_lo_W(vp1);
        t11[r] = Q6_V_hi_W(vp1);
    }
}

static inline void gdn_unpack_64xS_tiles_to_f32(
    float * restrict dst_f32,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < 2; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

static inline void gdn_unpack_64xS_tiles_to_f16(
    __fp16 * restrict dst_f16,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_col_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < 2; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                __fp16 * out0 = dst_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                __fp16 * out1 = dst_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0) = Q6_V_lo_W(vp01);
                hvx_vmem(out1) = Q6_V_hi_W(vp01);
            }
        }
    }
}

static inline void gdn_unpack_SxS_tiles_to_f32(
    float * restrict dst_f32,
    const __fp16 * restrict tiles,
    uint32_t S_v
) {
    const uint32_t n_tiles = S_v / 32;
    for (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        for (uint32_t d = 0; d < S_v / 64; ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * out0 = dst_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * out1 = dst_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                hvx_vmem(out0 + 0)  = Q6_V_lo_W(p0);
                hvx_vmem(out0 + 32) = Q6_V_hi_W(p0);
                hvx_vmem(out1 + 0)  = Q6_V_lo_W(p1);
                hvx_vmem(out1 + 32) = Q6_V_hi_W(p1);
            }
        }
    }
}

static inline void gdn_f32_to_hmx_row_tiles_and_f16(
    __fp16 * restrict dst_tiles,
    __fp16 * restrict dst_prime_tiles,
    __fp16 * restrict dst_f16,
    const float * restrict src,
    const __fp16 * restrict scale_per_row,
    uint32_t n_rows,
    uint32_t n_cols
) {
    const uint32_t n_col_tiles = n_cols / 32;
    const uint32_t * scale_pairs = (const uint32_t *) scale_per_row;

    for (uint32_t r = 0; r < n_rows; r += 2) {
        uint32_t r0 = r / 32;
        uint32_t r1 = (r % 32) / 2;
        const float * p0 = src + (r + 0) * n_cols;
        const float * p1 = src + (r + 1) * n_cols;

        HVX_Vector v_scale;
        if (dst_prime_tiles) {
            uint32_t scale_pair = scale_pairs ? scale_pairs[r / 2] : 0x3c003c00;
            v_scale = Q6_V_vsplat_R(scale_pair);
        }

        for (uint32_t c = 0; c < n_col_tiles; c += 2) {
            HVX_Vector v0_0 = hvx_vmem(p0 + (c + 0) * 32);
            HVX_Vector v1_0 = hvx_vmem(p1 + (c + 0) * 32);
            HVX_Vector v0_1 = hvx_vmem(p0 + (c + 1) * 32);
            HVX_Vector v1_1 = hvx_vmem(p1 + (c + 1) * 32);

            HVX_Vector vh0 = hvx_vec_f32_to_f16_shuff(v0_0, v1_0);
            HVX_Vector vh1 = hvx_vec_f32_to_f16_shuff(v0_1, v1_1);
            __fp16 * tile0 = dst_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile0)[r1] = vh0;
            ((HVX_Vector *) tile1)[r1] = vh1;

            if (dst_prime_tiles) {
                HVX_Vector vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
                HVX_Vector vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
                __fp16 * tile0_s = dst_prime_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
                __fp16 * tile1_s = dst_prime_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
                ((HVX_Vector *) tile0_s)[r1] = vh0_s;
                ((HVX_Vector *) tile1_s)[r1] = vh1_s;
            }

            if (dst_f16) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_cols + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_cols + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

static inline void hvx_transpose_32x32_words(HVX_Vector * restrict m, HVX_Vector * restrict tmp) {
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

static inline void gdn_pack_d_t_row_tiles(
    __fp16 * restrict dst_tiles,
    const __fp16 * restrict src_d,
    uint32_t S_v,
    HVX_Vector * restrict m,
    HVX_Vector * restrict tmp
) {
    for (uint32_t col_half = 0; col_half < S_v / 64; ++col_half) {
        uint32_t r0_base = col_half * 2;
        for (uint32_t c0 = 0; c0 < 2; ++c0) {
            for (uint32_t s_local = 0; s_local < 32; ++s_local) {
                uint32_t s = c0 * 32 + s_local;
                m[s_local] = hvx_vmem(src_d + s * S_v + col_half * 64);
            }

            hvx_transpose_32x32_words(m, tmp);

            uint32_t tile0_idx = (r0_base + 0) * 2 + c0;
            uint32_t tile1_idx = (r0_base + 1) * 2 + c0;
            HVX_Vector * t0 = (HVX_Vector *)(dst_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * t1 = (HVX_Vector *)(dst_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                t0[r] = m[r];
                t1[r] = m[16 + r];
            }
        }
    }
}

static __attribute__((noinline)) void gdn_build_inv_l_blocks(
    __fp16 * restrict inv_row_tiles,
    const HVX_Vector * restrict rows_kk,
    const __fp16 * restrict decay_m,
    const float * restrict beta,
    __fp16 * restrict l10_tile,
    __fp16 * restrict neg_a11_tile
) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq2_R(64);

    uint16_t beta_u16[64] __attribute__((aligned(128)));
    uint16_t l00[32][32]  __attribute__((aligned(128)));
    uint16_t l11[32][32]  __attribute__((aligned(128)));

    HVX_Vector * restrict p_l00 = (HVX_Vector *) l00;
    HVX_Vector * restrict p_l11 = (HVX_Vector *) l11;
    HVX_Vector * restrict p_l10_tile = (HVX_Vector *) l10_tile;

    HVX_Vector * restrict tile00 = (HVX_Vector *) (inv_row_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile01 = (HVX_Vector *) (inv_row_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile11 = (HVX_Vector *) (inv_row_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict p_neg_a11 = (HVX_Vector *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_to_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    for (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    for (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_Vector v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_Vector v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
        HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

        p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
    }

    for (uint32_t r = 0; r < 16; ++r) {
        uint32_t t0 = 32 + 2 * r;
        uint32_t t1 = t0 + 1;

        HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
        HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
        HVX_Vector v_b0 = Q6_Vh_vsplat_R(beta_u16[t0]);
        HVX_Vector v_b1 = Q6_Vh_vsplat_R(beta_u16[t1]);

        HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
        HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

        HVX_VectorPair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
        p_l10_tile[r] = Q6_V_lo_W(vp_l10);
        p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
    }

    HVX_Vector a_rows[32];
    for (uint32_t t = 0; t < 32; ++t) {
        HVX_Vector v_inv = Q6_V_vzero();
        for (uint32_t k = 0; k < t; ++k) {
            HVX_Vector v_lk = Q6_Vh_vsplat_R(l00[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_rows[k]));
        }
        HVX_VectorPred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_rows[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp = Q6_W_vshuff_VVR(a_rows[2 * r + 1], a_rows[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    for (uint32_t t = 0; t < 32; ++t) {
        HVX_Vector v_inv = Q6_V_vzero();
        for (uint32_t k = 0; k < t; ++k) {
            HVX_Vector v_lk = Q6_Vh_vsplat_R(l11[t][k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_rows[k]));
        }
        HVX_VectorPred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_rows[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp = Q6_W_vshuff_VVR(a_rows[2 * r + 1], a_rows[2 * r + 0], -2);
        tile11[r] = Q6_V_lo_W(vp);

        HVX_Vector n0 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_rows[2 * r + 0]);
        HVX_Vector n1 = hvx_vec_sub_f16_f16(Q6_V_vzero(), a_rows[2 * r + 1]);
        HVX_VectorPair vp_neg = Q6_W_vshuff_VVR(n1, n0, -2);
        p_neg_a11[r] = Q6_V_lo_W(vp_neg);
    }
}


static inline void gdn_dma_push_chunk_inputs(
    dma_queue * dma_q,
    float * vtcm_q,
    float * vtcm_k,
    float * vtcm_v,
    const struct htp_tensor * q,
    const struct htp_tensor * k,
    const struct htp_tensor * v,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_size,
    uint32_t S_v
) {
    const dma_addr_t q_dma = q->data + (uint64_t) iq3 * q->nb[3] + (uint64_t) t_chunk * q->nb[2] + (uint64_t) iq1 * q->nb[1];
    const dma_addr_t k_dma = k->data + (uint64_t) ik3 * k->nb[3] + (uint64_t) t_chunk * k->nb[2] + (uint64_t) ik1 * k->nb[1];
    const dma_addr_t v_dma = v->data + (uint64_t) iv3 * v->nb[3] + (uint64_t) t_chunk * v->nb[2] + (uint64_t) iv1 * v->nb[1];

    dma_queue_push(dma_q, dma_make_data(vtcm_q, q_dma), S_v * sizeof(float), q->nb[2], S_v * sizeof(float), chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_k, k_dma), S_v * sizeof(float), k->nb[2], S_v * sizeof(float), chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_v, v_dma), S_v * sizeof(float), v->nb[2], S_v * sizeof(float), chunk_size);
}

static inline void gdn_dma_push_chunk_gb(
    dma_queue * dma_q,
    float * vtcm_g_raw,
    float * vtcm_b_raw,
    const struct htp_tensor * g,
    const struct htp_tensor * beta,
    uint32_t iv3,
    uint32_t iv1,
    uint32_t t_chunk,
    uint32_t chunk_size,
    uint32_t n_batch
) {
    const dma_addr_t g_dma    = g->data + (uint64_t) iv3 * g->nb[3] + (uint64_t) t_chunk * g->nb[2] + (uint64_t) iv1 * g->nb[1];
    const dma_addr_t beta_dma = beta->data + (uint64_t) iv3 * beta->nb[3] + (uint64_t) t_chunk * beta->nb[2] + (uint64_t) iv1 * beta->nb[1];
    const uint32_t row_bytes  = n_batch * sizeof(float);

    dma_queue_push(dma_q, dma_make_data(vtcm_g_raw, g_dma), row_bytes, g->nb[2], row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_b_raw, beta_dma), row_bytes, beta->nb[2], row_bytes, chunk_size);
}

static inline void gdn_pack_s_col_tiles(
    __fp16 * restrict vtcm_s_col_tiles,
    __fp16 * restrict vtcm_s_f16,
    const float * restrict vtcm_s_state,
    uint32_t S_v
) {
    for (uint32_t j = 0; j < S_v; ++j) {
        for (uint32_t i = 0; i < S_v; i += 64) {
            HVX_Vector v0 = hvx_vmem(vtcm_s_state + j * S_v + i + 0);
            HVX_Vector v1 = (i + 32 < S_v) ? hvx_vmem(vtcm_s_state + j * S_v + i + 32) : Q6_V_vzero();
            hvx_vmem(vtcm_s_f16 + j * S_v + i) = hvx_vec_f32_to_f16(v0, v1);
        }
    }
    hmx_interleave_rows_to_tiles(vtcm_s_col_tiles, vtcm_s_f16, S_v, S_v, S_v, 0, S_v);
}

struct htp_gdn_head_ptrs {
    float *  s_state;
    __fp16 * s_f16;
    __fp16 * s_col_tiles;
    float *  s_update_f32;
    __fp16 * s_update_tiles;

    float * q_f32[2];
    float * k_f32[2];
    float * v_f32[2];
    float * g_f32[2];
    float * b_f32[2];
    float * o_f32[2];

    float * v_inter_f32;
    float * o_inter_f32;
    float * o_intra_f32;

    __fp16 * k_f16;
    __fp16 * v_prime_f16;
    __fp16 * delta_f16;
    __fp16 * d_f16;

    __fp16 * q_row_tiles;
    __fp16 * q_prime_row_tiles;
    __fp16 * k_row_tiles;
    __fp16 * k_col_tiles;
    __fp16 * k_prime_row_tiles;
    __fp16 * k_col_tiles_64x128;
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_row_tiles;
    __fp16 * a_row_tiles;
    __fp16 * v_prime_col_tiles;
    __fp16 * delta_tiles;
    __fp16 * delta_col_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_row_tiles;

    __fp16 * gamma;
    float *  lambda_init;
    __fp16 * lambda_init_f16;
    __fp16 * decay_m;
    __fp16 * decay_a;

    HVX_Vector * rows_kk;
    HVX_Vector * rows_qk;
    HVX_Vector * rows_inv;
    HVX_Vector * rows_a;

    HVX_Vector * vtcm_m;
    HVX_Vector * vtcm_tmp;

    uint32_t iv1;
    uint32_t iv3;
    uint32_t iq1;
    uint32_t ik1;
    uint32_t iq3;
    uint32_t ik3;
    dma_addr_t state_in_dma;
    dma_addr_t state_out_dma;
};

static inline void gdn_init_head_ptrs(
    struct htp_gdn_head_ptrs * head,
    const struct htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_base,
    uint32_t h,
    uint32_t base_iv1,
    uint32_t iv3,
    const struct htp_tensor * q,
    const struct htp_tensor * k,
    const struct htp_tensor * v,
    const struct htp_tensor * state,
    const struct htp_tensor * dst,
    const struct htp_tensor * dst_cache,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t S_v,
    uint32_t H,
    uint32_t n_tokens,
    uint32_t chunk_size
) {
    const size_t dma_scalar_sz = hex_round_up(chunk_size * sizeof(float), 128);
    const size_t decay_sz      = 64 * 64 * sizeof(__fp16);
    const size_t row_vecs_sz   = 64 * 128;

    head->s_state        = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_s_state + h * L->state_f32_bytes);
    head->s_f16          = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_f16 + h * L->state_f16_bytes);
    head->s_col_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_col_tiles + h * L->state_tiles_bytes);
    head->s_update_f32   = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_s_update_f32 + h * L->state_f32_bytes);
    head->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_update_tiles + h * L->state_tiles_bytes);

    head->q_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[0] + h * L->dma_chunk_bytes);
    head->q_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[1] + h * L->dma_chunk_bytes) : head->q_f32[0];
    head->k_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[0] + h * L->dma_chunk_bytes);
    head->k_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[1] + h * L->dma_chunk_bytes) : head->k_f32[0];
    head->v_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[0] + h * L->dma_chunk_bytes);
    head->v_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[1] + h * L->dma_chunk_bytes) : head->v_f32[0];
    head->g_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[0] + h * dma_scalar_sz);
    head->g_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[1] + h * dma_scalar_sz) : head->g_f32[0];
    head->b_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[0] + h * dma_scalar_sz);
    head->b_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[1] + h * dma_scalar_sz) : head->b_f32[0];
    head->o_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[0] + h * L->dma_chunk_bytes);
    head->o_f32[1] = L->pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[1] + h * L->dma_chunk_bytes) : head->o_f32[0];

    head->v_inter_f32 = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_inter_f32 + h * L->dma_chunk_bytes);
    head->o_inter_f32 = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_inter_f32 + h * L->dma_chunk_bytes);
    head->o_intra_f32 = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_intra_f32 + h * L->dma_chunk_bytes);

    head->k_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_f16 + h * L->act_f16_bytes);
    head->v_prime_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_prime_f16 + h * L->act_f16_bytes);
    head->delta_f16   = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_f16 + h * L->act_f16_bytes);
    head->d_f16       = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_d_f16 + h * L->act_f16_bytes);

    head->q_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_row_tiles + h * L->tile_64xSv_bytes);
    head->q_prime_row_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_prime_row_tiles + h * L->tile_64xSv_bytes);
    head->k_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_row_tiles + h * L->tile_64xSv_bytes);
    head->k_col_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles + h * L->tile_64xSv_bytes);
    head->k_prime_row_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_prime_row_tiles + h * L->tile_64xSv_bytes);
    head->k_col_tiles_64x128 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles_64x128 + h * L->tile_64xSv_bytes);
    head->kk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_kk_tiles + h * L->tile_64x64_bytes);
    head->qk_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_qk_tiles + h * L->tile_64x64_bytes);
    head->v_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_inter_tiles + h * L->tile_64xSv_bytes);
    head->o_inter_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_inter_tiles + h * L->tile_64xSv_bytes);
    head->inv_row_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_inv_row_tiles + h * L->tile_64x64_bytes);
    head->a_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_a_row_tiles + h * L->tile_64x64_bytes);
    head->v_prime_col_tiles  = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_prime_col_tiles + h * L->tile_64xSv_bytes);
    head->delta_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_tiles + h * L->tile_64xSv_bytes);
    head->delta_col_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_col_tiles + h * L->tile_64xSv_bytes);
    head->o_intra_tiles      = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_intra_tiles + h * L->tile_64xSv_bytes);
    head->d_row_tiles        = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_d_row_tiles + h * L->tile_64xSv_bytes);

    head->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * dma_scalar_sz);
    head->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * dma_scalar_sz + 128);
    head->lambda_init     = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_lambda_init + h * dma_scalar_sz);
    head->decay_m     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_m + h * decay_sz);
    head->decay_a     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_a + h * decay_sz);

    head->rows_kk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_kk + h * row_vecs_sz);
    head->rows_qk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_qk + h * row_vecs_sz);
    head->rows_inv = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_inv + h * row_vecs_sz);
    head->rows_a   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_a + h * row_vecs_sz);

    head->vtcm_m   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_thread_scratch + h * (64 * 128));
    head->vtcm_tmp = head->vtcm_m + 32;

    head->iv1 = base_iv1 + h;
    head->iv3 = iv3;
    head->iq1 = fastmodulo(head->iv1, q->ne[1], &kparams->div_q1);
    head->ik1 = fastmodulo(head->iv1, k->ne[1], &kparams->div_k1);
    head->iq3 = fastdiv(head->iv3, &kparams->div_rq3);
    head->ik3 = fastdiv(head->iv3, &kparams->div_rk3);

    head->state_in_dma = state->data +
        ((uint64_t) head->iv3 * kparams->state_seq_stride + (uint64_t) head->iv1 * S_v * S_v) * sizeof(float);

    head->state_out_dma = dst_cache ?
        (dst_cache->data + ((uint64_t) head->iv3 * H + head->iv1) * S_v * S_v * sizeof(float)) :
        (dst->data + ((uint64_t) S_v * H * n_tokens * kparams->n_seqs + (uint64_t) (head->iv3 * H + head->iv1) * S_v * S_v) * sizeof(float));
}

struct htp_gdn_batch_context {
    struct htp_gdn_head_ptrs * heads;
    const float *              vtcm_g_raw;
    const float *              vtcm_b_raw;
    uint32_t                   curr_buf;
    uint32_t                   c;
    uint32_t                   n_batch;
    uint32_t                   S_v;
    float                      scale;
    struct htp_ops_context *   octx;
    const struct htp_gdn_kernel_params * kparams;
};

static void gdn_hvx_init_state_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    gdn_pack_s_col_tiles(head->s_col_tiles, head->s_f16, head->s_state, bctx->S_v);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
}

static inline __attribute__((unused)) HVX_Vector hvx_clamp_neg20_0(HVX_Vector v, HVX_Vector v_zero, HVX_Vector v_neg20) {
    HVX_VectorPred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_zero);
    v = Q6_V_vmux_QVV(p_gt, v_zero, v);
    HVX_VectorPred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_neg20, v);
    return Q6_V_vmux_QVV(p_lt, v_neg20, v);
}

static inline HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    const HVX_Vector zero = Q6_V_vzero();

    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_vec_add_f32_f32(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_vec_add_f32_f32(v, carry_in);

    return v;
}

static inline HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

static void gdn_hvx_phase1a_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    const uint32_t n_batch = bctx->n_batch;

    if (n_batch == 1) {
        hvx_vmem(head->g_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_g_raw + 0);
        hvx_vmem(head->g_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_g_raw + 32);
        hvx_vmem(head->b_f32[curr_buf] + 0)  = hvx_vmem(bctx->vtcm_b_raw + 0);
        hvx_vmem(head->b_f32[curr_buf] + 32) = hvx_vmem(bctx->vtcm_b_raw + 32);
    } else {
        int32_t offsets[32] __attribute__((aligned(128)));
        for (int k = 0; k < 32; ++k) {
            offsets[k] = k * n_batch * sizeof(float);
        }
        HVX_Vector vv = *(const HVX_Vector *) offsets;
        const size_t rt_g = (size_t) ((const uint8_t *) bctx->vtcm_g_raw + i * sizeof(float));
        const size_t rt_b = (size_t) ((const uint8_t *) bctx->vtcm_b_raw + i * sizeof(float));
        const size_t mu   = 64 * n_batch * sizeof(float);

        Q6_vgather_ARMVw((HVX_Vector *) (head->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->g_f32[curr_buf] + 32), rt_g + 32 * n_batch * sizeof(float), mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->b_f32[curr_buf] + 32), rt_b + 32 * n_batch * sizeof(float), mu, vv);
    }

    const uint32_t t_chunk = bctx->c * 64;
    const uint32_t valid_tokens = hex_smin(64, bctx->kparams->n_tokens - t_chunk);
    if (valid_tokens < 64) {
        for (uint32_t t = valid_tokens; t < 64; ++t) {
            head->g_f32[curr_buf][t] = 0.0f;
            head->b_f32[curr_buf][t] = 0.0f;
        }
        const HVX_Vector vzero = Q6_V_vzero();
        for (uint32_t t = valid_tokens; t < 64; ++t) {
            for (uint32_t j = 0; j < S_v; j += 32) {
                hvx_vmem(head->q_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(head->k_f32[curr_buf] + t * S_v + j) = vzero;
                hvx_vmem(head->v_f32[curr_buf] + t * S_v + j) = vzero;
            }
        }
    }

    const HVX_Vector v_g0 = hvx_vmem(head->g_f32[curr_buf] + 0);
    const HVX_Vector v_g1 = hvx_vmem(head->g_f32[curr_buf] + 32);

    HVX_Vector v_gamma0 = hvx_prefix_scan_f32(v_g0, Q6_V_vzero());
    HVX_Vector v_carry  = hvx_splat_last_f32(v_gamma0);
    HVX_Vector v_gamma1 = hvx_prefix_scan_f32(v_g1, v_carry);

    const HVX_Vector v_zero  = Q6_V_vzero();
    const HVX_Vector v_neg20 = hvx_vec_splat_f32(-20.0f);

    hvx_vmem(head->gamma) = hvx_vec_f32_to_f16(v_gamma0, v_gamma1);

    HVX_Vector v_l0 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma0, v_zero, v_neg20));
    HVX_Vector v_l1 = hvx_vec_exp_f32(hvx_clamp_neg20_0(v_gamma1, v_zero, v_neg20));

    hvx_vmem(head->lambda_init + 0)  = v_l0;
    hvx_vmem(head->lambda_init + 32) = v_l1;
    hvx_vmem(head->lambda_init_f16)  = hvx_vec_f32_to_f16(v_l0, v_l1);

    gdn_f32_to_hmx_row_tiles_and_f16(head->k_row_tiles, head->k_prime_row_tiles, head->k_f16,
                                     head->k_f32[curr_buf], head->lambda_init_f16, 64, S_v);
    hmx_interleave_rows_to_tiles(head->k_col_tiles, head->k_f16, 64, S_v, S_v, 0, 64);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

static void gdn_hvx_phase1b_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;

    gdn_f32_to_hmx_row_tiles_and_f16(head->q_row_tiles, head->q_prime_row_tiles, NULL,
                                     head->q_f32[curr_buf], head->lambda_init_f16, 64, S_v);

    hmx_interleave_cols_to_tiles(head->k_col_tiles_64x128, head->k_f16, 64, S_v, S_v, 2, 0, 64);

    const uint16_t * gamma_u16 = (const uint16_t *) head->gamma;
    const HVX_Vector v_gamma   = hvx_vmem(head->gamma);

    const HVX_Vector v_zero_f16  = Q6_V_vzero();
    const HVX_Vector v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    const HVX_Vector v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(head->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(head->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    for (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_Vector v_gamma_t0  = Q6_Vh_vsplat_R(gamma_u16[t0]);
        HVX_Vector v_gamma_t1  = Q6_Vh_vsplat_R(gamma_u16[t1]);

        HVX_Vector diff0       = hvx_vec_sub_f16_f16(v_gamma_t0, v_gamma);
        HVX_Vector diff1       = hvx_vec_sub_f16_f16(v_gamma_t1, v_gamma);

        HVX_VectorPred p_gt0   = Q6_Q_vcmp_gt_VhfVhf(diff0, v_zero_f16);
        HVX_VectorPred p_gt1   = Q6_Q_vcmp_gt_VhfVhf(diff1, v_zero_f16);

        diff0                  = Q6_V_vmux_QVV(p_gt0, v_zero_f16, diff0);
        diff1                  = Q6_V_vmux_QVV(p_gt1, v_zero_f16, diff1);

        diff0                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff0);
        diff1                  = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff1);

        HVX_Vector diff_log2e0 = hvx_vec_mul_f16_f16(diff0, v_log2e_f16);
        HVX_Vector diff_log2e1 = hvx_vec_mul_f16_f16(diff1, v_log2e_f16);

        HVX_Vector v_exp0      = hvx_vec_exp2_f16(diff_log2e0);
        HVX_Vector v_exp1      = hvx_vec_exp2_f16(diff_log2e1);

        HVX_VectorPred mask_lt0 = Q6_Q_vsetq2_R(2 * t0);
        HVX_VectorPred mask_lt1 = Q6_Q_vsetq2_R(2 * t1);

        HVX_Vector v_m0         = Q6_V_vand_QV(mask_lt0, v_exp0);
        HVX_Vector v_m1         = Q6_V_vand_QV(mask_lt1, v_exp1);

        HVX_VectorPred mask_le0 = Q6_Q_vsetq2_R(2 * (t0 + 1));
        HVX_VectorPred mask_le1 = Q6_Q_vsetq2_R(2 * (t1 + 1));

        HVX_VectorPred mask_diag0 = Q6_Q_and_QQn(mask_le0, mask_lt0);
        HVX_VectorPred mask_diag1 = Q6_Q_and_QQn(mask_le1, mask_lt1);

        HVX_Vector v_a0         = Q6_V_vmux_QVV(mask_diag0, v_one_f16, v_m0);
        HVX_Vector v_a1         = Q6_V_vmux_QVV(mask_diag1, v_one_f16, v_m1);

        hvx_vmem(head->decay_m + t0 * 64) = v_m0;
        hvx_vmem(head->decay_a + t0 * 64) = v_a0;
        hvx_vmem(head->decay_m + t1 * 64) = v_m1;
        hvx_vmem(head->decay_a + t1 * 64) = v_a1;
    }

    {
        HVX_Vector v_gamma_t     = Q6_Vh_vsplat_R(gamma_u16[63]);
        HVX_Vector diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_VectorPred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_Vector diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_Vector v_exp         = hvx_vec_exp2_f16(diff_log2e);

        HVX_VectorPred mask_lt_t = Q6_Q_vsetq2_R(2 * 63);
        HVX_Vector v_m           = Q6_V_vand_QV(mask_lt_t, v_exp);

        HVX_VectorPred mask_le_t = Q6_Q_vcmp_eq_VhVh(v_zero_f16, v_zero_f16);
        HVX_VectorPred mask_diag = Q6_Q_and_QQn(mask_le_t, mask_lt_t);
        HVX_Vector v_a           = Q6_V_vmux_QVV(mask_diag, v_one_f16, v_m);

        hvx_vmem(head->decay_m + 63 * 64) = v_m;
        hvx_vmem(head->decay_a + 63 * 64) = v_a;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

static void gdn_hvx_phase2_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;

    gdn_unpack_64x64_tiles_to_vectors(head->rows_kk, head->kk_tiles);

    gdn_build_inv_l_blocks(
        head->inv_row_tiles,
        head->rows_kk,
        head->decay_m,
        head->b_f32[curr_buf],
        (__fp16 *) head->vtcm_m,
        (__fp16 *) head->vtcm_m + HMX_FP16_TILE_N_ELMS
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

static void gdn_hvx_phase3_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;

    gdn_unpack_64xS_tiles_to_f32(head->v_inter_f32, head->v_inter_tiles, S_v);

    HVX_VectorAlias local_b[2];
    local_b[0].v = hvx_vmem(head->b_f32[curr_buf] + 0);
    local_b[1].v = hvx_vmem(head->b_f32[curr_buf] + 32);

    for (uint32_t t = 0; t < 64; ++t) {
        HVX_Vector vb = hvx_vec_splat_f32(local_b[t / 32].fp32[t % 32]);
        for (uint32_t j = 0; j < S_v; j += 64) {
            HVX_Vector vv0 = hvx_vmem(head->v_f32[curr_buf] + t * S_v + j + 0);
            HVX_Vector vv1 = (j + 32 < S_v) ? hvx_vmem(head->v_f32[curr_buf] + t * S_v + j + 32) : Q6_V_vzero();
            HVX_Vector vi0 = hvx_vmem(head->v_inter_f32 + t * S_v + j + 0);
            HVX_Vector vi1 = (j + 32 < S_v) ? hvx_vmem(head->v_inter_f32 + t * S_v + j + 32) : Q6_V_vzero();

            HVX_Vector vp0 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0, vi0), vb);
            HVX_Vector vp1 = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1, vi1), vb);

            hvx_vmem(head->v_prime_f16 + t * S_v + j) = hvx_vec_f32_to_f16(vp0, vp1);
        }
    }

    hmx_interleave_cols_to_tiles(head->v_prime_col_tiles, head->v_prime_f16, 64, S_v, S_v, 2, 0, 64);

    gdn_unpack_64x64_tiles_to_vectors(head->rows_qk, head->qk_tiles);
    for (uint32_t t = 0; t < 64; ++t) {
        HVX_Vector v_decay_a = hvx_vmem(head->decay_a + t * 64);
        head->rows_a[t]      = hvx_vec_mul_f16_f16(head->rows_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_to_tiles(head->a_row_tiles, head->rows_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);
}

static void gdn_hvx_phase4_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t S_v = bctx->S_v;

    gdn_unpack_64xS_tiles_to_f16(head->delta_f16, head->delta_tiles, S_v);
    hmx_interleave_cols_to_tiles(head->delta_col_tiles, head->delta_f16, 64, S_v, S_v, 2, 0, 64);

    const uint16_t * decay_last = (const uint16_t *) (head->decay_a + 63 * 64);
    const HVX_Vector vzero      = Q6_V_vzero();

    for (uint32_t s = 0; s < 64; ++s) {
        HVX_Vector vs         = Q6_Vh_vsplat_R(decay_last[s]);
        HVX_VectorPred p_zero = Q6_Q_vcmp_eq_VhVh(vs, vzero);
        for (uint32_t j = 0; j < S_v; j += 64) {
            HVX_Vector vd   = hvx_vmem(head->delta_f16 + s * S_v + j);
            HVX_Vector prod = hvx_vec_mul_f16_f16(vd, vs);
            hvx_vmem(head->d_f16 + s * S_v + j) = Q6_V_vmux_QVV(p_zero, vzero, prod);
        }
    }

    gdn_pack_d_t_row_tiles(head->d_row_tiles, head->d_f16, S_v, head->vtcm_m, head->vtcm_tmp);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

static void gdn_hvx_phase5_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    const float scale = bctx->scale;

    gdn_unpack_64xS_tiles_to_f32(head->o_inter_f32, head->o_inter_tiles, S_v);
    gdn_unpack_64xS_tiles_to_f32(head->o_intra_f32, head->o_intra_tiles, S_v);

    HVX_Vector vscale = hvx_vec_splat_f32(scale);
    for (uint32_t j = 0; j < 64 * S_v / 32; ++j) {
        HVX_Vector vi = hvx_vmem(head->o_inter_f32 + j * 32);
        HVX_Vector va = hvx_vmem(head->o_intra_f32 + j * 32);
        hvx_vmem(head->o_f32[curr_buf] + j * 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(vi, va), vscale);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);
}

static void gdn_hvx_phase6_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = &bctx->octx->ctx->trace[i];
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t S_v = bctx->S_v;
    const uint32_t c = bctx->c;
    const uint32_t n_chunks = bctx->kparams->n_chunks;

    gdn_unpack_SxS_tiles_to_f32(head->s_update_f32, head->s_update_tiles, S_v);

    HVX_VectorAlias last_lambda;
    last_lambda.v = hvx_vmem(head->lambda_init + 32);
    HVX_Vector v_l_final = hvx_vec_splat_f32(last_lambda.fp32[31]);

    for (uint32_t j = 0; j < S_v * S_v / 32; ++j) {
        HVX_Vector vs_old = hvx_vmem(head->s_state + j * 32);
        HVX_Vector vsu    = hvx_vmem(head->s_update_f32 + j * 32);
        hvx_vmem(head->s_state + j * 32) = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vs_old, v_l_final), vsu);
    }

    if (c + 1 < n_chunks) {
        gdn_pack_s_col_tiles(head->s_col_tiles, head->s_f16, head->s_state, S_v);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}


static int gated_delta_net_f32_hmx_chunked(
    struct htp_ops_context * octx,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t row_start,
    uint32_t nrows
) {
    const struct htp_tensor * q         = octx->src[0];
    const struct htp_tensor * k         = octx->src[1];
    const struct htp_tensor * v         = octx->src[2];
    const struct htp_tensor * g         = octx->src[3];
    const struct htp_tensor * beta      = octx->src[4];
    const struct htp_tensor * state     = octx->src[5];
    const struct htp_tensor * dst       = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v        = kparams->S_v;
    const uint32_t H          = kparams->H;
    const uint32_t n_tokens   = kparams->n_tokens;
    const float    scale      = kparams->scale;
    const uint32_t chunk_size = kparams->chunk_size;
    const uint32_t n_chunks   = kparams->n_chunks;
    const uint32_t n_sv_tiles = S_v / 32;

    struct htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_size, kparams->n_heads_batch, kparams->n_threads, kparams->pipeline != 0);

    if (L.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    float * vtcm_g_raw[2] = {
        VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[1]) : VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[0])
    };
    float * vtcm_b_raw[2] = {
        VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[0]),
        L.pipeline ? VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[1]) : VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[0])
    };

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L.off_scales_1);
    hmx_init_column_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_queue_t  hmx_q = octx->ctx->hmx_queue;
    dma_queue *  dma_q = octx->ctx->dma[0];
    work_queue_t wp    = octx->ctx->work_queue;

    struct htp_gdn_head_ptrs heads[8];
    struct htp_gdn_hmx_gemm_task gemm_tasks[8][9];

    uint32_t n_batch = 1;
    for (uint32_t r = row_start; r < row_start + nrows; r += n_batch) {
        const uint32_t head_in_seq         = fastmodulo(r, H, &kparams->div_H);
        const uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        const uint32_t heads_left_in_seq   = H - head_in_seq;
        const uint32_t heads_left_in_range = (row_start + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_left_in_seq, heads_left_in_range));

        for (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_head_ptrs(&heads[h], &L, vtcm_base, h, head_in_seq, iv3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_tokens, chunk_size);
        }

        struct htp_gdn_batch_context bctx;
        bctx.heads      = heads;
        bctx.vtcm_g_raw = NULL;
        bctx.vtcm_b_raw = NULL;
        bctx.curr_buf   = 0;
        bctx.c          = 0;
        bctx.n_batch    = n_batch;
        bctx.S_v        = S_v;
        bctx.scale      = scale;
        bctx.octx       = octx;
        bctx.kparams    = kparams;

        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_push(dma_q, dma_make_data(heads[h].s_state, heads[h].state_in_dma),
                           S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_pop(dma_q);
        }

        if (n_chunks > 0) {
            work_queue_run(wp, gdn_hvx_init_state_worker, &bctx, n_batch);

            const uint32_t chunk0_tokens = hex_smin(chunk_size, n_tokens);
            for (uint32_t h = 0; h < n_batch; ++h) {
                gdn_dma_push_chunk_inputs(dma_q, heads[h].q_f32[0], heads[h].k_f32[0], heads[h].v_f32[0],
                                          q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                          heads[h].iv3, heads[h].iv1, 0, chunk0_tokens, S_v);
            }
            gdn_dma_push_chunk_gb(dma_q, vtcm_g_raw[0], vtcm_b_raw[0], g, beta, iv3, head_in_seq, 0, chunk0_tokens, n_batch);
        }

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t curr_buf = c & 1;
            const uint32_t next_buf = (c + 1) & 1;
            const uint32_t t_chunk  = c * chunk_size;

            bctx.curr_buf   = curr_buf;
            bctx.c          = c;
            bctx.vtcm_g_raw = vtcm_g_raw[curr_buf];
            bctx.vtcm_b_raw = vtcm_b_raw[curr_buf];

            for (uint32_t h = 0; h < n_batch; ++h) {
                dma_queue_pop(dma_q);
                dma_queue_pop(dma_q);
                dma_queue_pop(dma_q);
            }
            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);

            if (c + 1 < n_chunks) {
                const uint32_t next_t_chunk = (c + 1) * chunk_size;
                const uint32_t next_tokens  = hex_smin(chunk_size, n_tokens - next_t_chunk);
                for (uint32_t h = 0; h < n_batch; ++h) {
                    gdn_dma_push_chunk_inputs(dma_q, heads[h].q_f32[next_buf], heads[h].k_f32[next_buf], heads[h].v_f32[next_buf],
                                              q, k, v, heads[h].iq3, heads[h].iq1, heads[h].ik3, heads[h].ik1,
                                              heads[h].iv3, heads[h].iv1, next_t_chunk, next_tokens, S_v);
                }
                gdn_dma_push_chunk_gb(dma_q, vtcm_g_raw[next_buf], vtcm_b_raw[next_buf],
                                      g, beta, iv3, head_in_seq, next_t_chunk, next_tokens, n_batch);
            }

            if (c > 0) {
                for (uint32_t h = 0; h < n_batch; ++h) {
                    dma_queue_pop(dma_q);
                }
            }

            work_queue_run(wp, gdn_hvx_phase1a_worker, &bctx, n_batch);

            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][0], heads[h].k_row_tiles, heads[h].k_col_tiles, heads[h].kk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][2], heads[h].k_prime_row_tiles, heads[h].s_col_tiles, heads[h].v_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_queue_run(wp, gdn_hvx_phase1b_worker, &bctx, n_batch);

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][1], heads[h].q_row_tiles, heads[h].k_col_tiles, heads[h].qk_tiles, 2, 2, n_sv_tiles, vtcm_scales_1);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][3], heads[h].q_prime_row_tiles, heads[h].s_col_tiles, heads[h].o_inter_tiles, 2, n_sv_tiles, n_sv_tiles, vtcm_scales_1);
            }

            work_queue_run(wp, gdn_hvx_phase2_worker, &bctx, n_batch);

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][7],
                    (__fp16 *) heads[h].vtcm_m,
                    heads[h].inv_row_tiles + 0 * HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    1, 1, 1, vtcm_scales_1
                );
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(
                    hmx_q, &gemm_tasks[h][8],
                    (__fp16 *) heads[h].vtcm_m + HMX_FP16_TILE_N_ELMS,
                    (__fp16 *) heads[h].vtcm_tmp,
                    heads[h].inv_row_tiles + 2 * HMX_FP16_TILE_N_ELMS,
                    1, 1, 1, vtcm_scales_1
                );
            }

            work_queue_run(wp, gdn_hvx_phase3_worker, &bctx, n_batch);

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][4], heads[h].inv_row_tiles, heads[h].v_prime_col_tiles, heads[h].delta_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            work_queue_run(wp, gdn_hvx_phase4_worker, &bctx, n_batch);

            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][5], heads[h].a_row_tiles, heads[h].delta_col_tiles, heads[h].o_intra_tiles, 2, n_sv_tiles, 2, vtcm_scales_1);
            }
            for (uint32_t h = 0; h < n_batch; ++h) {
                htp_gdn_push_hmx_gemm_task(hmx_q, &gemm_tasks[h][6], heads[h].d_row_tiles, heads[h].k_col_tiles_64x128, heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2, vtcm_scales_1);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            work_queue_run(wp, gdn_hvx_phase5_worker, &bctx, n_batch);

            const uint32_t valid_tokens = hex_smin(chunk_size, n_tokens - t_chunk);
            for (uint32_t h = 0; h < n_batch; ++h) {
                const dma_addr_t attn_chunk_dma = dst->data +
                    ((uint64_t) heads[h].iv3 * n_tokens * H + (uint64_t) t_chunk * H + heads[h].iv1) * S_v * sizeof(float);
                dma_queue_push(dma_q, dma_make_data(attn_chunk_dma, heads[h].o_f32[curr_buf]),
                               dst->nb[1], S_v * sizeof(float), S_v * sizeof(float), valid_tokens);
            }

            for (uint32_t h = 0; h < n_batch; ++h) {
                hmx_queue_pop(hmx_q);
            }

            work_queue_run(wp, gdn_hvx_phase6_worker, &bctx, n_batch);
        }

        if (n_chunks > 0) {
            for (uint32_t h = 0; h < n_batch; ++h) {
                dma_queue_pop(dma_q);
            }
        }

        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_push(dma_q, dma_make_data(heads[h].state_out_dma, heads[h].s_state),
                           S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
        }
        for (uint32_t h = 0; h < n_batch; ++h) {
            dma_queue_pop(dma_q);
        }
    }

    dma_queue_flush(dma_q);
    return HTP_STATUS_OK;
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
        const bool can_use_hmx = (octx->ctx->hmx_enabled) &&
                                 (S_v % 64 == 0) &&
                                 (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                                 (g->ne[0] == 1) &&
                                 (K == 1);

        struct htp_gdn_hmx_vtcm_layout hmx_layout_local;
        struct htp_gdn_vtcm_layout hvx_layout_local;
        uint32_t n_heads_batch = 1;

        if (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout_local, S_v, HTP_GDN_CHUNK_SIZE, total_rows, octx->ctx->vtcm_size, n_threads, true, &n_heads_batch)) {
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
            kparams_local.pipeline        = hmx_layout_local.pipeline ? 1 : 0;
            kparams_local.chunk_size      = HTP_GDN_CHUNK_SIZE;
            kparams_local.n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
            kparams_local.n_heads_batch   = (uint16_t) n_heads_batch;
            kparams_local.vtcm_size       = (uint32_t) hmx_layout_local.total_bytes;
            kparams_local.state_aligned   = (uint32_t) hmx_layout_local.state_f32_bytes;
            kparams_local.vtcm_per_thread = (uint32_t) (hmx_layout_local.total_bytes / (n_threads > 0 ? n_threads : 1));
        } else {
            htp_gdn_vtcm_layout_build(&hvx_layout_local, S_v, n_threads);
            kparams_local.kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
            kparams_local.pipeline        = 0;
            kparams_local.n_heads_batch   = 1;
            kparams_local.state_aligned   = (uint32_t) hvx_layout_local.state_aligned;
            kparams_local.vtcm_per_thread = (uint32_t) hvx_layout_local.bytes_per_thread;
            kparams_local.vtcm_size       = (uint32_t) hvx_layout_local.total_bytes;
        }
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

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows,
            can_split ? 1 : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.count,
            &octx->ctx->mdev.count_div
        );
        row_start = range.start;
        nrows     = range.count;
    } else if (octx->op_params[1] != 0) {
        row_start = octx->op_params[1];
        nrows     = octx->op_params[2];
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        return gated_delta_net_f32_hmx_chunked(octx, kparams, row_start, nrows);
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
