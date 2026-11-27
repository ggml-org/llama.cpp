#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <HAP_farf.h>

#include "hvx-base.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-exp.h"
#include "hvx-repl.h"
#include "dma-queue.h"
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hmx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"
#include "gated-delta-net-ops.h"

struct htp_gdn_context;

typedef void (*gdn_load_vec_fn_t)(HVX_Vector * restrict dst, const void * restrict src, uint32_t n);
typedef void (*gdn_store_vec_fn_t)(void * restrict dst, const HVX_Vector * restrict src, uint32_t n);
typedef void (*gdn_step_fn_t)(
    const struct htp_gdn_context * gctx,
    float * restrict s_work,
    void * restrict attn_out,
    const void * restrict q_t,
    const void * restrict k_t,
    const void * restrict v_t,
    const float * restrict g_t,
    const float * restrict beta_t,
    HVX_Vector * restrict vtcm_scratch,
    float scale,
    uint32_t S_v,
    const void * restrict k_next,
    bool has_memory
);

struct htp_gdn_context {
    struct htp_ops_context * octx;
    const struct htp_gdn_kernel_params * kparams;
    struct htp_gdn_vtcm_layout layout;
    uint8_t * vtcm_base;
    const void * ids_data;
    const int32_t * n_rep_data;
    uint32_t ids_type;
    uint32_t n_pend;
    bool replay;
    bool keep;
    uint32_t row_start;
    uint32_t nrows;
    gdn_load_vec_fn_t     load_q;
    gdn_load_vec_fn_t     load_k;
    gdn_load_vec_fn_t     load_v;
    gdn_store_vec_fn_t    store_out;
    gdn_step_fn_t         step_fn;
};

static __attribute__((noinline)) HVX_Vector gdn_mul_dot_f32(float * restrict dst, const HVX_Vector * restrict mul, const HVX_Vector * restrict dot, uint32_t n) {
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

static __attribute__((noinline)) HVX_Vector gdn_mul_scalar_dot_f32(float * restrict dst, HVX_Vector vmul, const HVX_Vector * restrict dot, uint32_t n) {
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

static __attribute__((noinline)) HVX_Vector gdn_add_scaled_dot_f32(float * restrict dst, const HVX_Vector * restrict src,
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

static __attribute__((noinline)) HVX_Vector gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
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

static __attribute__((noinline)) HVX_Vector gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
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

static __attribute__((noinline)) HVX_Vector gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));

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

static __attribute__((noinline)) HVX_Vector gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
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

static __attribute__((noinline)) HVX_Vector gdn_mul_scalar_dot8_f32(float * restrict dst0, float * restrict dst1,
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

static __attribute__((noinline)) HVX_Vector gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));
    const HVX_Vector scale4 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 4 * sizeof(float)));
    const HVX_Vector scale5 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 5 * sizeof(float)));
    const HVX_Vector scale6 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 6 * sizeof(float)));
    const HVX_Vector scale7 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 7 * sizeof(float)));

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

static __attribute__((noinline)) void gdn_add_scaled_f32(float * restrict dst,
        const HVX_Vector * restrict src, HVX_Vector vscale, uint32_t n) {
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd  = hvx_vmemu(dst + i * epv);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(src[i], vscale));
        hvx_vmemu(dst + i * epv) = out;
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector out = hvx_vec_add_f32_f32(hvx_vmemu(dst + off), hvx_vec_mul_f32_f32(src[nvec], vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
    }
}

static __attribute__((noinline)) void gdn_add_scaled4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3,
        const HVX_Vector * restrict src, HVX_Vector vscale, uint32_t n) {
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = src[i];
        hvx_vmemu(dst0 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        hvx_vmemu(dst1 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        hvx_vmemu(dst2 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        hvx_vmemu(dst3 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = src[nvec];
        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0)));
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1)));
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2)));
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3)));
    }
}

static __attribute__((noinline)) void gdn_add_scaled8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const HVX_Vector * restrict src, HVX_Vector vscale, uint32_t n) {
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));
    const HVX_Vector scale4 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 4 * sizeof(float)));
    const HVX_Vector scale5 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 5 * sizeof(float)));
    const HVX_Vector scale6 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 6 * sizeof(float)));
    const HVX_Vector scale7 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 7 * sizeof(float)));

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = src[i];
        hvx_vmemu(dst0 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        hvx_vmemu(dst1 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        hvx_vmemu(dst2 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        hvx_vmemu(dst3 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        hvx_vmemu(dst4 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        hvx_vmemu(dst5 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        hvx_vmemu(dst6 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        hvx_vmemu(dst7 + i * epv) = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = src[nvec];
        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0)));
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1)));
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2)));
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3)));
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4)));
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5)));
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6)));
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7)));
    }
}

static __attribute__((noinline)) HVX_Vector gdn_dot_f32(const float * restrict src,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(hvx_vmemu(src + i * epv), dot[i]));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src + off), dot[nvec]), zero));
    }
    return hvx_vec_reduce_sum_f32(acc);
}

static __attribute__((noinline)) HVX_Vector gdn_dot4_f32(const float * restrict src0, const float * restrict src1,
        const float * restrict src2, const float * restrict src3,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = dot[i];
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(hvx_vmemu(src0 + i * epv), vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(hvx_vmemu(src1 + i * epv), vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(hvx_vmemu(src2 + i * epv), vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(hvx_vmemu(src3 + i * epv), vdot));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();
        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src0 + off), vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src1 + off), vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src2 + off), vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src3 + off), vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    return hvx_vec_reduce_sum_f32x4(acc);
}

static __attribute__((noinline)) HVX_Vector gdn_dot8_f32(const float * restrict src0, const float * restrict src1,
        const float * restrict src2, const float * restrict src3, const float * restrict src4,
        const float * restrict src5, const float * restrict src6, const float * restrict src7,
        const HVX_Vector * restrict dot, uint32_t n) {
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
        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(hvx_vmemu(src0 + i * epv), vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(hvx_vmemu(src1 + i * epv), vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(hvx_vmemu(src2 + i * epv), vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(hvx_vmemu(src3 + i * epv), vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(hvx_vmemu(src4 + i * epv), vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(hvx_vmemu(src5 + i * epv), vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(hvx_vmemu(src6 + i * epv), vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(hvx_vmemu(src7 + i * epv), vdot));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = dot[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();
        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src0 + off), vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src1 + off), vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src2 + off), vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src3 + off), vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src4 + off), vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src5 + off), vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src6 + off), vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(hvx_vmemu(src7 + off), vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    HVX_Vector rA = hvx_vec_reduce_sum_f32x4(accA);
    HVX_Vector rB = hvx_vec_reduce_sum_f32x4(accB);
    HVX_VectorPred q16 = Q6_Q_vsetq2_R(16);
    return Q6_V_vmux_QVV(q16, rA, Q6_V_vror_VR(rB, 128 - 16));
}

static __attribute__((noinline)) HVX_Vector gdn_scale_add_dot_f32(float * restrict dst, HVX_Vector vgate,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd   = hvx_vec_mul_f32_f32(hvx_vmemu(dst + i * epv), vgate);
        HVX_Vector out  = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(src[i], vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, dot[i]));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector vd   = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vgate);
        HVX_Vector out  = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(src[nvec], vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, dot[nvec]), zero));
    }
    return hvx_vec_reduce_sum_f32(acc);
}

static __attribute__((noinline)) HVX_Vector gdn_scale_add_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, HVX_Vector vgate,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot, uint32_t n) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs   = src[i];
        HVX_Vector vdot = dot[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale3));

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

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vgate), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vgate), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vgate), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vgate), hvx_vec_mul_f32_f32(vs, scale3));

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

static __attribute__((noinline)) HVX_VectorPair gdn_scale_add_dual_dot_f32(float * restrict dst, HVX_Vector vgate,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot_q, const HVX_Vector * restrict dot_k, uint32_t n) {
    HVX_Vector acc_q = Q6_V_vzero();
    HVX_Vector acc_k = Q6_V_vzero();
    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd  = hvx_vec_mul_f32_f32(hvx_vmemu(dst + i * epv), vgate);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(src[i], vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc_q = hvx_vec_add_f32_f32(acc_q, hvx_vec_mul_f32_f32(out, dot_q[i]));
        acc_k = hvx_vec_add_f32_f32(acc_k, hvx_vec_mul_f32_f32(out, dot_k[i]));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector vd  = hvx_vec_mul_f32_f32(hvx_vmemu(dst + off), vgate);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(src[nvec], vscale));
        hvx_vec_store_u(dst + off, nloe * sizeof(float), out);
        acc_q = hvx_vec_add_f32_f32(acc_q, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, dot_q[nvec]), zero));
        acc_k = hvx_vec_add_f32_f32(acc_k, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out, dot_k[nvec]), zero));
    }
    return Q6_W_vcombine_VV(hvx_vec_reduce_sum_f32(acc_k), hvx_vec_reduce_sum_f32(acc_q));
}

static __attribute__((noinline)) HVX_VectorPair gdn_scale_add_dual_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, HVX_Vector vgate,
        const HVX_Vector * restrict src, HVX_Vector vscale,
        const HVX_Vector * restrict dot_q, const HVX_Vector * restrict dot_k, uint32_t n) {
    HVX_Vector aq0 = Q6_V_vzero();
    HVX_Vector aq1 = Q6_V_vzero();
    HVX_Vector aq2 = Q6_V_vzero();
    HVX_Vector aq3 = Q6_V_vzero();
    HVX_Vector ak0 = Q6_V_vzero();
    HVX_Vector ak1 = Q6_V_vzero();
    HVX_Vector ak2 = Q6_V_vzero();
    HVX_Vector ak3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_repl_f32(vscale);
    const HVX_Vector scale1 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 1 * sizeof(float)));
    const HVX_Vector scale2 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 2 * sizeof(float)));
    const HVX_Vector scale3 = hvx_vec_repl_f32(Q6_V_vror_VR(vscale, 3 * sizeof(float)));

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = src[i];
        HVX_Vector vq = dot_q[i];
        HVX_Vector vk = dot_k[i];

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vgate), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        aq0 = hvx_vec_add_f32_f32(aq0, hvx_vec_mul_f32_f32(out0, vq));
        aq1 = hvx_vec_add_f32_f32(aq1, hvx_vec_mul_f32_f32(out1, vq));
        aq2 = hvx_vec_add_f32_f32(aq2, hvx_vec_mul_f32_f32(out2, vq));
        aq3 = hvx_vec_add_f32_f32(aq3, hvx_vec_mul_f32_f32(out3, vq));

        ak0 = hvx_vec_add_f32_f32(ak0, hvx_vec_mul_f32_f32(out0, vk));
        ak1 = hvx_vec_add_f32_f32(ak1, hvx_vec_mul_f32_f32(out1, vk));
        ak2 = hvx_vec_add_f32_f32(ak2, hvx_vec_mul_f32_f32(out2, vk));
        ak3 = hvx_vec_add_f32_f32(ak3, hvx_vec_mul_f32_f32(out3, vk));
    }
    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = src[nvec];
        HVX_Vector vq = dot_q[nvec];
        HVX_Vector vk = dot_k[nvec];
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vgate), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vgate), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vgate), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vgate), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        aq0 = hvx_vec_add_f32_f32(aq0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vq), zero));
        aq1 = hvx_vec_add_f32_f32(aq1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vq), zero));
        aq2 = hvx_vec_add_f32_f32(aq2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vq), zero));
        aq3 = hvx_vec_add_f32_f32(aq3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vq), zero));

        ak0 = hvx_vec_add_f32_f32(ak0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vk), zero));
        ak1 = hvx_vec_add_f32_f32(ak1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vk), zero));
        ak2 = hvx_vec_add_f32_f32(ak2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vk), zero));
        ak3 = hvx_vec_add_f32_f32(ak3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vk), zero));
    }

    HVX_Vector_x4 acc_q = { .v = { aq0, aq1, aq2, aq3 } };
    HVX_Vector_x4 acc_k = { .v = { ak0, ak1, ak2, ak3 } };
    return Q6_W_vcombine_VV(hvx_vec_reduce_sum_f32x4(acc_k), hvx_vec_reduce_sum_f32x4(acc_q));
}

static inline HVX_Vector gdn_clamp_f16_range_f32(HVX_Vector v, HVX_Vector v_min, HVX_Vector v_max) {
    HVX_VectorPred p_gt = Q6_Q_vcmp_gt_VsfVsf(v, v_max);
    v = Q6_V_vmux_QVV(p_gt, v_max, v);
    HVX_VectorPred p_lt = Q6_Q_vcmp_gt_VsfVsf(v_min, v);
    return Q6_V_vmux_QVV(p_lt, v_min, v);
}

static inline HVX_Vector hvx_vec_load_u(const void * restrict src, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) ((uintptr_t) src & ~(uintptr_t) 127);
    const uint32_t left_off  = (uint32_t) ((uintptr_t) src & 127);
    const uint32_t right_off = left_off + n;
    HVX_Vector v = (right_off > 128) ? Q6_V_valign_VVR(vsrc[1], vsrc[0], left_off) : Q6_V_vror_VR(vsrc[0], left_off);
    return Q6_V_vand_QV(Q6_Q_vsetq2_R(n), v);
}

static __attribute__((noinline)) void gdn_load_vec_f16_to_f32(HVX_Vector * restrict dst, const void * restrict src, uint32_t n) {
    const __fp16 * restrict s16 = (const __fp16 *) src;
    const uint32_t n64 = n / 64;
    const uint32_t rem = n % 64;
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < n64; ++i) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(s16 + i * 64));
        dst[2 * i + 0] = Q6_V_lo_W(p);
        dst[2 * i + 1] = Q6_V_hi_W(p);
    }
    if (rem) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vec_load_u(s16 + n64 * 64, rem * sizeof(__fp16)));
        dst[2 * n64 + 0] = Q6_V_lo_W(p);
        dst[2 * n64 + 1] = Q6_V_hi_W(p);
    }
}

static __attribute__((noinline)) void gdn_load_vec_f32(HVX_Vector * restrict dst, const void * restrict src, uint32_t n) {
    const float * restrict s32 = (const float *) src;
    const uint32_t n32 = n / 32;
    const uint32_t rem = n % 32;
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < n32; ++i) {
        dst[i] = hvx_vmemu(s32 + i * 32);
    }
    if (rem) {
        dst[n32] = hvx_vec_load_u(s32 + n32 * 32, rem * sizeof(float));
    }
}

static __attribute__((noinline)) void gdn_store_vec_f16(void * restrict dst, const HVX_Vector * restrict src, uint32_t n) {
    __fp16 * restrict d16 = (__fp16 *) dst;
    const HVX_Vector v_min = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector v_max = hvx_vec_splat_f32(65504.0f);
    const uint32_t n64 = n / 64;
    const uint32_t rem = n % 64;
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < n64; ++i) {
        HVX_Vector v0 = gdn_clamp_f16_range_f32(src[2 * i + 0], v_min, v_max);
        HVX_Vector v1 = gdn_clamp_f16_range_f32(src[2 * i + 1], v_min, v_max);
        hvx_vmemu(d16 + i * 64) = hvx_vec_f32_to_f16(v0, v1);
    }
    if (rem) {
        HVX_Vector v0 = gdn_clamp_f16_range_f32(src[2 * n64 + 0], v_min, v_max);
        HVX_Vector v1 = (rem > 32) ? gdn_clamp_f16_range_f32(src[2 * n64 + 1], v_min, v_max) : Q6_V_vzero();
        hvx_vec_store_u(d16 + n64 * 64, rem * sizeof(__fp16), hvx_vec_f32_to_f16(v0, v1));
    }
}

static __attribute__((noinline)) void gdn_store_vec_f32(void * restrict dst, const HVX_Vector * restrict src, uint32_t n) {
    float * restrict d32 = (float *) dst;
    const uint32_t n32 = n / 32;
    const uint32_t rem = n % 32;
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < n32; ++i) {
        hvx_vmemu(d32 + i * 32) = src[i];
    }
    if (rem) {
        hvx_vec_store_u(d32 + n32 * 32, rem * sizeof(float), src[n32]);
    }
}

static __attribute__((noinline)) HVX_Vector gdn_vec_exp_f32(HVX_Vector v) {
    return hvx_vec_exp_f32(v);
}

static __attribute__((noinline)) void gdn_step_kda_f32(
    const struct htp_gdn_context * gctx,
    float * restrict s_work,
    void * restrict attn_out,
    const void * restrict q_t,
    const void * restrict k_t,
    const void * restrict v_t,
    const float * restrict g_t,
    const float * restrict beta_t,
    HVX_Vector * restrict vtcm_scratch,
    float scale,
    uint32_t S_v,
    const void * restrict k_next,
    bool has_memory
) {
    (void) k_next;
    (void) has_memory;
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector * restrict vq  = vtcm_scratch + 0;
    HVX_Vector * restrict vk  = vtcm_scratch + 4;
    HVX_Vector * restrict vg  = vtcm_scratch + 8;
    float * restrict v_buf    = (float *) (vtcm_scratch + 12);
    float * restrict out_buf  = (float *) (vtcm_scratch + 16);

    gctx->load_q(vq, q_t, S_v);
    gctx->load_k(vk, k_t, S_v);
    gctx->load_v((HVX_Vector *) v_buf, v_t, S_v);

    for (uint32_t i = 0; i < nvec; ++i) {
        vg[i] = gdn_vec_exp_f32(hvx_vmemu(g_t + i * epv));
    }
    if (nloe) {
        vg[nvec] = gdn_vec_exp_f32(hvx_vec_load_u(g_t + nvec * epv, nloe * sizeof(float)));
    }

    const HVX_Vector vbeta  = hvx_vec_repl_f32(hvx_vmemu(beta_t));
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (j + 0) * S_v;
        float * row1 = s_work + (j + 1) * S_v;
        float * row2 = s_work + (j + 2) * S_v;
        float * row3 = s_work + (j + 3) * S_v;
        float * row4 = s_work + (j + 4) * S_v;
        float * row5 = s_work + (j + 5) * S_v;
        float * row6 = s_work + (j + 6) * S_v;
        float * row7 = s_work + (j + 7) * S_v;

        HVX_Vector vsums = gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                            vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);

        HVX_Vector vattn = gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                                   vk, vdelta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(out_buf + j, 8 * sizeof(float), res_attn);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (j + 0) * S_v;
        float * row1 = s_work + (j + 1) * S_v;
        float * row2 = s_work + (j + 2) * S_v;
        float * row3 = s_work + (j + 3) * S_v;

        HVX_Vector vsums = gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
        HVX_Vector diff   = hvx_vec_sub_f32_f32(vv_t, vsums);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(diff, vbeta);

        HVX_Vector vattn = gdn_add_scaled_dot4_f32(row0, row1, row2, row3, vk, vdelta, vq, S_v);

        HVX_Vector res_attn = hvx_vec_mul_f32_f32(vattn, vscale);
        hvx_vec_store_u(out_buf + j, 4 * sizeof(float), res_attn);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + j * S_v;
        HVX_Vector vsum = gdn_mul_dot_f32(row, vg, vk, S_v);
        HVX_Vector vv_t = hvx_vec_repl_f32(hvx_vmemu(v_buf + j));
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        HVX_Vector vres = gdn_add_scaled_dot_f32(row, vk, vdj, vq, S_v);
        hvx_vec_store_u(out_buf + j, sizeof(float), hvx_vec_mul_f32_f32(vres, vscale));
    }

    gctx->store_out(attn_out, (const HVX_Vector *) out_buf, S_v);
}

static __attribute__((noinline)) void gdn_step_scalar_f32(
    const struct htp_gdn_context * gctx,
    float * restrict s_work,
    void * restrict attn_out,
    const void * restrict q_t,
    const void * restrict k_t,
    const void * restrict v_t,
    const float * restrict g_t,
    const float * restrict beta_t,
    HVX_Vector * restrict vtcm_scratch,
    float scale,
    uint32_t S_v,
    const void * restrict k_next,
    bool has_memory
) {
    HVX_Vector * restrict vq      = vtcm_scratch + 0;
    HVX_Vector * restrict vk      = vtcm_scratch + 4;
    HVX_Vector * restrict vk_next = vtcm_scratch + 8;
    float * restrict v_buf        = (float *) (vtcm_scratch + 12);
    float * restrict out_buf      = (float *) (vtcm_scratch + 16);
    float * restrict mem_buf      = (float *) (vtcm_scratch + 20);

    gctx->load_q(vq, q_t, S_v);
    gctx->load_k(vk, k_t, S_v);
    gctx->load_v((HVX_Vector *) v_buf, v_t, S_v);

    const HVX_Vector vgate  = gdn_vec_exp_f32(hvx_vec_repl_f32(hvx_vmemu(g_t)));
    const HVX_Vector vbeta  = hvx_vec_repl_f32(hvx_vmemu(beta_t));
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    if (!has_memory) {
        if (!k_next) {
            uint32_t j = 0;
            for (; j + 8 <= S_v; j += 8) {
                float * row0 = s_work + (j + 0) * S_v;
                float * row1 = s_work + (j + 1) * S_v;
                float * row2 = s_work + (j + 2) * S_v;
                float * row3 = s_work + (j + 3) * S_v;
                float * row4 = s_work + (j + 4) * S_v;
                float * row5 = s_work + (j + 5) * S_v;
                float * row6 = s_work + (j + 6) * S_v;
                float * row7 = s_work + (j + 7) * S_v;

                HVX_Vector vmem   = gdn_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7, vk, S_v);
                HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
                HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);

                HVX_Vector vattn0 = gdn_scale_add_dot4_f32(row0, row1, row2, row3, vgate, vk, vdelta, vq, S_v);
                HVX_Vector vattn1 = gdn_scale_add_dot4_f32(row4, row5, row6, row7, vgate, vk,
                                                           Q6_V_vror_VR(vdelta, 4 * sizeof(float)), vq, S_v);
                hvx_vec_store_u(out_buf + j + 0, 4 * sizeof(float), hvx_vec_mul_f32_f32(vattn0, vscale));
                hvx_vec_store_u(out_buf + j + 4, 4 * sizeof(float), hvx_vec_mul_f32_f32(vattn1, vscale));
            }
            for (; j + 4 <= S_v; j += 4) {
                float * row0 = s_work + (j + 0) * S_v;
                float * row1 = s_work + (j + 1) * S_v;
                float * row2 = s_work + (j + 2) * S_v;
                float * row3 = s_work + (j + 3) * S_v;

                HVX_Vector vmem   = gdn_dot4_f32(row0, row1, row2, row3, vk, S_v);
                HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
                HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);

                HVX_Vector vattn = gdn_scale_add_dot4_f32(row0, row1, row2, row3, vgate, vk, vdelta, vq, S_v);
                hvx_vec_store_u(out_buf + j, 4 * sizeof(float), hvx_vec_mul_f32_f32(vattn, vscale));
            }
            for (; j < S_v; ++j) {
                float * row     = s_work + j * S_v;
                HVX_Vector vmem = gdn_dot_f32(row, vk, S_v);
                HVX_Vector vv_t = hvx_vec_repl_f32(hvx_vmemu(v_buf + j));
                HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);
                HVX_Vector vres = gdn_scale_add_dot_f32(row, vgate, vk, vdj, vq, S_v);
                hvx_vec_store_u(out_buf + j, sizeof(float), hvx_vec_mul_f32_f32(vres, vscale));
            }
            gctx->store_out(attn_out, (const HVX_Vector *) out_buf, S_v);
            return;
        }

        // Token 0 of multi-token tile: read-only Pass 1 to seed mem_buf
        uint32_t j = 0;
        for (; j + 8 <= S_v; j += 8) {
            float * row0 = s_work + (j + 0) * S_v;
            float * row1 = s_work + (j + 1) * S_v;
            float * row2 = s_work + (j + 2) * S_v;
            float * row3 = s_work + (j + 3) * S_v;
            float * row4 = s_work + (j + 4) * S_v;
            float * row5 = s_work + (j + 5) * S_v;
            float * row6 = s_work + (j + 6) * S_v;
            float * row7 = s_work + (j + 7) * S_v;

            HVX_Vector vmem = gdn_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7, vk, S_v);
            hvx_vec_store_u(mem_buf + j, 8 * sizeof(float), vmem);
        }
        for (; j + 4 <= S_v; j += 4) {
            float * row0 = s_work + (j + 0) * S_v;
            float * row1 = s_work + (j + 1) * S_v;
            float * row2 = s_work + (j + 2) * S_v;
            float * row3 = s_work + (j + 3) * S_v;

            HVX_Vector vmem = gdn_dot4_f32(row0, row1, row2, row3, vk, S_v);
            hvx_vec_store_u(mem_buf + j, 4 * sizeof(float), vmem);
        }
        for (; j < S_v; ++j) {
            HVX_Vector vmem = gdn_dot_f32(s_work + j * S_v, vk, S_v);
            hvx_vec_store_u(mem_buf + j, sizeof(float), vmem);
        }
    }

    if (k_next) {
        gctx->load_k(vk_next, k_next, S_v);

        uint32_t j = 0;
        for (; j + 4 <= S_v; j += 4) {
            float * row0 = s_work + (j + 0) * S_v;
            float * row1 = s_work + (j + 1) * S_v;
            float * row2 = s_work + (j + 2) * S_v;
            float * row3 = s_work + (j + 3) * S_v;

            HVX_Vector vmem   = hvx_vmemu(mem_buf + j);
            HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
            HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);

            HVX_VectorPair vp = gdn_scale_add_dual_dot4_f32(row0, row1, row2, row3,
                                                            vgate, vk, vdelta, vq, vk_next, S_v);
            hvx_vec_store_u(out_buf + j, 4 * sizeof(float), hvx_vec_mul_f32_f32(Q6_V_lo_W(vp), vscale));
            hvx_vec_store_u(mem_buf + j, 4 * sizeof(float), Q6_V_hi_W(vp));
        }
        for (; j < S_v; ++j) {
            float * row       = s_work + j * S_v;
            HVX_Vector vmem   = hvx_vec_repl_f32(hvx_vmemu(mem_buf + j));
            HVX_Vector vv_t   = hvx_vec_repl_f32(hvx_vmemu(v_buf + j));
            HVX_Vector vdj    = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);
            HVX_VectorPair vp = gdn_scale_add_dual_dot_f32(row, vgate, vk, vdj, vq, vk_next, S_v);
            hvx_vec_store_u(out_buf + j, sizeof(float), hvx_vec_mul_f32_f32(Q6_V_lo_W(vp), vscale));
            hvx_vec_store_u(mem_buf + j, sizeof(float), Q6_V_hi_W(vp));
        }
    } else {
        uint32_t j = 0;
        for (; j + 4 <= S_v; j += 4) {
            float * row0 = s_work + (j + 0) * S_v;
            float * row1 = s_work + (j + 1) * S_v;
            float * row2 = s_work + (j + 2) * S_v;
            float * row3 = s_work + (j + 3) * S_v;

            HVX_Vector vmem   = hvx_vmemu(mem_buf + j);
            HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
            HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);

            HVX_Vector vattn = gdn_scale_add_dot4_f32(row0, row1, row2, row3, vgate, vk, vdelta, vq, S_v);
            hvx_vec_store_u(out_buf + j, 4 * sizeof(float), hvx_vec_mul_f32_f32(vattn, vscale));
        }
        for (; j < S_v; ++j) {
            float * row     = s_work + j * S_v;
            HVX_Vector vmem = hvx_vec_repl_f32(hvx_vmemu(mem_buf + j));
            HVX_Vector vv_t = hvx_vec_repl_f32(hvx_vmemu(v_buf + j));
            HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, hvx_vec_mul_f32_f32(vmem, vgate)), vbeta);
            HVX_Vector vres = gdn_scale_add_dot_f32(row, vgate, vk, vdj, vq, S_v);
            hvx_vec_store_u(out_buf + j, sizeof(float), hvx_vec_mul_f32_f32(vres, vscale));
        }
    }

    gctx->store_out(attn_out, (const HVX_Vector *) out_buf, S_v);
}

static inline uint32_t gdn_read_seq_idx(const void * ids_data, uint32_t ids_type, uint32_t iv3) {
    if (!ids_data) {
        return iv3;
    }
    return (ids_type == HTP_TYPE_I64)
        ? (uint32_t) (((const int64_t *) ids_data)[iv3])
        : (uint32_t) (((const int32_t *) ids_data)[iv3]);
}

static inline bool gdn_has_replay(const struct htp_ops_context * octx) {
    return octx->src[6] != NULL && octx->src[7] != NULL;
}

static inline int gdn_prepare_aux_data(
    struct htp_ops_context * octx,
    const struct htp_tensor * t,
    void ** out_buf
) {
    if (!t) {
        *out_buf = NULL;
        return HTP_STATUS_OK;
    }
    const size_t nbytes = hex_round_up(htp_tensor_nbytes(t), 128);
    void * buf = (nbytes <= octx->ctx->ddr_spad_size) ? octx->ctx->ddr_spad_base : memalign(128, nbytes);
    if (!buf) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    if (htp_tensor_read(&octx->ctx->main_th, buf, nbytes, t) == 0) {
        if (buf != octx->ctx->ddr_spad_base) {
            free(buf);
        }
        return HTP_STATUS_INTERNAL_ERR;
    }
    *out_buf = buf;
    return HTP_STATUS_OK;
}

static inline void gdn_dma_push_tile_inputs(
    dma_queue * dma_q,
    void * vtcm_q,
    void * vtcm_k,
    void * vtcm_v,
    void * vtcm_g,
    void * vtcm_b,
    const struct htp_tensor * q,
    const struct htp_tensor * k,
    const struct htp_tensor * v,
    const struct htp_tensor * g,
    const struct htp_tensor * beta,
    uint32_t iq3, uint32_t iq1,
    uint32_t ik3, uint32_t ik1,
    uint32_t iv3, uint32_t iv1,
    uint32_t t_base,
    uint32_t tile_tokens,
    uint32_t q_row_bytes,
    uint32_t k_row_bytes,
    uint32_t v_row_bytes,
    uint32_t g_row_bytes
) {
    const dma_addr_t q_dma = q->data + ((size_t) iq3 * q->nb[3] + (size_t) t_base * q->nb[2] + (size_t) iq1 * q->nb[1]);
    const dma_addr_t k_dma = k->data + ((size_t) ik3 * k->nb[3] + (size_t) t_base * k->nb[2] + (size_t) ik1 * k->nb[1]);
    const dma_addr_t v_dma = v->data + ((size_t) iv3 * v->nb[3] + (size_t) t_base * v->nb[2] + (size_t) iv1 * v->nb[1]);
    const dma_addr_t g_dma = g->data + ((size_t) iv3 * g->nb[3] + (size_t) t_base * g->nb[2] + (size_t) iv1 * g->nb[1]);
    const dma_addr_t b_dma = beta->data + ((size_t) iv3 * beta->nb[3] + (size_t) t_base * beta->nb[2] + (size_t) iv1 * beta->nb[1]);

    dma_queue_push(dma_q, dma_make_data(vtcm_q, q_dma), q_row_bytes, q->nb[2], q_row_bytes, tile_tokens);
    dma_queue_push(dma_q, dma_make_data(vtcm_k, k_dma), k_row_bytes, k->nb[2], k_row_bytes, tile_tokens);
    dma_queue_push(dma_q, dma_make_data(vtcm_v, v_dma), v_row_bytes, v->nb[2], v_row_bytes, tile_tokens);
    dma_queue_push(dma_q, dma_make_data(vtcm_g, g_dma), g_row_bytes, g->nb[2], g_row_bytes, tile_tokens);
    dma_queue_push(dma_q, dma_make_data(vtcm_b, b_dma), sizeof(float), beta->nb[2], sizeof(float), tile_tokens);
}

static __attribute__((noinline)) void gdn_replay_step_f32(
    float * restrict s_work,
    const float * restrict k_t,
    const float * restrict v_t,
    const float * restrict g_t,
    const float * restrict beta_t,
    HVX_Vector * restrict vtcm_scratch,
    bool kda,
    uint32_t S_v
) {
    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = S_v / epv;
    const uint32_t nloe = S_v % epv;

    HVX_Vector * restrict vk = vtcm_scratch + 0;
    HVX_Vector * restrict vg = vtcm_scratch + 4;
    float * restrict v_buf   = (float *) (vtcm_scratch + 8);

    gdn_load_vec_f32(vk, k_t, S_v);
    gdn_load_vec_f32((HVX_Vector *) v_buf, v_t, S_v);

    HVX_Vector vgate = Q6_V_vzero();
    if (kda) {
        for (uint32_t i = 0; i < nvec; ++i) {
            vg[i] = gdn_vec_exp_f32(hvx_vmemu(g_t + i * epv));
        }
        if (nloe) {
            vg[nvec] = gdn_vec_exp_f32(hvx_vec_load_u(g_t + nvec * epv, nloe * sizeof(float)));
        }
    } else {
        vgate = gdn_vec_exp_f32(hvx_vec_repl_f32(hvx_vmemu(g_t)));
    }

    const HVX_Vector vbeta = hvx_vec_repl_f32(hvx_vmemu(beta_t));

    uint32_t j = 0;
    for (; j + 8 <= S_v; j += 8) {
        float * row0 = s_work + (j + 0) * S_v;
        float * row1 = s_work + (j + 1) * S_v;
        float * row2 = s_work + (j + 2) * S_v;
        float * row3 = s_work + (j + 3) * S_v;
        float * row4 = s_work + (j + 4) * S_v;
        float * row5 = s_work + (j + 5) * S_v;
        float * row6 = s_work + (j + 6) * S_v;
        float * row7 = s_work + (j + 7) * S_v;

        HVX_Vector vsums = kda
            ? gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7, vg, vk, S_v)
            : gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsums), vbeta);

        gdn_add_scaled8_f32(row0, row1, row2, row3, row4, row5, row6, row7, vk, vdelta, S_v);
    }
    for (; j + 4 <= S_v; j += 4) {
        float * row0 = s_work + (j + 0) * S_v;
        float * row1 = s_work + (j + 1) * S_v;
        float * row2 = s_work + (j + 2) * S_v;
        float * row3 = s_work + (j + 3) * S_v;

        HVX_Vector vsums = kda
            ? gdn_mul_dot4_f32(row0, row1, row2, row3, vg, vk, S_v)
            : gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, vgate, vk, S_v);

        HVX_Vector vv_t   = hvx_vmemu(v_buf + j);
        HVX_Vector vdelta = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsums), vbeta);

        gdn_add_scaled4_f32(row0, row1, row2, row3, vk, vdelta, S_v);
    }
    for (; j < S_v; ++j) {
        float * row = s_work + j * S_v;
        HVX_Vector vsum = kda
            ? gdn_mul_dot_f32(row, vg, vk, S_v)
            : gdn_mul_scalar_dot_f32(row, vgate, vk, S_v);
        HVX_Vector vv_t = hvx_vec_repl_f32(hvx_vmemu(v_buf + j));
        HVX_Vector vdj  = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv_t, vsum), vbeta);
        gdn_add_scaled_f32(row, vk, vdj, S_v);
    }
}

static void gdn_replay_f32(
    const struct htp_ops_context * octx,
    dma_queue * dma_q,
    float * restrict s_work,
    float * restrict k_buf,
    float * restrict v_buf,
    float * restrict g_buf,
    float * restrict b_buf,
    HVX_Vector * restrict vtcm_scratch,
    uint32_t tile_cap,
    uint32_t ik1,
    uint32_t iv1,
    uint32_t iv3,
    uint32_t n_steps,
    uint32_t n_pend,
    bool kda,
    uint32_t S_v
) {
    const struct htp_tensor * pend_k  = octx->src[6];
    const struct htp_tensor * pend_v  = octx->src[7];
    const struct htp_tensor * pend_gb = octx->src[8];

    const uint32_t vec_bytes = S_v * sizeof(float);
    const uint32_t g_elems   = kda ? S_v : 1;
    const uint32_t g_bytes   = g_elems * sizeof(float);

    const dma_addr_t pk_base  = pend_k->data  + (size_t) iv3 * pend_k->nb[3]  + (size_t) ik1 * pend_k->nb[1];
    const dma_addr_t pv_base  = pend_v->data  + (size_t) iv3 * pend_v->nb[3]  + (size_t) iv1 * pend_v->nb[1];
    const dma_addr_t pgb_base = pend_gb->data + (size_t) iv3 * pend_gb->nb[3] + (size_t) iv1 * pend_gb->nb[1];

    for (uint32_t t_base = 0; t_base < n_steps; t_base += tile_cap) {
        const uint32_t tile_tokens = hex_smin(tile_cap, n_steps - t_base);

        const dma_addr_t pk_dma = pk_base  + (size_t) t_base * pend_k->nb[2];
        const dma_addr_t pv_dma = pv_base  + (size_t) t_base * pend_v->nb[2];
        const dma_addr_t pg_dma = pgb_base + (size_t) t_base * pend_gb->nb[2];
        const dma_addr_t pb_dma = pgb_base + (size_t) (n_pend + t_base) * pend_gb->nb[2];

        dma_queue_push(dma_q, dma_make_data(k_buf, pk_dma), vec_bytes, pend_k->nb[2], vec_bytes, tile_tokens);
        dma_queue_push(dma_q, dma_make_data(v_buf, pv_dma), vec_bytes, pend_v->nb[2], vec_bytes, tile_tokens);
        dma_queue_push(dma_q, dma_make_data(g_buf, pg_dma), g_bytes, pend_gb->nb[2], g_bytes, tile_tokens);
        dma_queue_push(dma_q, dma_make_data(b_buf, pb_dma), sizeof(float), pend_gb->nb[2], sizeof(float), tile_tokens);
        for (int d = 0; d < 4; ++d) {
            dma_queue_pop(dma_q);
        }

        for (uint32_t t_local = 0; t_local < tile_tokens; ++t_local) {
            gdn_replay_step_f32(
                s_work,
                k_buf + (size_t) t_local * S_v,
                v_buf + (size_t) t_local * S_v,
                g_buf + (size_t) t_local * g_elems,
                b_buf + t_local,
                vtcm_scratch, kda, S_v);
        }
    }
}

static void gated_delta_net_f32_pp_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
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

    const uint32_t q_row_bytes    = S_v * ((q->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t k_row_bytes    = S_v * ((k->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t v_row_bytes    = S_v * ((v->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t g_row_bytes    = (kparams->kda ? S_v : 1) * sizeof(float);
    const uint32_t out_row_bytes  = S_v * ((dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t dst_row_stride = H * out_row_bytes;

    dma_queue * dma_q = th->dma_q;
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    uint8_t * th_vtcm = gctx->vtcm_base + layout->bytes_per_hvx * ith;
    float * s_work_curr = (float *) th_vtcm;
    HVX_Vector * vtcm_scratch = (HVX_Vector *) (th_vtcm + layout->off_scratch);

    uint8_t * q_spad[2];
    uint8_t * k_spad[2];
    uint8_t * v_spad[2];
    uint8_t * g_spad[2];
    uint8_t * o_spad[2];
    float *   beta_spad[2];
    for (int b = 0; b < 2; ++b) {
        uint8_t * spad = th_vtcm + layout->off_spad + (size_t) b * layout->spad_stride;
        q_spad[b]    = spad + 0 * layout->tile_vec_bytes;
        k_spad[b]    = spad + 1 * layout->tile_vec_bytes;
        v_spad[b]    = spad + 2 * layout->tile_vec_bytes;
        g_spad[b]    = spad + 3 * layout->tile_vec_bytes;
        o_spad[b]    = spad + 4 * layout->tile_vec_bytes;
        beta_spad[b] = (float *) (spad + 5 * layout->tile_vec_bytes);
    }

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride    = kparams->state_seq_stride;
    const uint32_t state_size_per_snap = kparams->state_size_per_snap;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + (size_t) S_v * H * n_tokens * n_seqs * sizeof(float));
    const bool replay = gctx->replay;
    const bool keep   = gctx->keep;

    struct htp_thread_trace * tr = th->trace;
    const gdn_step_fn_t step_fn  = gctx->step_fn;
    const uint32_t n_tiles       = (n_tokens + HTP_GDN_TOKEN_TILE - 1) / HTP_GDN_TOKEN_TILE;

    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const uint32_t iq1 = fastmodulo(iv1, q->ne[1], fd_q1);
        const uint32_t ik1 = fastmodulo(iv1, k->ne[1], fd_k1);
        const uint32_t iq3 = fastdiv(iv3, fd_rq3);
        const uint32_t ik3 = fastdiv(iv3, fd_rk3);
        const uint32_t is3 = gdn_read_seq_idx(gctx->ids_data, gctx->ids_type, iv3);

        const dma_addr_t s_in  = state->data + ((size_t) is3 * state_seq_stride + (size_t) iv1 * S_v * S_v) * sizeof(float);
        const dma_addr_t s_out = state_out_dma_base + ((size_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);

        dma_queue_push(dma_q, dma_make_data(s_work_curr, s_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);
        dma_queue_pop(dma_q);

        if (replay) {
            const int32_t raw_n_rep = gctx->n_rep_data[iv3];
            const uint32_t n_steps = raw_n_rep <= 0 ? 0 : hex_smin((uint32_t) raw_n_rep, gctx->n_pend);
            if (n_steps > 0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
                gdn_replay_f32(octx, dma_q, s_work_curr,
                               (float *) k_spad[0], (float *) v_spad[0], (float *) g_spad[0], beta_spad[0],
                               vtcm_scratch, HTP_GDN_TOKEN_TILE,
                               ik1, iv1, iv3, n_steps, gctx->n_pend,
                               kparams->kda != 0, S_v);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
            }
            if (keep) {
                dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                               S_v * sizeof(float), S_v * sizeof(float),
                               S_v * sizeof(float), S_v);
                dma_queue_pop(dma_q);
            }
        }

        const uint32_t first_tokens = hex_smin(HTP_GDN_TOKEN_TILE, n_tokens);
        gdn_dma_push_tile_inputs(dma_q, q_spad[0], k_spad[0], v_spad[0], g_spad[0], beta_spad[0],
                                 q, k, v, g, beta, iq3, iq1, ik3, ik1, iv3, iv1,
                                 0, first_tokens, q_row_bytes, k_row_bytes, v_row_bytes, g_row_bytes);

        for (uint32_t tile_idx = 0; tile_idx < n_tiles; ++tile_idx) {
            const uint32_t curr_buf    = tile_idx & 1;
            const uint32_t next_buf    = curr_buf ^ 1;
            const uint32_t t_base      = tile_idx * HTP_GDN_TOKEN_TILE;
            const uint32_t tile_tokens = hex_smin(HTP_GDN_TOKEN_TILE, n_tokens - t_base);

            for (int d = 0; d < 5; ++d) {
                dma_queue_pop(dma_q);
            }
            if (tile_idx > 0) {
                dma_queue_pop(dma_q);
            }

            if (K <= 1 && tile_idx + 1 < n_tiles) {
                const uint32_t next_t_base = (tile_idx + 1) * HTP_GDN_TOKEN_TILE;
                const uint32_t next_tokens = hex_smin(HTP_GDN_TOKEN_TILE, n_tokens - next_t_base);
                gdn_dma_push_tile_inputs(dma_q, q_spad[next_buf], k_spad[next_buf], v_spad[next_buf], g_spad[next_buf], beta_spad[next_buf],
                                         q, k, v, g, beta, iq3, iq1, ik3, ik1, iv3, iv1,
                                         next_t_base, next_tokens, q_row_bytes, k_row_bytes, v_row_bytes, g_row_bytes);
            }

            const uint8_t * q_curr = q_spad[curr_buf];
            const uint8_t * k_curr = k_spad[curr_buf];
            const uint8_t * v_curr = v_spad[curr_buf];
            const uint8_t * g_curr = g_spad[curr_buf];
            const float *   b_curr = beta_spad[curr_buf];
            uint8_t *       o_curr = o_spad[curr_buf];

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
            for (uint32_t t_local = 0; t_local < tile_tokens; ++t_local) {
                const void * k_next = (t_local + 1 < tile_tokens) ? (k_curr + (t_local + 1) * k_row_bytes) : NULL;
                step_fn(gctx, s_work_curr,
                        o_curr + t_local * out_row_bytes,
                        q_curr + t_local * q_row_bytes,
                        k_curr + t_local * k_row_bytes,
                        v_curr + t_local * v_row_bytes,
                        (const float *) (g_curr + t_local * g_row_bytes),
                        b_curr + t_local,
                        vtcm_scratch, scale, S_v, k_next, t_local > 0);

                if (K > 1) {
                    const uint32_t t = t_base + t_local;
                    const int32_t target_slot = (int32_t) n_tokens - 1 - (int32_t) t;
                    if (target_slot > 0 && target_slot < (int32_t) K) {
                        const dma_addr_t snap_dma = state_out_dma_base +
                            ((size_t) target_slot * state_size_per_snap + ((size_t) iv3 * H + iv1) * S_v * S_v) * sizeof(float);
                        dma_queue_push(dma_q, dma_make_data(snap_dma, s_work_curr),
                                       S_v * sizeof(float), S_v * sizeof(float),
                                       S_v * sizeof(float), S_v);
                        dma_queue_pop(dma_q);
                    }
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

            if (K > 1 && tile_idx + 1 < n_tiles) {
                const uint32_t next_t_base = (tile_idx + 1) * HTP_GDN_TOKEN_TILE;
                const uint32_t next_tokens = hex_smin(HTP_GDN_TOKEN_TILE, n_tokens - next_t_base);
                gdn_dma_push_tile_inputs(dma_q, q_spad[next_buf], k_spad[next_buf], v_spad[next_buf], g_spad[next_buf], beta_spad[next_buf],
                                         q, k, v, g, beta, iq3, iq1, ik3, ik1, iv3, iv1,
                                         next_t_base, next_tokens, q_row_bytes, k_row_bytes, v_row_bytes, g_row_bytes);
            }

            const dma_addr_t attn_out_dma = dst->data +
                ((size_t) (iv3 * n_tokens + t_base) * H + iv1) * out_row_bytes;
            dma_queue_push(dma_q, dma_make_data(attn_out_dma, o_curr),
                           dst_row_stride, out_row_bytes, out_row_bytes, tile_tokens);
        }

        dma_queue_pop(dma_q);

        if (!keep) {
            dma_queue_push(dma_q, dma_make_data(s_out, s_work_curr),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);
            dma_queue_pop(dma_q);
        }
    }
    dma_queue_flush(dma_q);
}

static void gated_delta_net_f32_tg_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
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

    const uint32_t q_row_bytes   = S_v * ((q->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t k_row_bytes   = S_v * ((k->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t v_row_bytes   = S_v * ((v->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t g_row_bytes   = (kparams->kda ? S_v : 1) * sizeof(float);
    const uint32_t out_row_bytes = S_v * ((dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));

    dma_queue * dma_q = th->dma_q;
    const struct htp_gdn_vtcm_layout * layout = &gctx->layout;
    uint8_t * th_vtcm = gctx->vtcm_base + layout->bytes_per_hvx * ith;
    float * s_work[2];
    s_work[0] = (float *) th_vtcm;
    s_work[1] = (float *) (th_vtcm + layout->state_aligned);
    HVX_Vector * vtcm_scratch = (HVX_Vector *) (th_vtcm + layout->off_scratch);

    uint8_t * q_spad[2];
    uint8_t * k_spad[2];
    uint8_t * v_spad[2];
    uint8_t * g_spad[2];
    uint8_t * o_spad[2];
    float *   beta_spad[2];
    for (int b = 0; b < 2; ++b) {
        uint8_t * spad = th_vtcm + layout->off_spad + (size_t) b * layout->spad_stride;
        q_spad[b]    = spad + 0 * layout->tile_vec_bytes;
        k_spad[b]    = spad + 1 * layout->tile_vec_bytes;
        v_spad[b]    = spad + 2 * layout->tile_vec_bytes;
        g_spad[b]    = spad + 3 * layout->tile_vec_bytes;
        o_spad[b]    = spad + 4 * layout->tile_vec_bytes;
        beta_spad[b] = (float *) (spad + 5 * layout->tile_vec_bytes);
    }

    const struct fastdiv_values * fd_H   = &kparams->div_H;
    const struct fastdiv_values * fd_q1  = &kparams->div_q1;
    const struct fastdiv_values * fd_k1  = &kparams->div_k1;
    const struct fastdiv_values * fd_rq3 = &kparams->div_rq3;
    const struct fastdiv_values * fd_rk3 = &kparams->div_rk3;

    const uint32_t state_seq_stride = kparams->state_seq_stride;
    const dma_addr_t state_out_dma_base = dst_cache ? dst_cache->data : (dst->data + (size_t) S_v * H * n_seqs * sizeof(float));

    uint32_t ir_prefetch = gctx->row_start + ith;
    int spad_idx = 0;

    // Prefetch preamble (up to 2 steps)
    for (int step = 0; step < 2 && ir_prefetch < row_end; step++) {
        const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
        const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
        const uint32_t piq1 = fastmodulo(piv1, q->ne[1], fd_q1);
        const uint32_t pik1 = fastmodulo(piv1, k->ne[1], fd_k1);
        const uint32_t piq3 = fastdiv(piv3, fd_rq3);
        const uint32_t pik3 = fastdiv(piv3, fd_rk3);
        const uint32_t pis3 = gdn_read_seq_idx(gctx->ids_data, gctx->ids_type, piv3);

        const dma_addr_t ps_in  = state->data + ((size_t) pis3 * state_seq_stride + (size_t) piv1 * S_v * S_v) * sizeof(float);
        const dma_addr_t ps_out = state_out_dma_base + ((size_t) piv3 * H + piv1) * S_v * S_v * sizeof(float);
        const dma_addr_t po_out = dst->data + ((size_t) piv3 * H + piv1) * out_row_bytes;

        dma_queue_push(dma_q, dma_make_data(ps_out, s_work[spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), 0);
        dma_queue_push(dma_q, dma_make_data(po_out, o_spad[spad_idx]),
                       out_row_bytes, out_row_bytes, out_row_bytes, 0);

        dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);
        gdn_dma_push_tile_inputs(dma_q, q_spad[spad_idx], k_spad[spad_idx], v_spad[spad_idx], g_spad[spad_idx], beta_spad[spad_idx],
                                 q, k, v, g, beta, piq3, piq1, pik3, pik1, piv3, piv1,
                                 0, 1, q_row_bytes, k_row_bytes, v_row_bytes, g_row_bytes);

        ir_prefetch += nth;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = th->trace;
    const gdn_step_fn_t step_fn  = gctx->step_fn;

    int curr_spad_idx = 0;
    for (uint32_t ir = gctx->row_start + ith; ir < row_end; ir += nth) {
        for (int d = 0; d < 8; ++d) {
            dma_queue_pop(dma_q);
        }

        const uint32_t iv1 = fastmodulo(ir, H, fd_H);
        const uint32_t iv3 = fastdiv(ir, fd_H);

        const dma_addr_t s_out  = state_out_dma_base + ((size_t) iv3 * H + iv1) * S_v * S_v * sizeof(float);
        const dma_addr_t po_out = dst->data + ((size_t) iv3 * H + iv1) * out_row_bytes;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        step_fn(gctx, s_work[curr_spad_idx], o_spad[curr_spad_idx],
                q_spad[curr_spad_idx], k_spad[curr_spad_idx], v_spad[curr_spad_idx],
                (const float *) g_spad[curr_spad_idx], beta_spad[curr_spad_idx],
                vtcm_scratch, scale, S_v, NULL, false);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_push(dma_q, dma_make_data(s_out, s_work[curr_spad_idx]),
                       S_v * sizeof(float), S_v * sizeof(float),
                       S_v * sizeof(float), S_v);
        dma_queue_push(dma_q, dma_make_data(po_out, o_spad[curr_spad_idx]),
                       out_row_bytes, out_row_bytes, out_row_bytes, 1);

        if (ir_prefetch < row_end) {
            const uint32_t piv1 = fastmodulo(ir_prefetch, H, fd_H);
            const uint32_t piv3 = fastdiv(ir_prefetch, fd_H);
            const uint32_t piq1 = fastmodulo(piv1, q->ne[1], fd_q1);
            const uint32_t pik1 = fastmodulo(piv1, k->ne[1], fd_k1);
            const uint32_t piq3 = fastdiv(piv3, fd_rq3);
            const uint32_t pik3 = fastdiv(piv3, fd_rk3);
            const uint32_t pis3 = gdn_read_seq_idx(gctx->ids_data, gctx->ids_type, piv3);

            const dma_addr_t ps_in = state->data + ((size_t) pis3 * state_seq_stride + (size_t) piv1 * S_v * S_v) * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(s_work[spad_idx], ps_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);
            gdn_dma_push_tile_inputs(dma_q, q_spad[spad_idx], k_spad[spad_idx], v_spad[spad_idx], g_spad[spad_idx], beta_spad[spad_idx],
                                     q, k, v, g, beta, piq3, piq1, pik3, pik1, piv3, piv1,
                                     0, 1, q_row_bytes, k_row_bytes, v_row_bytes, g_row_bytes);

            ir_prefetch += nth;
            spad_idx ^= 1;
        }

        curr_spad_idx ^= 1;
    }
    dma_queue_flush(dma_q);
}

struct htp_gdn_head_ptrs {
    float *  s_state;
    __fp16 * s_col_tiles;
    __fp16 * s_update_tiles;

    float * q_f32[2];
    float * k_f32[2];
    float * v_f32[2];
    float * g_f32[2];
    float * b_f32[2];
    float * o_f32[2];

    __fp16 * k_f16;

    __fp16 * q_row_tiles;
    __fp16 * q_prime_row_tiles;
    __fp16 * k_row_tiles;
    __fp16 * k_col_tiles;
    __fp16 * k_prime_row_tiles;
    __fp16 * k_col_tiles_64x128[2];
    __fp16 * kk_tiles;
    __fp16 * qk_tiles;
    __fp16 * v_inter_tiles;
    __fp16 * o_inter_tiles;
    __fp16 * inv_row_tiles;
    __fp16 * a_row_tiles;
    __fp16 * v_prime_col_tiles;
    __fp16 * delta_tiles;
    __fp16 * o_intra_tiles;
    __fp16 * d_row_tiles;

    __fp16 * gamma;
    float *  lambda_init[2];
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

struct htp_gdn_batch_context {
    struct htp_gdn_head_ptrs *           heads;
    float *                              vtcm_g_raw[2];
    float *                              vtcm_b_raw[2];
    uint8_t *                            hmx_scales;
    const int32_t *                      n_rep_data;
    uint32_t                             n_pend;
    bool                                 replay;
    bool                                 keep;
    uint32_t                             head_in_seq;
    uint32_t                             iv3;
    uint32_t                             curr_buf;
    uint32_t                             s_update_buf;
    uint32_t                             c;
    uint32_t                             n_batch;
    uint32_t                             S_v;
    float                                scale;
    struct htp_ops_context *             octx;
    const struct htp_gdn_kernel_params * kparams;
};

static inline void gdn_hmx_run_gemm(
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 *       out_tiles,
    uint32_t       n_row_tiles,
    uint32_t       n_col_tiles,
    uint32_t       n_dot_tiles
) {
    const size_t dot_stride = (size_t) n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (uint32_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * r_tiles = row_tiles + r * dot_stride;
        const __fp16 * c_tiles = col_tiles;
        __fp16 *       o_tile  = out_tiles + r * n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, n_dot_tiles);
            c_tiles += dot_stride;
            o_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

static void gdn_hmx_kk_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].k_row_tiles, bctx->heads[h].k_col_tiles, bctx->heads[h].kk_tiles, 2u, 2u, n_sv_tiles);
    }
}

static void gdn_hmx_qk_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].q_row_tiles, bctx->heads[h].k_col_tiles, bctx->heads[h].qk_tiles, 2u, 2u, n_sv_tiles);
    }
}

static void gdn_hmx_ks_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].k_prime_row_tiles, bctx->heads[h].s_col_tiles, bctx->heads[h].v_inter_tiles, 2u, n_sv_tiles, n_sv_tiles);
    }
}

static void gdn_hmx_qs_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].q_prime_row_tiles, bctx->heads[h].s_col_tiles, bctx->heads[h].o_inter_tiles, 2u, n_sv_tiles, n_sv_tiles);
    }
}

static void gdn_hmx_inv10_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        const struct htp_gdn_head_ptrs * head = &bctx->heads[h];
        hmx_fa_qk_dot_tile(
            (const __fp16 *) head->vtcm_m,
            head->inv_row_tiles + 0 * HMX_FP16_TILE_N_ELMS,
            (__fp16 *) head->vtcm_tmp,
            1u
        );
        hmx_fa_qk_dot_tile(
            (const __fp16 *) head->vtcm_m + HMX_FP16_TILE_N_ELMS,
            (const __fp16 *) head->vtcm_tmp,
            head->inv_row_tiles + 2 * HMX_FP16_TILE_N_ELMS,
            1u
        );
    }
}

static void gdn_hmx_delta_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    const size_t   dot_stride = 2u * HMX_FP16_TILE_N_ELMS;
    // Write delta_tiles in col-major order [c][r] so O_INTRA reads in place
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        const struct htp_gdn_head_ptrs * head = &bctx->heads[h];
        for (uint32_t r = 0; r < 2u; ++r) {
            const __fp16 * r_tiles = head->inv_row_tiles + r * dot_stride;
            const __fp16 * c_tiles = head->v_prime_col_tiles;
            for (uint32_t c = 0; c < n_sv_tiles; ++c) {
                __fp16 * o_tile = head->delta_tiles + (c * 2u + r) * HMX_FP16_TILE_N_ELMS;
                hmx_fa_qk_dot_tile(r_tiles, c_tiles, o_tile, 2u);
                c_tiles += dot_stride;
            }
        }
    }
}

static void gdn_hmx_o_intra_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].a_row_tiles, bctx->heads[h].delta_tiles, bctx->heads[h].o_intra_tiles, 2u, n_sv_tiles, 2u);
    }
}

static void gdn_hmx_s_update_worker(const htp_thread_t * th, void * data) {
    (void) th;
    const struct htp_gdn_batch_context * bctx = (const struct htp_gdn_batch_context *) data;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) bctx->hmx_scales));
    const uint32_t n_sv_tiles = bctx->S_v >> 5;
    const uint32_t buf_idx    = bctx->s_update_buf;
    for (uint32_t h = 0; h < bctx->n_batch; ++h) {
        gdn_hmx_run_gemm(bctx->heads[h].d_row_tiles, bctx->heads[h].k_col_tiles_64x128[buf_idx], bctx->heads[h].s_update_tiles, n_sv_tiles, n_sv_tiles, 2u);
    }
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

static inline float gdn_vtcm_load_f32(const float * vtcm_arr, uint32_t idx) {
    const HVX_Vector v = hvx_vmem(&vtcm_arr[idx & ~31u]);
    union {
        int32_t i;
        float   f;
    } u;
    u.i = Q6_R_vextract_VR(v, (idx & 31u) * 4u);
    return u.f;
}

static inline uint16_t gdn_hvx_extract_u16(HVX_Vector v, uint32_t idx) {
    const uint32_t w = (uint32_t) Q6_R_vextract_VR(v, (idx & ~1u) * 2u);
    return (uint16_t) ((idx & 1u) ? (w >> 16) : (w & 0xFFFFu));
}

static inline void gdn_f32_to_hmx_row_tiles_and_f16(
    __fp16 * restrict dst_tiles,
    __fp16 * restrict dst_prime_tiles,
    __fp16 * restrict dst_f16,
    const float * restrict src,
    const __fp16 * restrict scale_per_row,
    uint32_t n_rows,
    uint32_t n_cols,
    const bool write_f16
) {
    const uint32_t   n_col_tiles = n_cols >> 5;
    const HVX_Vector v_scale_f16 = hvx_vmem(scale_per_row);

    for (uint32_t r = 0; r < n_rows; r += 2) {
        uint32_t r0 = r >> 5;
        uint32_t r1 = (r & 31u) >> 1;
        const float * p0 = src + (r + 0) * n_cols;
        const float * p1 = src + (r + 1) * n_cols;

        const uint32_t pair = (uint32_t) Q6_R_vextract_VR(v_scale_f16, r * 2u);
        HVX_Vector v_scale = Q6_V_vsplat_R(pair);

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

            HVX_Vector vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
            HVX_Vector vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
            __fp16 * tile0_s = dst_prime_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1_s = dst_prime_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile0_s)[r1] = vh0_s;
            ((HVX_Vector *) tile1_s)[r1] = vh1_s;

            if (write_f16) {
                HVX_VectorPair vp01 = Q6_W_vdeal_VVR(vh1, vh0, -2);
                hvx_vmem(dst_f16 + (r + 0) * n_cols + c * 32) = Q6_V_lo_W(vp01);
                hvx_vmem(dst_f16 + (r + 1) * n_cols + c * 32) = Q6_V_hi_W(vp01);
            }
        }
    }
}

static inline void gdn_f16_to_hmx_row_tiles(
    __fp16 * restrict dst_tiles,
    __fp16 * restrict dst_prime_tiles,
    const __fp16 * restrict src,
    const __fp16 * restrict scale_per_row,
    uint32_t n_rows,
    uint32_t n_cols
) {
    const uint32_t   n_col_tiles = n_cols >> 5;
    const HVX_Vector v_scale_f16 = hvx_vmem(scale_per_row);

    for (uint32_t r = 0; r < n_rows; r += 2) {
        uint32_t r0 = r >> 5;
        uint32_t r1 = (r & 31u) >> 1;
        const __fp16 * p0 = src + (r + 0) * n_cols;
        const __fp16 * p1 = src + (r + 1) * n_cols;

        const uint32_t pair = (uint32_t) Q6_R_vextract_VR(v_scale_f16, r * 2u);
        HVX_Vector v_scale = Q6_V_vsplat_R(pair);

        for (uint32_t c = 0; c < n_col_tiles; c += 2) {
            HVX_Vector vr0 = hvx_vmem(p0 + c * 32);
            HVX_Vector vr1 = hvx_vmem(p1 + c * 32);
            HVX_VectorPair vp = Q6_W_vshuff_VVR(vr1, vr0, -2);
            HVX_Vector vh0 = Q6_V_lo_W(vp);
            HVX_Vector vh1 = Q6_V_hi_W(vp);

            __fp16 * tile0 = dst_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1 = dst_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile0)[r1] = vh0;
            ((HVX_Vector *) tile1)[r1] = vh1;

            HVX_Vector vh0_s = hvx_vec_mul_f16_f16(vh0, v_scale);
            HVX_Vector vh1_s = hvx_vec_mul_f16_f16(vh1, v_scale);
            __fp16 * tile0_s = dst_prime_tiles + (r0 * n_col_tiles + c + 0) * HMX_FP16_TILE_N_ELMS;
            __fp16 * tile1_s = dst_prime_tiles + (r0 * n_col_tiles + c + 1) * HMX_FP16_TILE_N_ELMS;
            ((HVX_Vector *) tile0_s)[r1] = vh0_s;
            ((HVX_Vector *) tile1_s)[r1] = vh1_s;
        }
    }
}

// Solve (I + L) * A = I for 32x32 unit lower-triangular L in L1-cached stack, writing A rows to VTCM
static __attribute__((noinline)) void gdn_invert_lower_32x32(
    HVX_Vector * restrict a_rows,
    const uint16_t * restrict l_l2
) {
    const HVX_Vector     v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_VectorPred q_mask64  = Q6_Q_vsetq2_R(64);

    for (uint32_t t = 0; t < 32; ++t) {
        const uint32_t * restrict l_row_u32 = (const uint32_t *) (l_l2 + t * 32u);
        HVX_Vector v_inv0 = Q6_V_vzero();
        HVX_Vector v_inv1 = Q6_V_vzero();
        uint32_t k = 0;
        #pragma nounroll
        for (; k + 1u < t; k += 2u) {
            const uint32_t w = l_row_u32[k >> 1];
            HVX_Vector v_lk0 = Q6_Vh_vsplat_R(w & 0xFFFFu);
            HVX_Vector v_lk1 = Q6_Vh_vsplat_R(w >> 16);
            v_inv0 = hvx_vec_sub_f16_f16(v_inv0, hvx_vec_mul_f16_f16(v_lk0, a_rows[k + 0u]));
            v_inv1 = hvx_vec_sub_f16_f16(v_inv1, hvx_vec_mul_f16_f16(v_lk1, a_rows[k + 1u]));
        }
        HVX_Vector v_inv = hvx_vec_add_f16_f16(v_inv0, v_inv1);
        if (k < t) {
            HVX_Vector v_lk = Q6_Vh_vsplat_R(l_l2[t * 32u + k]);
            v_inv = hvx_vec_sub_f16_f16(v_inv, hvx_vec_mul_f16_f16(v_lk, a_rows[k]));
        }
        HVX_VectorPred q_diag = (t == 0) ? Q6_Q_vsetq2_R(2) : Q6_Q_and_QQn(Q6_Q_vsetq2_R(2 * (t + 1)), Q6_Q_vsetq2_R(2 * t));
        a_rows[t] = Q6_V_vand_QV(q_mask64, Q6_V_vmux_QVV(q_diag, v_one_f16, v_inv));
    }
}

static __attribute__((noinline)) void gdn_build_inv_l_blocks(
    __fp16 * restrict inv_row_tiles,
    const HVX_Vector * restrict rows_kk,
    const __fp16 * restrict decay_m,
    const float * restrict beta,
    __fp16 * restrict l10_tile,
    __fp16 * restrict neg_a11_tile,
    HVX_Vector * restrict a_rows_vtcm
) {
    // Keep l00, l11, and beta_u16 in L1-cached stack to avoid scalar VTCM loads in inversion
    uint16_t l_l2_buf[2048 + 64] __attribute__((aligned(128)));

    HVX_Vector * restrict a_rows     = a_rows_vtcm;
    HVX_Vector * restrict p_l00      = (HVX_Vector *) (l_l2_buf + 0);
    HVX_Vector * restrict p_l11      = (HVX_Vector *) (l_l2_buf + 1024);
    uint16_t * restrict   beta_u16   = l_l2_buf + 2048;
    HVX_Vector * restrict p_l10_tile = (HVX_Vector *) l10_tile;

    HVX_Vector * restrict tile00    = (HVX_Vector *) (inv_row_tiles + 0 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile01    = (HVX_Vector *) (inv_row_tiles + 1 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict tile11    = (HVX_Vector *) (inv_row_tiles + 3 * HMX_FP16_TILE_N_ELMS);
    HVX_Vector * restrict p_neg_a11 = (HVX_Vector *) neg_a11_tile;

    hvx_vmem(beta_u16) = hvx_vec_f32_to_f16(hvx_vmem(beta + 0), hvx_vmem(beta + 32));

    for (uint32_t r = 0; r < 16; ++r) {
        tile01[r] = Q6_V_vzero();
    }

    {
        const HVX_VectorPred q_mask64 = Q6_Q_vsetq2_R(64);
        const uint32_t * restrict beta_u32 = (const uint32_t *) beta_u16;

        for (uint32_t r = 0; r < 16; ++r) {
            uint32_t t0 = 2 * r;
            uint32_t t1 = t0 + 1;

            HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
            HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
            const uint32_t bw = beta_u32[r];
            HVX_Vector v_b0 = Q6_Vh_vsplat_R(bw & 0xFFFFu);
            HVX_Vector v_b1 = Q6_Vh_vsplat_R(bw >> 16);

            HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
            HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

            p_l00[r] = Q6_V_vmux_QVV(q_mask64, r0, Q6_V_vror_VR(r1, 64));
        }

        for (uint32_t r = 0; r < 16; ++r) {
            uint32_t t0 = 32 + 2 * r;
            uint32_t t1 = t0 + 1;

            HVX_Vector v_d0 = hvx_vmem(decay_m + t0 * 64);
            HVX_Vector v_d1 = hvx_vmem(decay_m + t1 * 64);
            const uint32_t bw = beta_u32[16 + r];
            HVX_Vector v_b0 = Q6_Vh_vsplat_R(bw & 0xFFFFu);
            HVX_Vector v_b1 = Q6_Vh_vsplat_R(bw >> 16);

            HVX_Vector r0 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t0], v_d0), v_b0);
            HVX_Vector r1 = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(rows_kk[t1], v_d1), v_b1);

            HVX_VectorPair vp_l10 = Q6_W_vshuff_VVR(r1, r0, -2);
            p_l10_tile[r] = Q6_V_lo_W(vp_l10);
            p_l11[r] = Q6_V_vmux_QVV(q_mask64, Q6_V_vror_VR(r0, 64), r1);
        }
    }

    gdn_invert_lower_32x32(a_rows, l_l2_buf + 0);

    for (uint32_t r = 0; r < 16; ++r) {
        HVX_VectorPair vp = Q6_W_vshuff_VVR(a_rows[2 * r + 1], a_rows[2 * r + 0], -2);
        tile00[r] = Q6_V_lo_W(vp);
    }

    gdn_invert_lower_32x32(a_rows, l_l2_buf + 1024);

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
    const dma_addr_t q_dma = q->data + (iq3 * q->nb[3] + t_chunk * q->nb[2] + iq1 * q->nb[1]);
    const dma_addr_t k_dma = k->data + (ik3 * k->nb[3] + t_chunk * k->nb[2] + ik1 * k->nb[1]);
    const dma_addr_t v_dma = v->data + (iv3 * v->nb[3] + t_chunk * v->nb[2] + iv1 * v->nb[1]);

    const uint32_t q_row_bytes = S_v * ((q->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t k_row_bytes = S_v * ((k->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const uint32_t v_row_bytes = S_v * ((v->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));

    dma_queue_push(dma_q, dma_make_data(vtcm_q, q_dma), q_row_bytes, q->nb[2], q_row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_k, k_dma), k_row_bytes, k->nb[2], k_row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_v, v_dma), v_row_bytes, v->nb[2], v_row_bytes, chunk_size);
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
    const dma_addr_t g_dma    = g->data + (iv3 * g->nb[3] + t_chunk * g->nb[2] + iv1 * g->nb[1]);
    const dma_addr_t beta_dma = beta->data + (iv3 * beta->nb[3] + t_chunk * beta->nb[2] + iv1 * beta->nb[1]);
    const uint32_t row_bytes  = n_batch * sizeof(float);
    const uint32_t dst_stride = (n_batch > 1) ? row_bytes + sizeof(float) : row_bytes;

    dma_queue_push(dma_q, dma_make_data(vtcm_g_raw, g_dma), dst_stride, g->nb[2], row_bytes, chunk_size);
    dma_queue_push(dma_q, dma_make_data(vtcm_b_raw, beta_dma), dst_stride, beta->nb[2], row_bytes, chunk_size);
}

static inline void gdn_pack_s_col_tiles(
    __fp16 * restrict vtcm_s_col_tiles,
    const float * restrict vtcm_s_state,
    uint32_t S_v
) {
    const uint32_t   n_k_tiles   = S_v >> 5;
    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const size_t     pair_region = (size_t) (2 * HMX_FP16_TILE_SIZE - 1);

    for (uint32_t r = 0; r < S_v; r += 2) {
        const uint32_t   ct      = r >> 5;
        const uint32_t   local_r = r & 31u;
        const HVX_Vector v_off0  = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
        const HVX_Vector v_off1  = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

        __fp16 *      tile_base = vtcm_s_col_tiles + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
        const float * s0        = vtcm_s_state + (r + 0) * S_v;
        const float * s1        = vtcm_s_state + (r + 1) * S_v;

        for (uint32_t d = 0; d < (S_v >> 6); ++d) {
            const HVX_Vector v0 = hvx_vec_f32_to_f16(hvx_vmem(s0 + d * 64 + 0), hvx_vmem(s0 + d * 64 + 32));
            const HVX_Vector v1 = hvx_vec_f32_to_f16(hvx_vmem(s1 + d * 64 + 0), hvx_vmem(s1 + d * 64 + 32));
            Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
            Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
            tile_base += 2 * HMX_FP16_TILE_N_ELMS;
        }
    }
    hvx_gather_sync(vtcm_s_col_tiles);
}

static inline void gdn_init_head_ptrs(
    struct htp_gdn_head_ptrs * head,
    const struct htp_gdn_hmx_vtcm_layout * L,
    uint8_t * vtcm_base,
    uint32_t h,
    uint32_t base_iv1,
    uint32_t iv3,
    uint32_t is3,
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
    (void) v;
    const size_t dma_scalar_sz = hex_round_up(chunk_size * sizeof(float), 128);
    const size_t decay_sz      = 64 * 64 * sizeof(__fp16);
    const size_t row_vecs_sz   = 64 * 128;

    head->s_state        = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_s_state + h * L->state_f32_bytes);
    head->s_col_tiles    = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_col_tiles + h * L->state_tiles_bytes);
    head->s_update_tiles = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_s_update_tiles + h * L->state_tiles_bytes);

    head->q_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[0] + h * L->dma_chunk_bytes);
    head->q_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_q_f32[1] + h * L->dma_chunk_bytes);
    head->k_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[0] + h * L->dma_chunk_bytes);
    head->k_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_k_f32[1] + h * L->dma_chunk_bytes);
    head->v_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[0] + h * L->dma_chunk_bytes);
    head->v_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_v_f32[1] + h * L->dma_chunk_bytes);
    head->g_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[0] + h * dma_scalar_sz);
    head->g_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_g_f32[1] + h * dma_scalar_sz);
    head->b_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[0] + h * dma_scalar_sz);
    head->b_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_b_f32[1] + h * dma_scalar_sz);
    head->o_f32[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[0] + h * L->dma_chunk_bytes);
    head->o_f32[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_o_f32[1] + h * L->dma_chunk_bytes);

    head->k_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_f16 + h * L->act_f16_bytes);

    head->q_row_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_row_tiles + h * L->tile_64xSv_bytes);
    head->q_prime_row_tiles     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_q_prime_row_tiles + h * L->tile_64xSv_bytes);
    head->k_row_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_row_tiles + h * L->tile_64xSv_bytes);
    head->k_col_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles + h * L->tile_64xSv_bytes);
    head->k_prime_row_tiles     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_prime_row_tiles + h * L->tile_64xSv_bytes);
    head->k_col_tiles_64x128[0] = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles_64x128[0] + h * L->tile_64xSv_bytes);
    head->k_col_tiles_64x128[1] = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_k_col_tiles_64x128[1] + h * L->tile_64xSv_bytes);
    head->kk_tiles              = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_kk_tiles + h * L->tile_64x64_bytes);
    head->qk_tiles              = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_qk_tiles + h * L->tile_64x64_bytes);
    head->v_inter_tiles         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_inter_tiles + h * L->tile_64xSv_bytes);
    head->o_inter_tiles         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_inter_tiles + h * L->tile_64xSv_bytes);
    head->inv_row_tiles         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_inv_row_tiles + h * L->tile_64x64_bytes);
    head->a_row_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_a_row_tiles + h * L->tile_64x64_bytes);
    head->v_prime_col_tiles     = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_v_prime_col_tiles + h * L->tile_64xSv_bytes);
    head->delta_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_delta_tiles + h * L->tile_64xSv_bytes);
    head->o_intra_tiles         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_o_intra_tiles + h * L->tile_64xSv_bytes);
    head->d_row_tiles           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_d_row_tiles + h * L->tile_64xSv_bytes);

    head->gamma           = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * dma_scalar_sz);
    head->lambda_init_f16 = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_gamma + h * dma_scalar_sz + 128);
    head->lambda_init[0]  = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_lambda_init[0] + h * dma_scalar_sz);
    head->lambda_init[1]  = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_lambda_init[1] + h * dma_scalar_sz);
    head->decay_m         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_m + h * decay_sz);
    head->decay_a         = VTCM_LAYOUT_PTR(__fp16, vtcm_base, L->off_decay_a + h * decay_sz);

    head->rows_kk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_kk + h * row_vecs_sz);
    head->rows_qk  = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_qk + h * row_vecs_sz);
    head->rows_inv = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_inv + h * row_vecs_sz);
    head->rows_a   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_rows_a + h * row_vecs_sz);

    head->vtcm_m   = VTCM_LAYOUT_PTR(HVX_Vector, vtcm_base, L->off_hvx_scratch + h * (64 * 128));
    head->vtcm_tmp = head->vtcm_m + 32;

    head->iv1 = base_iv1 + h;
    head->iv3 = iv3;
    head->iq1 = fastmodulo(head->iv1, q->ne[1], &kparams->div_q1);
    head->ik1 = fastmodulo(head->iv1, k->ne[1], &kparams->div_k1);
    head->iq3 = fastdiv(head->iv3, &kparams->div_rq3);
    head->ik3 = fastdiv(head->iv3, &kparams->div_rk3);

    head->state_in_dma = state->data +
        ((size_t) is3 * kparams->state_seq_stride + (size_t) head->iv1 * S_v * S_v) * sizeof(float);

    head->state_out_dma = dst_cache ?
        (dst_cache->data + (head->iv3 * H + head->iv1) * S_v * S_v * sizeof(float)) :
        (dst->data + (S_v * H * n_tokens * kparams->n_seqs + (head->iv3 * H + head->iv1) * S_v * S_v) * sizeof(float));
}

static void gdn_hvx_prologue_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context *       bctx    = (struct htp_gdn_batch_context *) data;
    struct htp_ops_context *             octx    = bctx->octx;
    const struct htp_gdn_kernel_params * kparams = bctx->kparams;
    struct htp_thread_trace *            tr      = th->trace;
    dma_queue *                          dma_q   = th->dma_q;
    struct htp_gdn_head_ptrs *           head    = &bctx->heads[i];

    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * g    = octx->src[3];
    const struct htp_tensor * beta = octx->src[4];

    const uint32_t S_v        = bctx->S_v;
    const uint32_t chunk_size = kparams->chunk_size;
    const uint32_t n_tokens   = kparams->n_tokens;
    const uint32_t n_chunks   = kparams->n_chunks;

    dma_queue_push(dma_q, dma_make_data(head->s_state, head->state_in_dma),
                   S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);

    if (bctx->replay) {
        dma_queue_pop(dma_q);
        const int32_t  raw_n_rep = bctx->n_rep_data[head->iv3];
        const uint32_t n_steps   = raw_n_rep <= 0 ? 0 : hex_smin((uint32_t) raw_n_rep, bctx->n_pend);
        if (n_steps > 0) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            gdn_replay_f32(octx, dma_q, head->s_state,
                           head->k_f32[0], head->v_f32[0], head->g_f32[0], head->b_f32[0],
                           head->vtcm_m, chunk_size,
                           head->ik1, head->iv1, head->iv3, n_steps, bctx->n_pend,
                           kparams->kda != 0, S_v);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
        }
        if (bctx->keep) {
            dma_queue_push(dma_q, dma_make_data(head->state_out_dma, head->s_state),
                           S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
            dma_queue_pop(dma_q);
        }
    }

    const uint32_t chunk0_tokens = hex_smin(chunk_size, n_tokens);
    gdn_dma_push_chunk_inputs(dma_q, head->q_f32[0], head->k_f32[0], head->v_f32[0],
                              q, k, v, head->iq3, head->iq1, head->ik3, head->ik1,
                              head->iv3, head->iv1, 0u, chunk0_tokens, S_v);
    if (i == 0u) {
        gdn_dma_push_chunk_gb(dma_q, bctx->vtcm_g_raw[0], bctx->vtcm_b_raw[0],
                              g, beta, bctx->iv3, bctx->head_in_seq, 0u, chunk0_tokens, bctx->n_batch);
    }

    if (!bctx->replay) {
        dma_queue_pop(dma_q);
    }
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    gdn_pack_s_col_tiles(head->s_col_tiles, head->s_state, S_v);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    dma_queue_pop(dma_q);
    dma_queue_pop(dma_q);
    dma_queue_pop(dma_q);
    if (i == 0u) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
    }

    if (n_chunks > 1u) {
        const uint32_t c1_tokens = hex_smin(chunk_size, n_tokens - chunk_size);
        gdn_dma_push_chunk_inputs(dma_q, head->q_f32[1], head->k_f32[1], head->v_f32[1],
                                  q, k, v, head->iq3, head->iq1, head->ik3, head->ik1,
                                  head->iv3, head->iv1, chunk_size, c1_tokens, S_v);
        if (i == 0u) {
            gdn_dma_push_chunk_gb(dma_q, bctx->vtcm_g_raw[1], bctx->vtcm_b_raw[1],
                                  g, beta, bctx->iv3, bctx->head_in_seq, chunk_size, c1_tokens, bctx->n_batch);
        }
    }
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

static void gdn_hvx_phase1a_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = th->trace;
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;
    const uint32_t n_batch = bctx->n_batch;
    const float *  vtcm_g_raw = bctx->vtcm_g_raw[curr_buf];
    const float *  vtcm_b_raw = bctx->vtcm_b_raw[curr_buf];

    if (n_batch == 1) {
        hvx_vmem(head->g_f32[curr_buf] + 0)  = hvx_vmem(vtcm_g_raw + 0);
        hvx_vmem(head->g_f32[curr_buf] + 32) = hvx_vmem(vtcm_g_raw + 32);
        hvx_vmem(head->b_f32[curr_buf] + 0)  = hvx_vmem(vtcm_b_raw + 0);
        hvx_vmem(head->b_f32[curr_buf] + 32) = hvx_vmem(vtcm_b_raw + 32);
    } else {
        const uint32_t vtcm_stride = (n_batch + 1) * sizeof(float);
        const HVX_Vector vv = hvx_vec_gather_offsets_w(vtcm_stride);
        const size_t rt_g = (size_t) ((const uint8_t *) vtcm_g_raw + i * sizeof(float));
        const size_t rt_b = (size_t) ((const uint8_t *) vtcm_b_raw + i * sizeof(float));
        const size_t mu   = 32 * vtcm_stride - 1;

        Q6_vgather_ARMVw((HVX_Vector *) (head->g_f32[curr_buf] + 0),  rt_g, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->g_f32[curr_buf] + 32), rt_g + 32 * vtcm_stride, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->b_f32[curr_buf] + 0),  rt_b, mu, vv);
        Q6_vgather_ARMVw((HVX_Vector *) (head->b_f32[curr_buf] + 32), rt_b + 32 * vtcm_stride, mu, vv);
        hvx_gather_sync(head->b_f32[curr_buf] + 32);
    }

    const uint32_t t_chunk = bctx->c * 64;
    const uint32_t valid_tokens = hex_smin(64, bctx->kparams->n_tokens - t_chunk);
    if (valid_tokens < 64) {
        const HVX_Vector vzero = Q6_V_vzero();
        if (valid_tokens <= 32) {
            const HVX_VectorPred mask = Q6_Q_vsetq2_R(valid_tokens * sizeof(float));
            hvx_vmem(head->g_f32[curr_buf] + 0)  = Q6_V_vand_QV(mask, hvx_vmem(head->g_f32[curr_buf] + 0));
            hvx_vmem(head->b_f32[curr_buf] + 0)  = Q6_V_vand_QV(mask, hvx_vmem(head->b_f32[curr_buf] + 0));
            hvx_vmem(head->g_f32[curr_buf] + 32) = vzero;
            hvx_vmem(head->b_f32[curr_buf] + 32) = vzero;
        } else {
            const HVX_VectorPred mask = Q6_Q_vsetq2_R((valid_tokens - 32) * sizeof(float));
            hvx_vmem(head->g_f32[curr_buf] + 32) = Q6_V_vand_QV(mask, hvx_vmem(head->g_f32[curr_buf] + 32));
            hvx_vmem(head->b_f32[curr_buf] + 32) = Q6_V_vand_QV(mask, hvx_vmem(head->b_f32[curr_buf] + 32));
        }
        const uint32_t q_words = (bctx->octx->src[0]->type == HTP_TYPE_F16) ? (S_v / 2) : S_v;
        const uint32_t k_words = (bctx->octx->src[1]->type == HTP_TYPE_F16) ? (S_v / 2) : S_v;
        const uint32_t v_words = (bctx->octx->src[2]->type == HTP_TYPE_F16) ? (S_v / 2) : S_v;
        for (uint32_t t = valid_tokens; t < 64; ++t) {
            for (uint32_t j = 0; j < q_words; j += 32) {
                hvx_vmem(head->q_f32[curr_buf] + t * q_words + j) = vzero;
            }
            for (uint32_t j = 0; j < k_words; j += 32) {
                hvx_vmem(head->k_f32[curr_buf] + t * k_words + j) = vzero;
            }
            for (uint32_t j = 0; j < v_words; j += 32) {
                hvx_vmem(head->v_f32[curr_buf] + t * v_words + j) = vzero;
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

    hvx_vmem(head->lambda_init[curr_buf] + 0)  = v_l0;
    hvx_vmem(head->lambda_init[curr_buf] + 32) = v_l1;
    hvx_vmem(head->lambda_init_f16)            = hvx_vec_f32_to_f16(v_l0, v_l1);

    if (bctx->octx->src[1]->type == HTP_TYPE_F16) {
        const __fp16 * k_f16 = (const __fp16 *) head->k_f32[curr_buf];
        gdn_f16_to_hmx_row_tiles(head->k_row_tiles, head->k_prime_row_tiles,
                                 k_f16, head->lambda_init_f16, 64, S_v);
        hmx_interleave_rows_to_tiles(head->k_col_tiles, k_f16, 64, S_v, S_v, 0, 64);
    } else {
        gdn_f32_to_hmx_row_tiles_and_f16(head->k_row_tiles, head->k_prime_row_tiles, head->k_f16,
                                         head->k_f32[curr_buf], head->lambda_init_f16, 64, S_v, true);
        hmx_interleave_rows_to_tiles(head->k_col_tiles, head->k_f16, 64, S_v, S_v, 0, 64);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);
}

static void gdn_hvx_phase1b_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = th->trace;
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf = bctx->curr_buf;
    const uint32_t S_v = bctx->S_v;

    if (bctx->octx->src[0]->type == HTP_TYPE_F16) {
        gdn_f16_to_hmx_row_tiles(head->q_row_tiles, head->q_prime_row_tiles,
                                 (const __fp16 *) head->q_f32[curr_buf], head->lambda_init_f16, 64, S_v);
    } else {
        gdn_f32_to_hmx_row_tiles_and_f16(head->q_row_tiles, head->q_prime_row_tiles, NULL,
                                         head->q_f32[curr_buf], head->lambda_init_f16, 64, S_v, false);
    }

    const __fp16 * k_f16_src = (bctx->octx->src[1]->type == HTP_TYPE_F16)
        ? (const __fp16 *) head->k_f32[curr_buf]
        : head->k_f16;
    hmx_interleave_cols_to_tiles(head->k_col_tiles_64x128[curr_buf], k_f16_src, 64, S_v, S_v, 2, 0, 64);

    const HVX_Vector v_gamma = hvx_vmem(head->gamma);

    const HVX_Vector v_zero_f16  = Q6_V_vzero();
    const HVX_Vector v_neg20_f16 = hvx_vec_splat_f16(-20.0f);
    const HVX_Vector v_log2e_f16 = hvx_vec_splat_f16(1.4426950408889634f);
    const HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0f);

    hvx_vmem(head->decay_m + 0) = Q6_V_vzero();
    hvx_vmem(head->decay_a + 0) = Q6_V_vand_QV(Q6_Q_vsetq2_R(2), v_one_f16);

    for (uint32_t t = 1; t < 63; t += 2) {
        uint32_t t0 = t;
        uint32_t t1 = t + 1;

        HVX_Vector v_gamma_t0  = Q6_Vh_vsplat_R(gdn_hvx_extract_u16(v_gamma, t0));
        HVX_Vector v_gamma_t1  = Q6_Vh_vsplat_R(gdn_hvx_extract_u16(v_gamma, t1));

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

        HVX_Vector v_exp0      = hvx_vec_exp2_neg_f16(diff_log2e0);
        HVX_Vector v_exp1      = hvx_vec_exp2_neg_f16(diff_log2e1);

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
        HVX_Vector v_gamma_t     = Q6_Vh_vsplat_R(gdn_hvx_extract_u16(v_gamma, 63));
        HVX_Vector diff          = hvx_vec_sub_f16_f16(v_gamma_t, v_gamma);
        HVX_VectorPred p_gt      = Q6_Q_vcmp_gt_VhfVhf(diff, v_zero_f16);
        diff                     = Q6_V_vmux_QVV(p_gt, v_zero_f16, diff);
        diff                     = Q6_Vhf_vmax_VhfVhf(v_neg20_f16, diff);

        HVX_Vector diff_log2e    = hvx_vec_mul_f16_f16(diff, v_log2e_f16);
        HVX_Vector v_exp         = hvx_vec_exp2_neg_f16(diff_log2e);

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

static void gdn_hvx_phase2_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = th->trace;
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
        (__fp16 *) head->vtcm_m + HMX_FP16_TILE_N_ELMS,
        head->rows_inv
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_SOLVE, info);
}

static void gdn_hvx_phase3_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context *       bctx    = (struct htp_gdn_batch_context *) data;
    struct htp_ops_context *             octx    = bctx->octx;
    const struct htp_gdn_kernel_params * kparams = bctx->kparams;
    struct htp_thread_trace *            tr      = th->trace;
    const uint32_t                       c       = bctx->c;
    const uint16_t                       info    = (uint16_t) c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t   curr_buf    = bctx->curr_buf;
    const uint32_t   S_v         = bctx->S_v;
    const uint32_t   n_col_tiles = S_v >> 5;
    const HVX_Vector v_b_lo      = hvx_vmem(head->b_f32[curr_buf] + 0);
    const HVX_Vector v_b_hi      = hvx_vmem(head->b_f32[curr_buf] + 32);

    if (octx->src[2]->type == HTP_TYPE_F16) {
        const __fp16 * v_f16 = (const __fp16 *) head->v_f32[curr_buf];
        for (uint32_t r = 0; r < 64; r += 2) {
            const uint32_t   r0  = r >> 5;
            const uint32_t   r1  = (r & 31u) >> 1;
            const HVX_Vector v_b = r0 ? v_b_hi : v_b_lo;
            const HVX_Vector vb0 = Q6_V_vsplat_R(Q6_R_vextract_VR(v_b, ((r + 0u) & 31u) * 4u));
            const HVX_Vector vb1 = Q6_V_vsplat_R(Q6_R_vextract_VR(v_b, ((r + 1u) & 31u) * 4u));

            for (uint32_t d = 0; d < (S_v >> 6); ++d) {
                const uint32_t c0 = 2 * d;
                const HVX_Vector * t_vi0 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + c0 + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * t_vi1 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + c0 + 1) * HMX_FP16_TILE_N_ELMS);
                const HVX_VectorPair vp_vi = Q6_W_vdeal_VVR(t_vi1[r1], t_vi0[r1], -2);

                const HVX_VectorPair p_vi0  = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_vi));
                const HVX_VectorPair p_vv0  = hvx_vec_f16_to_f32(hvx_vmem(v_f16 + (r + 0) * S_v + d * 64));
                const HVX_Vector     vp0_lo = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(Q6_V_lo_W(p_vv0), Q6_V_lo_W(p_vi0)), vb0);
                const HVX_Vector     vp0_hi = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(Q6_V_hi_W(p_vv0), Q6_V_hi_W(p_vi0)), vb0);

                const HVX_VectorPair p_vi1  = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_vi));
                const HVX_VectorPair p_vv1  = hvx_vec_f16_to_f32(hvx_vmem(v_f16 + (r + 1) * S_v + d * 64));
                const HVX_Vector     vp1_lo = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(Q6_V_lo_W(p_vv1), Q6_V_lo_W(p_vi1)), vb1);
                const HVX_Vector     vp1_hi = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(Q6_V_hi_W(p_vv1), Q6_V_hi_W(p_vi1)), vb1);

                HVX_Vector * t_vp0 = (HVX_Vector *) (head->v_prime_col_tiles + ((c0 + 0) * 2 + r0) * HMX_FP16_TILE_N_ELMS);
                HVX_Vector * t_vp1 = (HVX_Vector *) (head->v_prime_col_tiles + ((c0 + 1) * 2 + r0) * HMX_FP16_TILE_N_ELMS);
                t_vp0[r1] = hvx_vec_f32_to_f16_shuff(vp0_lo, vp1_lo);
                t_vp1[r1] = hvx_vec_f32_to_f16_shuff(vp0_hi, vp1_hi);
            }
        }
    } else {
        const float * v_f32 = head->v_f32[curr_buf];
        for (uint32_t r = 0; r < 64; r += 2) {
            const uint32_t   r0  = r >> 5;
            const uint32_t   r1  = (r & 31u) >> 1;
            const HVX_Vector v_b = r0 ? v_b_hi : v_b_lo;
            const HVX_Vector vb0 = Q6_V_vsplat_R(Q6_R_vextract_VR(v_b, ((r + 0u) & 31u) * 4u));
            const HVX_Vector vb1 = Q6_V_vsplat_R(Q6_R_vextract_VR(v_b, ((r + 1u) & 31u) * 4u));

            for (uint32_t d = 0; d < (S_v >> 6); ++d) {
                const uint32_t c0 = 2 * d;
                const HVX_Vector * t_vi0 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + c0 + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * t_vi1 = (const HVX_Vector *) (head->v_inter_tiles + (r0 * n_col_tiles + c0 + 1) * HMX_FP16_TILE_N_ELMS);
                const HVX_VectorPair vp_vi = Q6_W_vdeal_VVR(t_vi1[r1], t_vi0[r1], -2);

                const HVX_VectorPair p_vi0  = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_vi));
                const HVX_Vector     vv0_lo = hvx_vmem(v_f32 + (r + 0) * S_v + d * 64 + 0);
                const HVX_Vector     vv0_hi = hvx_vmem(v_f32 + (r + 0) * S_v + d * 64 + 32);
                const HVX_Vector     vp0_lo = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0_lo, Q6_V_lo_W(p_vi0)), vb0);
                const HVX_Vector     vp0_hi = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv0_hi, Q6_V_hi_W(p_vi0)), vb0);

                const HVX_VectorPair p_vi1  = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_vi));
                const HVX_Vector     vv1_lo = hvx_vmem(v_f32 + (r + 1) * S_v + d * 64 + 0);
                const HVX_Vector     vv1_hi = hvx_vmem(v_f32 + (r + 1) * S_v + d * 64 + 32);
                const HVX_Vector     vp1_lo = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1_lo, Q6_V_lo_W(p_vi1)), vb1);
                const HVX_Vector     vp1_hi = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(vv1_hi, Q6_V_hi_W(p_vi1)), vb1);

                HVX_Vector * t_vp0 = (HVX_Vector *) (head->v_prime_col_tiles + ((c0 + 0) * 2 + r0) * HMX_FP16_TILE_N_ELMS);
                HVX_Vector * t_vp1 = (HVX_Vector *) (head->v_prime_col_tiles + ((c0 + 1) * 2 + r0) * HMX_FP16_TILE_N_ELMS);
                t_vp0[r1] = hvx_vec_f32_to_f16_shuff(vp0_lo, vp1_lo);
                t_vp1[r1] = hvx_vec_f32_to_f16_shuff(vp0_hi, vp1_hi);
            }
        }
    }

    gdn_unpack_64x64_tiles_to_vectors(head->rows_qk, head->qk_tiles);
    for (uint32_t t = 0; t < 64; ++t) {
        HVX_Vector v_decay_a = hvx_vmem(head->decay_a + t * 64);
        head->rows_a[t]      = hvx_vec_mul_f16_f16(head->rows_qk[t], v_decay_a);
    }
    gdn_pack_64x64_vectors_to_tiles(head->a_row_tiles, head->rows_a);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_V_PREP, info);

    dma_queue * dma_q = th->dma_q;
    const uint32_t n_chunks = kparams->n_chunks;
    if (c + 1u < n_chunks) {
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        if (i == 0u) {
            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);
        }
    }
    if (c > 0u) {
        dma_queue_pop(dma_q);
    }
    if (c + 2u < n_chunks) {
        const uint32_t chunk_size = kparams->chunk_size;
        const uint32_t t_c2       = (c + 2u) * chunk_size;
        const uint32_t c2_tokens  = hex_smin(chunk_size, kparams->n_tokens - t_c2);
        gdn_dma_push_chunk_inputs(dma_q, head->q_f32[curr_buf], head->k_f32[curr_buf], head->v_f32[curr_buf],
                                  octx->src[0], octx->src[1], octx->src[2],
                                  head->iq3, head->iq1, head->ik3, head->ik1,
                                  head->iv3, head->iv1, t_c2, c2_tokens, S_v);
        if (i == 0u) {
            gdn_dma_push_chunk_gb(dma_q, bctx->vtcm_g_raw[curr_buf], bctx->vtcm_b_raw[curr_buf],
                                  octx->src[3], octx->src[4], bctx->iv3, bctx->head_in_seq,
                                  t_c2, c2_tokens, bctx->n_batch);
        }
    }
}

static void gdn_hvx_phase4_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = th->trace;
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t S_v = bctx->S_v;

    const HVX_Vector v_decay_last = hvx_vmem(head->decay_a + 63 * 64);
    const HVX_Vector vzero        = Q6_V_vzero();
    HVX_Vector * restrict m       = head->vtcm_m;
    HVX_Vector * restrict tmp     = head->vtcm_tmp;

    for (uint32_t col_half = 0; col_half < (S_v >> 6); ++col_half) {
        const uint32_t r0_base = col_half * 2;
        for (uint32_t c0 = 0; c0 < 2; ++c0) {
            // delta_tiles is stored in col-major tile order [col_tile][row_tile]
            const HVX_Vector * src0 = (const HVX_Vector *) (head->delta_tiles + ((2 * col_half + 0) * 2 + c0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * src1 = (const HVX_Vector *) (head->delta_tiles + ((2 * col_half + 1) * 2 + c0) * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                const uint32_t s0 = c0 * 32 + 2 * r + 0;
                const uint32_t s1 = c0 * 32 + 2 * r + 1;
                const HVX_VectorPair vp = Q6_W_vdeal_VVR(src1[r], src0[r], -2);

                const HVX_Vector     vs0  = Q6_Vh_vsplat_R(gdn_hvx_extract_u16(v_decay_last, s0));
                const HVX_Vector     vs1  = Q6_Vh_vsplat_R(gdn_hvx_extract_u16(v_decay_last, s1));
                const HVX_VectorPred p_z0 = Q6_Q_vcmp_eq_VhVh(vs0, vzero);
                const HVX_VectorPred p_z1 = Q6_Q_vcmp_eq_VhVh(vs1, vzero);

                m[2 * r + 0] = Q6_V_vmux_QVV(p_z0, vzero, hvx_vec_mul_f16_f16(Q6_V_lo_W(vp), vs0));
                m[2 * r + 1] = Q6_V_vmux_QVV(p_z1, vzero, hvx_vec_mul_f16_f16(Q6_V_hi_W(vp), vs1));
            }

            hvx_transpose_32x32_w_shuff_inplace(m, tmp);

            const uint32_t tile0_idx = (r0_base + 0) * 2 + c0;
            const uint32_t tile1_idx = (r0_base + 1) * 2 + c0;
            HVX_Vector * dst0 = (HVX_Vector *) (head->d_row_tiles + tile0_idx * HMX_FP16_TILE_N_ELMS);
            HVX_Vector * dst1 = (HVX_Vector *) (head->d_row_tiles + tile1_idx * HMX_FP16_TILE_N_ELMS);

            for (uint32_t r = 0; r < 16; ++r) {
                dst0[r] = m[r];
                dst1[r] = m[16 + r];
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_D_PREP, info);
}

static void gdn_hvx_phase5_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context *       bctx    = (struct htp_gdn_batch_context *) data;
    const struct htp_gdn_kernel_params * kparams = bctx->kparams;
    const struct htp_tensor *            dst     = bctx->octx->dst;
    struct htp_thread_trace *            tr      = th->trace;
    const uint32_t                       c       = bctx->c;
    const uint16_t                       info    = (uint16_t) c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf    = bctx->curr_buf;
    const uint32_t S_v         = bctx->S_v;
    const uint32_t n_col_tiles = S_v >> 5;
    const HVX_Vector vscale    = hvx_vec_splat_f32(bctx->scale);

    if (dst->type == HTP_TYPE_F16) {
        __fp16 * o_f16 = (__fp16 *) head->o_f32[curr_buf];
        const HVX_Vector v_min = hvx_vec_splat_f32(-65504.0f);
        const HVX_Vector v_max = hvx_vec_splat_f32(65504.0f);

        for (uint32_t r0 = 0; r0 < 2; ++r0) {
            for (uint32_t d = 0; d < (S_v >> 6); ++d) {
                const HVX_Vector * ti0 = (const HVX_Vector *) (head->o_inter_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ti1 = (const HVX_Vector *) (head->o_inter_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ta0 = (const HVX_Vector *) (head->o_intra_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ta1 = (const HVX_Vector *) (head->o_intra_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

                for (uint32_t r = 0; r < 16; ++r) {
                    const HVX_VectorPair vp_i = Q6_W_vdeal_VVR(ti1[r], ti0[r], -2);
                    const HVX_VectorPair vp_a = Q6_W_vdeal_VVR(ta1[r], ta0[r], -2);

                    const HVX_VectorPair pi0    = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_i));
                    const HVX_VectorPair pa0    = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_a));
                    const HVX_Vector     vo0_lo = gdn_clamp_f16_range_f32(hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_lo_W(pi0), Q6_V_lo_W(pa0)), vscale), v_min, v_max);
                    const HVX_Vector     vo0_hi = gdn_clamp_f16_range_f32(hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_hi_W(pi0), Q6_V_hi_W(pa0)), vscale), v_min, v_max);

                    const HVX_VectorPair pi1    = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_i));
                    const HVX_VectorPair pa1    = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_a));
                    const HVX_Vector     vo1_lo = gdn_clamp_f16_range_f32(hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_lo_W(pi1), Q6_V_lo_W(pa1)), vscale), v_min, v_max);
                    const HVX_Vector     vo1_hi = gdn_clamp_f16_range_f32(hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_hi_W(pi1), Q6_V_hi_W(pa1)), vscale), v_min, v_max);

                    __fp16 * out0 = o_f16 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                    __fp16 * out1 = o_f16 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                    hvx_vmem(out0) = hvx_vec_f32_to_f16(vo0_lo, vo0_hi);
                    hvx_vmem(out1) = hvx_vec_f32_to_f16(vo1_lo, vo1_hi);
                }
            }
        }
    } else {
        float * o_f32 = head->o_f32[curr_buf];

        for (uint32_t r0 = 0; r0 < 2; ++r0) {
            for (uint32_t d = 0; d < (S_v >> 6); ++d) {
                const HVX_Vector * ti0 = (const HVX_Vector *) (head->o_inter_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ti1 = (const HVX_Vector *) (head->o_inter_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ta0 = (const HVX_Vector *) (head->o_intra_tiles + (r0 * n_col_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
                const HVX_Vector * ta1 = (const HVX_Vector *) (head->o_intra_tiles + (r0 * n_col_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);

                for (uint32_t r = 0; r < 16; ++r) {
                    const HVX_VectorPair vp_i = Q6_W_vdeal_VVR(ti1[r], ti0[r], -2);
                    const HVX_VectorPair vp_a = Q6_W_vdeal_VVR(ta1[r], ta0[r], -2);

                    const HVX_VectorPair pi0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_i));
                    const HVX_VectorPair pa0 = hvx_vec_f16_to_f32(Q6_V_lo_W(vp_a));
                    const HVX_VectorPair pi1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_i));
                    const HVX_VectorPair pa1 = hvx_vec_f16_to_f32(Q6_V_hi_W(vp_a));

                    float * out0 = o_f32 + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                    float * out1 = o_f32 + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                    hvx_vmem(out0 + 0)  = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_lo_W(pi0), Q6_V_lo_W(pa0)), vscale);
                    hvx_vmem(out0 + 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_hi_W(pi0), Q6_V_hi_W(pa0)), vscale);
                    hvx_vmem(out1 + 0)  = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_lo_W(pi1), Q6_V_lo_W(pa1)), vscale);
                    hvx_vmem(out1 + 32) = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(Q6_V_hi_W(pi1), Q6_V_hi_W(pa1)), vscale);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_OUT, info);

    const uint32_t   chunk_size     = kparams->chunk_size;
    const uint32_t   t_chunk        = c * chunk_size;
    const uint32_t   valid_tokens   = hex_smin(chunk_size, kparams->n_tokens - t_chunk);
    const uint32_t   row_out_bytes  = S_v * ((dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float));
    const dma_addr_t attn_chunk_dma = dst->data +
        ((size_t) head->iv3 * kparams->n_tokens * kparams->H + (size_t) t_chunk * kparams->H + head->iv1) * row_out_bytes;
    dma_queue_push(th->dma_q, dma_make_data(attn_chunk_dma, head->o_f32[curr_buf]),
                   dst->nb[1], row_out_bytes, row_out_bytes, valid_tokens);
}

static void gdn_hvx_phase6_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx = (struct htp_gdn_batch_context *) data;
    struct htp_thread_trace * tr = th->trace;
    const uint16_t info = (uint16_t) bctx->c;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);

    struct htp_gdn_head_ptrs * head = &bctx->heads[i];
    const uint32_t curr_buf    = bctx->curr_buf;
    const uint32_t S_v         = bctx->S_v;
    const uint32_t c           = bctx->c;
    const uint32_t n_chunks    = bctx->kparams->n_chunks;
    const uint32_t n_tiles     = S_v >> 5;
    const HVX_Vector v_l_final = hvx_vec_splat_f32(gdn_vtcm_load_f32(head->lambda_init[curr_buf], 63));
    const bool pack_next       = (c + 1 < n_chunks);
    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const size_t     pair_region = (size_t) (2 * HMX_FP16_TILE_SIZE - 1);

    for (uint32_t r0 = 0; r0 < n_tiles; ++r0) {
        for (uint32_t d = 0; d < (S_v >> 6); ++d) {
            const HVX_Vector * t0 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_tiles + 2 * d + 0) * HMX_FP16_TILE_N_ELMS);
            const HVX_Vector * t1 = (const HVX_Vector *) (head->s_update_tiles + (r0 * n_tiles + 2 * d + 1) * HMX_FP16_TILE_N_ELMS);
            __fp16 * tile_base = head->s_col_tiles + (size_t) (r0 * n_tiles + 2 * d) * HMX_FP16_TILE_N_ELMS;

            for (uint32_t r = 0; r < 16; ++r) {
                const HVX_VectorPair vp01 = Q6_W_vdeal_VVR(t1[r], t0[r], -2);
                const HVX_VectorPair p0   = hvx_vec_f16_to_f32(Q6_V_lo_W(vp01));
                const HVX_VectorPair p1   = hvx_vec_f16_to_f32(Q6_V_hi_W(vp01));

                float * s0 = head->s_state + (r0 * 32 + 2 * r + 0) * S_v + d * 64;
                float * s1 = head->s_state + (r0 * 32 + 2 * r + 1) * S_v + d * 64;

                const HVX_Vector ns0_lo = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s0 + 0),  v_l_final), Q6_V_lo_W(p0));
                const HVX_Vector ns0_hi = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s0 + 32), v_l_final), Q6_V_hi_W(p0));
                const HVX_Vector ns1_lo = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s1 + 0),  v_l_final), Q6_V_lo_W(p1));
                const HVX_Vector ns1_hi = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vmem(s1 + 32), v_l_final), Q6_V_hi_W(p1));

                hvx_vmem(s0 + 0)  = ns0_lo;
                hvx_vmem(s0 + 32) = ns0_hi;
                hvx_vmem(s1 + 0)  = ns1_lo;
                hvx_vmem(s1 + 32) = ns1_hi;

                if (pack_next) {
                    const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(2 * r * 4));
                    const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
                    const HVX_Vector v0_f16 = hvx_vec_f32_to_f16(ns0_lo, ns0_hi);
                    const HVX_Vector v1_f16 = hvx_vec_f32_to_f16(ns1_lo, ns1_hi);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0_f16);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1_f16);
                }
            }
        }
    }

    if (pack_next) {
        hvx_gather_sync(head->s_col_tiles);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_GDN_STATE, info);
}

static void gdn_hvx_epilogue_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct htp_gdn_batch_context * bctx  = (struct htp_gdn_batch_context *) data;
    struct htp_gdn_head_ptrs *     head  = &bctx->heads[i];
    dma_queue *                    dma_q = th->dma_q;
    const uint32_t                 S_v   = bctx->S_v;

    dma_queue_pop(dma_q);
    if (!bctx->keep) {
        dma_queue_push(dma_q, dma_make_data(head->state_out_dma, head->s_state),
                       S_v * sizeof(float), S_v * sizeof(float), S_v * sizeof(float), S_v);
    }
    dma_queue_flush(dma_q);
}

static int gated_delta_net_f32_hmx_chunked(
    struct htp_ops_context * octx,
    const struct htp_gdn_kernel_params * kparams,
    uint32_t row_start,
    uint32_t nrows
) {
    if (kparams->n_chunks == 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_tensor * q         = octx->src[0];
    const struct htp_tensor * k         = octx->src[1];
    const struct htp_tensor * v         = octx->src[2];
    const struct htp_tensor * state     = octx->src[5];
    const bool replay                   = gdn_has_replay(octx);
    const struct htp_tensor * ids       = replay ? NULL : octx->src[6];
    const struct htp_tensor * n_rep     = replay ? octx->src[9] : NULL;
    const struct htp_tensor * dst       = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    const uint32_t S_v        = kparams->S_v;
    const uint32_t H          = kparams->H;
    const uint32_t n_tokens   = kparams->n_tokens;
    const float    scale      = kparams->scale;
    const uint32_t chunk_size = kparams->chunk_size;
    const uint32_t n_chunks   = kparams->n_chunks;
    const uint32_t n_pend     = replay ? (uint32_t) octx->op_params[1] : 0;
    const bool     keep       = replay && octx->op_params[2] != 0;

    struct htp_gdn_hmx_vtcm_layout L;
    htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_size, kparams->n_heads_batch, kparams->n_hvx);

    if (L.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    void *          aux_buf   = NULL;
    const int       aux_err   = gdn_prepare_aux_data(octx, replay ? n_rep : ids, &aux_buf);
    if (aux_err != HTP_STATUS_OK) {
        return aux_err;
    }
    const void *    ids_buf   = replay ? NULL : aux_buf;
    const int32_t * n_rep_buf = replay ? (const int32_t *) aux_buf : NULL;
    const uint32_t  ids_type  = ids ? ids->type : 0;

    uint8_t * vtcm_scales_1 = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L.off_scales_1);
    hmx_init_column_scales(vtcm_scales_1, Q6_V_vsplat_R(0x3c00));

    hmx_queue_t  hmx_q = octx->ctx->hmx_queue;
    work_queue_t wp    = octx->ctx->hvx_queue;

    struct htp_gdn_head_ptrs heads[8];

    struct htp_gdn_batch_context bctx;
    bctx.heads         = heads;
    bctx.vtcm_g_raw[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[0]);
    bctx.vtcm_g_raw[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L.off_g_raw[1]);
    bctx.vtcm_b_raw[0] = VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[0]);
    bctx.vtcm_b_raw[1] = VTCM_LAYOUT_PTR(float, vtcm_base, L.off_b_raw[1]);
    bctx.hmx_scales    = vtcm_scales_1;
    bctx.n_rep_data    = n_rep_buf;
    bctx.n_pend        = n_pend;
    bctx.replay        = replay;
    bctx.keep          = keep;
    bctx.S_v           = S_v;
    bctx.scale         = scale;
    bctx.octx          = octx;
    bctx.kparams       = kparams;

    uint32_t n_batch = 1;
    for (uint32_t r = row_start; r < row_start + nrows; r += n_batch) {
        const uint32_t head_in_seq         = fastmodulo(r, H, &kparams->div_H);
        const uint32_t iv3                 = fastdiv(r, &kparams->div_H);
        const uint32_t is3                 = gdn_read_seq_idx(ids_buf, ids_type, iv3);
        const uint32_t heads_left_in_seq   = H - head_in_seq;
        const uint32_t heads_left_in_range = (row_start + nrows) - r;
        n_batch = hex_smin((uint32_t) kparams->n_heads_batch, hex_smin(heads_left_in_seq, heads_left_in_range));

        for (uint32_t h = 0; h < n_batch; ++h) {
            gdn_init_head_ptrs(&heads[h], &L, vtcm_base, h, head_in_seq, iv3, is3,
                               q, k, v, state, dst, dst_cache, kparams, S_v, H, n_tokens, chunk_size);
        }

        bctx.head_in_seq = head_in_seq;
        bctx.iv3         = iv3;
        bctx.n_batch     = n_batch;

        work_queue_run(wp, gdn_hvx_prologue_worker, &bctx, n_batch);

        // Prologue: prepare chunk 0 (phase1a, KK, phase1b, QK + KS in-flight)
        bctx.c        = 0u;
        bctx.curr_buf = 0u;
        work_queue_run(wp, gdn_hvx_phase1a_worker, &bctx, n_batch);
        hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_kk_worker, &bctx));

        work_queue_run(wp, gdn_hvx_phase1b_worker, &bctx, n_batch);
        hmx_queue_pop(hmx_q);
        hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_qk_worker, &bctx));
        hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_ks_worker, &bctx));

        // Pipelined chunk loop: overlap chunk c tail with chunk c+1 prep
        for (uint32_t c = 0u; c < n_chunks; ++c) {
            const uint32_t curr_buf = c & 1u;
            const uint32_t next_buf = (c + 1u) & 1u;
            const bool     has_next = (c + 1u < n_chunks);

            // Step 1: HMX QK(c) + KS(c) || HVX phase2(c)
            bctx.c        = c;
            bctx.curr_buf = curr_buf;
            work_queue_run(wp, gdn_hvx_phase2_worker, &bctx, n_batch);
            hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_inv10_worker, &bctx));
            hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_qs_worker,    &bctx));
            hmx_queue_pop(hmx_q); // QK(c)
            hmx_queue_pop(hmx_q); // KS(c)

            // Step 2: HMX INV_10(c) + QS(c) || HVX phase3(c) + DMA pop(c+1)/push(c+2)
            work_queue_run(wp, gdn_hvx_phase3_worker, &bctx, n_batch);
            hmx_queue_pop(hmx_q); // INV_10(c)
            hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_delta_worker, &bctx));

            // Step 3: HMX QS(c) tail + DELTA(c) || HVX phase1a(c+1)
            if (has_next) {
                bctx.c        = c + 1u;
                bctx.curr_buf = next_buf;
                work_queue_run(wp, gdn_hvx_phase1a_worker, &bctx, n_batch);
            }
            hmx_queue_pop(hmx_q); // QS(c)
            hmx_queue_pop(hmx_q); // DELTA(c)
            if (has_next) {
                hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_kk_worker, &bctx));
            }

            // Step 4: HMX KK(c+1) || HVX phase4(c)
            bctx.c            = c;
            bctx.curr_buf     = curr_buf;
            bctx.s_update_buf = curr_buf;
            work_queue_run(wp, gdn_hvx_phase4_worker, &bctx, n_batch);
            hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_o_intra_worker,  &bctx));
            hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_s_update_worker, &bctx));

            // Step 5: HMX O_INTRA(c) + S_UPDATE(c) || HVX phase1b(c+1)
            if (has_next) {
                bctx.c        = c + 1u;
                bctx.curr_buf = next_buf;
                work_queue_run(wp, gdn_hvx_phase1b_worker, &bctx, n_batch);
                hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_qk_worker, &bctx));
                hmx_queue_pop(hmx_q); // KK(c+1)
            }
            hmx_queue_pop(hmx_q); // O_INTRA(c)

            // Step 6: HMX S_UPDATE(c) tail + QK(c+1) || HVX phase5(c) + DMA store O(c)
            bctx.c        = c;
            bctx.curr_buf = curr_buf;
            work_queue_run(wp, gdn_hvx_phase5_worker, &bctx, n_batch);
            hmx_queue_pop(hmx_q); // S_UPDATE(c)

            // Step 7: HMX QK(c+1) tail || HVX phase6(c)
            work_queue_run(wp, gdn_hvx_phase6_worker, &bctx, n_batch);
            if (has_next) {
                hmx_queue_push(hmx_q, hmx_queue_make_desc(gdn_hmx_ks_worker, &bctx));
            }
        }

        work_queue_run(wp, gdn_hvx_epilogue_worker, &bctx, n_batch);
    }

    if (aux_buf && aux_buf != octx->ctx->ddr_spad_base) {
        free(aux_buf);
    }
    return HTP_STATUS_OK;
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * dst       = octx->dst;
    const struct htp_tensor * dst_cache = octx->dsts[1];

    for (int i = 0; i < 3; i++) {
        if (octx->src[i]->type != HTP_TYPE_F32 && octx->src[i]->type != HTP_TYPE_F16) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    for (int i = 3; i < 6; i++) {
        if (octx->src[i]->type != HTP_TYPE_F32) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    const bool replay = gdn_has_replay(octx);
    const struct htp_tensor * ids   = replay ? NULL : octx->src[6];
    const struct htp_tensor * n_rep = replay ? octx->src[9] : NULL;
    if (replay) {
        if (!octx->src[8] || !octx->src[9] ||
            octx->src[6]->type != HTP_TYPE_F32 ||
            octx->src[7]->type != HTP_TYPE_F32 ||
            octx->src[8]->type != HTP_TYPE_F32 ||
            octx->src[9]->type != HTP_TYPE_I32) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        if (octx->src[7] || octx->src[8] || octx->src[9]) {
            return HTP_STATUS_NO_SUPPORT;
        }
        if (ids && ids->type != HTP_TYPE_I32 && ids->type != HTP_TYPE_I64) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (dst_cache) {
        if (dst_cache->type != HTP_TYPE_F32) {
            return HTP_STATUS_NO_SUPPORT;
        }
        if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else if (dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_gdn_kernel_params * kparams = (const struct htp_gdn_kernel_params *) octx->kernel_params;
    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t S_v        = kparams->S_v;
    const uint32_t n_tokens   = kparams->n_tokens;
    const uint32_t total_rows = kparams->total_rows;
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows,
            1,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.count,
            &octx->ctx->mdev.count_div
        );
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) {
        return gated_delta_net_f32_hmx_chunked(octx, kparams, row_start, nrows);
    }

    const uint32_t n_hvx = (nrows < octx->n_hvx) ? nrows : octx->n_hvx;

    struct htp_gdn_context gctx;
    gctx.octx      = octx;
    gctx.kparams   = kparams;
    gctx.row_start = row_start;
    gctx.nrows     = nrows;
    gctx.load_q    = (octx->src[0]->type == HTP_TYPE_F16) ? gdn_load_vec_f16_to_f32 : gdn_load_vec_f32;
    gctx.load_k    = (octx->src[1]->type == HTP_TYPE_F16) ? gdn_load_vec_f16_to_f32 : gdn_load_vec_f32;
    gctx.load_v    = (octx->src[2]->type == HTP_TYPE_F16) ? gdn_load_vec_f16_to_f32 : gdn_load_vec_f32;
    gctx.store_out = (dst->type == HTP_TYPE_F16) ? gdn_store_vec_f16 : gdn_store_vec_f32;
    gctx.step_fn   = kparams->kda ? gdn_step_kda_f32 : gdn_step_scalar_f32;

    htp_gdn_vtcm_layout_build(&gctx.layout, S_v, n_hvx);
    gctx.vtcm_base = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, gctx.layout.off_state);

    if (gctx.layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    void * aux_buf = NULL;
    const int aux_err = gdn_prepare_aux_data(octx, replay ? n_rep : ids, &aux_buf);
    if (aux_err != HTP_STATUS_OK) {
        return aux_err;
    }
    gctx.ids_type   = ids ? ids->type : 0;
    gctx.ids_data   = replay ? NULL : aux_buf;
    gctx.n_rep_data = replay ? (const int32_t *) aux_buf : NULL;
    gctx.n_pend     = replay ? (uint32_t) octx->op_params[1] : 0;
    gctx.replay     = replay;
    gctx.keep       = replay && octx->op_params[2] != 0;

    FARF(HIGH, "gated-delta-net-f32: q(%ux%ux%ux%u) k(%ux%ux%ux%u) v(%ux%ux%ux%u) state(%ux%ux%ux%u) -> (%ux%ux%ux%u) : "
         "vtcm-size %zu n_hvx %u\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->src[2]->ne[0], octx->src[2]->ne[1], octx->src[2]->ne[2], octx->src[2]->ne[3],
         octx->src[5]->ne[0], octx->src[5]->ne[1], octx->src[5]->ne[2], octx->src[5]->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gctx.layout.total_bytes, n_hvx);

    bool has_replay_work = gctx.keep;
    if (replay && !has_replay_work) {
        for (uint32_t s = 0; s < kparams->n_seqs; ++s) {
            if (gctx.n_rep_data[s] > 0) {
                has_replay_work = true;
                break;
            }
        }
    }

    if (n_tokens == 1 && !has_replay_work) {
        work_queue_run(octx->ctx->hvx_queue, gated_delta_net_f32_tg_thread, &gctx, n_hvx);
    } else {
        work_queue_run(octx->ctx->hvx_queue, gated_delta_net_f32_pp_thread, &gctx, n_hvx);
    }

    if (aux_buf && aux_buf != octx->ctx->ddr_spad_base) {
        free(aux_buf);
    }

    return HTP_STATUS_OK;
}
