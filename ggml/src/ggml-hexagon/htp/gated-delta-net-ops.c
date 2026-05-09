#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define HTP_GDN_MAX_SV 128

struct htp_gdn_context {
    struct htp_ops_context * octx;
    uint32_t rows_per_thread;
    size_t state_bytes;
    bool use_vtcm;
    uint8_t * vtcm_state_base;
    size_t vtcm_state_per_thread;
};

static inline float gdn_mul_dot_f32(float * restrict dst, const float * restrict mul,
        const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vd = hvx_vmemu(dst + off);
        HVX_Vector vm = hvx_vmem(mul + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_Vector out = hvx_vec_mul_f32_f32(vd, vm);
        hvx_vec_store_u(dst + off, tail * sizeof(float), out);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector prod = hvx_vec_mul_f32_f32(out, vdot);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, prod, Q6_V_vzero()));
    }

    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
}

static inline float gdn_mul_scalar_dot_f32(float * restrict dst, float mul,
        const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vd = hvx_vmemu(dst + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_Vector out = hvx_vec_mul_f32_f32(vd, vmul);
        hvx_vec_store_u(dst + off, tail * sizeof(float), out);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector prod = hvx_vec_mul_f32_f32(out, vdot);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, prod, Q6_V_vzero()));
    }

    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
}

static inline float gdn_add_scaled_dot_f32(float * restrict dst, const float * restrict src,
        float scale, const float * restrict dot, uint32_t n) {
    HVX_Vector acc = Q6_V_vzero();
    const HVX_Vector vscale = hvx_vec_splat_f32(scale);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vd = hvx_vmemu(dst + i * epv);
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vmemu(dst + i * epv) = out;
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(out, vdot));
    }

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vd = hvx_vmemu(dst + off);
        HVX_Vector vs = hvx_vmem(src + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_Vector out = hvx_vec_add_f32_f32(vd, hvx_vec_mul_f32_f32(vs, vscale));
        hvx_vec_store_u(dst + off, tail * sizeof(float), out);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector prod = hvx_vec_mul_f32_f32(out, vdot);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(mask, prod, Q6_V_vzero()));
    }

    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(acc));
}

static inline void gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict mul,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = hvx_vmem(mul + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static inline void gdn_mul_scalar_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float mul,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static inline void gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict src,
        const float * restrict scale, const float * restrict dot, uint32_t n,
        float * restrict sums) {
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
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = hvx_vmem(src + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static inline void gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const float * restrict mul, const float * restrict dot, uint32_t n,
        float * restrict sums) {
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
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = hvx_vmem(mul + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, tail * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, tail * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, tail * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, tail * sizeof(float), out7);

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
    hvx_vec_store_u(sums + 0, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accA));
    hvx_vec_store_u(sums + 4, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accB));
}

static inline void gdn_mul_scalar_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        float mul, const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector vmul = hvx_vec_splat_f32(mul);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vmul);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vmul);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vmul);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vmul);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vmul);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vmul);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vmul);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vmul);

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, tail * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, tail * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, tail * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, tail * sizeof(float), out7);

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
    hvx_vec_store_u(sums + 0, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accA));
    hvx_vec_store_u(sums + 4, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accB));
}

static inline void gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const float * restrict src, const float * restrict scale,
        const float * restrict dot, uint32_t n, float * restrict sums) {
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
    const uint32_t tail = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

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

    if (tail) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = hvx_vmem(src + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, tail * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, tail * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, tail * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, tail * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, tail * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, tail * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, tail * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, tail * sizeof(float), out7);

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
    hvx_vec_store_u(sums + 0, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accA));
    hvx_vec_store_u(sums + 4, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accB));
}

static void gated_delta_net_f32_pp_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];

    const uint32_t total_rows = H * n_seqs;
    if (ith >= total_rows) {
        return;
    }

    const uint32_t rq3 = n_seqs / q->ne[3];
    const uint32_t rk3 = n_seqs / k->ne[3];
    const float scale = 1.0f / sqrtf((float) S_v);

    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_base + (uint64_t) S_v * H * n_tokens * n_seqs;
    const float * state_in_base = (const float *) (uintptr_t) state->data;

    const bool kda = (g->ne[0] == S_v);
    float local_gate[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_q[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_k[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_sums[4] __attribute__((aligned(128)));

    for (uint32_t ir = ith; ir < total_rows; ir += nth) {
        const uint32_t iv1 = ir % H;
        const uint32_t iv3 = ir / H;

        const uint32_t iq1 = iv1 % q->ne[1];
        const uint32_t ik1 = iv1 % k->ne[1];
        const uint32_t iq3 = iv3 / rq3;
        const uint32_t ik3 = iv3 / rk3;

        float * s_out = state_out_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
        const float * s_in = state_in_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;

        memcpy(s_out, s_in, gctx->state_bytes);
        float * s_work = s_out;

        float * attn_data = dst_base + ((uint64_t) iv3 * n_tokens * H + iv1) * S_v;

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

            memcpy(local_q, q_t, (size_t) S_v * sizeof(float));
            memcpy(local_k, k_t, (size_t) S_v * sizeof(float));

            if (kda) {
                hvx_exp_f32((uint8_t *) local_gate, (const uint8_t *) g_t, S_v, false);

                uint32_t j = 0;
                for (; j + 4 <= S_v; j += 4) {
                    float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                    float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                    float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                    float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                    gdn_mul_dot4_f32(row0, row1, row2, row3, local_gate, local_k, S_v, local_sums);
                    float local_delta_b[4] __attribute__((aligned(128)));
                    for (uint32_t r = 0; r < 4; ++r) {
                        local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                    }
                    gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                    for (uint32_t r = 0; r < 4; ++r) {
                        attn_data[j + r] = local_sums[r] * scale;
                    }
                }
                for (; j < S_v; ++j) {
                    float * row = s_work + (uint64_t) j * S_v;
                    const float sum = gdn_mul_dot_f32(row, local_gate, local_k, S_v);
                    const float dj = (v_t[j] - sum) * beta_val;
                    attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
                }
            } else {
                const float gate = expf(g_t[0]);
                uint32_t j = 0;
                for (; j + 4 <= S_v; j += 4) {
                    float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                    float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                    float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                    float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                    gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, gate, local_k, S_v, local_sums);
                    float local_delta_b[4] __attribute__((aligned(128)));
                    for (uint32_t r = 0; r < 4; ++r) {
                        local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                    }
                    gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                    for (uint32_t r = 0; r < 4; ++r) {
                        attn_data[j + r] = local_sums[r] * scale;
                    }
                }
                for (; j < S_v; ++j) {
                    float * row = s_work + (uint64_t) j * S_v;
                    const float sum = gdn_mul_scalar_dot_f32(row, gate, local_k, S_v);
                    const float dj = (v_t[j] - sum) * beta_val;
                    attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
                }
            }

            attn_data += (uint64_t) S_v * H;
        }
    }
}

static void gated_delta_net_f32_tg_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_gdn_context * gctx = (struct htp_gdn_context *) data;
    struct htp_ops_context * octx = gctx->octx;

    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_seqs   = v->ne[3];

    const uint32_t total_rows = H * n_seqs;
    if (ith >= total_rows) {
        return;
    }

    const uint32_t rq3 = n_seqs / q->ne[3];
    const uint32_t rk3 = n_seqs / k->ne[3];
    const float scale = 1.0f / sqrtf((float) S_v);

    float * dst_base       = (float *) (uintptr_t) dst->data;
    float * state_out_base = dst_base + (uint64_t) S_v * H * n_seqs;
    const float * state_in_base = (const float *) (uintptr_t) state->data;

    const bool kda = (g->ne[0] == S_v);
    float local_gate[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_q[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_k[HTP_GDN_MAX_SV] __attribute__((aligned(128)));
    float local_sums[8] __attribute__((aligned(128)));

    dma_queue * dma = octx->ctx->dma[ith];

    uint8_t * spad = NULL;
    if (gctx->use_vtcm) {
        spad = gctx->vtcm_state_base + gctx->vtcm_state_per_thread * ith;
    }

    for (uint32_t ir = ith; ir < total_rows; ir += nth) {
        const uint32_t iv1 = ir % H;
        const uint32_t iv3 = ir / H;

        const uint32_t iq1 = iv1 % q->ne[1];
        const uint32_t ik1 = iv1 % k->ne[1];
        const uint32_t iq3 = iv3 / rq3;
        const uint32_t ik3 = iv3 / rk3;

        float * s_out = state_out_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
        const float * s_in = state_in_base + ((uint64_t) iv3 * H + iv1) * S_v * S_v;
        float * s_work;

        if (spad) {
            dma_queue_push(dma, dma_make_ptr(spad, s_in),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);
            dma_queue_pop(dma);
            s_work = (float *) spad;
        } else {
            s_work = s_out;
            memcpy(s_work, s_in, gctx->state_bytes);
        }

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

        memcpy(local_q, q_t, (size_t) S_v * sizeof(float));
        memcpy(local_k, k_t, (size_t) S_v * sizeof(float));

        if (kda) {
            hvx_exp_f32((uint8_t *) local_gate, (const uint8_t *) g_t, S_v, false);

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
                gdn_mul_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                 local_gate, local_k, S_v, local_sums);
                float local_delta_b[8] __attribute__((aligned(128)));
                for (uint32_t r = 0; r < 8; ++r) {
                    local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                }
                gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                        local_k, local_delta_b, local_q, S_v, local_sums);
                for (uint32_t r = 0; r < 8; ++r) {
                    attn_data[j + r] = local_sums[r] * scale;
                }
            }
            for (; j + 4 <= S_v; j += 4) {
                float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                gdn_mul_dot4_f32(row0, row1, row2, row3, local_gate, local_k, S_v, local_sums);
                float local_delta_b[4] __attribute__((aligned(128)));
                for (uint32_t r = 0; r < 4; ++r) {
                    local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                }
                gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                for (uint32_t r = 0; r < 4; ++r) {
                    attn_data[j + r] = local_sums[r] * scale;
                }
            }
            for (; j < S_v; ++j) {
                float * row = s_work + (uint64_t) j * S_v;
                const float sum = gdn_mul_dot_f32(row, local_gate, local_k, S_v);
                const float dj = (v_t[j] - sum) * beta_val;
                attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
            }
        } else {
            const float gate = expf(g_t[0]);
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
                gdn_mul_scalar_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                        gate, local_k, S_v, local_sums);
                float local_delta_b[8] __attribute__((aligned(128)));
                for (uint32_t r = 0; r < 8; ++r) {
                    local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                }
                gdn_add_scaled_dot8_f32(row0, row1, row2, row3, row4, row5, row6, row7,
                                        local_k, local_delta_b, local_q, S_v, local_sums);
                for (uint32_t r = 0; r < 8; ++r) {
                    attn_data[j + r] = local_sums[r] * scale;
                }
            }
            for (; j + 4 <= S_v; j += 4) {
                float * row0 = s_work + (uint64_t) (j + 0) * S_v;
                float * row1 = s_work + (uint64_t) (j + 1) * S_v;
                float * row2 = s_work + (uint64_t) (j + 2) * S_v;
                float * row3 = s_work + (uint64_t) (j + 3) * S_v;
                gdn_mul_scalar_dot4_f32(row0, row1, row2, row3, gate, local_k, S_v, local_sums);
                float local_delta_b[4] __attribute__((aligned(128)));
                for (uint32_t r = 0; r < 4; ++r) {
                    local_delta_b[r] = (v_t[j + r] - local_sums[r]) * beta_val;
                }
                gdn_add_scaled_dot4_f32(row0, row1, row2, row3, local_k, local_delta_b, local_q, S_v, local_sums);
                for (uint32_t r = 0; r < 4; ++r) {
                    attn_data[j + r] = local_sums[r] * scale;
                }
            }
            for (; j < S_v; ++j) {
                float * row = s_work + (uint64_t) j * S_v;
                const float sum = gdn_mul_scalar_dot_f32(row, gate, local_k, S_v);
                const float dj = (v_t[j] - sum) * beta_val;
                attn_data[j] = gdn_add_scaled_dot_f32(row, local_k, dj, local_q, S_v) * scale;
            }
        }

        if (spad) {
            dma_queue_push(dma, dma_make_ptr(s_out, spad),
                           S_v * sizeof(float), S_v * sizeof(float),
                           S_v * sizeof(float), S_v);
            dma_queue_pop(dma);
        }
    }
}

int op_gated_delta_net(struct htp_ops_context * octx) {
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * g     = octx->src[3];
    const struct htp_tensor * beta  = octx->src[4];
    const struct htp_tensor * state = octx->src[5];
    const struct htp_tensor * dst   = octx->dst;

    if (!q || !k || !v || !g || !beta || !state || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F32 || v->type != HTP_TYPE_F32 ||
        g->type != HTP_TYPE_F32 || beta->type != HTP_TYPE_F32 || state->type != HTP_TYPE_F32 ||
        dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t S_v      = v->ne[0];
    const uint32_t H        = v->ne[1];
    const uint32_t n_tokens = v->ne[2];
    const uint32_t n_seqs   = v->ne[3];

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
    if (state->ne[0] * state->ne[1] * state->ne[2] * state->ne[3] != S_v * S_v * H * n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct htp_gdn_context gctx;
    gctx.octx = octx;
    gctx.rows_per_thread = (H * n_seqs + octx->n_threads - 1) / octx->n_threads;
    gctx.state_bytes = (size_t) S_v * S_v * sizeof(float);

    size_t state_aligned = (size_t) S_v * S_v * sizeof(float);
    state_aligned = (state_aligned + 127) & ~(size_t)127;

    gctx.use_vtcm = false;
    gctx.vtcm_state_base = NULL;
    gctx.vtcm_state_per_thread = 0;

    if (n_tokens == 1 && octx->ctx->vtcm_base) {
        size_t vtcm_total = state_aligned * octx->n_threads;
        if (octx->ctx->vtcm_size >= vtcm_total) {
            gctx.use_vtcm = true;
            gctx.vtcm_state_base = octx->ctx->vtcm_base;
            gctx.vtcm_state_per_thread = state_aligned;
        }
    }

    if (n_tokens == 1) {
        worker_pool_run_func(octx->ctx->worker_pool, gated_delta_net_f32_tg_thread, &gctx, octx->n_threads);
    } else {
        worker_pool_run_func(octx->ctx->worker_pool, gated_delta_net_f32_pp_thread, &gctx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
