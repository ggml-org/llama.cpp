#ifndef HVX_SIGMOID_H
#define HVX_SIGMOID_H

#include "hvx-base.h"
#include "hvx-inverse.h"
#include "hvx-exp.h"

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_f32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_VectorPair hvx_vec_fast_sigmoid_f32_guard_pair(HVX_Vector x0,
                                                                 HVX_Vector x1,
                                                                 HVX_Vector one,
                                                                 HVX_Vector max_exp,
                                                                 HVX_Vector min_exp) {
    const HVX_Vector c_log2f   = Q6_V_vsplat_R(FAST_SIGMOID_LOG2F);
    const HVX_Vector c1        = Q6_V_vsplat_R(FAST_SIGMOID_C1);
    const HVX_Vector c2        = Q6_V_vsplat_R(FAST_SIGMOID_C2);
    const HVX_Vector c3        = Q6_V_vsplat_R(FAST_SIGMOID_C3);
    const HVX_Vector inv_aprox = Q6_V_vsplat_R(0x7EEEEBB3);
    const HVX_Vector two_sf    = hvx_vec_splat_f32(2.0f);
    const HVX_Vector zero      = Q6_V_vzero();

    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(x0, c_log2f);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(x1, c_log2f);
    v0            = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v0), c3);
    v1            = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v1), c3);

    HVX_Vector in_int0 = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v0));
    HVX_Vector in_int1 = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v1));
    HVX_Vector f0      = Q6_Vqf32_vsub_Vqf32Vsf(v0, Q6_Vsf_equals_Vw(in_int0));
    HVX_Vector f1      = Q6_Vqf32_vsub_Vqf32Vsf(v1, Q6_Vsf_equals_Vw(in_int1));
    HVX_Vector ff0     = Q6_Vqf32_vmpy_Vqf32Vqf32(f0, f0);
    HVX_Vector ff1     = Q6_Vqf32_vmpy_Vqf32Vqf32(f1, f1);

    HVX_Vector p1_0 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ff0), c2);
    HVX_Vector p1_1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ff1), c2);
    p1_0            = Q6_Vqf32_vadd_Vqf32Vsf(p1_0, c_log2f);
    p1_1            = Q6_Vqf32_vadd_Vqf32Vsf(p1_1, c_log2f);

    HVX_Vector p2_0 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(f0), c1);
    HVX_Vector p2_1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(f1), c1);
    p2_0            = Q6_Vqf32_vmpy_Vqf32Vqf32(p2_0, ff0);
    p2_1            = Q6_Vqf32_vmpy_Vqf32Vqf32(p2_1, ff1);
    p2_0            = Q6_Vqf32_vadd_Vqf32Vqf32(p2_0, f0);
    p2_1            = Q6_Vqf32_vadd_Vqf32Vqf32(p2_1, f1);

    HVX_Vector v3_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p2_0, p1_0));
    HVX_Vector v3_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p2_1, p1_1));
    v3_0            = Q6_Vw_vaslacc_VwVwR(v3_0, in_int0, 24);
    v3_1            = Q6_Vw_vaslacc_VwVwR(v3_1, in_int1, 24);

    HVX_Vector v4_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(p2_0, p1_0));
    HVX_Vector v4_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(p2_1, p1_1));
    HVX_Vector v5_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3_0, v4_0));
    HVX_Vector v5_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3_1, v4_1));

    HVX_Vector i0 = Q6_Vw_vsub_VwVw(inv_aprox, v5_0);
    HVX_Vector i1 = Q6_Vw_vsub_VwVw(inv_aprox, v5_1);

    HVX_Vector r0 = Q6_Vqf32_vmpy_VsfVsf(
        i0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i0, v5_0)))));
    HVX_Vector r1 = Q6_Vqf32_vmpy_VsfVsf(
        i1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i1, v5_1)))));

    r0 = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r0, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r0), v5_0))));
    r1 = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r1, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r1), v5_1))));

    r0 = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r0, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r0), v5_0))));
    r1 = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r1, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r1), v5_1))));

    HVX_Vector out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3_0, Q6_Vsf_equals_Vqf32(r0)));
    HVX_Vector out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3_1, Q6_Vsf_equals_Vqf32(r1)));

    const HVX_VectorPred pmax0 = Q6_Q_vcmp_gt_VsfVsf(max_exp, x0);
    const HVX_VectorPred pmin0 = Q6_Q_vcmp_gt_VsfVsf(x0, min_exp);
    const HVX_VectorPred pmax1 = Q6_Q_vcmp_gt_VsfVsf(max_exp, x1);
    const HVX_VectorPred pmin1 = Q6_Q_vcmp_gt_VsfVsf(x1, min_exp);

    out0 = Q6_V_vmux_QVV(pmin0, Q6_V_vmux_QVV(pmax0, out0, one), zero);
    out1 = Q6_V_vmux_QVV(pmin1, Q6_V_vmux_QVV(pmax1, out1, one), zero);

    return Q6_W_vcombine_VV(out1, out0);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_2it(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_Vector two_sf = hvx_vec_splat_f32(2.0f);
    HVX_Vector i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_Vector r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_Vector res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard_2it(HVX_Vector v,
                                                            HVX_Vector one,
                                                            HVX_Vector max_exp,
                                                            HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32_2it(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_f32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_f32(2.0f);
    HVX_Vector one = hvx_vec_splat_f32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_sigmoid_f32(HVX_Vector x) {
    HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);
    return hvx_vec_fast_sigmoid_f32_guard(x, one, max_exp, min_exp);
}

static inline HVX_Vector hvx_vec_silu_f32(HVX_Vector x) {
    HVX_Vector sig = hvx_vec_sigmoid_f32(x);
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, sig));
}

#define hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    do {                                                                                      \
        dst_type * restrict vdst = (dst_type *) dst;                                          \
        src_type * restrict vsrc = (src_type *) src;                                          \
                                                                                              \
        const HVX_Vector one     = hvx_vec_splat_f32(1.f);                                    \
        const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);                                   \
        const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        const uint32_t epv  = 128 / sizeof(float);                                            \
        const uint32_t nvec = n / epv;                                                        \
        const uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        for (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        if (nloe) {                                                                           \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp);                         \
        }                                                                                     \
    } while(0)

#define hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    do {                                                              \
        dst_type * restrict vdst = (dst_type *) dst;                  \
        src_type * restrict vsrc = (src_type *) src;                  \
                                                                      \
        const uint32_t epv  = 128 / sizeof(float);                    \
        const uint32_t nvec = n / epv;                                \
        const uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        for (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        if (nloe) {                                                   \
             HVX_Vector tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp); \
        }                                                             \
    } while(0)

static inline void hvx_sigmoid_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sigmoid_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_tanh_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_tanh_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

#if __HVX_ARCH__ >= 81
// Centered degree-4 minimax polynomial for 1 / (1 + e) in t = e - 0.5 on [-0.5, 0.5]
#define HVX_SIGMOID_RECIP_C0 (0x3956)
#define HVX_SIGMOID_RECIP_C1 (0xB711)
#define HVX_SIGMOID_RECIP_C2 (0x34AF)
#define HVX_SIGMOID_RECIP_C3 (0xB365)
#define HVX_SIGMOID_RECIP_C4 (0x3113)

static inline HVX_Vector hvx_vec_fast_sigmoid_f16(HVX_Vector x_v) {
    const HVX_Vector a   = Q6_V_vand_VV(x_v, Q6_Vh_vsplat_R(0x7FFF));
    const HVX_Vector z   = Q6_Vuh_vmin_VuhVuh(a, Q6_Vh_vsplat_R(0x4C50));  // min(|x|, 17.25) so e underflows to exact 0 in fp16
    const HVX_Vector y   = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(z, Q6_Vh_vsplat_R(0x65C5)));
    const HVX_Vector t   = Q6_Vh_equals_Vhf_rnd(y);
    const HVX_Vector rho = Q6_V_vand_VV(t, Q6_Vh_vsplat_R(0x03FF));
    const HVX_Vector r15 = Q6_Vh_vasl_VhR(rho, 5);
    HVX_Vector       rp  = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhRh_s1_rnd_sat(r15, HVX_EXP2_FIXED_G2_PAIR), Q6_Vh_vsplat_R(HVX_EXP2_FIXED_G1));
    rp                   = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp, r15), Q6_Vh_vsplat_R(HVX_EXP2_FIXED_G0));
    const HVX_Vector b   = Q6_Vh_vadd_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x6400), t), Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp, rho));
    const HVX_Vector e   = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(b, Q6_Vh_vsplat_R(0x1400)));
    const HVX_Vector tc  = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(e, Q6_Vh_vsplat_R(0xB800)));

    HVX_Vector p = Q6_Vqf16_vmpy_VhfVhf(tc, Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C4));
    p            = Q6_Vqf16_vadd_Vqf16Vhf(p, Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C3));
    p            = Q6_Vqf16_vmpy_Vqf16Vhf(p, tc);
    p            = Q6_Vqf16_vadd_Vqf16Vhf(p, Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C2));
    p            = Q6_Vqf16_vmpy_Vqf16Vhf(p, tc);
    p            = Q6_Vqf16_vadd_Vqf16Vhf(p, Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C1));
    p            = Q6_Vqf16_vmpy_Vqf16Vhf(p, tc);
    const HVX_Vector r_q = Q6_Vqf16_vadd_Vqf16Vhf(p, Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C0));

    const HVX_Vector     r      = Q6_Vhf_equals_Vqf16(r_q);
    const HVX_Vector     er     = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(r_q, e));
    const HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), x_v);
    return Q6_V_vmux_QVV(is_neg, er, r);
}

static inline HVX_VectorPair hvx_vec_fast_sigmoid_f16_pair(HVX_Vector x0, HVX_Vector x1) {
    const HVX_Vector em_mask = Q6_Vh_vsplat_R(0x7FFF);
    const HVX_Vector v_clamp = Q6_Vh_vsplat_R(0x4C50);  // min(|x|, 17.25) so e underflows to exact 0 in fp16
    const HVX_Vector v_scale = Q6_Vh_vsplat_R(0x65C5);
    const HVX_Vector v_frac  = Q6_Vh_vsplat_R(0x03FF);
    const HVX_Vector v_g1    = Q6_Vh_vsplat_R(HVX_EXP2_FIXED_G1);
    const HVX_Vector v_g0    = Q6_Vh_vsplat_R(HVX_EXP2_FIXED_G0);
    const HVX_Vector v_base  = Q6_Vh_vsplat_R(0x6400);
    const HVX_Vector v_last  = Q6_Vh_vsplat_R(0x1400);
    const HVX_Vector v_nhalf = Q6_Vh_vsplat_R(0xB800);

    const HVX_Vector z0 = Q6_Vuh_vmin_VuhVuh(Q6_V_vand_VV(x0, em_mask), v_clamp);
    const HVX_Vector z1 = Q6_Vuh_vmin_VuhVuh(Q6_V_vand_VV(x1, em_mask), v_clamp);
    const HVX_Vector y0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(z0, v_scale));
    const HVX_Vector y1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(z1, v_scale));
    const HVX_Vector t0 = Q6_Vh_equals_Vhf_rnd(y0);
    const HVX_Vector t1 = Q6_Vh_equals_Vhf_rnd(y1);
    const HVX_Vector rho0 = Q6_V_vand_VV(t0, v_frac);
    const HVX_Vector rho1 = Q6_V_vand_VV(t1, v_frac);
    const HVX_Vector r15_0 = Q6_Vh_vasl_VhR(rho0, 5);
    const HVX_Vector r15_1 = Q6_Vh_vasl_VhR(rho1, 5);

    HVX_Vector rp0 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhRh_s1_rnd_sat(r15_0, HVX_EXP2_FIXED_G2_PAIR), v_g1);
    HVX_Vector rp1 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhRh_s1_rnd_sat(r15_1, HVX_EXP2_FIXED_G2_PAIR), v_g1);
    rp0            = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp0, r15_0), v_g0);
    rp1            = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp1, r15_1), v_g0);

    const HVX_Vector b0 = Q6_Vh_vadd_VhVh(Q6_Vh_vsub_VhVh(v_base, t0), Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp0, rho0));
    const HVX_Vector b1 = Q6_Vh_vadd_VhVh(Q6_Vh_vsub_VhVh(v_base, t1), Q6_Vh_vmpy_VhVh_s1_rnd_sat(rp1, rho1));
    const HVX_Vector e0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(b0, v_last));
    const HVX_Vector e1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(b1, v_last));
    const HVX_Vector tc0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(e0, v_nhalf));
    const HVX_Vector tc1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(e1, v_nhalf));

    const HVX_Vector c4 = Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C4);
    const HVX_Vector c3 = Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C3);
    const HVX_Vector c2 = Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C2);
    const HVX_Vector c1 = Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C1);
    const HVX_Vector c0 = Q6_Vh_vsplat_R(HVX_SIGMOID_RECIP_C0);

    HVX_Vector p0 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(tc0, c4), c3);
    HVX_Vector p1 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(tc1, c4), c3);
    p0            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p0, tc0), c2);
    p1            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p1, tc1), c2);
    p0            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p0, tc0), c1);
    p1            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p1, tc1), c1);
    const HVX_Vector rq0 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p0, tc0), c0);
    const HVX_Vector rq1 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(p1, tc1), c0);

    const HVX_Vector zero   = Q6_V_vzero();
    const HVX_Vector r0     = Q6_Vhf_equals_Vqf16(rq0);
    const HVX_Vector r1     = Q6_Vhf_equals_Vqf16(rq1);
    const HVX_Vector er0    = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(rq0, e0));
    const HVX_Vector er1    = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(rq1, e1));
    const HVX_Vector out0   = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(zero, x0), er0, r0);
    const HVX_Vector out1   = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(zero, x1), er1, r1);
    return Q6_W_vcombine_VV(out1, out0);
}
#else
// Clamping u >= -14.0f keeps 2^u in normal fp16 range while 1.0f + 2^-14 == 1.0f in fp16
static inline HVX_Vector hvx_vec_fast_sigmoid_f16(HVX_Vector x_v) {
    const HVX_Vector v_one       = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    const HVX_Vector em_mask     = Q6_Vh_vsplat_R(0x7FFF);
    const HVX_Vector half_hf_v   = Q6_Vh_vsplat_R(0x3800);
    const HVX_Vector v_clamp_min = hvx_vec_splat_f16(-14.0f);

    HVX_Vector     abs_x  = Q6_V_vand_VV(x_v, em_mask);
    HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);
    HVX_Vector     u      = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);
    u                     = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_Vector x_minus_half = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(u, half_hf_v));
    HVX_Vector k_v          = Q6_Vh_equals_Vhf(x_minus_half);
    HVX_Vector f_v          = Q6_Vhf_equals_Vh(k_v);
    HVX_Vector x_qf16       = Q6_Vqf16_vsub_VhfVhf(u, f_v);

    // Degree-4 polynomial for 2^f on [-0.5, 0.5] (max error < 4.2e-5, well below fp16 0.5 ULP)
    HVX_Vector y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_Vh_vsplat_R(0x20ed), Q6_V_vzero()), x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, v_one);

    HVX_Vector exp_val = Q6_Vh_vaslacc_VhVhR(Q6_Vhf_equals_Vqf16(y), k_v, 10);
    HVX_Vector denom   = hvx_vec_add_f16_f16(v_one, exp_val);
    HVX_Vector sig_abs = hvx_vec_inverse_pos_f16(denom);
    HVX_Vector sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    return Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

static inline HVX_VectorPair hvx_vec_fast_sigmoid_f16_pair(HVX_Vector x0, HVX_Vector x1) {
    const HVX_Vector v_one       = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    const HVX_Vector em_mask     = Q6_Vh_vsplat_R(0x7FFF);
    const HVX_Vector half_hf_v   = Q6_Vh_vsplat_R(0x3800);
    const HVX_Vector v_clamp_min = hvx_vec_splat_f16(-14.0f);

    const HVX_Vector     abs0    = Q6_V_vand_VV(x0, em_mask);
    const HVX_Vector     abs1    = Q6_V_vand_VV(x1, em_mask);
    const HVX_VectorPred is_neg0 = Q6_Q_vcmp_gt_VhVh(abs0, x0);
    const HVX_VectorPred is_neg1 = Q6_Q_vcmp_gt_VhVh(abs1, x1);

    HVX_Vector u0 = Q6_Vhf_vmax_VhfVhf(v_clamp_min, hvx_vec_mul_f16_f16(abs0, v_neg_log2e));
    HVX_Vector u1 = Q6_Vhf_vmax_VhfVhf(v_clamp_min, hvx_vec_mul_f16_f16(abs1, v_neg_log2e));

    HVX_Vector k0 = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(u0, half_hf_v)));
    HVX_Vector k1 = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(u1, half_hf_v)));
    HVX_Vector q0 = Q6_Vqf16_vsub_VhfVhf(u0, Q6_Vhf_equals_Vh(k0));
    HVX_Vector q1 = Q6_Vqf16_vsub_VhfVhf(u1, Q6_Vhf_equals_Vh(k1));

    // Degree-4 polynomial for 2^f on [-0.5, 0.5] (max error < 4.2e-5, well below fp16 0.5 ULP)
    const HVX_Vector c3 = Q6_Vqf16_vadd_VhfVhf(Q6_Vh_vsplat_R(0x20ed), Q6_V_vzero());
    const HVX_Vector c2 = Q6_Vh_vsplat_R(0x2b1b);
    const HVX_Vector c1 = Q6_Vh_vsplat_R(0x33b0);
    const HVX_Vector c0 = Q6_Vh_vsplat_R(0x398c);

    HVX_Vector y0 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(c3, q0), c2);
    HVX_Vector y1 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(c3, q1), c2);
    y0            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y0, q0), c1);
    y1            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y1, q1), c1);
    y0            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y0, q0), c0);
    y1            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y1, q1), c0);
    y0            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y0, q0), v_one);
    y1            = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vqf16(y1, q1), v_one);

    HVX_Vector e0   = Q6_Vh_vaslacc_VhVhR(Q6_Vhf_equals_Vqf16(y0), k0, 10);
    HVX_Vector e1   = Q6_Vh_vaslacc_VhVhR(Q6_Vhf_equals_Vqf16(y1), k1, 10);
    HVX_Vector d0   = hvx_vec_add_f16_f16(v_one, e0);
    HVX_Vector d1   = hvx_vec_add_f16_f16(v_one, e1);
    HVX_Vector sa0  = hvx_vec_inverse_pos_f16(d0);
    HVX_Vector sa1  = hvx_vec_inverse_pos_f16(d1);
    HVX_Vector sn0  = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sa0));
    HVX_Vector sn1  = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sa1));
    HVX_Vector out0 = Q6_V_vmux_QVV(is_neg0, sn0, sa0);
    HVX_Vector out1 = Q6_V_vmux_QVV(is_neg1, sn1, sa1);
    return Q6_W_vcombine_VV(out1, out0);
}
#endif

static inline HVX_Vector hvx_vec_tanh_f16(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    const HVX_Vector v_two = hvx_vec_splat_f16(2.0f);

    HVX_Vector x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f16(x2);

    const HVX_Vector v_neg_one = hvx_vec_splat_f16(-1.0f);
    return hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#define hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    do {                                                                 \
        dst_type * restrict vdst = (dst_type *) dst;                     \
        src_type * restrict vsrc = (src_type *) src;                     \
                                                                         \
        const uint32_t epv  = 128 / sizeof(_Float16);                    \
        const uint32_t nvec = n / epv;                                   \
        const uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        for (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        if (nloe) {                                                      \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((void *) &vdst[i], nloe * sizeof(_Float16), tmp); \
        }                                                                \
    } while(0)

#define hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    do {                                                                 \
        dst_type * restrict vdst = (dst_type *) dst;                     \
        src_type * restrict vsrc = (src_type *) src;                     \
                                                                         \
        const uint32_t epv  = 128 / sizeof(_Float16);                    \
        const uint32_t nvec = n / epv;                                   \
        const uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        for (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        if (nloe) {                                                      \
             HVX_Vector tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((void *) &vdst[i], nloe * sizeof(_Float16), tmp); \
        }                                                                \
    } while(0)

static inline void hvx_sigmoid_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_tanh_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

#endif /* HVX_SIGMOID_H */
