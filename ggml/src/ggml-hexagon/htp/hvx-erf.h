#ifndef HVX_ERF_H
#define HVX_ERF_H

#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
static __attribute__((noinline)) HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const HVX_Vector ax = hvx_vec_abs_f32(x);
    HVX_Vector t = hvx_vec_inverse_f32(hvx_vec_add_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_Vector poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    const HVX_Vector exp_term = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_Vector result = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_term));

    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    result = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(result), result);
    return result;
}

// GELU_ERF uses the erf definition.
static inline HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
    const HVX_Vector scale = hvx_vec_splat_f32(0.7071067811865475f);
    const HVX_Vector half  = hvx_vec_splat_f32(0.5f);
    const HVX_Vector one   = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_x = hvx_vec_splat_f32(10.0f);
    const HVX_Vector min_x = hvx_vec_splat_f32(-10.0f);
    const HVX_VectorPred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    const HVX_VectorPred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_Vector x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    const HVX_Vector erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_Vector result = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_add_f32_f32(one, erf));

    result = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), result);
    result = Q6_V_vmux_QVV(pos_large, x, result);
    return result;
}

static inline void hvx_gelu_erf_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    const uint32_t elem_size = sizeof(float);
    const uint32_t epv       = 128 / elem_size;
    const uint32_t nvec      = n / epv;
    const uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    if (nloe) {
        HVX_Vector v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((void *) &vdst[i], nloe * elem_size, v);
    }
}

#endif /* HVX_ERF_H */
