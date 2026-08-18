#ifndef HVX_LOG_H
#define HVX_LOG_H

#include "hvx-base.h"

// Approximates ln(x) element-wise for float vectors.
// x must contain positive float elements.
// Uses Abramowitz & Stegun polynomial approximation 4.1.44 for ln(1+y) over [0, 1].
static inline HVX_Vector hvx_vec_log_f32(HVX_Vector x) {
    // x = m * 2^e, where m in [1, 2)
    HVX_Vector biased_e = Q6_Vuw_vlsr_VuwR(x, 23);
    HVX_Vector e_int = Q6_Vw_vsub_VwVw(biased_e, Q6_V_vsplat_R(127));
    HVX_Vector e_float = Q6_Vsf_equals_Vw(e_int);

    // Extract mantissa and set exponent to 127 (which represents float value in [1.0, 2.0))
    HVX_Vector mant_mask = Q6_V_vsplat_R(0x007FFFFF);
    HVX_Vector exp_127 = Q6_V_vsplat_R(0x3F800000);
    HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(x, mant_mask), exp_127);

    // y = m - 1.0f, y in [0, 1)
    HVX_Vector y = hvx_vec_sub_f32_f32(m, hvx_vec_splat_f32(1.0f));

    // Abramowitz & Stegun 4.1.44 polynomial approximation of ln(1+y)
    HVX_Vector c;
    HVX_Vector res;

    c   = hvx_vec_splat_f32(-0.0064535442f);
    res = hvx_vec_mul_f32_f32(y, c);

    c   = hvx_vec_splat_f32(0.0360884937f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.0953293897f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.1676540711f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.2407338084f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.3317990258f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.4998741238f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.9999964239f);
    res = hvx_vec_add_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    // ln(x) = e * ln(2) + ln(1+y)
    HVX_Vector ln2 = hvx_vec_splat_f32(0.69314718056f);
    HVX_Vector term_e = hvx_vec_mul_f32_f32(e_float, ln2);

    return hvx_vec_add_f32_f32(term_e, res);
}

#endif /* HVX_LOG_H */
