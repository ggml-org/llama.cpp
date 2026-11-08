#ifndef HVX_POW_H
#define HVX_POW_H

#include <math.h>
#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
static inline HVX_Vector hvx_vec_pow_const_base_f32(float base, HVX_Vector exponent) {
    float ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.7228f;

    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    return hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
static inline HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.7228f;

    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    return hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
