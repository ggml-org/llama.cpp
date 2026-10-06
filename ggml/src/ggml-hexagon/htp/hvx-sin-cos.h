#ifndef HVX_SIN_COS_H
#define HVX_SIN_COS_H

#include "hvx-base.h"
#include "hvx-floor.h"

// Range-reduce x to y in [-pi/2, pi/2] and the quadrant sign (-1)^n.
// Floor/truncate need IEEE bits, so convert qf32 back to sf before them.
static inline void hvx_vec_sincos_reduce_f32(HVX_Vector x, HVX_Vector * y, HVX_Vector * sign) {
    HVX_Vector const_inv_pi  = hvx_vec_splat_f32(0.3183098861837907f);
    HVX_Vector const_half    = hvx_vec_splat_f32(0.5f);
    HVX_Vector const_pi      = hvx_vec_splat_f32(3.141592653589793f);
    HVX_Vector const_one     = hvx_vec_splat_f32(1.0f);
    HVX_Vector const_neg_one = hvx_vec_splat_f32(-1.0f);
    HVX_Vector const_one_i   = Q6_V_vsplat_R(1);

    HVX_Vector x_over_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, const_inv_pi));
    x_over_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x_over_pi, const_half));

    HVX_Vector n_float = hvx_vec_floor_f32(x_over_pi);
    HVX_Vector n_int   = hvx_vec_truncate_f32(n_float);

    HVX_Vector n_pi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(n_float, const_pi));
    *y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, n_pi));

    HVX_VectorPred is_odd = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(n_int, const_one_i), const_one_i);
    *sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);
}

static inline void hvx_vec_sincos_f32(HVX_Vector x, HVX_Vector * vcos, HVX_Vector * vsin) {
    HVX_Vector y;
    HVX_Vector sign;
    hvx_vec_sincos_reduce_f32(x, &y, &sign);

    HVX_Vector z = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, y));

    HVX_Vector c4 = hvx_vec_splat_f32(2.3557242013849433e-05f);
    HVX_Vector c3 = hvx_vec_splat_f32(-0.0013871428263450528f);
    HVX_Vector c2 = hvx_vec_splat_f32(0.041665895266688284f);
    HVX_Vector c1 = hvx_vec_splat_f32(-0.4999999360426369f);
    HVX_Vector c0 = hvx_vec_splat_f32(0.9999999999071725f);

    HVX_Vector cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, c4))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));
    cos_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, cos_y))));

    HVX_Vector s4 = hvx_vec_splat_f32(2.642186986152672e-06f);
    HVX_Vector s3 = hvx_vec_splat_f32(-0.00019825318964070864f);
    HVX_Vector s2 = hvx_vec_splat_f32(0.00833326283319605f);
    HVX_Vector s1 = hvx_vec_splat_f32(-0.16666666082087775f);
    HVX_Vector s0 = hvx_vec_splat_f32(0.999999999915155f);

    HVX_Vector sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, s4))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(s0, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(z, sin_y))));
    sin_y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, sin_y));

    *vcos = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(cos_y, sign));
    *vsin = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sin_y, sign));
}

static inline HVX_Vector hvx_vec_cos_f32(HVX_Vector x) {
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(x, &vcos, &vsin);
    return vcos;
}

static inline HVX_Vector hvx_vec_sin_f32(HVX_Vector x) {
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(x, &vcos, &vsin);
    return vsin;
}

#endif /* HVX_SIN_COS_H */
