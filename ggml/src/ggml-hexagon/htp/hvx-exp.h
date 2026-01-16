#ifndef HVX_EXP_H
#define HVX_EXP_H

#include <stdbool.h>
#include <stdint.h>

#include "hvx-base.h"
#include "hvx-floor.h"

#define EXP_COEFF_5 (0x39506967)  // 0.000198757 = 1/(7!)
#define EXP_COEFF_4 (0x3AB743CE)  // 0.0013982   = 1/(6!)
#define EXP_COEFF_3 (0x3C088908)  // 0.00833345  = 1/(5!)
#define EXP_COEFF_2 (0x3D2AA9C1)  // 0.416658    = 1/(4!)
#define EXP_COEFF_1 (0x3E2AAAAA)  // 0.16666667  = 1/(3!)
#define EXP_COEFF_0 (0x3F000000)  // 0.5         = 1/(2!)
#define EXP_LOGN2   (0x3F317218)  // ln(2)   = 0.6931471805
#define EXP_LOG2E   (0x3FB8AA3B)  // log2(e) = 1/ln(2) = 1.4426950408
#define EXP_ONE     (0x3f800000)  // 1.0
#define EXP_RANGE_R (0x41a00000)  // 20.0
#define EXP_RANGE_L (0xc1a00000)  // -20.0

static inline HVX_Vector hvx_vec_exp_f32(HVX_Vector in_vec) {
    HVX_Vector z_qf32_v;
    HVX_Vector x_v;
    HVX_Vector x_qf32_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector epsilon_v;
    HVX_Vector log2e = Q6_V_vsplat_R(EXP_LOG2E);
    HVX_Vector logn2 = Q6_V_vsplat_R(EXP_LOGN2);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x) ~= 1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x) ~= 1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =  1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    HVX_Vector temp_v = in_vec;

    // Clamp inputs to (-20.0, 20.0)
    HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, Q6_V_vsplat_R(EXP_RANGE_R));
    HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(EXP_RANGE_L), in_vec);

    in_vec = Q6_V_vmux_QVV(pred_cap_right, Q6_V_vsplat_R(EXP_RANGE_R), temp_v);
    in_vec = Q6_V_vmux_QVV(pred_cap_left, Q6_V_vsplat_R(EXP_RANGE_L), temp_v);

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = hvx_vec_floor_f32(epsilon_v);
    k_v = hvx_vec_truncate_f32(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);

    //  x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    // normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    // z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    // y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(EXP_COEFF_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E3 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E2 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E1 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E0 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = y + 1.0;
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(EXP_ONE));

    // insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent

    y_v = Q6_Vsf_equals_Vqf32(y_v);

    // add k_v to the exponent of y_v
    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, IEEE_VSF_MANTLEN + 1);
    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, IEEE_VSF_MANTLEN);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    return y_v;
}

static inline HVX_Vector hvx_vec_exp_f32_guard(HVX_Vector in_vec, HVX_Vector max_exp, HVX_Vector inf) {
    const HVX_VectorPred pred0 = Q6_Q_vcmp_gt_VsfVsf(in_vec, max_exp);

    HVX_Vector out = hvx_vec_exp_f32(in_vec);

    return Q6_V_vmux_QVV(pred0, inf, out);
}

static inline void hvx_exp_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems, bool negate) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == hex_is_aligned((void *) src, VLEN)) || (0 == hex_is_aligned((void *) dst, VLEN))) {
        unaligned_addr = 1;
    }
    // assert((0 == unaligned_addr) || (0 == num_elems_whole));
    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
    }

    HVX_Vector vec_out = Q6_V_vzero();

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.02f;  // log(INF)

    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    if (0 == unaligned_loop) {
        HVX_Vector * p_vec_in1 = (HVX_Vector *) src;
        HVX_Vector * p_vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            if (true == negate) {
                HVX_Vector neg_vec_in = hvx_vec_neg_f32(*p_vec_in1++);
                *p_vec_out++          = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
            } else {
                *p_vec_out++ = hvx_vec_exp_f32_guard(*p_vec_in1++, max_exp, inf);
            }
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            if (true == negate) {
                HVX_Vector neg_vec_in                    = hvx_vec_neg_f32(in);
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
            } else {
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(in, max_exp, inf);
            }
        }
    }

    if (left_over > 0) {
        const float * srcf = (float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        if (true == negate) {
            HVX_Vector neg_vec_in = hvx_vec_neg_f32(in);

            vec_out = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
        } else {
            vec_out = hvx_vec_exp_f32_guard(in, max_exp, inf);
        }

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, vec_out);
    }
}

#endif /* HVX_EXP_H */
