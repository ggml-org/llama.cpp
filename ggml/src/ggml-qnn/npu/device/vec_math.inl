#pragma once

#include "hexagon_npu.h"

#include <hexagon_types.h>

#include <cstdint>

// TODO: move this macros to a common header
#define IEEE_VSF_EXPLEN   (8)
#define IEEE_VSF_EXPBIAS  (127)
#define IEEE_VSF_EXPMASK  (0xFF)
#define IEEE_VSF_MANTLEN  (23)
#define IEEE_VSF_MANTMASK (0x7FFFFF)
#define IEEE_VSF_MIMPMASK (0x800000)

#define IEEE_VHF_EXPLEN   (5)
#define IEEE_VHF_EXPBIAS  (15)
#define IEEE_VHF_EXPMASK  (0x1F)
#define IEEE_VHF_MANTLEN  (10)
#define IEEE_VHF_MANTMASK (0x3FF)
#define IEEE_VHF_MIMPMASK (0x400)

#define COEFF_EXP_5 0x39506967  // 0.000198757 = 1/(7!)
#define COEFF_EXP_4 0x3AB743CE  // 0.0013982   = 1/(6!)
#define COEFF_EXP_3 0x3C088908  // 0.00833345  = 1/(5!)
#define COEFF_EXP_2 0x3D2AA9C1  // 0.416658    = 1/(4!)
#define COEFF_EXP_1 0x3E2AAAAA  // 0.16666667  = 1/(3!)
#define COEFF_EXP_0 0x3F000000  // 0.5         = 1/(2!)
#define LOGN2       0x3F317218  // ln(2)   = 0.6931471805
#define LOG2E       0x3FB8AA3B  // log2(e) = 1/ln(2) = 1.4426950408

#define COEFF_EXP_5_HF 0x0A83   // 0.000198757 = 1/(7!)
#define COEFF_EXP_4_HF 0x15BA   // 0.0013982   = 1/(6!)
#define COEFF_EXP_3_HF 0x2044   // 0.00833345  = 1/(5!)
#define COEFF_EXP_2_HF 0x36AB   // 0.416658    = 1/(4!)
#define COEFF_EXP_1_HF 0x3155   // 0.16666667  = 1/(3!)
#define COEFF_EXP_0_HF 0x3800   // 0.5         = 1/(2!)
#define LOGN2_HF       0x398C   // ln(2)   = 0.693147
#define LOG2E_HF       0x3DC5   // log2(e) = 1/ln(2) = 1.4427

namespace hexagon::vec::math {

inline HVX_Vector qhmath_hvx_vsf_floor_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v    = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(vin, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);               // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

//  truncate(x)
//  given a vector of float x,
//  return the vector of integers resulting from dropping all fractional bits
//  no checking performed for overflow - could be extended to return maxint
//
// truncate float to int
inline HVX_Vector qhmath_hvx_vw_truncate_vsf(HVX_Vector vin) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, vin);

    HVX_Vector expval_v = vin >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;                // fractional bits - exp shift

    HVX_Vector mant_v = vin & mask_mant_v;                            // obtain mantissa
    HVX_Vector vout   = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);         // add implicit 1.0
    vout              = Q6_Vw_vasr_VwVw(vout, rshift_v);              // shift to obtain truncated integer
    vout              = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0 -> 0

    HVX_Vector neg_vout = -vout;
    vout                = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives
    return (vout);
}

// qhmath_hvx_vhf_floor_vhf(x)
//  given a vector of half float x,
//  return the vector of largest integer valued half float <= x
//
inline HVX_Vector qhmath_hvx_vhf_floor_vhf(HVX_Vector vin) {
    HVX_Vector mask_mant_v    = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector const_emask_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v  = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_Vh_vsplat_R(0xbc00);  // -1 IEEE vhf

    // initialization (no changes)
    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhR(vin, IEEE_VHF_MANTLEN);
    expval_v            = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v            = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VhVh(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVhVh(q_negexp, vin, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVhVh(q_negexp, const_zero_v, vin);

    // if expval < 0 (q_negexp)   // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval          // fraction bits to mask off
    //    vout = ~(mask)          // apply mask to remove fraction
    //    if (qneg) // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                       // already an integer
    //    ; // no change

    // compute floor
    mask_mant_v               = Q6_Vh_vasr_VhVh(mask_mant_v, expval_v);
    HVX_Vector neg_addin_v    = Q6_Vh_vasr_VhVh(mask_impl_v, expval_v);
    HVX_Vector vout_neg_addin = Q6_Vh_vadd_VhVh(vin, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, vin);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(vin, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VhVh(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = vin;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, vin, vout);               // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1
    return vout;
}

// truncate half float to short
inline HVX_Vector qhmath_hvx_vh_truncate_vhf(HVX_Vector vin) {
    HVX_Vector const_mnlen_v = Q6_Vh_vsplat_R(IEEE_VHF_MANTLEN);
    HVX_Vector mask_mant_v   = Q6_Vh_vsplat_R(IEEE_VHF_MANTMASK);
    HVX_Vector mask_impl_v   = Q6_Vh_vsplat_R(IEEE_VHF_MIMPMASK);
    HVX_Vector const_emask_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPMASK);
    HVX_Vector const_ebias_v = Q6_Vh_vsplat_R(IEEE_VHF_EXPBIAS);
    HVX_Vector const_zero_v  = Q6_V_vzero();
    HVX_Vector const_one_v   = Q6_Vh_vsplat_R(1);

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VhVh(const_zero_v, vin);

    HVX_Vector expval_v = Q6_Vh_vasr_VhVh(vin, const_mnlen_v);
    expval_v            = Q6_V_vand_VV(expval_v, const_emask_v);
    expval_v            = Q6_Vh_vsub_VhVh(expval_v, const_ebias_v);

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VhVh(const_zero_v, expval_v);

    // fractional bits - exp shift
    HVX_Vector rshift_v = Q6_Vh_vsub_VhVh(const_mnlen_v, expval_v);

    HVX_Vector mant_v = vin & mask_mant_v;                            // obtain mantissa
    HVX_Vector vout   = Q6_Vh_vadd_VhVh(mant_v, mask_impl_v);         // add implicit 1.0
    vout              = Q6_Vh_vasr_VhVh(vout, rshift_v);              // shift to obtain truncated integer
    vout              = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);  // expval<0 -> 0

    // HVX_Vector neg_vout = -vout;
    HVX_Vector not_vout = Q6_V_vnot_V(vout);
    HVX_Vector neg_vout = Q6_Vh_vadd_VhVh(not_vout, const_one_v);
    vout                = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives
    return (vout);
}

/*
 * This function computes the exponent on all IEEE 32-bit float elements of an HVX_Vector
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_exp_vf(HVX_Vector sline) {
    HVX_Vector z_qf32_v;
    HVX_Vector x_v;
    HVX_Vector x_qf32_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector epsilon_v;
    HVX_Vector log2e = Q6_V_vsplat_R(LOG2E);
    HVX_Vector logn2 = Q6_V_vsplat_R(LOGN2);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // 1) clipping + uint input
    //        if (x > MAXLOG)
    //            return (MAXNUM);
    //        if (x < MINLOG)
    //            return (0.0);
    //
    // 2) exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x)~=1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x)~=1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, sline);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = qhmath_hvx_vsf_floor_vsf(epsilon_v);
    k_v = qhmath_hvx_vw_truncate_vsf(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(sline, zero_v);

    //    x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    //    normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    //    z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    //    y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(COEFF_EXP_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E3 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E2 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E1 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = E0 + y * x;
    E_const = Q6_V_vsplat_R(COEFF_EXP_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    //    y = y + 1.0;
    E_const = Q6_V_vsplat_R(0x3f800000);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);

    //insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent
    y_v = Q6_Vsf_equals_Vqf32(y_v);

    //    add k_v to the exponent of y_v
    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);

    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    //    exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    return y_v;
}

/*
 * This function computes the exponent on all IEEE 16-bit float elements of an HVX_Vector
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_exp_vhf(HVX_Vector sline) {
    HVX_Vector z_qf16_v;
    HVX_Vector x_qf16_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector tmp_v;
    HVX_Vector log2e = Q6_Vh_vsplat_R(LOG2E_HF);
    HVX_Vector logn2 = Q6_Vh_vsplat_R(LOGN2_HF);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // 1) clipping + uint input
    //        if (x > MAXLOG)
    //            return (MAXNUM);
    //        if (x < MINLOG)
    //            return (0.0);

    // 2) round to int
    //    k = (int) (x * log2e);
    //    f = (float) k;
    //    k = Q6_R_convert_sf2w_R(log2e * x); //f = floorf( log2e * x + 0.5);
    //    f = Q6_R_convert_w2sf_R(k);         //k = (int)f;

    tmp_v            = Q6_Vqf16_vmpy_VhfVhf(log2e, sline);
    //    float16's 0.5 is 0x3800
    HVX_Vector cp5_v = Q6_Vh_vsplat_R(0x3800);
    tmp_v            = Q6_Vqf16_vadd_Vqf16Vhf(tmp_v, cp5_v);
    tmp_v            = Q6_Vhf_equals_Vqf16(tmp_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = qhmath_hvx_vhf_floor_vhf(tmp_v);
    k_v = qhmath_hvx_vh_truncate_vhf(f_v);

    x_qf16_v = Q6_Vqf16_vadd_VhfVhf(sline, zero_v);

    //    x = x - f * logn2;
    tmp_v    = Q6_Vqf16_vmpy_VhfVhf(f_v, logn2);
    x_qf16_v = Q6_Vqf16_vsub_Vqf16Vqf16(x_qf16_v, tmp_v);

    //    normalize before every QFloat's vmpy
    x_qf16_v = Q6_Vqf16_vadd_Vqf16Vhf(x_qf16_v, zero_v);

    //    z = x * x;
    z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, x_qf16_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vhf(z_qf16_v, zero_v);

    //    y = E4 + E5 * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_5_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vhf(x_qf16_v, E_const);
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_4_HF);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E3 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_3_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E2 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_2_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E1 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_1_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = E0 + y * x;
    E_const = Q6_Vh_vsplat_R(COEFF_EXP_0_HF);
    y_v     = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = x + y * z;
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, z_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, zero_v);

    //    y = y + 1.0;
    E_const = Q6_Vh_vsplat_R(0x3C00);
    y_v     = Q6_Vqf16_vadd_Vqf16Vhf(y_v, E_const);

    // insert exponents
    //    y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent
    y_v = Q6_Vhf_equals_Vqf16(y_v);

    // add k_v to the exponent of y_v
    // shift away sign bit
    HVX_Vector y_v_exponent = Q6_Vh_vasl_VhR(y_v, 1);

    // shift back by sign bit + 10-bit mantissa
    y_v_exponent = Q6_Vuh_vlsr_VuhR(y_v_exponent, 11);

    y_v_exponent = Q6_Vh_vadd_VhVh(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VhVh(zero_v, y_v_exponent);

    // max IEEE hf exponent; if overflow detected, result is set to infinity
    HVX_Vector     exp_max_v              = Q6_Vh_vsplat_R(0x1e);
    // INF in 16-bit float is 0x7C00
    HVX_Vector     inf_v                  = Q6_Vh_vsplat_R(0x7C00);
    HVX_VectorPred qy_v_overflow_exponent = Q6_Q_vcmp_gt_VhVh(y_v_exponent, exp_max_v);

    // update exponent
    y_v = Q6_Vh_vaslacc_VhVhR(y_v, k_v, 10);

    // clip to min/max values
    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
    y_v = Q6_V_vmux_QVV(qy_v_overflow_exponent, inf_v, y_v);

    return y_v;
}

inline HVX_VectorPair_x4 qhmath_load_div_sf_ltu() {
    /* Coefficients in float representation */
    alignas(hexagon::kBytesPerVector) constexpr const float c0_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.882601794814435,
        3.6625422144222575,
        3.464451548227971,
        3.2869700047974098,
        3.126105117815294,
        2.9797652947122333,
        2.846287833147896,
        2.7247270166228237,
        2.614282526778659,
        2.5119448279766914,
        2.4168240690138916,
        2.3287715099556494,
        2.2470044371606255,
        2.1705097010458525,
        2.0993232550771013,
        2.032425103348979,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c1_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -5.65213466274883,
        -5.029649818173625,
        -4.500359068222728,
        -4.051125252469975,
        -3.6643282495304743,
        -3.3293252513210945,
        -3.0377500909629918,
        -2.78384542029156,
        -2.562751394984757,
        -2.3660481944625364,
        -2.1902579830702398,
        -2.033579850063907,
        -1.8932880190031018,
        -1.7665817851802996,
        -1.6526109646324616,
        -1.5489652830974667,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c2_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.6564123863772062,
        3.0693863078484034,
        2.5979108429264546,
        2.2188401136904137,
        1.90879196515026,
        1.6531365145318937,
        1.4408072849395228,
        1.2640160009581791,
        1.1164726565567085,
        0.9904366133906549,
        0.8821387892416702,
        0.7892039810345458,
        0.7089644931002874,
        0.6390020714403465,
        0.5781761255999769,
        0.5246475096790261,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c3_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.8868796162009371,
        -0.7023245532864408,
        -0.5623148115716742,
        -0.45568061400557225,
        -0.3728293181808119,
        -0.30778916969628956,
        -0.25624427383670373,
        -0.21520836864975557,
        -0.18238585316003267,
        -0.1554651987039696,
        -0.133224398745864,
        -0.11484835534787588,
        -0.09954996553138899,
        -0.08667244996867919,
        -0.07585106425203664,
        -0.06663557250850614,
    };

    /* Load coefficients */
    HVX_Vector c0_coeff_v = *((HVX_Vector *) (c0_coeffs));
    HVX_Vector c1_coeff_v = *((HVX_Vector *) (c1_coeffs));
    HVX_Vector c2_coeff_v = *((HVX_Vector *) (c2_coeffs));
    HVX_Vector c3_coeff_v = *((HVX_Vector *) (c3_coeffs));

    /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
    hexagon::HVX_VectorPair_x4 result;
    result.val[0] = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    result.val[1] = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    result.val[2] = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    result.val[3] = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

    return result;
}

inline HVX_Vector qhmath_hvx_div_vf(HVX_Vector num, HVX_Vector denom, HVX_VectorPair_x4 coeffs) {
    HVX_Vector     sline1;
    HVX_Vector     sline2;
    HVX_Vector     norm_factor;
    HVX_Vector     tmp_v;
    HVX_Vector     idx1_v;
    HVX_Vector     idx2_v;
    HVX_Vector     output_v;
    HVX_Vector     input_shifted_v_qf32;
    HVX_Vector     input_scaled_v_qf32;
    HVX_VectorPair c0_coeff_vp;
    HVX_VectorPair c1_coeff_vp;
    HVX_VectorPair c2_coeff_vp;
    HVX_VectorPair c3_coeff_vp;

    /*
     * Splat scale factor in order to be used later for finding indexes of coefficients.
     * Scale factor is represented in IEEE 16-bit floating-point format and it is
     * calculated using the following formula:
     *    scale_factor = (16.0 / (b0 - a0))
     * NOTE: Calculated value is slightly decreased in order to avoid out of bound
     *       indexes during VLUT lookup.
     */
    HVX_Vector scale_v = Q6_V_vsplat_R(0x417ffffe);

    /*
     * Vector of zeroes used as neutral element in sf to qf32 conversions.
     * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
     *       can be avoided in real-time, but this is not done in order to don't
     *       sacrify code readibility in expense of insignificant performance improvement.
     */
    HVX_Vector zero_v_sf = Q6_V_vzero();

    /* Set sign = 0, exp = 254, mant = 0 */
    HVX_Vector exp = Q6_V_vsplat_R(0x7F000000);

    /* Set mask for sign and exponent */
    HVX_Vector signexp_mask = Q6_V_vsplat_R(0xFF800000);

    /* Mask for extracting only 4 bits of mantissa */
    HVX_Vector mask_idx1_v = Q6_V_vsplat_R(0x0000000F);
    HVX_Vector mask_idx2_v = Q6_V_vsplat_R(0x00000010);

    /* 16.0 in IEEE 16-bit floating-point representation */
    HVX_Vector const16_0_v_sf = Q6_V_vsplat_R(0x41800000);

    /*
     * Prepare vector of input_min values, that is used later in shifting input range.
     * input_min is low boundary of specified input range.
     */
    HVX_Vector input_min_v_f = Q6_V_vsplat_R(0x3f800000);

    /* Convert scale factor from sf to q32. Use the same vector for both formats */
    scale_v = Q6_Vqf32_vadd_VsfVsf(scale_v, zero_v_sf);

    /* Calculate normalization factor */
    norm_factor = Q6_V_vand_VV(denom, signexp_mask);
    norm_factor = Q6_Vw_vsub_VwVw(exp, norm_factor);

    /* Normalize denominators */
    sline2 = Q6_Vqf32_vmpy_VsfVsf(denom, norm_factor);
    sline2 = Q6_Vsf_equals_Vqf32(sline2);

    /* Convert normalization factor and numerator to qf32 */
    norm_factor = Q6_Vqf32_vadd_VsfVsf(norm_factor, zero_v_sf);
    sline1      = Q6_Vqf32_vadd_VsfVsf(num, zero_v_sf);

    /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
    input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline2, input_min_v_f);

    /*
             * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
             * in order to get corresponding coefficient indexes
             */
    input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

    /*
             * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
             * to [16.0,32.0) in order to convert float indexes to integer values.
             * Float values, represented in IEEE 754, in range [16.0,32.0] have the
             * same exponent, which means 4 MSB of mantissa carry information about
             * integer index.
             */
    input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

    /* Convert back from qf32 to sf in order to extract integer index */
    tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

    /* Only 4 MSB bits of mantissa represent segment index */
    idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

    idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
    idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
    idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

    /* Obtain the polynomial coefficients from lookup table */
    c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[0]), 1);
    c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[0]), 1);
    c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[1]), 1);
    c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[1]), 1);
    c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[2]), 1);
    c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[2]), 1);
    c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[3]), 1);
    c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[3]), 1);

    /* Perform evaluation of polynomial using Horner's method */
    output_v = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(c3_coeff_vp), sline2);
    output_v = Q6_Vqf32_vadd_Vqf32Vsf(output_v, Q6_V_lo_W(c2_coeff_vp));
    output_v = Q6_Vsf_equals_Vqf32(output_v);

    output_v = Q6_Vqf32_vmpy_VsfVsf(output_v, sline2);
    output_v = Q6_Vqf32_vadd_Vqf32Vsf(output_v, Q6_V_lo_W(c1_coeff_vp));
    output_v = Q6_Vsf_equals_Vqf32(output_v);

    output_v = Q6_Vqf32_vmpy_VsfVsf(output_v, sline2);
    output_v = Q6_Vqf32_vadd_Vqf32Vsf(output_v, Q6_V_lo_W(c0_coeff_vp));

    /* Multiply result by same normalization factor applied to input earlier */
    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, norm_factor);

    /* Calculate num * 1/den */
    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, sline1);

    return Q6_Vsf_equals_Vqf32(output_v);
}

inline HVX_VectorPair_x4 qhmath_load_div_hf_ltu() {
    /* Coefficients in float representation */
    alignas(hexagon::kBytesPerVector) constexpr const float c0_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.8807721943716516,
        3.6618209528616856,
        3.4657742282097708,
        3.2853461610022414,
        3.1229570908314015,
        2.976379865829892,
        2.8438614274889833,
        2.723793061029549,
        2.613859154046634,
        2.5119508509784287,
        2.4167270706641473,
        2.3286721812015188,
        2.2462659531748064,
        2.1692490555028736,
        2.0981551828382417,
        2.0319234960945,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c1_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -5.646783581176797,
        -5.027704168781284,
        -4.5037889029173535,
        -4.0470997487793445,
        -3.6569569537789364,
        -3.3217563552211695,
        -3.03258650196419,
        -2.781935505534812,
        -2.5619261358961922,
        -2.3660577978107398,
        -2.190083163030879,
        -2.033405493468989,
        -1.8920413948588666,
        -1.7645298754188785,
        -1.6507730169513504,
        -1.5482028127706613,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c2_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.6511964773849632,
        3.0676375988553106,
        2.6008750952258324,
        2.215514199159397,
        1.9030391013295935,
        1.6474963735373633,
        1.4371447652517673,
        1.2627141904289978,
        1.11593649827749,
        0.9904415490260164,
        0.882033772823834,
        0.7891019704346331,
        0.7082630629776306,
        0.6378888508693012,
        0.5772121720355701,
        0.524261196551401,
    };
    alignas(hexagon::kBytesPerVector) constexpr const float c3_coeffs[32] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.8851851956149304,
        -0.7018008948429424,
        -0.5631686602024177,
        -0.4547647803673564,
        -0.37133287830029976,
        -0.3063883382130307,
        -0.255378412302572,
        -0.2149126167280633,
        -0.18226975346347984,
        -0.15546600267845986,
        -0.13320337246909697,
        -0.11482846255803722,
        -0.0994184164975366,
        -0.08647114157420362,
        -0.07568254923048714,
        -0.06657033258736733,
    };

    /* Load coefficients */
    HVX_Vector c0_coeff_v = *((HVX_Vector *) (c0_coeffs));
    HVX_Vector c1_coeff_v = *((HVX_Vector *) (c1_coeffs));
    HVX_Vector c2_coeff_v = *((HVX_Vector *) (c2_coeffs));
    HVX_Vector c3_coeff_v = *((HVX_Vector *) (c3_coeffs));

    /* Convert coefficients from hf to qf32 format. Use the same vector for both representations */
    HVX_Vector zero_v_hf = Q6_V_vzero();
    c0_coeff_v           = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_hf);
    c1_coeff_v           = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_hf);
    c2_coeff_v           = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_hf);
    c3_coeff_v           = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_hf);

    /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
    hexagon::HVX_VectorPair_x4 result;
    result.val[0] = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    result.val[1] = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    result.val[2] = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    result.val[3] = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

    return result;
}

inline HVX_Vector qhmath_hvx_div_vhf(HVX_Vector num, HVX_Vector denom, HVX_VectorPair_x4 coeffs) {
    HVX_Vector     sline2;
    HVX_Vector     norm_factor;
    HVX_VectorPair norm_factor_qf32;
    HVX_Vector     tmp_v;
    HVX_Vector     idx1_v;
    HVX_Vector     idx2_v;
    HVX_DV         output_dv;
    HVX_Vector     input_shifted_v_hf;
    HVX_Vector     input_scaled_v;
    HVX_VectorPair input_vp_qf32;
    HVX_VectorPair input_n_vp_qf32;
    HVX_VectorPair c0_coeff_vp;
    HVX_VectorPair c1_coeff_vp;
    HVX_VectorPair c2_coeff_vp;
    HVX_VectorPair c3_coeff_vp;

    /*
      * Splat scale factor in order to be used later for finding indexes of coefficients.
      * Scale factor is represented in IEEE 16-bit floating-point format and it is
      * calculated using the following formula:
      *    scale_factor = (convert_sf_to_hf) (16.0 / (b0 - a0))
      * NOTE: Calculated value is slightly decreased in order to avoid out of bound
      *       indexes during VLUT lookup.
      */
    HVX_Vector scale_v = Q6_Vh_vsplat_R(0x4bfb);

    /* Vector of ones used as mpy neutral element in conversions from hf vector to qf32 vector pair */
    HVX_Vector one_v_hf = Q6_Vh_vsplat_R(0x3c00);

    /*
     * Vector of zeroes used as neutral element in hf to qf16 conversions.
     * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
     *       can be avoided in real-time, but this is not done in order to don't
     *       sacrify code readibility in expense of insignificant performance improvement.
     */
    HVX_Vector zero_v_hf = Q6_V_vzero();

    /* Set sign = 0, exp = 30, mant = 0 */
    HVX_Vector exp = Q6_Vh_vsplat_R(0x7800);

    /* Set mask for sign and exponent */
    HVX_Vector signexp_mask = Q6_Vh_vsplat_R(0xFC00);

    /* Mask for extracting only 4 bits of mantissa */
    HVX_Vector mask_idx1_v = Q6_Vh_vsplat_R(0x000F);
    HVX_Vector mask_idx2_v = Q6_V_vsplat_R(0x00001010);

    /* 16.0 in IEEE 16-bit floating-point representation */
    HVX_Vector const16_0_v_hf = Q6_Vh_vsplat_R(0x4c00);

    /*
     * Prepare vector of input_min values, that is used later in shifting input range.
     * input_min is low boundary of specified input range.
     */
    HVX_Vector input_min_v_hf = Q6_Vh_vsplat_R(0x3c00);

    /* Convert scale factor from hf to q16. Use the same vector for both formats */
    scale_v = Q6_Vqf16_vadd_VhfVhf(scale_v, zero_v_hf);

    /* Calculate normalization factor */
    norm_factor = Q6_V_vand_VV(denom, signexp_mask);
    norm_factor = Q6_Vh_vsub_VhVh(exp, norm_factor);

    /* Normalize denominators */
    sline2 = Q6_Vqf16_vmpy_VhfVhf(denom, norm_factor);

    /* Convert normalization factor to qf32 */
    norm_factor_qf32 = Q6_Wqf32_vmpy_VhfVhf(norm_factor, one_v_hf);

    /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
    tmp_v              = Q6_Vh_vdeal_Vh(sline2);
    input_shifted_v_hf = Q6_Vqf16_vsub_Vqf16Vhf(tmp_v, input_min_v_hf);

    /*
             * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
             * in order to get corresponding coefficient indexes
             */
    input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

    /*
             * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
             * to [16.0,32.0) in order to convert float indexes to integer values.
             * Float values, represented in IEEE 754, in range [16.0,32.0] have the
             * same exponent, which means 4 MSB of mantissa carry information about
             * integer index.
             */
    /* Use the same input_scaled_v vector for hf and qf16 representation */
    input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

    /* Convert back from qf16 to hf in order to extract integer index  */
    tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

    /* Only 4 MSB bits of mantissa represent segment index */
    idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

    /* Ensure only 4 MSB bits of mantissa are used as indexes */
    idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);

    idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
    idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
    idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

    /* Obtain the polynomial coefficients from lookup table */
    c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[0]), 1);
    c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[0]), 1);
    c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[1]), 1);
    c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[1]), 1);
    c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[2]), 1);
    c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[2]), 1);
    c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(coeffs.val[3]), 1);
    c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(coeffs.val[3]), 1);

    /* Convert inputs from hf vector to qf32 vector pair for Horner's method*/
    input_vp_qf32   = Q6_Wqf32_vmpy_Vqf16Vhf(sline2, one_v_hf);
    input_n_vp_qf32 = Q6_Wqf32_vmpy_VhfVhf(num, one_v_hf);

    /* Perform evaluation of polynomial using Horner's method */
    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c3_coeff_vp), Q6_V_lo_W(input_vp_qf32));
    output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
    output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
    output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c3_coeff_vp), Q6_V_hi_W(input_vp_qf32));
    output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
    output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
    output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

    /* Multiply result by same normalization factor applied to input earlier */
    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(norm_factor_qf32));
    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(norm_factor_qf32));

    /* Calculate num * 1/den */
    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_n_vp_qf32));
    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_n_vp_qf32));

    return Q6_Vhf_equals_Wqf32(output_dv.VV);
}

/*
 * This function converts a vector of IEEE float elements to a vector of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf32_convert_vsf(HVX_Vector vin) {
    return Q6_Vqf32_vadd_VsfVsf(vin, Q6_V_vzero());
}

/*
 * This function converts a vector of IEEE half float elements to a vector of qf16 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vqf16_convert_vhf(HVX_Vector vin) {
    return Q6_Vqf16_vadd_VhfVhf(vin, Q6_V_vzero());
}

/*
 * This function converts a pair of vectors of qf32 elements to a vector of IEEE half float elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_Vector qhmath_hvx_vhf_convert_vqf32(HVX_VectorPair vin_vp) {
    return Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(vin_vp));
}

/*
 * This function converts a vector of qf16 elements to a pair of vectors of qf32 elements
 * See also: libs\qfe\inc\qhmath_hvx_convert.h
 */
inline HVX_VectorPair qhmath_hvx_vqf32_convert_vqf16(HVX_Vector vxl) {
    HVX_VectorPair vxw_vp, exponent_vp;
    HVX_Vector     mantissa_mask = Q6_Vh_vsplat_R(0xffe0);
    HVX_Vector     exp_mask      = Q6_Vh_vsplat_R(0x1f);
    HVX_Vector     exp_offset    = Q6_Vh_vsplat_R(0x70);
    HVX_Vector     mant32_shift  = Q6_Vh_vsplat_R(0x10);
    HVX_Vector     reql, reqh, vxl_w, vxh_w, mantissa;
    HVX_Vector     el_exponent, eh_exponent;

    el_exponent = Q6_V_vand_VV(exp_mask, vxl);
    // Obtain the mantissa part: bits (5-15)
    mantissa    = Q6_V_vand_VV(mantissa_mask, vxl);
    // Convert qf16 biassed exponent to qf32 biased exponent
    // new exp = exp + ( 127 (qf32 bias) -15(qf16 biass) ) = 112
    el_exponent = Q6_Vh_vadd_VhVh(exp_offset, el_exponent);

    vxw_vp = Q6_Ww_vunpack_Vh(mantissa);
    vxl_w  = Q6_V_lo_W(vxw_vp);
    vxh_w  = Q6_V_hi_W(vxw_vp);

    exponent_vp = Q6_Ww_vunpack_Vh(el_exponent);
    el_exponent = Q6_V_lo_W(exponent_vp);
    eh_exponent = Q6_V_hi_W(exponent_vp);
    // Convert q16 mantiss to q32 mantissa
    reql        = Q6_Vw_vasl_VwVw(vxl_w, mant32_shift);
    reqh        = Q6_Vw_vasl_VwVw(vxh_w, mant32_shift);
    // Add the exponent
    vxl_w       = Q6_Vw_vadd_VwVw(reql, el_exponent);
    vxh_w       = Q6_Vw_vadd_VwVw(reqh, eh_exponent);

    return Q6_W_vcombine_VV(vxh_w, vxl_w);
}

inline HVX_VectorPair hvx_vqf32_convert_vhf(HVX_Vector vxl) {
    return qhmath_hvx_vqf32_convert_vqf16(qhmath_hvx_vqf16_convert_vhf(vxl));
}

inline HVX_Vector_x2 hvx_vsf_convert_vhf(HVX_Vector vxl, HVX_Vector one) {
    HVX_VectorPair res = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vxl), one);

    HVX_Vector_x2 ret;
    ret.val[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(res));
    ret.val[1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(res));
    return ret;
}

/**
 * @brief Calculates exponential (e^x) for vector elements with infinity guard
 *
 * This function computes the exponential value for each element in the input vector.
 * For input values greater than kMaxExp (88.02f), the function returns the provided
 * infinity value instead of attempting to calculate an exponential that would overflow.
 *
 * @param sline The input vector containing values to compute exponential for
 * @param inf The vector containing the infinity representation to use for guarded values
 * @return HVX_Vector containing exponential values, with values > kMaxExp replaced by inf
 *
 * @note Input values greater than 88.02f will return the specified infinity value
 */
inline HVX_Vector qhmath_hvx_exp_vf_guard_inf(HVX_Vector sline, const HVX_Vector inf) {
    constexpr float  kMaxExp = 88.02f;
    const HVX_Vector max_exp = Q6_V_vsplat_R(reinterpret_cast<const uint32_t &>(kMaxExp));

    HVX_VectorPred pred_gt_max_exp = Q6_Q_vcmp_gt_VsfVsf(sline, max_exp);

    HVX_Vector out = qhmath_hvx_exp_vf(sline);

    out = Q6_V_vmux_QVV(pred_gt_max_exp, inf, out);
    return out;
}

/**
 * @brief Vectorized division with guard for infinite denominators on HVX.
 *
 * Performs element-wise division num/denom using qhmath_hvx_div_vf and then
 * masks out lanes where denom equals the provided inf value, forcing those
 * lanes of the result to zero. This is a temporary guard until proper INF
 * handling is implemented in the underlying division routine.
 *
 * @param num    Numerator vector (per-lane).
 * @param denom  Denominator vector (per-lane); lanes equal to inf are zeroed in the output.
 * @param coeffs Coefficients used by qhmath_hvx_div_vf for the reciprocal/division approximation.
 * @param inf    Lane value representing +INF to compare against denom.
 * @return       Vector of num/denom with lanes set to zero where denom == inf.
 *
 * @note NaNs, negative infinity, zero denominators, and subnormals are not explicitly handled.
 * @see qhmath_hvx_div_vf
 */
inline HVX_Vector qhmath_hvx_div_vf_guard_inf(HVX_Vector        num,
                                              HVX_Vector        denom,
                                              HVX_VectorPair_x4 coeffs,
                                              const HVX_Vector  inf) {
    HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(denom, inf);

    // TODO: fix the inf in div
    HVX_Vector out = qhmath_hvx_div_vf(num, denom, coeffs);

    out = Q6_V_vmux_QVV(pred_inf, Q6_V_vzero(), out);
    return out;
}

inline HVX_Vector Q6_Vsf_vadd_VsfVsf_guard_inf(HVX_Vector num0, HVX_Vector num1, const HVX_Vector inf) {
    HVX_VectorPred pred0 = Q6_Q_vcmp_eq_VwVw(num0, inf);

    HVX_Vector out = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(num0, num1));

    out = Q6_V_vmux_QVV(pred0, inf, out);
    return out;
}

}  // namespace hexagon::vec::math
