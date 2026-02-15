#ifndef HVX_FLOOR_H
#define HVX_FLOOR_H

#include <stdbool.h>
#include <stdint.h>

#include "hvx-base.h"

#define IEEE_VSF_EXPLEN   (8)
#define IEEE_VSF_EXPBIAS  (127)
#define IEEE_VSF_EXPMASK  (0xFF)
#define IEEE_VSF_MANTLEN  (23)
#define IEEE_VSF_MANTMASK (0x7FFFFF)
#define IEEE_VSF_MIMPMASK (0x800000)

static inline HVX_Vector hvx_vec_truncate_f32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_Vector mant_v = in_vec & mask_mant_v;                  // obtain mantissa
    HVX_Vector vout   = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_Vector neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    return (vout);
}

static inline HVX_Vector hvx_vec_floor_f32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, in_vec, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, in_vec);

    // if expval < 0 (q_negexp)         // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval                // fraction bits to mask off
    //    vout = ~(mask)                // apply mask to remove fraction
    //    if (qneg)                     // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                             // already an integer
    //    ;                             // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v    = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(in_vec, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, in_vec);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(in_vec, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = in_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, in_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    return vout;
}

#endif /* HVX_FLOOR_H */
