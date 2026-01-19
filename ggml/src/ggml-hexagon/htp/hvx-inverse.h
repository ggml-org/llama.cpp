#ifndef HVX_INVERSE_H
#define HVX_INVERSE_H

#include <HAP_farf.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"

// ====================================================
// FUNCTION: 1/(x+1)     y(0) = 1,  y(0.5) = 0.6667, y(1) = 0.5
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 14
// Peak Error: 1.1295e-04  Rms Error: 2.8410e-05   Mean Error: 1.1370e-05
//      32769  -32706   31252  -10589
//      32590  -30635   22793   -4493
//      32066  -27505   16481   -2348
//      31205  -24054   11849   -1306

static inline HVX_Vector hvx_vec_recip_xp1_O3_unsigned(HVX_Vector vx) {
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh(vx, 0xFAE6F6D4EE73D6A3ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x2E49406159097A14ull);
    p = Q6_Vh_vmps_VhVhVuhPuh_sat(p, vx, 0x5DF66B7177AB7FC2ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x79E57D427F4E8001ull);
    return p;  // signed result, 14 fractional bits
}

// Find reciprocal of fp16.
// (1) first, convert to fp32, multiplying by 1.0; this is done to
//    handle denormals. Ignoring sign and zero, result should be at
//    least 5.9604645e-08 (32-bit code 0x33800000) and at most 131008 (0x47ffe000)
//    (exponent in range [103,143])
// (2) extract the mantissa into 16-bit unsigned; find reciprocal using a fitted poly
// (3) put this, along with '253-exp' (exp from (1)) together to make an qf32
// (4) convert that to fp16
// (5) put sign back in. Also, if the original value (w/o sign) was <0x81, replace
//     the result with the max value.
static inline HVX_Vector hvx_vec_inverse_f16(HVX_Vector vals) {
    HVX_Vector     em_mask  = Q6_Vh_vsplat_R(0x7FFF);
    HVX_Vector     avals    = Q6_V_vand_VV(vals, em_mask);
    HVX_VectorPred is_neg   = Q6_Q_vcmp_gt_VhVh(avals, vals);
    // is too small to 1/x ? for 'standard' fp16, this would be 0x101
    HVX_VectorPred is_small = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0x101), avals);

    HVX_VectorPair to_qf32  = Q6_Wqf32_vmpy_VhfVhf(avals, Q6_Vh_vsplat_R(0x3C00));  // *1.0
    HVX_Vector     to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(to_qf32));
    HVX_Vector     to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(to_qf32));

    // bits 22..13 contain the mantissa now (w/o hidden bit); move to bit 14..5 of a 16-bit vector
    HVX_Vector mant_u16 = Q6_Vh_vshuffo_VhVh(Q6_Vw_vasl_VwR(to_f32_1, 9), Q6_Vw_vasl_VwR(to_f32_0, 9));
    // likewise extract the upper 16 from each, containing the exponents in range 103..142
    HVX_Vector exp_u16  = Q6_Vh_vshuffo_VhVh(to_f32_1, to_f32_0);
    //Get exponent in IEEE 32-bit representation
    exp_u16             = Q6_Vuh_vlsr_VuhR(exp_u16, 7);

    // so, mant_u16 contains an unbiased mantissa in upper 10 bits of each u16 lane
    // We can consider it to be x-1.0, with 16 fractional bits, where 'x' is in range [1.0,2.0)
    // Use poly to transform to 1/x, with 14 fractional bits
    //
    HVX_Vector rm = hvx_vec_recip_xp1_O3_unsigned(mant_u16);

    HVX_Vector vcl0 = Q6_Vuh_vcl0_Vuh(rm);  //count leading zeros

    // Get mantissa for 16-bit represenation
    HVX_Vector mant_recip = Q6_V_vand_VV(Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhVh(rm, vcl0), 5), Q6_Vh_vsplat_R(0x03FF));

    //Compute Reciprocal Exponent
    HVX_Vector exp_recip =
        Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(254), exp_u16), Q6_Vh_vsub_VhVh(vcl0, Q6_Vh_vsplat_R(1)));
    //Convert it for 16-bit representation
    exp_recip = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsub_VhVh(exp_recip, Q6_Vh_vsplat_R(127)), Q6_Vh_vsplat_R(15));
    exp_recip = Q6_Vh_vasl_VhR(exp_recip, 10);

    //Merge exponent and mantissa for reciprocal
    HVX_Vector recip = Q6_V_vor_VV(exp_recip, mant_recip);
    // map 'small' inputs to standard largest value 0x7bff
    recip            = Q6_V_vmux_QVV(is_small, Q6_Vh_vsplat_R(0x7bff), recip);
    // add sign back
    recip            = Q6_V_vandor_VQR(recip, is_neg, 0x80008000);
    return recip;
}

static inline HVX_Vector hvx_vec_inverse_f32(HVX_Vector v_sf) {
    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_Vector two_sf       = hvx_vec_splat_f32(2.0);

    // First approximation
    HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);

    HVX_Vector r_qf;

    // Refine
    r_qf = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

    return Q6_Vsf_equals_Vqf32(r_qf);
}

static inline HVX_Vector hvx_vec_inverse_f32_guard(HVX_Vector v_sf, HVX_Vector nan_inf_mask) {
    HVX_Vector out = hvx_vec_inverse_f32(v_sf);

    HVX_Vector     masked_out = Q6_V_vand_VV(out, nan_inf_mask);
    const HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(nan_inf_mask, masked_out);

    return Q6_V_vmux_QVV(pred, Q6_V_vzero(), out);
}

#define hvx_inverse_f32_loop_body(dst_type, src_type, vec_store)             \
    do {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);           \
                                                                             \
        const uint32_t nvec = n / VLEN_FP32;                                 \
        const uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        for (; i < nvec; i++) {                                              \
             vdst[i] = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_mask);     \
        }                                                                    \
        if (nloe) {                                                          \
            HVX_Vector v = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_mask); \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, v);             \
        }                                                                    \
    } while(0)

static inline void hvx_inverse_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_inverse_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_inverse_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_inverse_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_inverse_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_inverse_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_inverse_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_inverse_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_inverse_f32(uint8_t * restrict dst, uint8_t * restrict src, const int num_elems) {
    if ((unsigned long) dst % 128 == 0) {
        if ((unsigned long) src % 128 == 0) {
            hvx_inverse_f32_aa(dst, src, num_elems);
        } else {
            hvx_inverse_f32_au(dst, src, num_elems);
        }
    } else {
        if ((unsigned long) src % 128 == 0) {
            hvx_inverse_f32_ua(dst, src, num_elems);
        } else {
            hvx_inverse_f32_uu(dst, src, num_elems);
        }
    }
}

#endif // HVX_INVERSE_H
