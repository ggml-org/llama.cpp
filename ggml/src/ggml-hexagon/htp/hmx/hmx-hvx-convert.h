// HVX data-type conversion helpers used by HMX operations.
// Ported from htp-ops-lib/include/dsp/hvx_convert.h.
//
// All public symbols prefixed with hmx_ to avoid collisions with the
// existing hexagon hvx-*.h headers.

#ifndef HMX_HVX_CONVERT_H
#define HMX_HVX_CONVERT_H

#include "hmx-hvx-internal.h"

// Convert two single-float vectors to one half-float vector.
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_wsf_to_vhf(HVX_Vector v1, HVX_Vector v0) {
    const HVX_Vector v_zero = Q6_V_vzero();
    HVX_Vector v0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0, v_zero);
    HVX_Vector v1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1, v_zero);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v1_qf32, v0_qf32));
}

// Convert one qf16 vector to two qf32 vectors (vector pair).
static HMX_HVX_INLINE_ALWAYS HVX_VectorPair hmx_hvx_vqf16_to_wqf32(HVX_Vector v_src) {
    const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
    const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
    const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

    // Extract packed exp & mantissa
    HVX_Vector exp_comp = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0x1f));
    HVX_Vector mantissa = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0xffe0));

    // Convert qf16 biased exponent to qf32 biased exponent: +112
    exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

    // Unpack even/odd
    HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);
    HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);

    HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
    HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);

    HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
    HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);

    return Q6_W_vcombine_VV(v1_qf32, v0_qf32);
}

// Convert one qf16 vector to two single-float vectors.
static HMX_HVX_INLINE_ALWAYS HVX_VectorPair hmx_hvx_vqf16_to_wsf(HVX_Vector v_src) {
    HVX_VectorPair vp = hmx_hvx_vqf16_to_wqf32(v_src);
    HVX_Vector v0_sf = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
    HVX_Vector v1_sf = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
    return Q6_W_vcombine_VV(v1_sf, v0_sf);
}

// Convert half-float to qf16.
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_vhf_to_vqf16(HVX_Vector vx) {
    return Q6_Vqf16_vadd_VhfVhf(vx, Q6_V_vzero());
}

// Convert half-float to two single-floats.
static HMX_HVX_INLINE_ALWAYS HVX_VectorPair hmx_hvx_vhf_to_wsf(HVX_Vector vx) {
    return hmx_hvx_vqf16_to_wsf(hmx_hvx_vhf_to_vqf16(vx));
}

// Convert half-float to two qf32.
static HMX_HVX_INLINE_ALWAYS HVX_VectorPair hmx_hvx_vhf_to_wqf32(HVX_Vector vx) {
    return hmx_hvx_vqf16_to_wqf32(hmx_hvx_vhf_to_vqf16(vx));
}

#endif // HMX_HVX_CONVERT_H
