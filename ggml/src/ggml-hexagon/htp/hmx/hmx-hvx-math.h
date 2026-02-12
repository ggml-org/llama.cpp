// HVX math helpers used by HMX operations (exp2, log2, inv).
// Ported from htp-ops-lib/include/dsp/hvx_math.h.
//
// All public symbols prefixed with hmx_ to avoid collisions with the
// existing hexagon hvx-exp.h / hvx-inverse.h headers which serve
// different (HVX-only) code paths.

#ifndef HMX_HVX_MATH_H
#define HMX_HVX_MATH_H

#include <stdint.h>
#include "hmx-hvx-internal.h"

// ---------------------------------------------------------------------------
// exp2 (half-float) — optimised for safe-softmax (x <= 0, no overflow)
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_exp2_vhf(HVX_Vector x_v) {
    const uint16_t e5_qf16 = 0x5082;
    const uint16_t e4_hf   = 0x157d;
    const uint16_t e3_hf   = 0x20ed;
    const uint16_t e2_hf   = 0x2b1b;
    const uint16_t e1_hf   = 0x33b0;
    const uint16_t e0_hf   = 0x398c;

    const HVX_Vector zero_v    = Q6_V_vzero();
    const HVX_Vector half_hf_v = Q6_Vh_vsplat_R(0x3800);

    HVX_Vector f_v, k_v, y_v, x_qf16_v;

#if __HVX_ARCH__ >= 73
    HVX_Vector x_minus_half_v = Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v);
    x_minus_half_v            = Q6_Vhf_equals_Vqf16(x_minus_half_v);
    k_v = Q6_Vh_equals_Vhf(x_minus_half_v);
    f_v = Q6_Vhf_equals_Vh(k_v);
#else
    HVX_Vector x_plus_half_v = Q6_Vqf16_vadd_VhfVhf(x_v, half_hf_v);
    x_plus_half_v            = Q6_Vhf_equals_Vqf16(x_plus_half_v);
    k_v = hmx_Vh_vfloor_VhfVhf(x_plus_half_v, &f_v);
#endif

    x_qf16_v = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);

    HVX_Vector e5_qf16_v = Q6_Vh_vsplat_R(e5_qf16);
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(e5_qf16_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e4_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e3_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e2_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e1_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e0_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(0x3c00)); // 1.0

    y_v = Q6_Vhf_equals_Vqf16(y_v);

    HVX_Vector y_v_exponent = Q6_Vh_vasl_VhR(y_v, 1);
    y_v_exponent            = Q6_Vuh_vlsr_VuhR(y_v_exponent, 11);
    y_v_exponent            = Q6_Vh_vadd_VhVh(k_v, y_v_exponent);

    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VhVh(zero_v, y_v_exponent);

    y_v = Q6_Vh_vaslacc_VhVhR(y_v, k_v, 10);
    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
    return y_v;
}

// ---------------------------------------------------------------------------
// log2 (half-float → qf16)
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_log2_vqf16_vhf(HVX_Vector x_v) {
    const uint16_t sqrt_half_hf = 0x39a8;
    const uint16_t log2e_m1_hf  = 0x3715;

    const uint16_t e9_hf = 0x2C81;
    const uint16_t e8_hf = 0xAF5F;
    const uint16_t e7_hf = 0x2F79;
    const uint16_t e6_hf = 0xAFF3;
    const uint16_t e5_hf = 0x308F;
    const uint16_t e4_hf = 0xB155;
    const uint16_t e3_hf = 0x3266;
    const uint16_t e2_hf = 0xB400;
    const uint16_t e1_hf = 0x3555;
    const uint16_t e0_hf = 0xB800;

    const HVX_Vector zero_v = Q6_V_vzero();

    HVX_Vector e_v, y_v, x_qf16_v, z_qf16_v, tmp_v;

    // frexp: extract exponent
    e_v = Q6_Vuh_vlsr_VuhR(x_v, 10);
    e_v = Q6_V_vand_VV(e_v, Q6_Vh_vsplat_R(0x1f));
    e_v = Q6_Vh_vsub_VhVh(e_v, Q6_Vh_vsplat_R(14));

    // fractional part
    x_v = Q6_V_vand_VV(x_v, Q6_Vh_vsplat_R(0x83ff));
    x_v = Q6_V_vor_VV(x_v, Q6_Vh_vsplat_R(0x3800));

    HVX_VectorPred q = Q6_Q_vcmp_gt_VhfVhf(Q6_Vh_vsplat_R(sqrt_half_hf), x_v);

    HVX_Vector tmp_e_v = Q6_Vh_vsub_VhVh(e_v, Q6_Vh_vsplat_R(1));
    e_v = Q6_V_vmux_QVV(q, tmp_e_v, e_v);

    HVX_Vector tmp_x_v = Q6_Vqf16_vmpy_VhfVhf(x_v, Q6_Vh_vsplat_R(0x4000)); // 2.0
    x_qf16_v = Q6_Vqf16_vadd_VhfVhf(x_v, zero_v);
    x_qf16_v = Q6_V_vmux_QVV(q, tmp_x_v, x_qf16_v);

    // log(1+x) polynomial
    x_qf16_v = Q6_Vqf16_vsub_Vqf16Vhf(x_qf16_v, Q6_Vh_vsplat_R(0x3c00)); // 1.0
    z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, x_qf16_v);
    x_v      = Q6_Vhf_equals_Vqf16(x_qf16_v);

    y_v = Q6_Vqf16_vmpy_VhfVhf(Q6_Vh_vsplat_R(e9_hf), x_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e8_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e7_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e6_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e5_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e4_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e3_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e2_hf));
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vadd_Vqf16Vhf(y_v, Q6_Vh_vsplat_R(e1_hf));

    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, x_qf16_v);
    y_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, z_qf16_v);

    tmp_v = Q6_Vqf16_vadd_VhfVhf(Q6_Vh_vsplat_R(e0_hf), zero_v);
    tmp_v = Q6_Vqf16_vmpy_Vqf16Vqf16(tmp_v, z_qf16_v);
    y_v   = Q6_Vqf16_vadd_Vqf16Vqf16(y_v, tmp_v);

    // z = (x + y) * log2(e) + e
    HVX_Vector log2e_m1_qf16_v = Q6_Vqf16_vadd_VhfVhf(Q6_Vh_vsplat_R(log2e_m1_hf), zero_v);

    z_qf16_v = Q6_Vqf16_vmpy_Vqf16Vqf16(y_v, log2e_m1_qf16_v);
    tmp_v    = Q6_Vqf16_vmpy_Vqf16Vqf16(x_qf16_v, log2e_m1_qf16_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, tmp_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, y_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, x_qf16_v);

    HVX_Vector qf16e_v = hmx_vqf16_from_int(e_v);
    z_qf16_v = Q6_Vqf16_vadd_Vqf16Vqf16(z_qf16_v, qf16e_v);
    return z_qf16_v;
}

// ---------------------------------------------------------------------------
// inv (half-float) — polynomial approximation via VLUT16
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_inv_vhf(HVX_Vector x_v) {
    static const float c0_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.8807721943716516, 3.6618209528616856, 3.4657742282097708, 3.2853461610022414,
        3.1229570908314015, 2.976379865829892,  2.8438614274889833, 2.723793061029549,
        2.613859154046634,  2.5119508509784287, 2.4167270706641473, 2.3286721812015188,
        2.2462659531748064, 2.1692490555028736, 2.0981551828382417, 2.0319234960945,
    };
    static const float c1_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -5.646783581176797, -5.027704168781284, -4.5037889029173535, -4.0470997487793445,
        -3.6569569537789364, -3.3217563552211695, -3.03258650196419,  -2.781935505534812,
        -2.5619261358961922, -2.3660577978107398, -2.190083163030879,  -2.033405493468989,
        -1.8920413948588666, -1.7645298754188785, -1.6507730169513504, -1.5482028127706613,
    };
    static const float c2_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.6511964773849632, 3.0676375988553106, 2.6008750952258324, 2.215514199159397,
        1.9030391013295935, 1.6474963735373633, 1.4371447652517673, 1.2627141904289978,
        1.11593649827749,   0.9904415490260164, 0.882033772823834,  0.7891019704346331,
        0.7082630629776306, 0.6378888508693012, 0.5772121720355701, 0.524261196551401,
    };
    static const float c3_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.8851851956149304, -0.7018008948429424, -0.5631686602024177, -0.4547647803673564,
        -0.37133287830029976, -0.3063883382130307, -0.255378412302572,  -0.2149126167280633,
        -0.18226975346347984, -0.15546600267845986, -0.13320337246909697, -0.11482846255803722,
        -0.0994184164975366,  -0.08647114157420362, -0.07568254923048714, -0.06657033258736733,
    };

    hmx_HVX_DV c0_coeff_dv, c1_coeff_dv, c2_coeff_dv, c3_coeff_dv, output_dv;
    HVX_VectorPair c0_coeff_vp, c1_coeff_vp, c2_coeff_vp, c3_coeff_vp;

    HVX_Vector scale_v      = Q6_Vh_vsplat_R(0x4bfb);
    HVX_Vector one_v_hf     = Q6_Vh_vsplat_R(0x3c00);
    HVX_Vector zero_v_hf    = Q6_V_vzero();
    HVX_Vector exp           = Q6_Vh_vsplat_R(0x7800);
    HVX_Vector signexp_mask  = Q6_Vh_vsplat_R(0xFC00);
    HVX_Vector mask_idx1_v   = Q6_Vh_vsplat_R(0x000F);
    HVX_Vector mask_idx2_v   = Q6_V_vsplat_R(0x00001010);
    HVX_Vector const16_0_v_hf = Q6_Vh_vsplat_R(0x4c00);
    HVX_Vector input_min_v_hf = Q6_Vh_vsplat_R(0x3c00);

    scale_v = Q6_Vqf16_vadd_VhfVhf(scale_v, zero_v_hf);

    HVX_Vector c0_coeff_v = *((HVX_Vector *)(c0_coeffs));
    HVX_Vector c1_coeff_v = *((HVX_Vector *)(c1_coeffs));
    HVX_Vector c2_coeff_v = *((HVX_Vector *)(c2_coeffs));
    HVX_Vector c3_coeff_v = *((HVX_Vector *)(c3_coeffs));

    c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_hf);
    c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_hf);
    c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_hf);
    c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_hf);

    c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

    HVX_Vector norm_factor = Q6_V_vand_VV(x_v, signexp_mask);
    norm_factor = Q6_Vh_vsub_VhVh(exp, norm_factor);

    x_v = Q6_Vqf16_vmpy_VhfVhf(x_v, norm_factor);

    HVX_VectorPair norm_factor_qf32 = Q6_Wqf32_vmpy_VhfVhf(norm_factor, one_v_hf);

    HVX_Vector tmp_v              = Q6_Vh_vdeal_Vh(x_v);
    HVX_Vector input_shifted_v_hf = Q6_Vqf16_vsub_Vqf16Vhf(tmp_v, input_min_v_hf);

    HVX_Vector input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);
    input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);
    tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

    HVX_Vector idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);
    idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
    idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
    idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);

    HVX_Vector idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

    c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
    c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
    c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
    c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
    c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
    c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
    c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
    c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);

    HVX_VectorPair input_vp_qf32 = Q6_Wqf32_vmpy_Vqf16Vhf(x_v, one_v_hf);

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

    output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(norm_factor_qf32));
    output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(norm_factor_qf32));

    return Q6_Vhf_equals_Wqf32(output_dv.VV);
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_exp2_vhf_vqf16(HVX_Vector x) {
    return hmx_hvx_exp2_vhf(Q6_Vhf_equals_Vqf16(x));
}

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_log2_vqf16(HVX_Vector x) {
    return hmx_hvx_log2_vqf16_vhf(Q6_Vhf_equals_Vqf16(x));
}

// ---------------------------------------------------------------------------
// exp2 (single-float)
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_exp2_vsf(HVX_Vector x_v) {
    const uint32_t e5_sf = 0x3920FDDE;
    const uint32_t e4_sf = 0x3AAF9F29;
    const uint32_t e3_sf = 0x3C1D96A6;
    const uint32_t e2_sf = 0x3D635774;
    const uint32_t e1_sf = 0x3E75FDEE;
    const uint32_t e0_sf = 0x3F317218;

    const HVX_Vector zero_v    = Q6_V_vzero();
    const HVX_Vector half_sf_v = Q6_V_vsplat_R(0x3F000000);
    const HVX_Vector one_sf_v  = Q6_V_vsplat_R(0x3F800000);

    HVX_Vector f_v, k_v, y_v, x_qf32_v;

#if __HVX_ARCH__ >= 73
    HVX_Vector x_minus_half_v = Q6_Vqf32_vsub_VsfVsf(x_v, half_sf_v);
    x_minus_half_v            = Q6_Vsf_equals_Vqf32(x_minus_half_v);
    k_v = Q6_Vw_equals_Vsf(x_minus_half_v);
    f_v = Q6_Vsf_equals_Vw(k_v);
#else
    HVX_Vector x_plus_half_v = Q6_Vqf32_vadd_VsfVsf(x_v, half_sf_v);
    x_plus_half_v            = Q6_Vsf_equals_Vqf32(x_plus_half_v);
    k_v = hmx_Vw_vfloor_VsfVsf(x_plus_half_v, &f_v);
#endif

    x_qf32_v = Q6_Vqf32_vsub_VsfVsf(x_v, f_v);
    x_v      = Q6_Vsf_equals_Vqf32(x_qf32_v);

    y_v = Q6_Vqf32_vmpy_VsfVsf(Q6_V_vsplat_R(e5_sf), x_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(e4_sf));
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(e3_sf));
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(e2_sf));
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(e1_sf));
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(e0_sf));
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, one_sf_v);

    y_v = Q6_Vsf_equals_Vqf32(y_v);

    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);
    y_v_exponent            = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);
    y_v_exponent            = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);
    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);
    return y_v;
}

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_exp2_vsf_vqf32(HVX_Vector x) {
    return hmx_hvx_exp2_vsf(Q6_Vsf_equals_Vqf32(x));
}

static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_exp2_vqf32(HVX_Vector x) {
    return Q6_Vqf32_vadd_VsfVsf(hmx_hvx_exp2_vsf_vqf32(x), Q6_V_vzero());
}

// ---------------------------------------------------------------------------
// inv (single-float → qf32)
// ---------------------------------------------------------------------------
static HMX_HVX_INLINE_ALWAYS HVX_Vector hmx_hvx_inv_vqf32_vsf(HVX_Vector x_v) {
    static const float c0_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.882601794814435,  3.6625422144222575, 3.464451548227971,  3.2869700047974098,
        3.126105117815294,  2.9797652947122333, 2.846287833147896,  2.7247270166228237,
        2.614282526778659,  2.5119448279766914, 2.4168240690138916, 2.3287715099556494,
        2.2470044371606255, 2.1705097010458525, 2.0993232550771013, 2.032425103348979,
    };
    static const float c1_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -5.65213466274883,  -5.029649818173625, -4.500359068222728,  -4.051125252469975,
        -3.6643282495304743, -3.3293252513210945, -3.0377500909629918, -2.78384542029156,
        -2.562751394984757,  -2.3660481944625364, -2.1902579830702398, -2.033579850063907,
        -1.8932880190031018, -1.7665817851802996, -1.6526109646324616, -1.5489652830974667,
    };
    static const float c2_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.6564123863772062, 3.0693863078484034, 2.5979108429264546, 2.2188401136904137,
        1.90879196515026,   1.6531365145318937, 1.4408072849395228, 1.2640160009581791,
        1.1164726565567085, 0.9904366133906549, 0.8821387892416702, 0.7892039810345458,
        0.7089644931002874, 0.6390020714403465, 0.5781761255999769, 0.5246475096790261,
    };
    static const float c3_coeffs[32] __attribute__((aligned(128))) = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.8868796162009371,  -0.7023245532864408, -0.5623148115716742,  -0.45568061400557225,
        -0.3728293181808119,  -0.30778916969628956, -0.25624427383670373, -0.21520836864975557,
        -0.18238585316003267, -0.1554651987039696,  -0.133224398745864,   -0.11484835534787588,
        -0.09954996553138899, -0.08667244996867919, -0.07585106425203664, -0.06663557250850614,
    };

    hmx_HVX_DV c0_coeff_dv, c1_coeff_dv, c2_coeff_dv, c3_coeff_dv;
    HVX_VectorPair c0_coeff_vp, c1_coeff_vp, c2_coeff_vp, c3_coeff_vp;

    HVX_Vector scale_v          = Q6_V_vsplat_R(0x417ffffe);
    const HVX_Vector zero_v_sf  = Q6_V_vzero();
    const HVX_Vector exp        = Q6_V_vsplat_R(0x7F000000);
    const HVX_Vector signexp_mask = Q6_V_vsplat_R(0xFF800000);
    const HVX_Vector mask_idx1_v  = Q6_V_vsplat_R(0x0000000F);
    const HVX_Vector mask_idx2_v  = Q6_V_vsplat_R(0x00000010);
    const HVX_Vector const16_0_v  = Q6_V_vsplat_R(0x41800000);
    const HVX_Vector input_min_v  = Q6_V_vsplat_R(0x3f800000);

    scale_v = Q6_Vqf32_vadd_VsfVsf(scale_v, zero_v_sf);

    HVX_Vector c0_coeff_v = *((HVX_Vector *)(c0_coeffs));
    HVX_Vector c1_coeff_v = *((HVX_Vector *)(c1_coeffs));
    HVX_Vector c2_coeff_v = *((HVX_Vector *)(c2_coeffs));
    HVX_Vector c3_coeff_v = *((HVX_Vector *)(c3_coeffs));

    c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_sf);
    c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_sf);
    c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_sf);
    c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_sf);

    c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);

    HVX_Vector norm_factor = Q6_V_vand_VV(x_v, signexp_mask);
    norm_factor = Q6_Vw_vsub_VwVw(exp, norm_factor);

    x_v = Q6_Vqf32_vmpy_VsfVsf(x_v, norm_factor);

    norm_factor = Q6_Vqf32_vadd_VsfVsf(norm_factor, zero_v_sf);

    HVX_Vector input_shifted_v = Q6_Vqf32_vsub_Vqf32Vsf(x_v, input_min_v);
    HVX_Vector input_scaled_v  = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v, scale_v);
    input_scaled_v = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v, const16_0_v);

    HVX_Vector tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v);

    HVX_Vector idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);
    idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
    idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);

    HVX_Vector idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

    c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
    c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
    c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
    c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
    c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
    c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
    c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
    c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);

    HVX_Vector output_v;
    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c3_coeff_vp), x_v);
    output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, x_v);
    output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, x_v);
    output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

    output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, norm_factor);

    return output_v;
}

#endif // HMX_HVX_MATH_H
