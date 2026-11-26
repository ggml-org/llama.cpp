#ifndef HMX_INT_UTILS_H
#define HMX_INT_UTILS_H

#include <stddef.h>
#include <stdint.h>

#include "hvx-utils.h"

#ifndef HMX_W8A8_ACT_TILE_SIZE
#define HMX_W8A8_ACT_TILE_SIZE 2048
#endif

static inline HVX_Vector hvx_vec_f32_to_qf32(HVX_Vector v) {
#if __HVX_ARCH__ >= 81
    return Q6_Vqf32_equals_Vsf(v);
#else
    return Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero());
#endif
}

static inline HVX_Vector hvx_qf32_mul_sf_scalar(HVX_Vector v_sf, uint32_t scalar_sf_bits) {
#if __HVX_ARCH__ >= 79
    return Q6_Vqf32_vmpy_VsfRsf(v_sf, (int32_t) scalar_sf_bits);
#else
    return Q6_Vqf32_vmpy_VsfVsf(v_sf, Q6_V_vsplat_R((int32_t) scalar_sf_bits));
#endif
}

static inline void emit_int_hmx_scales_and_bias(
        HVX_Vector v_max_sf,
        float scale_const,
        HVX_Vector v_sum_w,
        HVX_Vector v_l1_w,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out) {

    // |sum_k (a - zp) * w_hat| < 128 * l1_w = (l1_w << 7) <= 32768 * 2^p
    HVX_Vector v_max_mag = Q6_Vw_vasl_VwR(v_l1_w, 7);
    HVX_Vector v_p       = Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(17), Q6_Vuw_vcl0_Vuw(v_max_mag)));
    HVX_Vector v_ib      = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(32768), v_p), Q6_Vw_vasl_VwR(v_sum_w, 7));

    HVX_Vector v_dw_base       = hvx_vec_mul_f32_f32(v_max_sf, hvx_vec_splat_f32(scale_const));
    HVX_Vector v_pow2_sf       = Q6_Vsf_equals_Vw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), v_p));
    *(HVX_Vector *) dst_scales = hvx_vec_mul_f32_f32(v_dw_base, v_pow2_sf);

    HVX_Vector v_exp_hi = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
    HVX_Vector v_exp_lo = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(24), v_p);
    bias_out[0] = Q6_Vw_vasl_VwR(v_exp_hi, 10);
    bias_out[1] = v_ib;
    bias_out[2] = Q6_Vw_vasl_VwR(v_exp_lo, 10);
    bias_out[3] = v_ib;
}

static inline void emit_int_hmx_scales_and_bias_zp0(
        HVX_Vector v_max_sf,
        float scale_const,
        HVX_Vector v_l1_w,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out) {

    // Unsigned a_u8 in [0, 255] (zp = 0): |sum_k a_u8 * w_hat| <= 255 * l1_w < (l1_w << 8) <= 32768 * 2^p
    HVX_Vector v_max_mag = Q6_Vw_vasl_VwR(v_l1_w, 8);
    HVX_Vector v_p       = Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(17), Q6_Vuw_vcl0_Vuw(v_max_mag)));
    HVX_Vector v_ib      = Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(32768), v_p);

    HVX_Vector v_dw_base       = hvx_vec_mul_f32_f32(v_max_sf, hvx_vec_splat_f32(scale_const));
    HVX_Vector v_pow2_sf       = Q6_Vsf_equals_Vw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), v_p));
    *(HVX_Vector *) dst_scales = hvx_vec_mul_f32_f32(v_dw_base, v_pow2_sf);

    HVX_Vector v_exp_hi = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
    HVX_Vector v_exp_lo = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(24), v_p);
    bias_out[0] = Q6_Vw_vasl_VwR(v_exp_hi, 10);
    bias_out[1] = v_ib;
    bias_out[2] = Q6_Vw_vasl_VwR(v_exp_lo, 10);
    bias_out[3] = v_ib;
}

static inline HVX_Vector int_hmx_scale_qf_from_max_f32(HVX_Vector v_max_sf, float num) {
    HVX_VectorPred q_pos  = Q6_Q_vcmp_gt_VsfVsf(v_max_sf, hvx_vec_splat_f32(1e-30f));
    HVX_Vector v_inv_raw  = hvx_vec_inverse_f32(v_max_sf);
    HVX_Vector v_scale_sf = Q6_V_vmux_QVV(q_pos, hvx_vec_mul_f32_f32(v_inv_raw, hvx_vec_splat_f32(num)), Q6_V_vzero());
    return hvx_vec_f32_to_qf32(v_scale_sf);
}

static inline HVX_Vector int_hmx_inv127_from_max_f32(HVX_Vector v_max_sf) {
    return int_hmx_scale_qf_from_max_f32(v_max_sf, 127.0f);
}

static inline void int_hmx_inv127_from_max_f16(
        HVX_Vector v_max_sc,
        HVX_Vector * restrict v_max_sf_out,
        HVX_Vector * restrict v_inv127_qf_out) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_Vector v_max_sf        = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_max_sc), v_one_f16)));
    *v_max_sf_out              = v_max_sf;
    *v_inv127_qf_out           = int_hmx_inv127_from_max_f32(v_max_sf);
}

static inline HVX_Vector int_hmx_m_b4_from_sc_qf(HVX_Vector v_sc_qf, HVX_Vector v_inv127_qf) {
    const HVX_Vector v12M_sf = Q6_V_vsplat_R(0x4B400000);
    HVX_Vector v_m_biased    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(v_sc_qf, v_inv127_qf), v12M_sf));
    HVX_Vector v_m_h         = Q6_Vh_vpacke_VwVw(v_m_biased, v_m_biased);
    return Q6_Vh_vshuff_Vh(Q6_Vb_vshuff_Vb(Q6_Vb_vpacke_VhVh(v_m_h, v_m_h)));
}

static inline HVX_Vector int_hmx_m_b4_from_f16(HVX_Vector v_sc, HVX_Vector v_inv127_qf) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_Vector v_sc_qf         = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sc), v_one_f16));
    return int_hmx_m_b4_from_sc_qf(v_sc_qf, v_inv127_qf);
}

static inline void int_hmx_m_b4_pair_from_f16(
        HVX_Vector v_sc,
        HVX_Vector v_inv127_qf,
        HVX_Vector * restrict v_m_b0_out,
        HVX_Vector * restrict v_m_b1_out) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair vp_sc_qf    = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sc), v_one_f16);
    *v_m_b0_out                = int_hmx_m_b4_from_sc_qf(Q6_V_lo_W(vp_sc_qf), v_inv127_qf);
    *v_m_b1_out                = int_hmx_m_b4_from_sc_qf(Q6_V_hi_W(vp_sc_qf), v_inv127_qf);
}

static inline void int_hmx_store_and_accum_pair(
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector v_W0,
        HVX_Vector v_W1,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector v_ones_s8 = Q6_Vb_vsplat_R(1);
    tile_dst[2 * pair_idx + 0] = v_W0;
    tile_dst[2 * pair_idx + 1] = v_W1;

    *v_sum_w = Q6_Vw_vrmpyacc_VwVbVb(Q6_Vw_vrmpyacc_VwVbVb(*v_sum_w, v_W0, v_ones_s8), v_W1, v_ones_s8);
    *v_l1_w  = Q6_Vw_vrmpyacc_VwVubVb(Q6_Vw_vrmpyacc_VwVubVb(*v_l1_w, Q6_Vb_vabs_Vb(v_W0), v_ones_s8), Q6_Vb_vabs_Vb(v_W1), v_ones_s8);
}

static inline void int_hmx_rescale_pair_wb(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    HVX_Vector     v_neg1_h   = Q6_Vh_vsplat_R(-1);
    HVX_VectorPair vp_tie_adj = Q6_W_vcombine_VV(v_neg1_h, v_neg1_h);
    HVX_VectorPair vp_p01     = Q6_Wh_vmpy_VbVb(v_q0, v_m_b4);
    HVX_VectorPair vp_p23     = Q6_Wh_vmpyacc_WhVbVb(vp_tie_adj, v_q1, v_m_b4);

#if __HVX_ARCH__ >= 79
    HVX_Vector v_W0 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p01), Q6_V_lo_W(vp_p01), 7);
    HVX_Vector v_W1 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p23), Q6_V_lo_W(vp_p23), 7);
#else
    HVX_Vector v_W0 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p01), 1),
                                            Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p01), 1));
    HVX_Vector v_W1 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p23), 1),
                                            Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p23), 1));
#endif
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, v_W0, v_W1, v_sum_w, v_l1_w);
}

#if __HVX_ARCH__ >= 81
static inline void pack_hf_quad_to_u8_sm_pair(
        uint8_t * restrict tile_base,
        uint32_t kt,
        uint32_t rg,
        HVX_Vector hf0,
        HVX_Vector hf1,
        HVX_Vector hf2,
        HVX_Vector hf3) {
    const HVX_Vector v_zp_u8   = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_min_u8b = Q6_Vb_vsplat_R(1);

    HVX_Vector vb02 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf2), Q6_Vh_equals_Vhf_rnd(hf0)));
    HVX_Vector vb13 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf3), Q6_Vh_equals_Vhf_rnd(hf1)));
    HVX_VectorPair w_u8 = Q6_W_vshuff_VVR(vb13, vb02, -1);

    ((HVX_Vector *) (tile_base + (size_t) (kt + 0) * HMX_W8A8_ACT_TILE_SIZE))[rg] =
        Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_lo_W(w_u8), v_zp_u8), v_min_u8b);
    ((HVX_Vector *) (tile_base + (size_t) (kt + 1) * HMX_W8A8_ACT_TILE_SIZE))[rg] =
        Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_hi_W(w_u8), v_zp_u8), v_min_u8b);
}

static inline void hmx_pack_hf_quad_to_u8_cm_pair(
        uint8_t * restrict tile_base,
        uint32_t kt,
        uint32_t rg,
        HVX_Vector hf0,
        HVX_Vector hf1,
        HVX_Vector hf2,
        HVX_Vector hf3) {
    const HVX_Vector v_zp_u8   = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_min_u8b = Q6_Vb_vsplat_R(1);

    HVX_Vector     vb01 = Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf1), Q6_Vh_equals_Vhf_rnd(hf0));
    HVX_Vector     vb23 = Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf3), Q6_Vh_equals_Vhf_rnd(hf2));
    HVX_VectorPair w_u8 = Q6_W_vdeal_VVR(vb23, vb01, -32);

    ((HVX_Vector *) (tile_base + (size_t) (kt + 0) * HMX_W8A8_ACT_TILE_SIZE))[rg] =
        Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_lo_W(w_u8), v_zp_u8), v_min_u8b);
    ((HVX_Vector *) (tile_base + (size_t) (kt + 1) * HMX_W8A8_ACT_TILE_SIZE))[rg] =
        Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_hi_W(w_u8), v_zp_u8), v_min_u8b);
}
#endif

#endif /* HMX_INT_UTILS_H */
