#ifndef HMX_FA_W8A8_KERNELS_H
#define HMX_FA_W8A8_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hex-fastdiv.h"
#include "hmx-int-utils.h"
#include "hmx-utils.h"
#include "hvx-utils.h"

// Quantize one 4-row quad of FP32 Q into W8A8 U8 SM 64x32 tiles and emit d_q (with scale folded).
static __attribute__((noinline)) void hmx_fa_w8a8_q_prep_quad_f32(
        uint8_t * restrict tile_row_base,
        float * restrict d_q_out,
        const float * restrict p0,
        const float * restrict p1,
        const float * restrict p2,
        const float * restrict p3,
        uint32_t rg,
        uint32_t n_dot,
        uint32_t valid_rows,
        float scale_f32) {

    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        const uint32_t off = kt * 32;
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + off);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + off);
        HVX_Vector vx2 = *(const HVX_UVector *) (p2 + off);
        HVX_Vector vx3 = *(const HVX_UVector *) (p3 + off);
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
        vmax2 = Q6_Vsf_vmax_VsfVsf(vmax2, hvx_vec_abs_f32(vx2));
        vmax3 = Q6_Vsf_vmax_VsfVsf(vmax3, hvx_vec_abs_f32(vx3));
    }

    vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(vmax0) : Q6_V_vzero();
    vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(vmax1) : Q6_V_vzero();
    vmax2 = (valid_rows > 2) ? hvx_vec_reduce_max_f32(vmax2) : Q6_V_vzero();
    vmax3 = (valid_rows > 3) ? hvx_vec_reduce_max_f32(vmax3) : Q6_V_vzero();

    const HVX_Vector v_sc_inv127 = hvx_vec_splat_f32(scale_f32 * (1.0f / 127.0f));
    const HVX_Vector v_127       = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_sc_inv127);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_sc_inv127);
    HVX_Vector vd2   = hvx_vec_mul_f32_f32(vmax2, v_sc_inv127);
    HVX_Vector vd3   = hvx_vec_mul_f32_f32(vmax3, v_sc_inv127);
    HVX_Vector v_d02 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_q_out, 4 * sizeof(float), v_d);

    HVX_Vector v_inv0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero());
    HVX_Vector v_inv1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero());
    HVX_Vector v_inv2 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero());
    HVX_Vector v_inv3 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero());

#if __HVX_ARCH__ >= 81
    const uint32_t n_pairs = n_dot >> 1;
    for (uint32_t step = 0; step < n_pairs; ++step) {
        const uint32_t off = step * 64;
        HVX_Vector hf0 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p0 + off + 32), v_inv0),
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p0 + off + 0),  v_inv0))));
        HVX_Vector hf1 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p1 + off + 32), v_inv1),
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p1 + off + 0),  v_inv1))));
        HVX_Vector hf2 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p2 + off + 32), v_inv2),
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p2 + off + 0),  v_inv2))));
        HVX_Vector hf3 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p3 + off + 32), v_inv3),
            Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p3 + off + 0),  v_inv3))));

        pack_hf_quad_to_u8_sm_pair(tile_row_base, step * 2, rg, hf0, hf1, hf2, hf3);
    }
    if (n_dot & 1u) {
        const uint32_t kt  = n_dot - 1u;
        const uint32_t off = kt * 32;
        const HVX_Vector v_zp_u8   = Q6_Vb_vsplat_R(0x80);
        const HVX_Vector v_min_u8b = Q6_Vb_vsplat_R(1);
        HVX_Vector hf0 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p0 + off), v_inv0))));
        HVX_Vector hf1 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p1 + off), v_inv1))));
        HVX_Vector hf2 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p2 + off), v_inv2))));
        HVX_Vector hf3 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) (p3 + off), v_inv3))));

        HVX_Vector vb02 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf2), Q6_Vh_equals_Vhf_rnd(hf0)));
        HVX_Vector vb13 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf3), Q6_Vh_equals_Vhf_rnd(hf1)));
        HVX_VectorPair w_u8 = Q6_W_vshuff_VVR(vb13, vb02, -1);
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] =
            Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_lo_W(w_u8), v_zp_u8), v_min_u8b);
    }
#else
    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080); // 12582912.0f + 128.0f
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        const uint32_t off = kt * 32;
        HVX_Vector vx0 = *(const HVX_UVector *) (p0 + off);
        HVX_Vector vx1 = *(const HVX_UVector *) (p1 + off);
        HVX_Vector vx2 = *(const HVX_UVector *) (p2 + off);
        HVX_Vector vx3 = *(const HVX_UVector *) (p3 + off);

        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3), v_bias_u8));

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
#endif
}

// Quantize one 4-row quad of FP16 Q into W8A8 U8 SM 64x32 tiles and emit d_q (with scale folded).
static __attribute__((noinline)) void hmx_fa_w8a8_q_prep_quad_f16(
        uint8_t * restrict tile_row_base,
        float * restrict d_q_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        const __fp16 * restrict p2,
        const __fp16 * restrict p3,
        uint32_t rg,
        uint32_t n_dot,
        uint32_t valid_rows,
        float scale_f32) {

    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();
    HVX_Vector vmax_hf2 = Q6_V_vzero();
    HVX_Vector vmax_hf3 = Q6_V_vzero();

    const uint32_t n_pairs = n_dot >> 1;
    for (uint32_t step = 0; step < n_pairs; ++step) {
        const uint32_t off = step * 64;
        HVX_Vector vh0 = *(const HVX_UVector *) (p0 + off);
        HVX_Vector vh1 = *(const HVX_UVector *) (p1 + off);
        HVX_Vector vh2 = *(const HVX_UVector *) (p2 + off);
        HVX_Vector vh3 = *(const HVX_UVector *) (p3 + off);
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
        vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vand_VV(vh2, v_abs_f16));
        vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vand_VV(vh3, v_abs_f16));
    }
    if (n_dot & 1u) {
        const uint32_t off = (n_dot - 1u) * 32;
        const HVX_VectorPred q_32h = Q6_Q_vsetq2_R(32 * sizeof(__fp16));
        HVX_Vector vh0 = Q6_V_vmux_QVV(q_32h, *(const HVX_UVector *) (p0 + off), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(q_32h, *(const HVX_UVector *) (p1 + off), Q6_V_vzero());
        HVX_Vector vh2 = Q6_V_vmux_QVV(q_32h, *(const HVX_UVector *) (p2 + off), Q6_V_vzero());
        HVX_Vector vh3 = Q6_V_vmux_QVV(q_32h, *(const HVX_UVector *) (p3 + off), Q6_V_vzero());
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
        vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vand_VV(vh2, v_abs_f16));
        vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vand_VV(vh3, v_abs_f16));
    }

    vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vror_VR(vmax_hf0, 64));
    vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vror_VR(vmax_hf1, 64));
    vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vror_VR(vmax_hf2, 64));
    vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vror_VR(vmax_hf3, 64));

    HVX_Vector vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf0), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf1), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax2 = (valid_rows > 2) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf2), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax3 = (valid_rows > 3) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf3), v_one_f16)))) : Q6_V_vzero();

    const HVX_Vector v_sc_inv127 = hvx_vec_splat_f32(scale_f32 * (1.0f / 127.0f));
    const HVX_Vector v_127       = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_sc_inv127);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_sc_inv127);
    HVX_Vector vd2   = hvx_vec_mul_f32_f32(vmax2, v_sc_inv127);
    HVX_Vector vd3   = hvx_vec_mul_f32_f32(vmax3, v_sc_inv127);
    HVX_Vector v_d02 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d   = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_q_out, 4 * sizeof(float), v_d);

    HVX_Vector v_inv0_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero()));
    HVX_Vector v_inv1_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero()));
    HVX_Vector v_inv2_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero()));
    HVX_Vector v_inv3_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero()));

#if __HVX_ARCH__ >= 81
    for (uint32_t step = 0; step < n_pairs; ++step) {
        const uint32_t off = step * 64;
        HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(*(const HVX_UVector *) (p0 + off), v_one_f16);
        HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(*(const HVX_UVector *) (p1 + off), v_one_f16);
        HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(*(const HVX_UVector *) (p2 + off), v_one_f16);
        HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(*(const HVX_UVector *) (p3 + off), v_one_f16);

        HVX_Vector hf0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf)));
        HVX_Vector hf1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf)));
        HVX_Vector hf2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf)));
        HVX_Vector hf3 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf)));

        pack_hf_quad_to_u8_sm_pair(tile_row_base, step * 2, rg, hf0, hf1, hf2, hf3);
    }
    if (n_dot & 1u) {
        const uint32_t kt  = n_dot - 1u;
        const uint32_t off = kt * 32;
        const HVX_Vector v_zp_u8   = Q6_Vb_vsplat_R(0x80);
        const HVX_Vector v_min_u8b = Q6_Vb_vsplat_R(1);
        HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p0 + off)), v_one_f16);
        HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p1 + off)), v_one_f16);
        HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p2 + off)), v_one_f16);
        HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p3 + off)), v_one_f16);

        HVX_Vector hf0 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf))));
        HVX_Vector hf1 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf))));
        HVX_Vector hf2 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf))));
        HVX_Vector hf3 = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_V_vzero(), Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf))));

        HVX_Vector vb02 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf2), Q6_Vh_equals_Vhf_rnd(hf0)));
        HVX_Vector vb13 = Q6_Vb_vshuff_Vb(Q6_Vb_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf3), Q6_Vh_equals_Vhf_rnd(hf1)));
        HVX_VectorPair w_u8 = Q6_W_vshuff_VVR(vb13, vb02, -1);
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] =
            Q6_Vub_vmax_VubVub(Q6_V_vxor_VV(Q6_V_lo_W(w_u8), v_zp_u8), v_min_u8b);
    }
#else
    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080);
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t step = 0; step < n_pairs; ++step) {
        const uint32_t off = step * 64;
        const uint32_t kt  = step * 2;
        HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p0 + off)), v_one_f16);
        HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p1 + off)), v_one_f16);
        HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p2 + off)), v_one_f16);
        HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p3 + off)), v_one_f16);

        {
            HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf), v_bias_u8));
            HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf), v_bias_u8));
            HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf), v_bias_u8));
            HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf), v_bias_u8));
            HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
            HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
            ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
        }
        {
            HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf), v_bias_u8));
            HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf), v_bias_u8));
            HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf), v_bias_u8));
            HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf), v_bias_u8));
            HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
            HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
            ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
        }
    }
    if (n_dot & 1u) {
        const uint32_t kt  = n_dot - 1u;
        const uint32_t off = kt * 32;
        HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p0 + off)), v_one_f16);
        HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p1 + off)), v_one_f16);
        HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p2 + off)), v_one_f16);
        HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(*(const HVX_UVector *) (p3 + off)), v_one_f16);

        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf), v_bias_u8));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
#endif
}

static inline void hmx_fa_w8a8_q_prep_f32(
        uint8_t * restrict vtcm_q_tiles,
        float * restrict vtcm_q_scales,
        const uint8_t * restrict temp_q_vtcm,
        size_t start,
        size_t end,
        size_t g_rows_end,
        size_t DK,
        size_t G,
        size_t n_rows_q,
        const struct fastdiv_values * div_G,
        bool q_transposed,
        float scale_f32) {

    const uint32_t   n_dot   = (uint32_t) (DK / 32);
    const HVX_Vector v_zp_u8 = Q6_Vb_vsplat_R(0x80);

    for (size_t r = start; r < end; r += 4) {
        const size_t r0 = r >> 6;
        const uint32_t rg = (uint32_t) ((r & 63) >> 2);
        uint8_t * tile_row_base = vtcm_q_tiles + r0 * n_dot * HMX_W8A8_ACT_TILE_SIZE;

        if (r >= g_rows_end) {
            for (uint32_t kt = 0; kt < n_dot; ++kt) {
                ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] = v_zp_u8;
            }
            hvx_vec_store_u(vtcm_q_scales + r, 4 * sizeof(float), Q6_V_vzero());
            continue;
        }

        const uint32_t valid_rows = (uint32_t) hex_smin(4, g_rows_end - r);
        const float * ptrs[4];
        for (uint32_t k = 0; k < 4; ++k) {
            if (k < valid_rows) {
                const size_t q_idx  = fastdiv(r + k, div_G);
                const size_t h_idx  = fastmodulo(r + k, G, div_G);
                const size_t offset = q_transposed ? (h_idx * n_rows_q + q_idx) : (q_idx * G + h_idx);
                ptrs[k] = (const float *) (temp_q_vtcm + offset * DK * sizeof(float));
            } else {
                ptrs[k] = ptrs[0];
            }
        }

        hmx_fa_w8a8_q_prep_quad_f32(
            tile_row_base, vtcm_q_scales + r,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3],
            rg, n_dot, valid_rows, scale_f32);
    }
}

static inline void hmx_fa_w8a8_q_prep_f16(
        uint8_t * restrict vtcm_q_tiles,
        float * restrict vtcm_q_scales,
        const uint8_t * restrict temp_q_vtcm,
        size_t start,
        size_t end,
        size_t g_rows_end,
        size_t DK,
        size_t G,
        size_t n_rows_q,
        const struct fastdiv_values * div_G,
        bool q_transposed,
        float scale_f32) {

    const uint32_t   n_dot   = (uint32_t) (DK / 32);
    const HVX_Vector v_zp_u8 = Q6_Vb_vsplat_R(0x80);

    for (size_t r = start; r < end; r += 4) {
        const size_t r0 = r >> 6;
        const uint32_t rg = (uint32_t) ((r & 63) >> 2);
        uint8_t * tile_row_base = vtcm_q_tiles + r0 * n_dot * HMX_W8A8_ACT_TILE_SIZE;

        if (r >= g_rows_end) {
            for (uint32_t kt = 0; kt < n_dot; ++kt) {
                ((HVX_Vector *) (tile_row_base + (size_t) kt * HMX_W8A8_ACT_TILE_SIZE))[rg] = v_zp_u8;
            }
            hvx_vec_store_u(vtcm_q_scales + r, 4 * sizeof(float), Q6_V_vzero());
            continue;
        }

        const uint32_t valid_rows = (uint32_t) hex_smin(4, g_rows_end - r);
        const __fp16 * ptrs[4];
        for (uint32_t k = 0; k < 4; ++k) {
            if (k < valid_rows) {
                const size_t q_idx  = fastdiv(r + k, div_G);
                const size_t h_idx  = fastmodulo(r + k, G, div_G);
                const size_t offset = q_transposed ? (h_idx * n_rows_q + q_idx) : (q_idx * G + h_idx);
                ptrs[k] = (const __fp16 *) (temp_q_vtcm + offset * DK * sizeof(__fp16));
            } else {
                ptrs[k] = ptrs[0];
            }
        }

        hmx_fa_w8a8_q_prep_quad_f16(
            tile_row_base, vtcm_q_scales + r,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3],
            rg, n_dot, valid_rows, scale_f32);
    }
}

static inline HVX_Vector hmx_fa_w8a8_k_pack4(const uint8_t * p, size_t stride) {
    HVX_Vector v0  = *(const HVX_UVector *) (p + 0 * stride);
    HVX_Vector v1  = *(const HVX_UVector *) (p + 1 * stride);
    HVX_Vector v2  = *(const HVX_UVector *) (p + 2 * stride);
    HVX_Vector v3  = *(const HVX_UVector *) (p + 3 * stride);
    HVX_Vector p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -4));
    HVX_Vector p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -4));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -8));
}

// Same math as int_hmx_rescale_pair_wb, using vadd(-1) instead of vcombine+vmpyacc to avoid Vdd pair spills.
static inline void hmx_fa_w8a8_k_rescale_pair(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector v_neg1_h = Q6_Vh_vsplat_R(-1);
    HVX_VectorPair   vp_p01   = Q6_Wh_vmpy_VbVb(v_q0, v_m_b4);
    HVX_VectorPair   vp_p23   = Q6_Wh_vmpy_VbVb(v_q1, v_m_b4);
    HVX_Vector       v_p23_lo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(vp_p23), v_neg1_h);
    HVX_Vector       v_p23_hi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(vp_p23), v_neg1_h);

#if __HVX_ARCH__ >= 79
    HVX_Vector v_W0 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p01), Q6_V_lo_W(vp_p01), 7);
    HVX_Vector v_W1 = Q6_Vb_vasr_VhVhR_rnd_sat(v_p23_hi, v_p23_lo, 7);
#else
    HVX_Vector v_W0 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p01), 1),
                                            Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p01), 1));
    HVX_Vector v_W1 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(v_p23_hi, 1),
                                            Q6_Vh_vasl_VhR(v_p23_lo, 1));
#endif
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, v_W0, v_W1, v_sum_w, v_l1_w);
}

// Q8_0 K -> S8 weight.b 32x32 tiles for one column tile (32 KV rows).
static __attribute__((noinline)) void hmx_fa_w8a8_k_prep_tile(
        const uint8_t * restrict k_raw,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t valid_rows,
        uint32_t n_dot,
        size_t k_raw_row_padded) {

    uint16_t off_l1[64] __attribute__((aligned(128)));
    for (uint32_t c = 0; c < 32; ++c) {
        off_l1[c]      = (c < valid_rows) ? (uint16_t) (c * k_raw_row_padded) : 0;
        off_l1[c + 32] = 0;
    }
    const HVX_Vector     v_off_h   = *(const HVX_Vector *) off_l1;
    const uint32_t       mu        = (uint32_t) (32 * k_raw_row_padded - 1);
    const HVX_VectorPred q_valid_h = Q6_Q_vsetq2_R(valid_rows * sizeof(uint16_t));

    // Step 1: gather Q8_0 block scales into tile_dst[kt * 8 + 7] and compute per-channel max
    HVX_Vector * tile_dst_all = (HVX_Vector *) dst_wt;
    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        Q6_vgather_ARMVh(&tile_dst_all[kt * 8 + 7], (size_t) (k_raw + kt * 34), mu, v_off_h);
    }
    hvx_gather_sync(&tile_dst_all[(n_dot - 1) * 8 + 7]);

    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        HVX_Vector v_sc = Q6_V_vmux_QVV(q_valid_h, tile_dst_all[kt * 8 + 7], Q6_V_vzero());
        tile_dst_all[kt * 8 + 7] = v_sc;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(v_sc, v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        tile_dst_all[kt * 8 + 7] = int_hmx_m_b4_from_f16(tile_dst_all[kt * 8 + 7], v_inv127_qf);
    }

    // Step 2: 32x32-byte vshuffvdd transpose of int8 quants and rescale in-place to weight.b
    // Invalid rows (r >= valid_rows) have zeroed scales so v_m_b4 == 0 zeroes their weights.
    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < n_dot; ++kt) {
        HVX_Vector *    tile_dst = tile_dst_all + kt * 8;
        const uint8_t * qs_base  = k_raw + kt * 34 + 2;

        HVX_Vector b0, b1, b4, b5;
        {
            HVX_Vector     t0  = hmx_fa_w8a8_k_pack4(qs_base +  0 * k_raw_row_padded, k_raw_row_padded);
            HVX_Vector     t1  = hmx_fa_w8a8_k_pack4(qs_base +  4 * k_raw_row_padded, k_raw_row_padded);
            HVX_VectorPair w01 = Q6_W_vshuff_VVR(t1, t0, -16);
            HVX_Vector     t2  = hmx_fa_w8a8_k_pack4(qs_base +  8 * k_raw_row_padded, k_raw_row_padded);
            HVX_Vector     t3  = hmx_fa_w8a8_k_pack4(qs_base + 12 * k_raw_row_padded, k_raw_row_padded);
            HVX_VectorPair w23 = Q6_W_vshuff_VVR(t3, t2, -16);

            HVX_VectorPair u0 = Q6_W_vshuff_VVR(Q6_V_lo_W(w23), Q6_V_lo_W(w01), -32);
            HVX_VectorPair u2 = Q6_W_vshuff_VVR(Q6_V_hi_W(w23), Q6_V_hi_W(w01), -32);
            b0 = Q6_V_lo_W(u0);
            b1 = Q6_V_hi_W(u0);
            b4 = Q6_V_lo_W(u2);
            b5 = Q6_V_hi_W(u2);
        }

        {
            HVX_Vector     t4  = hmx_fa_w8a8_k_pack4(qs_base + 16 * k_raw_row_padded, k_raw_row_padded);
            HVX_Vector     t5  = hmx_fa_w8a8_k_pack4(qs_base + 20 * k_raw_row_padded, k_raw_row_padded);
            HVX_VectorPair w45 = Q6_W_vshuff_VVR(t5, t4, -16);
            HVX_Vector     t6  = hmx_fa_w8a8_k_pack4(qs_base + 24 * k_raw_row_padded, k_raw_row_padded);
            HVX_Vector     t7  = hmx_fa_w8a8_k_pack4(qs_base + 28 * k_raw_row_padded, k_raw_row_padded);
            HVX_VectorPair w67 = Q6_W_vshuff_VVR(t7, t6, -16);

            const HVX_Vector v_m_b4 = tile_dst[7];

            HVX_VectorPair u1  = Q6_W_vshuff_VVR(Q6_V_lo_W(w67), Q6_V_lo_W(w45), -32);
            HVX_VectorPair q01 = Q6_W_vshuff_VVR(Q6_V_lo_W(u1), b0, -64);
            hmx_fa_w8a8_k_rescale_pair(Q6_V_lo_W(q01), Q6_V_hi_W(q01), v_m_b4, tile_dst, 0, &v_sum_w, &v_l1_w);
            HVX_VectorPair q23 = Q6_W_vshuff_VVR(Q6_V_hi_W(u1), b1, -64);
            hmx_fa_w8a8_k_rescale_pair(Q6_V_lo_W(q23), Q6_V_hi_W(q23), v_m_b4, tile_dst, 1, &v_sum_w, &v_l1_w);

            HVX_VectorPair u3  = Q6_W_vshuff_VVR(Q6_V_hi_W(w67), Q6_V_hi_W(w45), -32);
            HVX_VectorPair q45 = Q6_W_vshuff_VVR(Q6_V_lo_W(u3), b4, -64);
            hmx_fa_w8a8_k_rescale_pair(Q6_V_lo_W(q45), Q6_V_hi_W(q45), v_m_b4, tile_dst, 2, &v_sum_w, &v_l1_w);
            HVX_VectorPair q67 = Q6_W_vshuff_VVR(Q6_V_hi_W(u3), b5, -64);
            hmx_fa_w8a8_k_rescale_pair(Q6_V_lo_W(q67), Q6_V_hi_W(q67), v_m_b4, tile_dst, 3, &v_sum_w, &v_l1_w);
        }
    }

    // Step 3: emit per-channel scales and HMX bias banks
    emit_int_hmx_scales_and_bias(v_max_sf, 128.0f / 127.0f, v_sum_w, v_l1_w, dst_scales, bias_out);
}

// Decode one 4-row x 32-col W8A8 S tile (out_hi + out_lo) into 4 FP32 vectors.
static inline void hmx_fa_w8a8_eval_s_tile_quad(
        const uint8_t * restrict tile_base,
        const float * restrict dw_base,
        size_t c,
        uint32_t da0,
        uint32_t da1,
        uint32_t da2,
        uint32_t da3,
        HVX_Vector sign_80,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1,
        HVX_Vector * v_out2,
        HVX_Vector * v_out3) {

    const size_t     c0        = c >> 5;
    const uint8_t *  tp        = tile_base + c0 * 2 * HMX_W8A8_ACT_TILE_SIZE;
    const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
    const HVX_Vector v_lo_u8   = *(const HVX_Vector *) (tp + HMX_W8A8_ACT_TILE_SIZE);
    const HVX_Vector v_dw_qf   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
    HVX_VectorPair   vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
    HVX_VectorPair   vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
    HVX_VectorPair   vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

    *v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01)), da0), v_dw_qf));
    *v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01)), da1), v_dw_qf));
    *v_out2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23)), da2), v_dw_qf));
    *v_out3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23)), da3), v_dw_qf));
}

// Pack 4 rows x 64 FP16 values in [0, 255] (zp = 0) into 1 or 2 U8 SM 64x32 tiles.
static inline void hmx_fa_w8a8_pack_p_quad_u8_sm(
        uint8_t * restrict tile_base,
        uint32_t kt,
        uint32_t rg,
        bool write_hi,
        HVX_Vector hf0,
        HVX_Vector hf1,
        HVX_Vector hf2,
        HVX_Vector hf3) {
#if __HVX_ARCH__ >= 81
    HVX_Vector vb02 = Q6_Vb_vshuff_Vb(Q6_Vub_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf2), Q6_Vh_equals_Vhf_rnd(hf0)));
    HVX_Vector vb13 = Q6_Vb_vshuff_Vb(Q6_Vub_vpack_VhVh_sat(Q6_Vh_equals_Vhf_rnd(hf3), Q6_Vh_equals_Vhf_rnd(hf1)));
    HVX_VectorPair w_u8 = Q6_W_vshuff_VVR(vb13, vb02, -1);

    ((HVX_Vector *) (tile_base + (size_t) (kt + 0) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_V_lo_W(w_u8);
    if (write_hi) {
        ((HVX_Vector *) (tile_base + (size_t) (kt + 1) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_V_hi_W(w_u8);
    }
#else
    const HVX_Vector v_one_f16  = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_bias_zp0 = Q6_V_vsplat_R(0x4B400000); // 12582912.0f
    const HVX_Vector v_zero     = Q6_V_vzero();
    const HVX_Vector v_max_u8   = Q6_Vh_vsplat_R(255);

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(hf0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(hf1), v_one_f16);
    HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(hf2), v_one_f16);
    HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(hf3), v_one_f16);

    {
        HVX_Vector y0   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(vp0), v_bias_zp0));
        HVX_Vector y1   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(vp1), v_bias_zp0));
        HVX_Vector y2   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(vp2), v_bias_zp0));
        HVX_Vector y3   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(vp3), v_bias_zp0));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_zero), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_zero), v_max_u8);
        ((HVX_Vector *) (tile_base + (size_t) (kt + 0) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
    if (write_hi) {
        HVX_Vector y0   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(vp0), v_bias_zp0));
        HVX_Vector y1   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(vp1), v_bias_zp0));
        HVX_Vector y2   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(vp2), v_bias_zp0));
        HVX_Vector y3   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(vp3), v_bias_zp0));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_zero), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_zero), v_max_u8);
        ((HVX_Vector *) (tile_base + (size_t) (kt + 1) * HMX_W8A8_ACT_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
#endif
}

// Zero-gather Q8_0 V -> S8 weight.b 32x32 tiles for one DV tile (32 output cols) across kv_rows.
static __attribute__((noinline)) void hmx_fa_w8a8_v_prep_tile(
        const uint8_t * restrict v_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t kv_rows,
        uint32_t n_col_tiles,
        size_t v_raw_row_padded) {

    uint16_t off_full[64] __attribute__((aligned(128)));
    for (uint32_t c = 0; c < 32; ++c) {
        off_full[c]      = (uint16_t) (c * v_raw_row_padded);
        off_full[c + 32] = 0;
    }
    const HVX_Vector v_off_full = *(const HVX_Vector *) off_full;
    const uint32_t   mu         = (uint32_t) (32 * v_raw_row_padded - 1);

    // Pass 1: gather Q8_0 block scales into tile_dst[kt * 8 + 7] and reduce max across kv_rows
    HVX_Vector * tile_dst_all = (HVX_Vector *) dst_wt;
    for (uint32_t kt = 0; kt < n_col_tiles; ++kt) {
        const uint32_t valid = (uint32_t) hex_smin(32, kv_rows - kt * 32);
        HVX_Vector v_off = v_off_full;
        if (valid < 32) {
            uint16_t off_tail[64] __attribute__((aligned(128)));
            for (uint32_t c = 0; c < 32; ++c) {
                off_tail[c]      = (c < valid) ? (uint16_t) (c * v_raw_row_padded) : 0;
                off_tail[c + 32] = 0;
            }
            v_off = *(const HVX_Vector *) off_tail;
        }
        Q6_vgather_ARMVh(&tile_dst_all[kt * 8 + 7], (size_t) (v_base + (size_t) kt * 32 * v_raw_row_padded), mu, v_off);
    }
    hvx_gather_sync(&tile_dst_all[(n_col_tiles - 1) * 8 + 7]);

    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n_col_tiles; ++kt) {
        const uint32_t       valid   = (uint32_t) hex_smin(32, kv_rows - kt * 32);
        const HVX_VectorPred q_valid = Q6_Q_vsetq2_R(valid * sizeof(uint16_t));
        HVX_Vector           v_sc    = Q6_V_vmux_QVV(q_valid, tile_dst_all[kt * 8 + 7], Q6_V_vzero());
        tile_dst_all[kt * 8 + 7]     = v_sc;
        v_max_sc                     = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(v_sc, v_abs_mask));
    }
    v_max_sc = hvx_vec_reduce_max_f16(v_max_sc);

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    // Pass 2: 4-row vshuff interleave + per-KV-row int8 scale multiply -> S8 weight.b
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v12M_sf   = Q6_V_vsplat_R(0x4B400000);
    const HVX_Vector v_ones_s8 = Q6_Vb_vsplat_R(1);
    HVX_Vector       v_l1_w    = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < n_col_tiles; ++kt) {
        HVX_Vector * tile_dst = tile_dst_all + kt * 8;

        HVX_Vector v_sc_qf    = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(tile_dst[7]), v_one_f16));
        HVX_Vector v_m_biased = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(v_sc_qf, v_inv127_qf), v12M_sf));
        HVX_Vector v_m_h      = Q6_Vh_vpacke_VwVw(v_m_biased, v_m_biased);
        HVX_Vector v_m_b      = Q6_Vb_vpacke_VhVh(v_m_h, v_m_h);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t g = 0; g < 8; ++g) {
            const uint32_t  k0  = kt * 32 + g * 4;
            const uint8_t * p0  = v_base + (size_t) k0 * v_raw_row_padded + 2;
            HVX_Vector r0 = (k0 + 0 < kv_rows) ? *(const HVX_UVector *) (p0 + 0 * v_raw_row_padded) : Q6_V_vzero();
            HVX_Vector r1 = (k0 + 1 < kv_rows) ? *(const HVX_UVector *) (p0 + 1 * v_raw_row_padded) : Q6_V_vzero();
            HVX_Vector r2 = (k0 + 2 < kv_rows) ? *(const HVX_UVector *) (p0 + 2 * v_raw_row_padded) : Q6_V_vzero();
            HVX_Vector r3 = (k0 + 3 < kv_rows) ? *(const HVX_UVector *) (p0 + 3 * v_raw_row_padded) : Q6_V_vzero();

            HVX_Vector v02 = Q6_V_lo_W(Q6_W_vshuff_VVR(r2, r0, -1));
            HVX_Vector v13 = Q6_V_lo_W(Q6_W_vshuff_VVR(r3, r1, -1));
            HVX_Vector v_q = Q6_V_lo_W(Q6_W_vshuff_VVR(v13, v02, -1));

            HVX_Vector     v_m = Q6_V_vsplat_R(Q6_R_vextract_VR(v_m_b, (int) (g * 4)));
            HVX_VectorPair vp  = Q6_Wh_vmpy_VbVb(v_q, v_m);
#if __HVX_ARCH__ >= 79
            HVX_Vector v_W = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp), Q6_V_lo_W(vp), 7);
#else
            HVX_Vector v_W = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp), 1),
                                                   Q6_Vh_vasl_VhR(Q6_V_lo_W(vp), 1));
#endif
            tile_dst[g] = v_W;
            v_l1_w      = Q6_Vw_vrmpyacc_VwVubVb(v_l1_w, Q6_Vb_vabs_Vb(v_W), v_ones_s8);
        }
    }

    // Pass 3: emit zp=0 per-channel scales and HMX bias banks
    emit_int_hmx_scales_and_bias_zp0(v_max_sf, 128.0f / 127.0f, v_l1_w, dst_scales, bias_out);
}

static inline void hmx_fa_w8a8_gemm_run(
        const uint8_t * restrict act_tiles,
        const uint8_t * restrict wt_tiles,
        const uint8_t * restrict wt_bias,
        uint8_t * restrict out_tiles,
        uint32_t n_row_tiles_64,
        uint32_t n_col_tiles,
        uint32_t n_dot_tiles) {

    __builtin_assume(n_row_tiles_64 > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    const uint32_t cvt_ctrl_hi = hmx_cvt_ctrl(0, 0, 0, 1);
#if __HVX_ARCH__ >= 79
    const uint32_t cvt_ctrl_lo = hmx_cvt_ctrl(1, 0, 1, 0);
#else
    const uint32_t cvt_ctrl_lo = hmx_cvt_ctrl(0, 0, 1, 0);
#endif
    const uint32_t a_rng      = hmx_act_sm_range(n_dot_tiles);
    const uint32_t w_rng      = hmx_wt_b_range(n_dot_tiles * HMX_W8A8_WT_TILE_SIZE);
    const uint32_t out_cm_rng = hmx_out_cm_range();

    for (uint32_t j = 0; j < n_col_tiles; ++j) {
        const uint8_t * wt_base = wt_tiles + (size_t) j * n_dot_tiles * HMX_W8A8_WT_TILE_SIZE;
        const uint8_t * bias_hi = wt_bias  + ((size_t) j * 2 + 0) * HMX_W8A8_BIAS_BANK_SIZE;
        const uint8_t * bias_lo = wt_bias  + ((size_t) j * 2 + 1) * HMX_W8A8_BIAS_BANK_SIZE;
#if __HVX_ARCH__ >= 79
        hmx_load_bias_bank(bias_hi, 0);
        hmx_load_bias_bank(bias_lo, 1);
#endif

        for (uint32_t i = 0; i < n_row_tiles_64; ++i) {
            const uint8_t * act_base = act_tiles + (size_t) i * n_dot_tiles * HMX_W8A8_ACT_TILE_SIZE;
            const size_t    out_idx  = ((size_t) i * n_col_tiles + j) * 2;
            uint8_t *       out_hi   = out_tiles + (out_idx + 0) * HMX_W8A8_ACT_TILE_SIZE;
            uint8_t *       out_lo   = out_tiles + (out_idx + 1) * HMX_W8A8_ACT_TILE_SIZE;

            asm volatile(HMX_CLRACC_INT());
            asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_SM("%0", "%1", "%2", "%3")
                         : : "r"(act_base), "r"(a_rng), "r"(wt_base), "r"(w_rng) : "memory");

#if __HVX_ARCH__ < 79
            hmx_load_bias_bank(bias_hi, 0);
#endif
            asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_hi) : "memory");
            asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_hi), "r"(out_cm_rng) : "memory");

#if __HVX_ARCH__ < 79
            hmx_load_bias_bank(bias_lo, 0);
#endif
            asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_lo) : "memory");
            asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_lo), "r"(out_cm_rng) : "memory");
        }
    }
}

typedef struct {
    const uint8_t * q_tiles;
    const uint8_t * k_tiles;
    const uint8_t * k_bias;
    uint8_t *       s_tiles;
    uint32_t        n_row_tiles_64;
    uint32_t        n_col_tiles;
    uint32_t        n_dot_tiles;  // DK / 32
} hmx_fa_w8a8_qk_job_t;

static void hmx_fa_w8a8_qk_dot_worker(const htp_thread_t * th, void * data) {
    (void) th;
    hmx_fa_w8a8_qk_job_t * job = (hmx_fa_w8a8_qk_job_t *) data;
    hmx_fa_w8a8_gemm_run(job->q_tiles, job->k_tiles, job->k_bias, job->s_tiles,
                         job->n_row_tiles_64, job->n_col_tiles, job->n_dot_tiles);
}

typedef struct {
    const uint8_t * p_tiles;
    const uint8_t * v_tiles;
    const uint8_t * v_bias;
    uint8_t *       do_tiles;
    uint32_t        n_row_tiles_64;
    uint32_t        n_v_tiles;    // DV / 32
    uint32_t        n_dot_tiles;  // ceil(kv_rows / 32)
} hmx_fa_w8a8_pv_job_t;

static void hmx_fa_w8a8_pv_worker(const htp_thread_t * th, void * data) {
    (void) th;
    hmx_fa_w8a8_pv_job_t * job = (hmx_fa_w8a8_pv_job_t *) data;
    hmx_fa_w8a8_gemm_run(job->p_tiles, job->v_tiles, job->v_bias, job->do_tiles,
                         job->n_row_tiles_64, job->n_v_tiles, job->n_dot_tiles);
}

#endif /* HMX_FA_W8A8_KERNELS_H */
