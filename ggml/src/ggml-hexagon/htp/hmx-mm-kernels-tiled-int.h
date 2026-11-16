// --- INT HMX (W8A8 SM and W8A16 :2x1) Track A kernels ---

typedef struct {
    uint8_t *             dst_wt_base;
    uint8_t *             dst_bias_base;
    float *               dst_scales_base;
    const uint8_t *       src_raw;
    uint32_t              n_col_tiles;
    uint32_t              n_k_tiles;
    uint32_t              n_groups;
    uint32_t              tiles_per_group;
    uint32_t              aligned_tile_size;
    int32_t               kernel_type;
    int32_t               weight_type;
    uint32_t              n_tot_units;
    uint32_t              n_units_per_task;
    uint32_t              n_tasks;
    struct fastdiv_values n_groups_div;
} int_weight_rescale_state_t;

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_x8_lut[128] = {
    0,    0, 8,    0, 16,   0, 24,   0, 32,   0, 48,   0, 64,   0, 96,   0,
    0,    0, 0xf8, 0, 0xf0, 0, 0xe8, 0, 0xe0, 0, 0xd0, 0, 0xc0, 0, 0xa0, 0,
};

static inline HVX_Vector hvx_vec_f32_to_qf32(HVX_Vector v) {
#if __HVX_ARCH__ >= 81
    return Q6_Vqf32_equals_Vsf(v);
#else
    return Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero());
#endif
}

// Convert two 32-float vectors (already scaled to [-32766, 32766]) directly to interleaved uint16 (even=vqf_0, odd=vqf_1) with +32768 offset.
static inline HVX_Vector hvx_vec_u16_zp32768_from_qf32_pair_shuff(HVX_Vector vqf_0, HVX_Vector vqf_1) {
    const HVX_Vector v_bias_u16 = Q6_V_vsplat_R(0x4B408000); // 12582912.0f + 32768.0f (1 ULP == 1.0f)
    HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vqf_0, v_bias_u16));
    HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vqf_1, v_bias_u16));
    return Q6_Vh_vshuffe_VhVh(y1, y0);
}

static inline void emit_int_hmx_scales_and_bias(
        HVX_Vector v_max_sf,
        float scale_const,
        HVX_Vector v_sum_w,
        HVX_Vector v_l1_w,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        bool is_w8a8) {

    // |sum_k (a - zp) * w_hat| < 128 * l1_w = (l1_w << 7) <= 32768 * 2^p
    HVX_Vector v_max_mag = Q6_Vw_vasl_VwR(v_l1_w, 7);
    HVX_Vector v_p       = Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(17), Q6_Vuw_vcl0_Vuw(v_max_mag)));
    HVX_Vector v_ib      = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(32768), v_p), Q6_Vw_vasl_VwR(v_sum_w, 7));

    HVX_Vector v_dw_base       = hvx_vec_mul_f32_f32(v_max_sf, hvx_vec_splat_f32(scale_const));
    HVX_Vector v_pow2_sf       = Q6_Vsf_equals_Vw(Q6_Vw_vasl_VwVw(Q6_V_vsplat_R(1), v_p));
    *(HVX_Vector *) dst_scales = hvx_vec_mul_f32_f32(v_dw_base, v_pow2_sf);

    if (is_w8a8) {
        HVX_Vector v_exp_hi = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
        HVX_Vector v_exp_lo = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(24), v_p);
        bias_out[0] = Q6_Vw_vasl_VwR(v_exp_hi, 10);
        bias_out[1] = v_ib;
        bias_out[2] = Q6_Vw_vasl_VwR(v_exp_lo, 10);
        bias_out[3] = v_ib;
    } else {
        HVX_Vector v_exp = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(16), v_p);
        bias_out[0] = Q6_Vw_vasl_VwR(v_exp, 10);
        bias_out[1] = v_ib;
    }
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

static inline HVX_Vector int_hmx_m_b2_from_sc_qf(HVX_Vector v_sc_qf, HVX_Vector v_inv127_qf) {
    const HVX_Vector v12M_sf = Q6_V_vsplat_R(0x4B400000);
    HVX_Vector v_m_biased    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(v_sc_qf, v_inv127_qf), v12M_sf));
    HVX_Vector v_m_h         = Q6_Vh_vpacke_VwVw(v_m_biased, v_m_biased);
    return Q6_Vb_vshuff_Vb(Q6_Vb_vpacke_VhVh(v_m_h, v_m_h));
}

static inline HVX_Vector int_hmx_m_b2_from_f16(HVX_Vector v_sc, HVX_Vector v_inv127_qf) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_Vector v_sc_qf         = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sc), v_one_f16));
    return int_hmx_m_b2_from_sc_qf(v_sc_qf, v_inv127_qf);
}

static inline HVX_Vector int_hmx_m_b4_from_f16(HVX_Vector v_sc, HVX_Vector v_inv127_qf) {
    return Q6_Vh_vshuff_Vh(int_hmx_m_b2_from_f16(v_sc, v_inv127_qf));
}

static inline void int_hmx_m_b2_pair_from_f16(
        HVX_Vector v_sc,
        HVX_Vector v_inv127_qf,
        HVX_Vector * restrict v_m_b0_out,
        HVX_Vector * restrict v_m_b1_out) {
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    HVX_VectorPair vp_sc_qf    = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_sc), v_one_f16);
    *v_m_b0_out                = int_hmx_m_b2_from_sc_qf(Q6_V_lo_W(vp_sc_qf), v_inv127_qf);
    *v_m_b1_out                = int_hmx_m_b2_from_sc_qf(Q6_V_hi_W(vp_sc_qf), v_inv127_qf);
}

static inline void int_hmx_m_b4_pair_from_f16(
        HVX_Vector v_sc,
        HVX_Vector v_inv127_qf,
        HVX_Vector * restrict v_m_b0_out,
        HVX_Vector * restrict v_m_b1_out) {
    HVX_Vector v_m0, v_m1;
    int_hmx_m_b2_pair_from_f16(v_sc, v_inv127_qf, &v_m0, &v_m1);
    *v_m_b0_out = Q6_Vh_vshuff_Vh(v_m0);
    *v_m_b1_out = Q6_Vh_vshuff_Vh(v_m1);
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

    HVX_Vector v_pair_sum = Q6_Vw_vrmpyacc_VwVbVb(Q6_Vw_vrmpy_VbVb(v_W0, v_ones_s8), v_W1, v_ones_s8);
    *v_sum_w              = Q6_Vw_vadd_VwVw(*v_sum_w, v_pair_sum);

    HVX_Vector v_pair_l1 = Q6_Vw_vrmpyacc_VwVubVb(Q6_Vw_vrmpy_VubVb(Q6_Vb_vabs_Vb(v_W0), v_ones_s8), Q6_Vb_vabs_Vb(v_W1), v_ones_s8);
    *v_l1_w              = Q6_Vw_vadd_VwVw(*v_l1_w, v_pair_l1);
}

static inline void int_hmx_rescale_pair_cp(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_m_b2,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    HVX_Vector     v_neg1_h   = Q6_Vh_vsplat_R(-1);
    HVX_VectorPair vp_tie_adj = Q6_W_vcombine_VV(v_neg1_h, v_neg1_h);
    HVX_VectorPair vp_p01     = Q6_Wh_vmpy_VbVb(v_q0, v_m_b2);
    HVX_VectorPair vp_p23     = Q6_Wh_vmpyacc_WhVbVb(vp_tie_adj, v_q1, v_m_b2);

#if __HVX_ARCH__ >= 79
    HVX_Vector v_b_01 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p01), Q6_V_lo_W(vp_p01), 7);
    HVX_Vector v_b_23 = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(vp_p23), Q6_V_lo_W(vp_p23), 7);
#else
    HVX_Vector v_b_01 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p01), 1),
                                              Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p01), 1));
    HVX_Vector v_b_23 = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(Q6_V_hi_W(vp_p23), 1),
                                              Q6_Vh_vasl_VhR(Q6_V_lo_W(vp_p23), 1));
#endif
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, Q6_Vh_vshuff_Vh(v_b_01), Q6_Vh_vshuff_Vh(v_b_23), v_sum_w, v_l1_w);
}

static inline void int_hmx_rescale_pair_wb(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_m_b4,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    HVX_VectorPair vp_p01 = Q6_Wh_vmpy_VbVb(v_q0, v_m_b4);
    HVX_VectorPair vp_p23 = Q6_Wh_vmpy_VbVb(v_q1, v_m_b4);

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

static inline void int_hmx_rescale_4bit_lut_step(
        HVX_Vector vq,
        HVX_Vector v_lut,
        HVX_Vector v_m_b2,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo          = Q6_V_vand_VV(vq, mask_0f);
    HVX_Vector v_hi          = Q6_Vub_vlsr_VubR(vq, 4);
    HVX_VectorPair vp_shuf   = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v_q0          = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(vp_shuf), v_lut, 0);
    HVX_Vector v_q1          = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(vp_shuf), v_lut, 0);
    int_hmx_rescale_pair_cp(v_q0, v_q1, v_m_b2, tile_dst, pair_idx, v_sum_w, v_l1_w);
}

// Rescale one (ct, g) superblock of Q4_0 tiles into int8 weight.b tiles,
// compute sum_w and exact l1_w per output channel, and write the Bias Register bank(s) and D_w_eff.
static __attribute__((noinline)) void rescale_q4_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_f0    = Q6_Vb_vsplat_R(0xF0);
    const HVX_Vector sign_80    = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);

    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector v_sc = Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask);
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, v_sc);
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_m_b    = int_hmx_m_b2_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);

        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vxor_VV(Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq, 4), mask_f0), sign_80);
            HVX_Vector v_hi = Q6_V_vxor_VV(Q6_V_vand_VV(vq, mask_f0), sign_80);
            HVX_VectorPair vp_shuf = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
            int_hmx_rescale_pair_cp(Q6_V_lo_W(vp_shuf), Q6_V_hi_W(vp_shuf), v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (8.0f / 127.0f) : (2048.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q8_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 1024), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_m_b    = int_hmx_m_b2_from_f16(hvx_vmem(tile_src + 1024), v_inv127_qf);

        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq0 = hvx_vmem(tile_src + (2 * i + 0) * 128);
            HVX_Vector vq1 = hvx_vmem(tile_src + (2 * i + 1) * 128);
            int_hmx_rescale_pair_cp(vq0, vq1, v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (128.0f / 127.0f) : (32768.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static inline HVX_Vector int_hmx_mxfp4_e8m0_to_f32(HVX_Vector v_raw) {
    HVX_Vector v_e_h     = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_raw));
    HVX_Vector v_e_w     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_e_h));
    HVX_VectorPred q_nan = Q6_Q_vcmp_eq_VwVw(v_e_w, Q6_V_vsplat_R(0xff));
    return Q6_V_vmux_QVV(q_nan, Q6_V_vzero(), Q6_Vw_vasl_VwR(v_e_w, 23));
}

static __attribute__((noinline)) void rescale_mxfp4_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_lut = *(const HVX_Vector *) kvalues_mxfp4_x8_lut;
    HVX_Vector v_max_sf = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sf = Q6_Vsf_vmax_VsfVsf(v_max_sf, int_hmx_mxfp4_e8m0_to_f32(hvx_vmem(tile_src + 512)));
    }

    HVX_Vector v_inv127_qf = int_hmx_inv127_from_max_f32(v_max_sf);
    HVX_Vector v_sum_w     = Q6_V_vzero();
    HVX_Vector v_l1_w      = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_sc_qf  = hvx_vec_f32_to_qf32(int_hmx_mxfp4_e8m0_to_f32(hvx_vmem(tile_src + 512)));
        HVX_Vector      v_m_b    = int_hmx_m_b2_from_sc_qf(v_sc_qf, v_inv127_qf);

        for (int i = 0; i < 4; ++i) {
            int_hmx_rescale_4bit_lut_step(hvx_vmem(tile_src + i * 128), v_lut, v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (8.0f / 127.0f) : (2048.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q5_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_78    = Q6_Vb_vsplat_R(0x78);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 640), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_plane  = hvx_vmem(tile_src + 512);
        HVX_Vector      v_m_b    = int_hmx_m_b2_from_f16(hvx_vmem(tile_src + 640), v_inv127_qf);

        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vand_VV(Q6_Vh_vasl_VhR(vq, 3), mask_78);
            HVX_Vector v_hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(vq, 1), mask_78);
            v_lo            = Q6_V_vandor_VQR(v_lo, Q6_Q_vand_VR(v_plane, 0x01010101u << (2 * i)),     0x80808080);
            v_hi            = Q6_V_vandor_VQR(v_hi, Q6_Q_vand_VR(v_plane, 0x01010101u << (2 * i + 1)), 0x80808080);
            HVX_VectorPair vp_shuf = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
            int_hmx_rescale_pair_cp(Q6_V_lo_W(vp_shuf), Q6_V_hi_W(vp_shuf), v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (16.0f / 127.0f) : (4096.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_iq4_nl_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector v_lut      = *(const HVX_Vector *) kvalues_iq4nl_lut;
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_m_b    = int_hmx_m_b2_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);

        for (int i = 0; i < 4; ++i) {
            int_hmx_rescale_4bit_lut_step(hvx_vmem(tile_src + i * 128), v_lut, v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (128.0f / 127.0f) : (32768.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q1_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const uint32_t   raw_tiles  = tiles_per_group >> 2;
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * aligned_tile_size;
        HVX_Vector      v_pos_m  = int_hmx_m_b4_from_f16(hvx_vmem(tile_src + 512), v_inv127_qf);
        HVX_Vector      v_neg_m  = Q6_Vb_vsub_VbVb(Q6_V_vzero(), v_pos_m);

        for (uint32_t q = 0; q < 4; ++q) {
            HVX_Vector   v_plane  = hvx_vmem(tile_src + q * 128);
            HVX_Vector * tile_dst = (HVX_Vector *) (dst_wt + (size_t) (4 * rt + q) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_W0 = unpack_q1_0_group(v_plane, 2 * i + 0, v_neg_m, v_pos_m);
                HVX_Vector v_W1 = unpack_q1_0_group(v_plane, 2 * i + 1, v_neg_m, v_pos_m);
                int_hmx_store_and_accum_pair(tile_dst, i, v_W0, v_W1, &v_sum_w, &v_l1_w);
            }
        }
    }

    const float scale_const = is_w8a8 ? (1.0f / 127.0f) : (256.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q2_0_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const uint32_t   raw_tiles  = tiles_per_group >> 1;
    const HVX_Vector mask_03    = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector i32        = Q6_Vb_vsplat_R(32);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const uint8_t * tile_src = src_base + (size_t) rt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 512), v_abs_mask));
    }

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t rt = 0; rt < raw_tiles; ++rt) {
        const HVX_Vector * vptr   = (const HVX_Vector *) (src_base + (size_t) rt * aligned_tile_size);
        HVX_Vector         v_m_b4 = int_hmx_m_b4_from_f16(vptr[4], v_inv127_qf);

        for (uint32_t half = 0; half < 2; ++half) {
            HVX_Vector * tile_dst = (HVX_Vector *) (dst_wt + (size_t) (2 * rt + half) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_q0 = Q6_Vb_vsub_VbVb(Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr + 2 * half, 2 * i + 0, mask_03), 5), i32);
                HVX_Vector v_q1 = Q6_Vb_vsub_VbVb(Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr + 2 * half, 2 * i + 1, mask_03), 5), i32);
                int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
            }
        }
    }

    const float scale_const = is_w8a8 ? (4.0f / 127.0f) : (1024.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_nvfp4_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_lut = *(const HVX_Vector *) kvalues_mxfp4_x8_lut;
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector v_sc = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, v_sc);
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_sc     = nvfp4_ue4m3_to_f16(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile_src + NVFP4_SCALE_OFF))));
        HVX_Vector      v_m_b0, v_m_b1;
        int_hmx_m_b2_pair_from_f16(v_sc, v_inv127_qf, &v_m_b0, &v_m_b1);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b = (i < 2) ? v_m_b0 : v_m_b1;
            int_hmx_rescale_4bit_lut_step(hvx_vmem(tile_src + i * 128), v_lut, v_m_b, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (8.0f / 127.0f) : (2048.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static inline HVX_Vector unpack_q3_k_group_shl5(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_03) {
    HVX_VectorPred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    HVX_Vector     v_lo2 = Q6_Vh_vasl_VhR(unpack_q3_k_low2(vptr, g, mask_03), 5);
    return Q6_V_vandor_VQR(v_lo2, q_neg, 0x80808080);
}

static __attribute__((noinline)) void rescale_q3_k_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_03    = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 384), v_abs_mask));
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const HVX_Vector * vptr     = (const HVX_Vector *) (src_base + (size_t) kt * aligned_tile_size);
        HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector         v_m_b0, v_m_b1;
        int_hmx_m_b4_pair_from_f16(vptr[3], v_inv127_qf, &v_m_b0, &v_m_b1);

        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b4 = (i < 2) ? v_m_b0 : v_m_b1;
            HVX_Vector v_q0   = unpack_q3_k_group_shl5(vptr, 2 * i + 0, mask_03);
            HVX_Vector v_q1   = unpack_q3_k_group_shl5(vptr, 2 * i + 1, mask_03);
            int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (4.0f / 127.0f) : (1024.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q6_k_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_0f    = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03    = Q6_Vb_vsplat_R(0x03);
    const HVX_Vector sign_80    = Q6_Vb_vsplat_R(0x80);
    const HVX_Vector v_abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector v_max_sc = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vand_VV(hvx_vmem(tile_src + 768), v_abs_mask));
    }
    v_max_sc = Q6_Vhf_vmax_VhfVhf(v_max_sc, Q6_V_vror_VR(v_max_sc, 64));

    HVX_Vector v_max_sf, v_inv127_qf;
    int_hmx_inv127_from_max_f16(v_max_sc, &v_max_sf, &v_inv127_qf);

    HVX_Vector v_sum_w = Q6_V_vzero();
    HVX_Vector v_l1_w  = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const HVX_Vector * vptr     = (const HVX_Vector *) (src_base + (size_t) kt * aligned_tile_size);
        HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector         v_m_b0, v_m_b1;
        int_hmx_m_b4_pair_from_f16(vptr[6], v_inv127_qf, &v_m_b0, &v_m_b1);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector v_m_b4 = (i < 2) ? v_m_b0 : v_m_b1;
            HVX_Vector v_q0   = Q6_V_vxor_VV(Q6_Vh_vasl_VhR(unpack_q6_k_group(vptr, 2 * i + 0, mask_0f, mask_03, Q6_V_vzero()), 2), sign_80);
            HVX_Vector v_q1   = Q6_V_vxor_VV(Q6_Vh_vasl_VhR(unpack_q6_k_group(vptr, 2 * i + 1, mask_0f, mask_03, Q6_V_vzero()), 2), sign_80);
            int_hmx_rescale_pair_wb(v_q0, v_q1, v_m_b4, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (32.0f / 127.0f) : (8192.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static inline void int_hmx_pack_q6_pair(
        HVX_Vector v_hi0,
        HVX_Vector v_lo0,
        HVX_Vector v_hi1,
        HVX_Vector v_lo1,
        HVX_Vector * restrict v_W0_out,
        HVX_Vector * restrict v_W1_out) {
#if __HVX_ARCH__ >= 79
    *v_W0_out = Q6_Vb_vasr_VhVhR_rnd_sat(v_hi0, v_lo0, 6);
    *v_W1_out = Q6_Vb_vasr_VhVhR_rnd_sat(v_hi1, v_lo1, 6);
#else
    *v_W0_out = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(v_hi0, 2), Q6_Vh_vasl_VhR(v_lo0, 2));
    *v_W1_out = Q6_Vb_vround_VhVh_sat(Q6_Vh_vasl_VhR(v_hi1, 2), Q6_Vh_vasl_VhR(v_lo1, 2));
#endif
}

static inline HVX_Vector int_hmx_asym_update_wmax_qf(
        HVX_Vector v_wmax_sf,
        HVX_Vector vd_qf,
        HVX_Vector vm_qf,
        HVX_Vector v_qmax_qf) {

    const HVX_Vector v_abs_mask_sf = Q6_V_vsplat_R(0x7fffffff);
    HVX_Vector vm_sf               = Q6_Vsf_equals_Vqf32(vm_qf);
    HVX_Vector v_end               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(vd_qf, v_qmax_qf), vm_qf));
    HVX_Vector v_ext0              = Q6_V_vand_VV(vm_sf, v_abs_mask_sf);
    HVX_Vector v_ext1              = Q6_V_vand_VV(v_end, v_abs_mask_sf);
    return Q6_Vsf_vmax_VsfVsf(v_wmax_sf, Q6_Vsf_vmax_VsfVsf(v_ext0, v_ext1));
}

static inline void int_hmx_asym_alpha_beta_cp(
        HVX_Vector vd_qf,
        HVX_Vector vm_qf,
        HVX_Vector v_scale_q6_qf,
        HVX_Vector * restrict v_alpha_h_out,
        HVX_Vector * restrict v_beta_h_out) {

    const HVX_Vector v12M_sf = Q6_V_vsplat_R(0x4B400000);
    HVX_Vector v_a_bias      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(vd_qf, v_scale_q6_qf), v12M_sf));
    HVX_Vector v_b_bias      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(vm_qf, v_scale_q6_qf), v12M_sf));
    *v_alpha_h_out           = Q6_Vh_vpacke_VwVw(v_a_bias, v_a_bias);
    *v_beta_h_out            = Q6_Vh_vpacke_VwVw(v_b_bias, v_b_bias);
}

static inline void int_hmx_asym_alpha_beta_wb(
        HVX_Vector vd_qf,
        HVX_Vector vm_qf,
        HVX_Vector v_scale_q6_qf,
        HVX_Vector * restrict v_alpha_h_out,
        HVX_Vector * restrict v_beta_h_out) {

    const HVX_Vector v12M_sf = Q6_V_vsplat_R(0x4B400000);
    HVX_Vector v_a_bias      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(vd_qf, v_scale_q6_qf), v12M_sf));
    HVX_Vector v_b_bias      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(vm_qf, v_scale_q6_qf), v12M_sf));
    *v_alpha_h_out           = Q6_Vh_vshuffe_VhVh(v_a_bias, v_a_bias);
    *v_beta_h_out            = Q6_Vh_vshuffe_VhVh(v_b_bias, v_b_bias);
}

static inline void int_hmx_rescale_asym_pair_cp(
        HVX_Vector v_lo,
        HVX_Vector v_hi,
        HVX_Vector v_alpha_h,
        HVX_Vector v_beta_h,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    HVX_VectorPair vp_lo = Q6_Wuh_vunpack_Vub(v_lo);
    HVX_VectorPair vp_hi = Q6_Wuh_vunpack_Vub(v_hi);
    HVX_Vector v_p0_e    = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp_lo), v_alpha_h);
    HVX_Vector v_p0_o    = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp_hi), v_alpha_h);
    HVX_Vector v_p1_e    = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp_lo), v_alpha_h);
    HVX_Vector v_p1_o    = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp_hi), v_alpha_h);
    HVX_Vector v_b_01, v_b_23;
    int_hmx_pack_q6_pair(v_p0_o, v_p0_e, v_p1_o, v_p1_e, &v_b_01, &v_b_23);
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, Q6_Vh_vshuff_Vh(v_b_01), Q6_Vh_vshuff_Vh(v_b_23), v_sum_w, v_l1_w);
}

static inline void int_hmx_rescale_asym_pair_wb(
        HVX_Vector v_q0,
        HVX_Vector v_q1,
        HVX_Vector v_alpha_h,
        HVX_Vector v_beta_h,
        HVX_Vector * restrict tile_dst,
        int pair_idx,
        HVX_Vector * restrict v_sum_w,
        HVX_Vector * restrict v_l1_w) {

    const HVX_Vector v_one_s8 = Q6_Vb_vsplat_R(1);
    HVX_VectorPair vp0        = Q6_Wh_vmpy_VbVb(v_q0, v_one_s8);
    HVX_VectorPair vp1        = Q6_Wh_vmpy_VbVb(v_q1, v_one_s8);
    HVX_Vector v_p0_e         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp0), v_alpha_h);
    HVX_Vector v_p0_o         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp0), v_alpha_h);
    HVX_Vector v_p1_e         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_lo_W(vp1), v_alpha_h);
    HVX_Vector v_p1_o         = Q6_Vh_vmpyiacc_VhVhVh(v_beta_h, Q6_V_hi_W(vp1), v_alpha_h);
    HVX_Vector v_W0, v_W1;
    int_hmx_pack_q6_pair(v_p0_o, v_p0_e, v_p1_o, v_p1_e, &v_W0, &v_W1);
    int_hmx_store_and_accum_pair(tile_dst, pair_idx, v_W0, v_W1, v_sum_w, v_l1_w);
}

static inline void int_hmx_iq2xxs_decode_pair(
        HVX_Vector v_c2,
        HVX_Vector v_sgn,
        int l,
        HVX_Vector v_nib_lut,
        HVX_Vector * restrict v_w0_out,
        HVX_Vector * restrict v_w1_out) {

    const HVX_Vector v_zero     = Q6_V_vzero();
    const HVX_Vector v_one      = Q6_V_vsplat_R(1);
    const HVX_Vector v_sgn_mask = Q6_V_vsplat_R(IQ2XXS_SIGN_MASK);

    HVX_Vector v_s7     = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v_sgn, IQ2XXS_SIGN_BITS * l), v_sgn_mask);
    HVX_Vector v_t      = Q6_Vh_vpopcount_Vh(v_s7);
    HVX_Vector v_s      = Q6_V_vor_VV(v_s7, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_t, v_one), IQ2XXS_PARITY_SHIFT));
    HVX_Vector v_s16    = Q6_V_vor_VV(v_s, Q6_Vw_vasl_VwR(v_s, 8));
    HVX_Vector v_rep    = Q6_V_vor_VV(v_s16, Q6_Vw_vasl_VwR(v_s16, 16));
    HVX_VectorPred q_lo = Q6_Q_vand_VR(v_rep, IQ2XXS_SIGN_PRED_LO);
    HVX_VectorPred q_hi = Q6_Q_vand_VR(v_rep, IQ2XXS_SIGN_PRED_HI);

    HVX_Vector v_d      = (l & 1) ? Q6_Vh_vshuffo_VhVh(v_c2, v_c2) : Q6_Vh_vshuffe_VhVh(v_c2, v_c2);
    HVX_Vector v_n      = Q6_V_vor_VV(Q6_V_vand_VV(v_d, Q6_V_vsplat_R(IQ2XXS_NIB_LO_MASK)),
                                      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v_d, 4), Q6_V_vsplat_R(IQ2XXS_NIB_HI_MASK)));
    HVX_VectorPair p_g  = Q6_Wh_vlut16_VbVhR(v_n, v_nib_lut, 0);
    HVX_Vector v_g0     = Q6_V_lo_W(p_g);
    HVX_Vector v_g1     = Q6_V_hi_W(p_g);

    *v_w0_out = Q6_V_vmux_QVV(q_lo, Q6_Vb_vsub_VbVb(v_zero, v_g0), v_g0);
    *v_w1_out = Q6_V_vmux_QVV(q_hi, Q6_Vb_vsub_VbVb(v_zero, v_g1), v_g1);
}

static inline HVX_Vector int_hmx_iq2xxs_gather_d(const uint8_t * restrict sb, uint32_t aligned_tile_size) {
    HVX_Vector v_dd = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (int j = 0; j < IQ2XXS_TILES_PER_SB; j++) {
        HVX_Vector v = hvx_vmem(sb + (size_t) j * aligned_tile_size + IQ2XXS_D_SLICE_OFF);
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(IQ2XXS_D_SLICE_BYTES * j), v_dd,
                             Q6_V_vror_VR(v, (VLEN - IQ2XXS_D_SLICE_BYTES * j) & (VLEN - 1)));
    }
    return v_dd;
}

static __attribute__((noinline)) void rescale_iq2_xxs_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_abs_mask_sf = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector v_one_w       = Q6_V_vsplat_R(1);
    const HVX_Vector v_one_s8      = Q6_Vb_vsplat_R(1);
    const HVX_Vector v12M_sf       = Q6_V_vsplat_R(0x4B400000);
    const HVX_Vector v_nib_lut     = hvx_vmem(iq2xxs_vtcm_lut);

    HVX_Vector v_max_sf = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += IQ2XXS_TILES_PER_SB) {
        const uint8_t * sb      = src_base + (size_t) sb_kt * aligned_tile_size;
        HVX_Vector v_d_sf       = Q6_V_lo_W(hvx_vec_f16_to_f32(int_hmx_iq2xxs_gather_d(sb, aligned_tile_size)));
        HVX_Vector v_d_abs_sf   = Q6_V_vand_VV(v_d_sf, v_abs_mask_sf);
        HVX_Vector v_max_sc_u32 = Q6_V_vzero();

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < IQ2XXS_TILES_PER_SB; ++j) {
            const uint8_t * tile_src = sb + (size_t) j * aligned_tile_size;
            HVX_Vector v_sgn         = Q6_V_hi_W(Q6_W_vdeal_VVR(hvx_vmem(tile_src + VLEN), hvx_vmem(tile_src), -IQ2XXS_WORD_BYTES));
            v_max_sc_u32             = Q6_Vw_vmax_VwVw(v_max_sc_u32, Q6_Vuw_vlsr_VuwR(v_sgn, IQ2XXS_SCALE_SHIFT));
        }

        HVX_Vector v_max_ls = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(v_max_sc_u32, 1), v_one_w);
        v_max_sf            = Q6_Vsf_vmax_VsfVsf(v_max_sf, hvx_vec_mul_f32_f32(v_d_abs_sf, Q6_Vsf_equals_Vw(v_max_ls)));
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_max_sf, 8128.0f / 43.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();
    HVX_Vector v_sgn         = iq2xxs_gather_issue(src_base);

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += IQ2XXS_TILES_PER_SB) {
        const uint8_t * sb    = src_base + (size_t) sb_kt * aligned_tile_size;
        HVX_Vector v_d_qf     = hvx_vec_f32_to_qf32(Q6_V_lo_W(hvx_vec_f16_to_f32(int_hmx_iq2xxs_gather_d(sb, aligned_tile_size))));
        HVX_Vector v_d_sc_qf  = Q6_Vqf32_vmpy_Vqf32Vqf32(v_d_qf, v_scale_q6_qf);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < IQ2XXS_TILES_PER_SB; ++j) {
            const uint32_t  kt       = sb_kt + j;
            const uint8_t * tile_src = sb + (size_t) j * aligned_tile_size;
            HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);

            HVX_Vector v_c[2];
            iq2xxs_gather_wait(tile_src, v_c);
            const HVX_Vector v_sgn_cur = v_sgn;
            if (kt + 1 < tiles_per_group) {
                v_sgn = iq2xxs_gather_issue(tile_src + aligned_tile_size);
            }

            HVX_Vector v_ls      = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vuw_vlsr_VuwR(v_sgn_cur, IQ2XXS_SCALE_SHIFT), 1), v_one_w);
            HVX_Vector v_ls_qf   = hvx_vec_f32_to_qf32(Q6_Vsf_equals_Vw(v_ls));
            HVX_Vector v_m_bias  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(v_d_sc_qf, v_ls_qf), v12M_sf));
            HVX_Vector v_alpha_h = Q6_Vh_vshuffe_VhVh(v_m_bias, v_m_bias);

            #pragma clang loop pipeline(disable) unroll(disable)
            for (int i = 0; i < 4; ++i) {
                HVX_Vector v_w0, v_w1;
                int_hmx_iq2xxs_decode_pair((i < 2) ? v_c[0] : v_c[1], v_sgn_cur, i, v_nib_lut, &v_w0, &v_w1);
                HVX_VectorPair vp0 = Q6_Wh_vmpy_VbVb(v_w0, v_one_s8);
                HVX_VectorPair vp1 = Q6_Wh_vmpy_VbVb(v_w1, v_one_s8);
                HVX_Vector v_p0_e  = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(vp0), v_alpha_h);
                HVX_Vector v_p0_o  = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(vp0), v_alpha_h);
                HVX_Vector v_p1_e  = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(vp1), v_alpha_h);
                HVX_Vector v_p1_o  = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(vp1), v_alpha_h);
                HVX_Vector v_W0, v_W1;
                int_hmx_pack_q6_pair(v_p0_o, v_p0_e, v_p1_o, v_p1_e, &v_W0, &v_W1);
                int_hmx_store_and_accum_pair(tile_dst, i, v_W0, v_W1, &v_sum_w, &v_l1_w);
            }
        }
    }

    const float scale_const = is_w8a8 ? (0.125f * 43.0f / 127.0f) : (32.0f * 43.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_max_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q4_1_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_0f   = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(15.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        v_wmax_sf                = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_qmax_qf);
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        HVX_Vector v_alpha_h, v_beta_h;
        int_hmx_asym_alpha_beta_cp(Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_scale_q6_qf, &v_alpha_h, &v_beta_h);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_0f);
            HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);
            int_hmx_rescale_asym_pair_cp(v_lo, v_hi, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (1.0f / 127.0f) : (256.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_wmax_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static __attribute__((noinline)) void rescale_q5_k_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector mask_0f   = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(31.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_VectorPair vp_dm_qf  = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        v_wmax_sf                = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_qmax_qf);
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t kt = 0; kt < tiles_per_group; ++kt) {
        const uint8_t * tile_src = src_base + (size_t) kt * aligned_tile_size;
        HVX_Vector    * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
        HVX_Vector      v_plane  = hvx_vmem(tile_src + 640);
        HVX_VectorPair  vp_dm_qf = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(tile_src + 512), v_one_f16);
        HVX_Vector v_alpha_h, v_beta_h;
        int_hmx_asym_alpha_beta_cp(Q6_V_lo_W(vp_dm_qf), Q6_V_hi_W(vp_dm_qf), v_scale_q6_qf, &v_alpha_h, &v_beta_h);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (int i = 0; i < 4; ++i) {
            HVX_Vector vq   = hvx_vmem(tile_src + i * 128);
            HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(vq, mask_0f), v_plane, 2 * i + 0);
            HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(vq, 4),  v_plane, 2 * i + 1);
            int_hmx_rescale_asym_pair_cp(v_lo, v_hi, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
        }
    }

    const float scale_const = is_w8a8 ? (1.0f / 127.0f) : (256.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_wmax_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static inline void int_hmx_q2_k_load_dmin(
        const uint8_t * restrict sb,
        uint32_t aligned_tile_size,
        HVX_Vector * restrict v_d,
        HVX_Vector * restrict v_m) {

    HVX_Vector v_dd = Q6_V_vzero();
    #pragma clang loop pipeline(disable) unroll(disable)
    for (int j = 0; j < 8; j++) {
        HVX_Vector v = hvx_vmem(sb + (size_t) j * aligned_tile_size + 256);
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }
    HVX_VectorPred q_lo = Q6_Q_vsetq_R(64);
    HVX_Vector     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

static __attribute__((noinline)) void rescale_q2_k_weight_superblock(
        const uint8_t * restrict src_base,
        uint8_t * restrict dst_wt,
        float * restrict dst_scales,
        HVX_Vector * restrict bias_out,
        uint32_t tiles_per_group,
        uint32_t aligned_tile_size,
        bool is_w8a8) {

    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_qmax_qf = hvx_vec_f32_to_qf32(hvx_vec_splat_f32(3.0f));

    HVX_Vector v_wmax_sf = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += 8) {
        const uint8_t * sb = src_base + (size_t) sb_kt * aligned_tile_size;
        HVX_Vector v_d, v_dm;
        int_hmx_q2_k_load_dmin(sb, aligned_tile_size, &v_d, &v_dm);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < 8; ++j) {
            const uint8_t * tile_src = sb + (size_t) j * aligned_tile_size;
            HVX_Vector v_D, v_M;
            q2_k_unpack_scales(tile_src, v_d, v_dm, &v_D, &v_M);
            HVX_VectorPair vp_d_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_D), v_one_f16);
            HVX_VectorPair vp_m_qf = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_M), v_one_f16);
            v_wmax_sf = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_lo_W(vp_d_qf), Q6_V_lo_W(vp_m_qf), v_qmax_qf);
            v_wmax_sf = int_hmx_asym_update_wmax_qf(v_wmax_sf, Q6_V_hi_W(vp_d_qf), Q6_V_hi_W(vp_m_qf), v_qmax_qf);
        }
    }

    HVX_Vector v_scale_q6_qf = int_hmx_scale_qf_from_max_f32(v_wmax_sf, 8128.0f);
    HVX_Vector v_sum_w       = Q6_V_vzero();
    HVX_Vector v_l1_w        = Q6_V_vzero();

    #pragma clang loop pipeline(disable) unroll(disable)
    for (uint32_t sb_kt = 0; sb_kt < tiles_per_group; sb_kt += 8) {
        const uint8_t * sb = src_base + (size_t) sb_kt * aligned_tile_size;
        HVX_Vector v_d, v_dm;
        int_hmx_q2_k_load_dmin(sb, aligned_tile_size, &v_d, &v_dm);

        #pragma clang loop pipeline(disable) unroll(disable)
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t     kt       = sb_kt + j;
            const uint8_t    * tile_src = sb + (size_t) j * aligned_tile_size;
            const HVX_Vector * vptr     = (const HVX_Vector *) tile_src;
            HVX_Vector       * tile_dst = (HVX_Vector *) (dst_wt + (size_t) kt * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);

            HVX_Vector v_D, v_M;
            q2_k_unpack_scales(tile_src, v_d, v_dm, &v_D, &v_M);

            #pragma clang loop pipeline(disable) unroll(disable)
            for (int sub = 0; sub < 2; ++sub) {
                HVX_Vector v_D_sub = (sub == 0) ? v_D : Q6_V_vror_VR(v_D, 64);
                HVX_Vector v_M_sub = (sub == 0) ? v_M : Q6_V_vror_VR(v_M, 64);
                HVX_Vector v_d_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_D_sub), v_one_f16));
                HVX_Vector v_m_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(v_M_sub), v_one_f16));
                HVX_Vector v_alpha_h, v_beta_h;
                int_hmx_asym_alpha_beta_wb(v_d_qf, v_m_qf, v_scale_q6_qf, &v_alpha_h, &v_beta_h);

                const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
                for (int p = 0; p < 2; ++p) {
                    const int  i    = 2 * sub + p;
                    HVX_Vector v_q0 = unpack_q3_k_low2(vptr, 2 * i + 0, mask_03);
                    HVX_Vector v_q1 = unpack_q3_k_low2(vptr, 2 * i + 1, mask_03);
                    int_hmx_rescale_asym_pair_wb(v_q0, v_q1, v_alpha_h, v_beta_h, tile_dst, i, &v_sum_w, &v_l1_w);
                }
            }
        }
    }

    const float scale_const = is_w8a8 ? (1.0f / 127.0f) : (256.0f / 127.0f);
    emit_int_hmx_scales_and_bias(v_wmax_sf, scale_const, v_sum_w, v_l1_w, dst_scales, bias_out, is_w8a8);
}

static void rescale_int_weight_to_int8_task(
        const int_weight_rescale_state_t * state,
        uint32_t start_unit,
        uint32_t end_unit) {

    const bool     is_w8a8     = htp_mm_kernel_is_hmx_w8a8(state->kernel_type);
    const size_t   bias_stride = is_w8a8 ? (2 * HTP_MM_HMX_BIAS_BANK_SIZE) : HTP_MM_HMX_BIAS_BANK_SIZE;
    const uint32_t raw_shift   = (state->weight_type == HTP_TYPE_Q1_0) ? 2u
                               : ((state->weight_type == HTP_TYPE_Q2_0) ? 1u : 0u);

    for (uint32_t u = start_unit; u < end_unit; ++u) {
        const uint32_t ct = fastdiv(u, &state->n_groups_div);
        const uint32_t g  = u - ct * state->n_groups;

        const size_t    kt_start   = (size_t) ct * state->n_k_tiles + (size_t) g * state->tiles_per_group;
        const uint8_t * src_base   = state->src_raw + (kt_start >> raw_shift) * state->aligned_tile_size;
        uint8_t       * dst_wt     = state->dst_wt_base + kt_start * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
        float         * dst_scales = state->dst_scales_base + (size_t) u * 32;
        HVX_Vector    * bias_out   = (HVX_Vector *) (state->dst_bias_base + (size_t) u * bias_stride);

        switch (state->weight_type) {
            case HTP_TYPE_Q4_0:
                rescale_q4_0_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q8_0:
                rescale_q8_0_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_IQ4_NL:
                rescale_iq4_nl_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_MXFP4:
                rescale_mxfp4_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q5_0:
                rescale_q5_0_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q1_0:
                rescale_q1_0_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q2_0:
                rescale_q2_0_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_NVFP4:
                rescale_nvfp4_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q3_K:
                rescale_q3_k_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q6_K:
                rescale_q6_k_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_IQ2_XXS:
                rescale_iq2_xxs_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q4_1:
            case HTP_TYPE_Q4_K:
                rescale_q4_1_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q5_1:
            case HTP_TYPE_Q5_K:
                rescale_q5_k_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            case HTP_TYPE_Q2_K:
                rescale_q2_k_weight_superblock(src_base, dst_wt, dst_scales, bias_out, state->tiles_per_group, state->aligned_tile_size, is_w8a8);
                break;
            default:
                break;
        }
    }
}

static inline void quantize_f16_quad_step_to_u8_sm(
        uint8_t * restrict tile_row_base,
        uint32_t kt,
        uint32_t rg,
        HVX_Vector vh0,
        HVX_Vector vh1,
        HVX_Vector vh2,
        HVX_Vector vh3,
        HVX_Vector v_inv0_qf,
        HVX_Vector v_inv1_qf,
        HVX_Vector v_inv2_qf,
        HVX_Vector v_inv3_qf,
        HVX_Vector v_one_f16,
        HVX_Vector v_bias_u8,
        HVX_Vector v_min_u8,
        HVX_Vector v_max_u8) {

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh1), v_one_f16);
    HVX_VectorPair vp2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh2), v_one_f16);
    HVX_VectorPair vp3 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh3), v_one_f16);

    {
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp2), v_inv2_qf), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp3), v_inv3_qf), v_bias_u8));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
    {
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp2), v_inv2_qf), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp3), v_inv3_qf), v_bias_u8));
        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HTP_MM_HMX_TILE_SIZE))[rg] = Q6_Vb_vshuffe_VbVb(vh13, vh02);
    }
}

static inline void quantize_f16_pair_step_to_u16_sm(
        uint16_t * restrict tile_row_base,
        uint32_t kt,
        uint32_t r1,
        HVX_Vector vh0,
        HVX_Vector vh1,
        HVX_Vector v_inv0_qf,
        HVX_Vector v_inv1_qf,
        HVX_Vector v_one_f16) {

    HVX_VectorPair vp0 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh0), v_one_f16);
    HVX_VectorPair vp1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vh1), v_one_f16);

    ((HVX_Vector *) (tile_row_base + (size_t) (kt + 0) * HTP_MM_HMX_TILE_N_ELMS))[r1] =
        hvx_vec_u16_zp32768_from_qf32_pair_shuff(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp0), v_inv0_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(vp1), v_inv1_qf));
    ((HVX_Vector *) (tile_row_base + (size_t) (kt + 1) * HTP_MM_HMX_TILE_N_ELMS))[r1] =
        hvx_vec_u16_zp32768_from_qf32_pair_shuff(
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp0), v_inv0_qf),
            Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(vp1), v_inv1_qf));
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 4-row FP32 group into W8A8 SM 64x32 tiles.
static __attribute__((noinline)) void quantize_act_superblock_quad_f32_to_u8_sm(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const float * restrict p0,
        const float * restrict p1,
        const float * restrict p2,
        const float * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));
    assert(hex_is_aligned(p2 + k_start, 128));
    assert(hex_is_aligned(p3 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~31u)) : k_start);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();
    HVX_Vector vmax2 = Q6_V_vzero();
    HVX_Vector vmax3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        HVX_Vector vx2 = hvx_vmem(p2 + c);
        HVX_Vector vx3 = hvx_vmem(p3 + c);
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
        vmax2 = Q6_Vsf_vmax_VsfVsf(vmax2, hvx_vec_abs_f32(vx2));
        vmax3 = Q6_Vsf_vmax_VsfVsf(vmax3, hvx_vec_abs_f32(vx3));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c    = c_end_full;
        const uint32_t rem  = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
        HVX_Vector vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        HVX_Vector vx2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
        HVX_Vector vx3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
        vmax2 = Q6_Vsf_vmax_VsfVsf(vmax2, hvx_vec_abs_f32(vx2));
        vmax3 = Q6_Vsf_vmax_VsfVsf(vmax3, hvx_vec_abs_f32(vx3));
    }

    vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(vmax0) : Q6_V_vzero();
    vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(vmax1) : Q6_V_vzero();
    vmax2 = (valid_rows > 2) ? hvx_vec_reduce_max_f32(vmax2) : Q6_V_vzero();
    vmax3 = (valid_rows > 3) ? hvx_vec_reduce_max_f32(vmax3) : Q6_V_vzero();

    const HVX_Vector v_inv_127 = hvx_vec_splat_f32(1.0f / 127.0f);
    const HVX_Vector v_127     = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps     = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0 = hvx_vec_mul_f32_f32(vmax0, v_inv_127);
    HVX_Vector vd1 = hvx_vec_mul_f32_f32(vmax1, v_inv_127);
    HVX_Vector vd2 = hvx_vec_mul_f32_f32(vmax2, v_inv_127);
    HVX_Vector vd3 = hvx_vec_mul_f32_f32(vmax3, v_inv_127);
    HVX_Vector v_d02   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_a_out, 4 * sizeof(float), v_d0123);

    HVX_Vector v_inv0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero());
    HVX_Vector v_inv1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero());
    HVX_Vector v_inv2 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero());
    HVX_Vector v_inv3 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero());

    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080); // 12582912.0f + 128.0f
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        HVX_Vector vx2 = hvx_vmem(p2 + c);
        HVX_Vector vx3 = hvx_vmem(p3 + c);

        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3), v_bias_u8));

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vshuffe_VbVb(vh13, vh02);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
    for (uint32_t c = c_end_full; c < k_end; c += 32) {
        HVX_Vector vx0 = Q6_V_vzero();
        HVX_Vector vx1 = Q6_V_vzero();
        HVX_Vector vx2 = Q6_V_vzero();
        HVX_Vector vx3 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem        = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
            vx2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
            vx3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        }
        HVX_Vector y0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0), v_bias_u8));
        HVX_Vector y1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1), v_bias_u8));
        HVX_Vector y2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx2, v_inv2), v_bias_u8));
        HVX_Vector y3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vx3, v_inv3), v_bias_u8));

        HVX_Vector vh02 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y2, y0), v_min_u8), v_max_u8);
        HVX_Vector vh13 = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(Q6_Vh_vshuffe_VhVh(y3, y1), v_min_u8), v_max_u8);
        HVX_Vector v_u8 = Q6_Vb_vshuffe_VbVb(vh13, vh02);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_u8;
    }
}

// Quantize one superblock g (k_group columns, multiple of 128) of a 4-row FP16 group into W8A8 SM 64x32 tiles.
static __attribute__((noinline)) void quantize_act_superblock_quad_f16_to_u8_sm(
        uint8_t * restrict tile_row_base,
        float * restrict d_a_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        const __fp16 * restrict p2,
        const __fp16 * restrict p3,
        uint32_t rg,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));
    assert(hex_is_aligned(p2 + k_start, 128));
    assert(hex_is_aligned(p3 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~63u)) : k_start);
    const HVX_Vector v_one_f16 = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16 = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();
    HVX_Vector vmax_hf2 = Q6_V_vzero();
    HVX_Vector vmax_hf3 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        HVX_Vector vh2 = hvx_vmem(p2 + c);
        HVX_Vector vh3 = hvx_vmem(p3 + c);
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
        vmax_hf2 = Q6_Vhf_vmax_VhfVhf(vmax_hf2, Q6_V_vand_VV(vh2, v_abs_f16));
        vmax_hf3 = Q6_Vhf_vmax_VhfVhf(vmax_hf3, Q6_V_vand_VV(vh3, v_abs_f16));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c     = c_end_full;
        const uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
        HVX_VectorPred mask  = Q6_Q_vsetq2_R(rem_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        HVX_Vector vh2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
        HVX_Vector vh3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
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

    const HVX_Vector v_inv_127 = hvx_vec_splat_f32(1.0f / 127.0f);
    const HVX_Vector v_127     = hvx_vec_splat_f32(127.0f);
    const HVX_Vector v_eps     = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0 = hvx_vec_mul_f32_f32(vmax0, v_inv_127);
    HVX_Vector vd1 = hvx_vec_mul_f32_f32(vmax1, v_inv_127);
    HVX_Vector vd2 = hvx_vec_mul_f32_f32(vmax2, v_inv_127);
    HVX_Vector vd3 = hvx_vec_mul_f32_f32(vmax3, v_inv_127);
    HVX_Vector v_d02   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
    HVX_Vector v_d13   = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
    HVX_Vector v_d0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
    hvx_vec_store_u(d_a_out, 4 * sizeof(float), v_d0123);

    HVX_Vector v_inv0_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_127), Q6_V_vzero()));
    HVX_Vector v_inv1_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_127), Q6_V_vzero()));
    HVX_Vector v_inv2_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax2, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax2), v_127), Q6_V_vzero()));
    HVX_Vector v_inv3_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax3, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax3), v_127), Q6_V_vzero()));

    const HVX_Vector v_bias_u8 = Q6_V_vsplat_R(0x4B400080);
    const HVX_Vector v_min_u8  = Q6_Vh_vsplat_R(1);
    const HVX_Vector v_max_u8  = Q6_Vh_vsplat_R(255);

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        HVX_Vector vh2 = hvx_vmem(p2 + c);
        HVX_Vector vh3 = hvx_vmem(p3 + c);
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_bias_u8, v_min_u8, v_max_u8);
    }
    for (uint32_t c = c_end_full; c < k_end; c += 64) {
        HVX_Vector vh0 = Q6_V_vzero();
        HVX_Vector vh1 = Q6_V_vzero();
        HVX_Vector vh2 = Q6_V_vzero();
        HVX_Vector vh3 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem_b      = (k_valid - c) * sizeof(__fp16);
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem_b);
            vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
            vh2 = Q6_V_vmux_QVV(mask, hvx_vmem(p2 + c), Q6_V_vzero());
            vh3 = Q6_V_vmux_QVV(mask, hvx_vmem(p3 + c), Q6_V_vzero());
        }
        quantize_f16_quad_step_to_u8_sm(tile_row_base, c >> 5, rg, vh0, vh1, vh2, vh3, v_inv0_qf, v_inv1_qf, v_inv2_qf, v_inv3_qf, v_one_f16, v_bias_u8, v_min_u8, v_max_u8);
    }
}

// Quantize one superblock g (k_group columns, multiple of 32) of a 2-row FP32 pair into W8A16 SM 32x32 uint16 tiles.
static __attribute__((noinline)) void quantize_act_superblock_pair_f32_to_u16_sm(
        uint16_t * restrict tile_row_base,
        float * restrict d_a_out,
        const float * restrict p0,
        const float * restrict p1,
        uint32_t r1,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));

    const uint32_t k_end      = k_start + k_group;
    const uint32_t c_end_full = (k_end <= k_valid) ? k_end
                              : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~31u)) : k_start);
    HVX_Vector vmax0 = Q6_V_vzero();
    HVX_Vector vmax1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0 = hvx_vmem(p0 + c);
        HVX_Vector vx1 = hvx_vmem(p1 + c);
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c    = c_end_full;
        const uint32_t rem  = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
        HVX_Vector vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        vmax0 = Q6_Vsf_vmax_VsfVsf(vmax0, hvx_vec_abs_f32(vx0));
        vmax1 = Q6_Vsf_vmax_VsfVsf(vmax1, hvx_vec_abs_f32(vx1));
    }

    vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(vmax0) : Q6_V_vzero();
    vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(vmax1) : Q6_V_vzero();

    const HVX_Vector v_inv_32766 = hvx_vec_splat_f32(1.0f / 32766.0f);
    const HVX_Vector v_32766     = hvx_vec_splat_f32(32766.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_inv_32766);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_inv_32766);
    HVX_Vector v_d01 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd1, vd0, -4));
    hvx_vec_store_u(d_a_out, 2 * sizeof(float), v_d01);

    HVX_Vector v_inv0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_32766), Q6_V_vzero());
    HVX_Vector v_inv1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_32766), Q6_V_vzero());

    for (uint32_t c = k_start; c < c_end_full; c += 32) {
        HVX_Vector vx0   = hvx_vmem(p0 + c);
        HVX_Vector vx1   = hvx_vmem(p1 + c);
        HVX_Vector vqf0  = Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0);
        HVX_Vector vqf1  = Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1);
        HVX_Vector v_u16 = hvx_vec_u16_zp32768_from_qf32_pair_shuff(vqf0, vqf1);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
    }
    for (uint32_t c = c_end_full; c < k_end; c += 32) {
        HVX_Vector vx0 = Q6_V_vzero();
        HVX_Vector vx1 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem        = k_valid - c;
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(float));
            vx0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vx1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        }
        HVX_Vector vqf0  = Q6_Vqf32_vmpy_VsfVsf(vx0, v_inv0);
        HVX_Vector vqf1  = Q6_Vqf32_vmpy_VsfVsf(vx1, v_inv1);
        HVX_Vector v_u16 = hvx_vec_u16_zp32768_from_qf32_pair_shuff(vqf0, vqf1);

        uint32_t kt = c >> 5;
        ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_u16;
    }
}

// Quantize one superblock g (k_group columns, multiple of 128) of a 2-row FP16 pair into W8A16 SM 32x32 uint16 tiles.
static __attribute__((noinline)) void quantize_act_superblock_pair_f16_to_u16_sm(
        uint16_t * restrict tile_row_base,
        float * restrict d_a_out,
        const __fp16 * restrict p0,
        const __fp16 * restrict p1,
        uint32_t r1,
        uint32_t k_start,
        uint32_t k_group,
        uint32_t k_valid,
        uint32_t valid_rows) {

    assert(hex_is_aligned(p0 + k_start, 128));
    assert(hex_is_aligned(p1 + k_start, 128));

    const uint32_t k_end          = k_start + k_group;
    const uint32_t c_end_full     = (k_end <= k_valid) ? k_end
                                  : ((k_valid > k_start) ? (k_start + ((k_valid - k_start) & ~63u)) : k_start);
    const HVX_Vector v_one_f16    = hvx_vec_splat_f16(1.0f);
    const HVX_Vector v_abs_f16    = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector vmax_hf0 = Q6_V_vzero();
    HVX_Vector vmax_hf1 = Q6_V_vzero();

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
    }
    if (c_end_full < k_end && c_end_full < k_valid) {
        const uint32_t c     = c_end_full;
        const uint32_t rem_b = (k_valid - c) * sizeof(__fp16);
        HVX_VectorPred mask  = Q6_Q_vsetq2_R(rem_b);
        HVX_Vector vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
        HVX_Vector vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vand_VV(vh0, v_abs_f16));
        vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vand_VV(vh1, v_abs_f16));
    }

    vmax_hf0 = Q6_Vhf_vmax_VhfVhf(vmax_hf0, Q6_V_vror_VR(vmax_hf0, 64));
    vmax_hf1 = Q6_Vhf_vmax_VhfVhf(vmax_hf1, Q6_V_vror_VR(vmax_hf1, 64));

    HVX_Vector vmax0 = (valid_rows > 0) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf0), v_one_f16)))) : Q6_V_vzero();
    HVX_Vector vmax1 = (valid_rows > 1) ? hvx_vec_reduce_max_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vmax_hf1), v_one_f16)))) : Q6_V_vzero();

    const HVX_Vector v_inv_32766 = hvx_vec_splat_f32(1.0f / 32766.0f);
    const HVX_Vector v_32766     = hvx_vec_splat_f32(32766.0f);
    const HVX_Vector v_eps       = hvx_vec_splat_f32(1e-30f);

    HVX_Vector vd0   = hvx_vec_mul_f32_f32(vmax0, v_inv_32766);
    HVX_Vector vd1   = hvx_vec_mul_f32_f32(vmax1, v_inv_32766);
    HVX_Vector v_d01 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd1, vd0, -4));
    hvx_vec_store_u(d_a_out, 2 * sizeof(float), v_d01);

    HVX_Vector v_inv0_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax0, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax0), v_32766), Q6_V_vzero()));
    HVX_Vector v_inv1_qf = hvx_vec_f32_to_qf32(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(vmax1, v_eps), hvx_vec_mul_f32_f32(hvx_vec_inverse_f32(vmax1), v_32766), Q6_V_vzero()));

    for (uint32_t c = k_start; c < c_end_full; c += 64) {
        HVX_Vector vh0 = hvx_vmem(p0 + c);
        HVX_Vector vh1 = hvx_vmem(p1 + c);
        quantize_f16_pair_step_to_u16_sm(tile_row_base, c >> 5, r1, vh0, vh1, v_inv0_qf, v_inv1_qf, v_one_f16);
    }
    for (uint32_t c = c_end_full; c < k_end; c += 64) {
        HVX_Vector vh0 = Q6_V_vzero();
        HVX_Vector vh1 = Q6_V_vzero();
        if (c < k_valid) {
            uint32_t rem_b      = (k_valid - c) * sizeof(__fp16);
            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem_b);
            vh0 = Q6_V_vmux_QVV(mask, hvx_vmem(p0 + c), Q6_V_vzero());
            vh1 = Q6_V_vmux_QVV(mask, hvx_vmem(p1 + c), Q6_V_vzero());
        }
        quantize_f16_pair_step_to_u16_sm(tile_row_base, c >> 5, r1, vh0, vh1, v_inv0_qf, v_inv1_qf, v_one_f16);
    }
}

static __attribute__((noinline)) void copy_gathered_act_row_to_vtcm(
        uint8_t * restrict dst_row,
        const uint8_t * restrict src_row,
        size_t valid_bytes) {
    size_t b = 0;
    for (; b + 128 <= valid_bytes; b += 128) {
        hvx_vmem(dst_row + b) = hvx_vmemu(src_row + b);
    }
    if (b < valid_bytes) {
        hvx_vmem(dst_row + b) = hvx_vec_load_u(src_row + b, valid_bytes - b);
    }
}

static __attribute__((noinline)) void quantize_act_step_rows_to_int_tiles(
        uint8_t * restrict vtcm_act_dst,
        float * restrict vtcm_act_scales,
        const uint8_t * restrict curr_buf,
        size_t row_bytes,
        uint32_t r,
        uint32_t valid_rows,
        uint32_t mc_pad,
        uint32_t n_k_tiles,
        uint32_t k_group,
        uint32_t n_groups,
        uint32_t k_valid,
        bool is_w8a8,
        bool act_is_f16) {
    if (is_w8a8) {
        const uint32_t r0 = r >> 6;
        const uint32_t rg = (r & 63u) >> 2;
        uint8_t * tile_row_base = vtcm_act_dst + (size_t) r0 * n_k_tiles * HTP_MM_HMX_TILE_SIZE;
        if (valid_rows == 0) {
            const HVX_Vector v_zero = Q6_V_vzero();
            const HVX_Vector v_zp   = Q6_Vb_vsplat_R(0x80);
            for (uint32_t g = 0; g < n_groups; ++g) {
                hvx_vec_store_u(vtcm_act_scales + (size_t) g * mc_pad + r, 4 * sizeof(float), v_zero);
            }
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_SIZE))[rg] = v_zp;
            }
        } else {
            const uint8_t * p0 = curr_buf;
            const uint8_t * p1 = (valid_rows > 1) ? (curr_buf + 1 * row_bytes) : curr_buf;
            const uint8_t * p2 = (valid_rows > 2) ? (curr_buf + 2 * row_bytes) : curr_buf;
            const uint8_t * p3 = (valid_rows > 3) ? (curr_buf + 3 * row_bytes) : curr_buf;

            for (uint32_t g = 0; g < n_groups; ++g) {
                float * d_a_out = vtcm_act_scales + (size_t) g * mc_pad + r;
                const uint32_t k_start = g * k_group;
                if (act_is_f16) {
                    quantize_act_superblock_quad_f16_to_u8_sm(
                        tile_row_base, d_a_out,
                        (const __fp16 *) p0, (const __fp16 *) p1, (const __fp16 *) p2, (const __fp16 *) p3,
                        rg, k_start, k_group, k_valid, valid_rows);
                } else {
                    quantize_act_superblock_quad_f32_to_u8_sm(
                        tile_row_base, d_a_out,
                        (const float *) p0, (const float *) p1, (const float *) p2, (const float *) p3,
                        rg, k_start, k_group, k_valid, valid_rows);
                }
            }
        }
    } else {
        const uint32_t r0 = r >> 5;
        const uint32_t r1 = (r & 31u) >> 1;
        uint16_t * tile_row_base = (uint16_t *) vtcm_act_dst + (size_t) r0 * n_k_tiles * HTP_MM_HMX_TILE_N_ELMS;
        if (valid_rows == 0) {
            const HVX_Vector v_zero = Q6_V_vzero();
            const HVX_Vector v_zp   = Q6_Vh_vsplat_R(0x8000);
            for (uint32_t g = 0; g < n_groups; ++g) {
                hvx_vec_store_u(vtcm_act_scales + (size_t) g * mc_pad + r, 2 * sizeof(float), v_zero);
            }
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                ((HVX_Vector *) (tile_row_base + (size_t) kt * HTP_MM_HMX_TILE_N_ELMS))[r1] = v_zp;
            }
        } else {
            const uint8_t * p0 = curr_buf;
            const uint8_t * p1 = (valid_rows > 1) ? (curr_buf + 1 * row_bytes) : curr_buf;

            for (uint32_t g = 0; g < n_groups; ++g) {
                float * d_a_out = vtcm_act_scales + (size_t) g * mc_pad + r;
                const uint32_t k_start = g * k_group;
                if (act_is_f16) {
                    quantize_act_superblock_pair_f16_to_u16_sm(
                        tile_row_base, d_a_out,
                        (const __fp16 *) p0, (const __fp16 *) p1,
                        r1, k_start, k_group, k_valid, valid_rows);
                } else {
                    quantize_act_superblock_pair_f32_to_u16_sm(
                        tile_row_base, d_a_out,
                        (const float *) p0, (const float *) p1,
                        r1, k_start, k_group, k_valid, valid_rows);
                }
            }
        }
    }
}

static __attribute__((noinline)) void transfer_activation_chunk_to_int_gathered(
        uint8_t * restrict vtcm_act_dst,
        float * restrict vtcm_act_scales,
        const void * restrict src,
        uint8_t * restrict thread_act,
        uint32_t start_row,
        uint32_t row_start,
        uint32_t row_end,
        uint32_t row_stride,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t k_block,
        uint32_t k_valid,
        uint32_t k_group,
        uint32_t n_groups,
        int32_t kernel_type,
        uint32_t act_elem_size,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        uint32_t ne11,
        const struct fastdiv_values * ne11_div,
        size_t nb11,
        size_t nb12) {
    const bool     is_w8a8     = htp_mm_kernel_is_hmx_w8a8(kernel_type);
    const uint32_t step_rows   = is_w8a8 ? 4u : 2u;
    const bool     act_is_f16  = (act_elem_size == sizeof(__fp16));
    const size_t   row_bytes   = hex_align_up((size_t) k_block * act_elem_size, 128);
    const size_t   valid_bytes = (size_t) k_valid * act_elem_size;
    const uint32_t n_k_tiles   = k_block >> 5;

    for (uint32_t r = row_start; r < row_end; r += row_stride) {
        const uint32_t valid_rows = (r < n_rows) ? hex_smin(n_rows - r, step_rows) : 0u;
        for (uint32_t s = 0; s < valid_rows; ++s) {
            const struct mmid_row_mapping mapping = matrix_rows[(size_t) cur_a * mapping_stride + start_row + r + s];
            const uint32_t i11 = (ne11 > 1) ? fastmodulo(mapping.i1, ne11, ne11_div) : 0u;
            const uint8_t * src_row = (const uint8_t *) src + (size_t) i11 * nb11 + (size_t) mapping.i2 * nb12;
            copy_gathered_act_row_to_vtcm(thread_act + (size_t) s * row_bytes, src_row, valid_bytes);
        }
        quantize_act_step_rows_to_int_tiles(
            vtcm_act_dst, vtcm_act_scales, thread_act, row_bytes,
            r, valid_rows, mc_pad, n_k_tiles, k_group, n_groups, k_valid,
            is_w8a8, act_is_f16);
    }
}

// Core HMX W8A8 SM dot-product kernel over all superblocks g = 0..n_groups-1.
static void core_dot_chunk_w8a8_sm(
        uint8_t * restrict vtcm_out,
        const uint8_t * restrict vtcm_act,
        const uint8_t * restrict vtcm_weight_tiles,
        const uint8_t * restrict vtcm_weight_bias,
        uint32_t n_row_tiles_64,
        uint32_t n_col_tiles,
        uint32_t n_k_tiles,
        uint32_t n_groups,
        uint32_t tiles_per_group) {

    assert(hex_is_aligned(vtcm_out, HTP_MM_HMX_TILE_SIZE));
    const uint32_t cvt_ctrl_hi = hmx_cvt_ctrl(0, 0, 0, 1);
    const uint32_t cvt_ctrl_lo = hmx_cvt_ctrl(1, 0, 1, 0);
    const uint32_t out_cm_rng  = hmx_out_cm_range();

    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t kt_base = g * tiles_per_group;

        for (uint32_t j = 0; j < n_col_tiles; ++j) {
            const uint8_t * wt_base = vtcm_weight_tiles + ((size_t) j * n_k_tiles + kt_base) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            const uint8_t * bias_hi = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 0) * HTP_MM_HMX_BIAS_BANK_SIZE;
            const uint8_t * bias_lo = vtcm_weight_bias  + (((size_t) j * n_groups + g) * 2 + 1) * HTP_MM_HMX_BIAS_BANK_SIZE;
            hmx_load_bias_bank(bias_hi, 0);
            hmx_load_bias_bank(bias_lo, 1);

            for (uint32_t i = 0; i < n_row_tiles_64; ++i) {
                const uint8_t * act_base = vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_SIZE;

                const size_t out_idx = (((size_t) g * n_row_tiles_64 + i) * n_col_tiles + j) * 2;
                uint8_t * out_hi = vtcm_out + (out_idx + 0) * HTP_MM_HMX_TILE_SIZE;
                uint8_t * out_lo = vtcm_out + (out_idx + 1) * HTP_MM_HMX_TILE_SIZE;

                asm volatile(HMX_CLRACC_INT());

                uint32_t rem = tiles_per_group;
                const uint8_t * a_ptr = act_base;
                const uint8_t * w_ptr = wt_base;
                while (rem > 0) {
                    const uint32_t step  = (rem > 64) ? 64 : rem;
                    const uint32_t a_rng = hmx_act_sm_range(step);
                    const uint32_t w_rng = hmx_wt_b_range(step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
                    asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_SM("%0", "%1", "%2", "%3")
                                 : : "r"(a_ptr), "r"(a_rng), "r"(w_ptr), "r"(w_rng) : "memory");
                    a_ptr += step * HTP_MM_HMX_TILE_SIZE;
                    w_ptr += step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
                    rem   -= step;
                }

                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_hi) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_hi), "r"(out_cm_rng) : "memory");

                asm volatile(HMX_CVT_UB_ACC("%0") : : "r"(cvt_ctrl_lo) : "memory");
                asm volatile(HMX_STORE_CVT_CM("%0", "%1") : : "r"(out_lo), "r"(out_cm_rng) : "memory");
            }
        }
    }
}

// Core HMX W8A16 :2x1 dot-product kernel over all superblocks g = 0..n_groups-1.
static void core_dot_chunk_w8a16_2x1(
        uint16_t * restrict vtcm_out,
        const uint16_t * restrict vtcm_act,
        const uint8_t * restrict vtcm_weight_tiles,
        const uint8_t * restrict vtcm_weight_bias,
        uint32_t n_row_tiles_32,
        uint32_t n_col_tiles,
        uint32_t n_k_tiles,
        uint32_t n_groups,
        uint32_t tiles_per_group) {

    const uint32_t cvt_ctrl_16 = hmx_cvt_ctrl(0, 0, 0, 0);
    const uint32_t out_sm_rng1 = hmx_act_sm_range(1);

    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t kt_base = g * tiles_per_group;

        for (uint32_t j = 0; j < n_col_tiles; ++j) {
            const uint8_t * wt_base  = vtcm_weight_tiles + ((size_t) j * n_k_tiles + kt_base) * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
            const uint8_t * bias_ptr = vtcm_weight_bias  + ((size_t) j * n_groups + g) * HTP_MM_HMX_BIAS_BANK_SIZE;
            hmx_load_bias_bank(bias_ptr, 0);

            for (uint32_t i = 0; i < n_row_tiles_32; ++i) {
                const uint8_t * act_base = (const uint8_t *) (vtcm_act + ((size_t) i * n_k_tiles + kt_base) * HTP_MM_HMX_TILE_N_ELMS);
                uint16_t      * out_tile = vtcm_out + (((size_t) g * n_row_tiles_32 + i) * n_col_tiles + j) * HTP_MM_HMX_TILE_N_ELMS;

                asm volatile(HMX_CLRACC_INT());

                uint32_t rem = tiles_per_group;
                const uint8_t * a_ptr = act_base;
                const uint8_t * w_ptr = wt_base;
                while (rem > 0) {
                    const uint32_t step  = (rem > 64) ? 64 : rem;
                    const uint32_t a_rng = hmx_act_sm_range(step);
                    const uint32_t w_rng = hmx_wt_b_range(step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE);
                    asm volatile(HMX_LOAD_MPY_DEEP_U8_S8_SM("%0", "%1", "%2", "%3")
                                 : : "r"(a_ptr), "r"(a_rng), "r"(w_ptr), "r"(w_rng) : "memory");
                    a_ptr += step * HTP_MM_HMX_TILE_SIZE;
                    w_ptr += step * HTP_MM_HMX_WEIGHT_B_TILE_SIZE;
                    rem   -= step;
                }

                asm volatile(HMX_CVT_UH_ACC_2X1("%0") : : "r"(cvt_ctrl_16) : "memory");
                asm volatile(HMX_STORE_CVT_SM("%0", "%1") : : "r"(out_tile), "r"(out_sm_rng1) : "memory");
            }
        }
    }
}

static inline HVX_Vector hvx_qf32_mul_sf_scalar(HVX_Vector v_sf, uint32_t scalar_sf_bits) {
#if __HVX_ARCH__ >= 79
    return Q6_Vqf32_vmpy_VsfRsf(v_sf, (int32_t) scalar_sf_bits);
#else
    return Q6_Vqf32_vmpy_VsfVsf(v_sf, Q6_V_vsplat_R((int32_t) scalar_sf_bits));
#endif
}

static inline uint32_t (*htp_da_buf_ptr(uint32_t (*buf)[32]))[32] {
    asm("%0 = or(%0, #0)" : "+r"(buf));
    return buf;
}

// HVX Epilogue for W8A8 CM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a8_sm_to_fp32_col_chunk(
        float * restrict dst,
        const void * restrict src2,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64  = mc_pad / 64;
    const size_t group_stride    = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_80     = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);
    const bool   dst_aligned_nt  = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(float))) & 127u) == 0);
    const size_t src2_row_bytes  = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;
                float *         out_row0  = dst + r * dst_stride;
                const uint8_t * src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool      full_quad = (r + 3 < n_rows);

                if (dst_aligned_nt && full_quad && !src2) {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t    c0         = c >> 5;
                        const uint8_t * tp         = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                        const HVX_Vector v_lo_u8   = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                        const HVX_Vector v_dw_qf   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf));
                        HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf));
                        HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf));
                        HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf));

                        hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_out0);
                        hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_out1);
                        hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_out2);
                        hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_out3);
                    }
                } else {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t    c0         = c >> 5;
                        const uint8_t * tp         = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                        const HVX_Vector v_lo_u8   = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                        const HVX_Vector v_dw_qf   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
                        HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_out0);
                        } else {
                            hvx_vmemu(out_row0 + c + 0 * dst_stride) = v_out0;
                        }

                        if (r + 1 < n_rows) {
                            HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
                            HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_out1);
                            } else {
                                hvx_vmemu(out_row0 + c + 1 * dst_stride) = v_out1;
                            }
                        }
                        if (r + 2 < n_rows) {
                            HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
                            HVX_Vector v_out2    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_out2);
                            } else {
                                hvx_vmemu(out_row0 + c + 2 * dst_stride) = v_out2;
                            }
                        }
                        if (r + 3 < n_rows) {
                            HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
                            HVX_Vector v_out3    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_out3);
                            } else {
                                hvx_vmemu(out_row0 + c + 3 * dst_stride) = v_out3;
                            }
                        }
                    }
                }

                if (limit_c_aligned < limit_c) {
                    const size_t    c           = limit_c_aligned;
                    const size_t    c0          = c >> 5;
                    const size_t    valid_bytes = (limit_c - c) * sizeof(float);
                    const uint8_t * tp          = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
                    const HVX_Vector v_hi_s8    = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
                    const HVX_Vector v_lo_u8    = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
                    const HVX_Vector v_dw_qf    = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

                    HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                    HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                    HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
                    HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                    hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, v_out0);

                    if (r + 1 < n_rows) {
                        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
                        HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                        hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, v_out1);
                    }
                    if (r + 2 < n_rows) {
                        HVX_Vector v_f32_2   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
                        HVX_Vector v_out2    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                        hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, v_out2);
                    }
                    if (r + 3 < n_rows) {
                        HVX_Vector v_f32_3   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));
                        HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
                        HVX_Vector v_out3    = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                        hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, v_out3);
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            float *         out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0      = c >> 5;
                const size_t c0_glob = c0_first + c0;

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();
                HVX_Vector v_acc2_qf = Q6_V_vzero();
                HVX_Vector v_acc3_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g         = g_base + gi;
                        const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
                        const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                        v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
                        v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c + 0 * dst_stride, v_out0);
                } else {
                    hvx_vmemu(out_row_base + c + 0 * dst_stride) = v_out0;
                }

                if (r + 1 < n_rows) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 1 * dst_stride, v_out1);
                    } else {
                        hvx_vmemu(out_row_base + c + 1 * dst_stride) = v_out1;
                    }
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 2 * dst_stride, v_out2);
                    } else {
                        hvx_vmemu(out_row_base + c + 2 * dst_stride) = v_out2;
                    }
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 3 * dst_stride, v_out3);
                    } else {
                        hvx_vmemu(out_row_base + c + 3 * dst_stride) = v_out3;
                    }
                }
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();
                HVX_Vector v_acc2_qf = Q6_V_vzero();
                HVX_Vector v_acc3_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g         = g_base + gi;
                        const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
                        const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
                        const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

                        HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
                        HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
                        HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
                        HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
                        HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                        v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
                        v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
                hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, v_out0);

                if (r + 1 < n_rows) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, v_out1);
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
                    hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, v_out2);
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
                    hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, v_out3);
                }
            }
        }
        r_base = r_end;
    }
}

static inline void w8a8_sm_eval_tile_quad_1g(
        const uint8_t * restrict tile_base,
        const float * restrict dw_base,
        size_t c,
        uint32_t da0,
        uint32_t da1,
        uint32_t da2,
        uint32_t da3,
        HVX_Vector sign_80,
        const uint8_t * restrict src2_row0,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        uint32_t n_rows,
        size_t r,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1,
        HVX_Vector * v_out2,
        HVX_Vector * v_out3) {
    const size_t     c0       = c >> 5;
    const uint8_t *  tp       = tile_base + c0 * 2 * HTP_MM_HMX_TILE_SIZE;
    const HVX_Vector v_hi_s8  = Q6_V_vxor_VV(*(const HVX_Vector *) tp, sign_80);
    const HVX_Vector v_lo_u8  = *(const HVX_Vector *) (tp + HTP_MM_HMX_TILE_SIZE);
    const HVX_Vector v_dw_qf  = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));

    HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
    HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
    HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw_qf);
    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row0 + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);

    if (r + 1 < n_rows) {
        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw_qf);
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row0 + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
    if (r + 2 < n_rows) {
        HVX_Vector v_f32_2   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
        HVX_Vector v_acc2_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da2), v_dw_qf);
        *v_out2 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row0 + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
    }
    if (r + 3 < n_rows) {
        HVX_Vector v_f32_3   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));
        HVX_Vector v_acc3_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da3), v_dw_qf);
        *v_out3 = Q6_Vsf_equals_Vqf32(src2_row0 ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row0 + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
    }
}

static inline void w8a8_sm_eval_tile_quad_ng(
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        size_t group_stride,
        size_t row_tile_stride,
        size_t r0,
        size_t r_blk,
        size_t rg,
        size_t lane,
        uint32_t mc_pad,
        size_t c0_glob,
        uint32_t n_groups,
        uint32_t (*da_ptr)[32],
        HVX_Vector sign_80,
        const uint8_t * restrict src2_row_base,
        size_t c,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        uint32_t n_rows,
        size_t r,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1,
        HVX_Vector * v_out2,
        HVX_Vector * v_out3) {
    HVX_Vector v_acc0_qf = Q6_V_vzero();
    HVX_Vector v_acc1_qf = Q6_V_vzero();
    HVX_Vector v_acc2_qf = Q6_V_vzero();
    HVX_Vector v_acc3_qf = Q6_V_vzero();

    for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
        const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
        if (n_groups > 16u) {
            for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r_blk << 5));
            }
        }
        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
            const uint32_t   g         = g_base + gi;
            const uint8_t *  tile_pair = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * 2 * HTP_MM_HMX_TILE_SIZE;
            const HVX_Vector v_hi_s8   = Q6_V_vxor_VV(((const HVX_Vector *) tile_pair)[rg], sign_80);
            const HVX_Vector v_lo_u8   = ((const HVX_Vector *) (tile_pair + HTP_MM_HMX_TILE_SIZE))[rg];

            HVX_VectorPair vp_i16    = Q6_W_vshuff_VVR(v_hi_s8, v_lo_u8, -1);
            HVX_VectorPair vp_i32_01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vp_i16));
            HVX_VectorPair vp_i32_23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vp_i16));

            HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_01));
            HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_01));
            HVX_Vector v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32_23));
            HVX_Vector v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32_23));

            const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
            const uint32_t * da_row  = da_ptr[gi] + lane;

            v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
            v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
            v_acc2_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc2_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_2, da_row[2]), v_dw_qf));
            v_acc3_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc3_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_3, da_row[3]), v_dw_qf));
        }
    }

    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base + 0 * src2_row_bytes, c, src2_elem_size)) : v_acc0_qf);
    if (r + 1 < n_rows) {
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + 1 * src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
    if (r + 2 < n_rows) {
        *v_out2 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc2_qf, hmx_load_src2_f32_32(src2_row_base + 2 * src2_row_bytes, c, src2_elem_size)) : v_acc2_qf);
    }
    if (r + 3 < n_rows) {
        *v_out3 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc3_qf, hmx_load_src2_f32_32(src2_row_base + 3 * src2_row_bytes, c, src2_elem_size)) : v_acc3_qf);
    }
}

// HVX Epilogue for W8A8 CM output tiles -> FP16 DDR output with superblock scale accumulation.
static __attribute__((noinline)) void transfer_output_chunk_w8a8_sm_to_fp16_col_chunk(
        __fp16 * restrict dst,
        const void * restrict src2,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64    = mc_pad / 64;
    const size_t group_stride      = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride   = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_80       = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);
    const bool   dst_aligned_nt    = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(__fp16))) & 127u) == 0);
    const size_t src2_row_bytes    = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;
                __fp16 *        out_row0  = dst + r * dst_stride;
                const uint8_t * src2_row0 = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                    HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row0 + c + 0 * dst_stride, v_f16_0);
                    } else {
                        hvx_vmemu(out_row0 + c + 0 * dst_stride) = v_f16_0;
                    }
                    if (r + 1 < n_rows) {
                        HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 1 * dst_stride, v_f16_1);
                        } else {
                            hvx_vmemu(out_row0 + c + 1 * dst_stride) = v_f16_1;
                        }
                    }
                    if (r + 2 < n_rows) {
                        HVX_Vector v_f16_2 = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 2 * dst_stride, v_f16_2);
                        } else {
                            hvx_vmemu(out_row0 + c + 2 * dst_stride) = v_f16_2;
                        }
                    }
                    if (r + 3 < n_rows) {
                        HVX_Vector v_f16_3 = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row0 + c + 3 * dst_stride, v_f16_3);
                        } else {
                            hvx_vmemu(out_row0 + c + 3 * dst_stride) = v_f16_3;
                        }
                    }
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    if (valid_c > 32) {
                        w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                                  src2_row0, src2_row_bytes, src2_elem_size, n_rows, r,
                                                  &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                    }

                    hvx_vec_store_u(out_row0 + c + 0 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (r + 1 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 1 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    }
                    if (r + 2 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 2 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                    }
                    if (r + 3 < n_rows) {
                        hvx_vec_store_u(out_row0 + c + 3 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            __fp16 *        out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          src2_row_base, c, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                          src2_row_base, c + 32, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c + 0 * dst_stride, v_f16_0);
                } else {
                    hvx_vmemu(out_row_base + c + 0 * dst_stride) = v_f16_0;
                }
                if (r + 1 < n_rows) {
                    HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 1 * dst_stride, v_f16_1);
                    } else {
                        hvx_vmemu(out_row_base + c + 1 * dst_stride) = v_f16_1;
                    }
                }
                if (r + 2 < n_rows) {
                    HVX_Vector v_f16_2 = hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 2 * dst_stride, v_f16_2);
                    } else {
                        hvx_vmemu(out_row_base + c + 2 * dst_stride) = v_f16_2;
                    }
                }
                if (r + 3 < n_rows) {
                    HVX_Vector v_f16_3 = hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + 3 * dst_stride, v_f16_3);
                    } else {
                        hvx_vmemu(out_row_base + c + 3 * dst_stride) = v_f16_3;
                    }
                }
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          src2_row_base, c, src2_row_bytes, src2_elem_size, n_rows, r,
                                          &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                if (valid_c > 32) {
                    w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                              group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                              mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                              src2_row_base, c + 32, src2_row_bytes, src2_elem_size, n_rows, r,
                                              &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                }

                hvx_vec_store_u(out_row_base + c + 0 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (r + 1 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 1 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }
                if (r + 2 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 2 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                }
                if (r + 3 < n_rows) {
                    hvx_vec_store_u(out_row_base + c + 3 * dst_stride, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                }
            }
        }
        r_base = r_end;
    }
}

// HVX Epilogue for W8A16 :2x1 SM output tiles -> FP32 DDR output with superblock scale accumulation.
static void transfer_output_chunk_w8a16_sm_to_fp32_col_chunk(
        float * restrict dst,
        const void * restrict src2,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32  = mc_pad / 32;
    const size_t group_stride    = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_8000   = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);
    const bool   dst_aligned_nt  = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(float))) & 127u) == 0);
    const size_t src2_row_bytes  = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane = (start_row + r) & 31u;
                const size_t   r1   = lane >> 1;
                const uint32_t da0  = da_ptr[0][lane + 0];
                const uint32_t da1  = da_ptr[0][lane + 1];

                const uint16_t * tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base       = vtcm_weight_scales + c0_first * 32;
                float *          out_row_base  = dst + r * dst_stride;
                const uint8_t *  src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool       has_row1      = (r + 1 < n_rows);

                if (dst_aligned_nt && has_row1 && !src2) {
                    #pragma unroll(2)
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t     c0     = c >> 5;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                        const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw));
                        HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw));

                        hvx_vmem_store_nt(out_row_base + c, v_out0);
                        hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                    }
                } else {
                    for (size_t c = 0; c < limit_c_aligned; c += 32) {
                        const size_t     c0     = c >> 5;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                        const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
                        HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row_base + c, v_out0);
                        } else {
                            hvx_vmemu(out_row_base + c) = v_out0;
                        }

                        if (has_row1) {
                            HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
                            HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
                            HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                            if (dst_aligned_nt) {
                                hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                            } else {
                                hvx_vmemu(out_row_base + c + dst_stride) = v_out1;
                            }
                        }
                    }
                }

                if (limit_c_aligned < limit_c) {
                    const size_t     c           = limit_c_aligned;
                    const size_t     c0          = c >> 5;
                    const size_t     valid_bytes = (limit_c - c) * sizeof(float);
                    const HVX_Vector v_i16       = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
                    const HVX_Vector v_dw        = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
                    HVX_VectorPair   vp_i32      = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
                    HVX_Vector v_out0    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                    hvx_vec_store_u(out_row_base + c, valid_bytes, v_out0);

                    if (has_row1) {
                        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
                        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
                        HVX_Vector v_out1    = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                        hvx_vec_store_u(out_row_base + c + dst_stride, valid_bytes, v_out1);
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane = (start_row + r) & 31u;
            const size_t r1   = lane >> 1;

            float *         out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
            const bool      has_row1      = (r + 1 < n_rows);

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0      = c >> 5;
                const size_t c0_glob = c0_first + c0;

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g      = g_base + gi;
                        const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c, v_out0);
                } else {
                    hvx_vmemu(out_row_base + c) = v_out0;
                }

                if (has_row1) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c + dst_stride, v_out1);
                    } else {
                        hvx_vmemu(out_row_base + c + dst_stride) = v_out1;
                    }
                }
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);

                HVX_Vector v_acc0_qf = Q6_V_vzero();
                HVX_Vector v_acc1_qf = Q6_V_vzero();

                for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
                    const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
                    if (n_groups > 16u) {
                        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                            *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
                        }
                    }
                    for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                        const uint32_t   g      = g_base + gi;
                        const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
                        const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
                        HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

                        HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
                        HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

                        const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
                        const uint32_t * da_row  = da_ptr[gi] + lane;

                        v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
                        v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
                    }
                }

                HVX_Vector v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
                hvx_vec_store_u(out_row_base + c, valid_bytes, v_out0);

                if (has_row1) {
                    HVX_Vector v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
                    hvx_vec_store_u(out_row_base + c + dst_stride, valid_bytes, v_out1);
                }
            }
        }
        r_base = r_end;
    }
}

static inline void w8a16_sm_eval_tile_pair_1g(
        const uint16_t * restrict tile_base,
        const float * restrict dw_base,
        size_t c,
        uint32_t da0,
        uint32_t da1,
        HVX_Vector sign_8000,
        const uint8_t * restrict src2_row_base,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        bool has_row1,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1) {
    const size_t     c0     = c >> 5;
    const HVX_Vector v_i16  = Q6_V_vxor_VV(*(const HVX_Vector *) (tile_base + c0 * HTP_MM_HMX_TILE_N_ELMS), sign_8000);
    const HVX_Vector v_dw   = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (dw_base + c));
    HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

    HVX_Vector v_f32_0   = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
    HVX_Vector v_acc0_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da0), v_dw);
    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);

    if (has_row1) {
        HVX_Vector v_f32_1   = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));
        HVX_Vector v_acc1_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da1), v_dw);
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
}

static inline void w8a16_sm_eval_tile_pair_ng(
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        size_t group_stride,
        size_t row_tile_stride,
        size_t r0,
        size_t r1,
        size_t lane,
        uint32_t mc_pad,
        size_t c0_glob,
        uint32_t n_groups,
        uint32_t (*da_ptr)[32],
        HVX_Vector sign_8000,
        const uint8_t * restrict src2_row_base,
        size_t c,
        size_t src2_row_bytes,
        size_t src2_elem_size,
        bool has_row1,
        HVX_Vector * v_out0,
        HVX_Vector * v_out1) {
    HVX_Vector v_acc0_qf = Q6_V_vzero();
    HVX_Vector v_acc1_qf = Q6_V_vzero();

    for (uint32_t g_base = 0; g_base < n_groups; g_base += 16u) {
        const uint32_t g_cnt = hex_smin(n_groups - g_base, 16u);
        if (n_groups > 16u) {
            for (uint32_t gi = 0; gi < g_cnt; ++gi) {
                *(HVX_Vector *) da_ptr[gi] = hvx_vmem(vtcm_act_scales + (size_t) (g_base + gi) * mc_pad + (r0 << 5));
            }
        }
        for (uint32_t gi = 0; gi < g_cnt; ++gi) {
            const uint32_t   g      = g_base + gi;
            const uint16_t * tile   = vtcm_src + (size_t) g * group_stride + r0 * row_tile_stride + c0_glob * HTP_MM_HMX_TILE_N_ELMS;
            const HVX_Vector v_i16  = Q6_V_vxor_VV(((const HVX_Vector *) tile)[r1], sign_8000);
            HVX_VectorPair   vp_i32 = Q6_Ww_vunpack_Vh(Q6_Vh_vdeal_Vh(v_i16));

            HVX_Vector v_f32_0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(vp_i32));
            HVX_Vector v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(vp_i32));

            const HVX_Vector v_dw_qf = hvx_vec_f32_to_qf32(*(const HVX_Vector *) (vtcm_weight_scales + (c0_glob * n_groups + g) * 32));
            const uint32_t * da_row  = da_ptr[gi] + lane;

            v_acc0_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc0_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_0, da_row[0]), v_dw_qf));
            v_acc1_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_acc1_qf, Q6_Vqf32_vmpy_Vqf32Vqf32(hvx_qf32_mul_sf_scalar(v_f32_1, da_row[1]), v_dw_qf));
        }
    }

    *v_out0 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc0_qf, hmx_load_src2_f32_32(src2_row_base, c, src2_elem_size)) : v_acc0_qf);
    if (has_row1) {
        *v_out1 = Q6_Vsf_equals_Vqf32(src2_row_base ? Q6_Vqf32_vadd_Vqf32Vsf(v_acc1_qf, hmx_load_src2_f32_32(src2_row_base + src2_row_bytes, c, src2_elem_size)) : v_acc1_qf);
    }
}

// HVX Epilogue for W8A16 :2x1 SM output tiles -> FP16 DDR output with superblock scale accumulation.
static __attribute__((noinline)) void transfer_output_chunk_w8a16_sm_to_fp16_col_chunk(
        __fp16 * restrict dst,
        const void * restrict src2,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_stride,
        uint32_t src2_stride,
        uint32_t dst_cols,
        size_t src2_elem_size) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32    = mc_pad / 32;
    const size_t group_stride      = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride   = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_8000     = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);
    const bool   dst_aligned_nt    = ((((uintptr_t) dst | ((size_t) dst_stride * sizeof(__fp16))) & 127u) == 0);
    const size_t src2_row_bytes    = (size_t) src2_stride * src2_elem_size;

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane = (start_row + r) & 31u;
                const size_t   r1   = lane >> 1;
                const uint32_t da0  = da_ptr[0][lane + 0];
                const uint32_t da1  = da_ptr[0][lane + 1];

                const uint16_t * tile_base     = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base       = vtcm_weight_scales + c0_first * 32;
                __fp16 *         out_row_base  = dst + r * dst_stride;
                const uint8_t *  src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
                const bool       has_row1      = (r + 1 < n_rows);

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_lo, &v1_lo);
                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_hi, &v1_hi);

                    HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + c, v_f16_0);
                    } else {
                        hvx_vmemu(out_row_base + c) = v_f16_0;
                    }
                    if (has_row1) {
                        HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                        if (dst_aligned_nt) {
                            hvx_vmem_store_nt(out_row_base + dst_stride + c, v_f16_1);
                        } else {
                            hvx_vmemu(out_row_base + dst_stride + c) = v_f16_1;
                        }
                    }
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_lo, &v1_lo);
                    if (valid_c > 32) {
                        w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                                   src2_row_base, src2_row_bytes, src2_elem_size, has_row1,
                                                   &v0_hi, &v1_hi);
                    }

                    hvx_vec_store_u(out_row_base + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (has_row1) {
                        hvx_vec_store_u(out_row_base + dst_stride + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    }
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane = (start_row + r) & 31u;
            const size_t r1   = lane >> 1;

            __fp16 *        out_row_base  = dst + r * dst_stride;
            const uint8_t * src2_row_base = src2 ? ((const uint8_t *) src2 + r * src2_row_bytes) : NULL;
            const bool      has_row1      = (r + 1 < n_rows);

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_lo, &v1_lo);
                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c + 32, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_hi, &v1_hi);

                HVX_Vector v_f16_0 = hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi);
                if (dst_aligned_nt) {
                    hvx_vmem_store_nt(out_row_base + c, v_f16_0);
                } else {
                    hvx_vmemu(out_row_base + c) = v_f16_0;
                }
                if (has_row1) {
                    HVX_Vector v_f16_1 = hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi);
                    if (dst_aligned_nt) {
                        hvx_vmem_store_nt(out_row_base + dst_stride + c, v_f16_1);
                    } else {
                        hvx_vmemu(out_row_base + dst_stride + c) = v_f16_1;
                    }
                }
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           src2_row_base, c, src2_row_bytes, src2_elem_size, has_row1,
                                           &v0_lo, &v1_lo);
                if (valid_c > 32) {
                    w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                               group_stride, row_tile_stride, r0, r1, lane,
                                               mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                               src2_row_base, c + 32, src2_row_bytes, src2_elem_size, has_row1,
                                               &v0_hi, &v1_hi);
                }

                hvx_vec_store_u(out_row_base + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (has_row1) {
                    hvx_vec_store_u(out_row_base + dst_stride + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }
            }
        }
        r_base = r_end;
    }
}

static inline void hmx_store_vec_nt_or_u(void * restrict ptr, bool aligned_128, HVX_Vector v) {
    if (aligned_128) {
        hvx_vmem_store_nt(ptr, v);
    } else {
        hvx_vmemu(ptr) = v;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_w8a8_cm_to_fp32_scattered(
        float * restrict dst,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t global_start_row,
        uint32_t vtcm_start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_cols,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64  = mc_pad / 64;
    const size_t group_stride    = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_80     = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = vtcm_start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = vtcm_start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;

                const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
                float * out_row0 = (float *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
                float * out_row1 = (r + 1 < n_rows) ? (float *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
                float * out_row2 = (r + 2 < n_rows) ? (float *) ((uint8_t *) dst + map_base[2].i1 * dst_nb1 + map_base[2].i2 * dst_nb2) : NULL;
                float * out_row3 = (r + 3 < n_rows) ? (float *) ((uint8_t *) dst + map_base[3].i1 * dst_nb1 + map_base[3].i2 * dst_nb2) : NULL;
                const bool aligned0 = hex_is_aligned(out_row0, 128);
                const bool aligned1 = out_row1 && hex_is_aligned(out_row1, 128);
                const bool aligned2 = out_row2 && hex_is_aligned(out_row2, 128);
                const bool aligned3 = out_row3 && hex_is_aligned(out_row3, 128);

                for (size_t c = 0; c < limit_c_aligned; c += 32) {
                    HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();
                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              NULL, 0, 0, n_rows, r, &v0, &v1, &v2, &v3);
                    hmx_store_vec_nt_or_u(out_row0 + c, aligned0, v0);
                    if (out_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, v1);
                    if (out_row2) hmx_store_vec_nt_or_u(out_row2 + c, aligned2, v2);
                    if (out_row3) hmx_store_vec_nt_or_u(out_row3 + c, aligned3, v3);
                }

                if (limit_c_aligned < limit_c) {
                    const size_t c           = limit_c_aligned;
                    const size_t valid_bytes = (limit_c - c) * sizeof(float);
                    HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();
                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              NULL, 0, 0, n_rows, r, &v0, &v1, &v2, &v3);
                    hvx_vec_store_u(out_row0 + c, valid_bytes, v0);
                    if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);
                    if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);
                    if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = vtcm_start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = vtcm_start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
            float * out_row0 = (float *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
            float * out_row1 = (r + 1 < n_rows) ? (float *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
            float * out_row2 = (r + 2 < n_rows) ? (float *) ((uint8_t *) dst + map_base[2].i1 * dst_nb1 + map_base[2].i2 * dst_nb2) : NULL;
            float * out_row3 = (r + 3 < n_rows) ? (float *) ((uint8_t *) dst + map_base[3].i1 * dst_nb1 + map_base[3].i2 * dst_nb2) : NULL;
            const bool aligned0 = hex_is_aligned(out_row0, 128);
            const bool aligned1 = out_row1 && hex_is_aligned(out_row1, 128);
            const bool aligned2 = out_row2 && hex_is_aligned(out_row2, 128);
            const bool aligned3 = out_row3 && hex_is_aligned(out_row3, 128);

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();
                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          NULL, c, 0, 0, n_rows, r, &v0, &v1, &v2, &v3);
                hmx_store_vec_nt_or_u(out_row0 + c, aligned0, v0);
                if (out_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, v1);
                if (out_row2) hmx_store_vec_nt_or_u(out_row2 + c, aligned2, v2);
                if (out_row3) hmx_store_vec_nt_or_u(out_row3 + c, aligned3, v3);
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);
                HVX_Vector v0, v1 = Q6_V_vzero(), v2 = Q6_V_vzero(), v3 = Q6_V_vzero();
                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          NULL, c, 0, 0, n_rows, r, &v0, &v1, &v2, &v3);
                hvx_vec_store_u(out_row0 + c, valid_bytes, v0);
                if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);
                if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, v2);
                if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, v3);
            }
        }
        r_base = r_end;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_w8a8_cm_to_fp16_scattered(
        __fp16 * restrict dst,
        const uint8_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t global_start_row,
        uint32_t vtcm_start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_cols,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_64    = mc_pad / 64;
    const size_t group_stride      = n_row_tiles_64 * total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t row_tile_stride   = total_col_tiles * 2 * HTP_MM_HMX_TILE_SIZE;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_80       = Q6_Vb_vsplat_R(0x80);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = vtcm_start_row + r_base;
            const size_t r_blk      = r_idx_base >> 5;
            const size_t r0         = r_idx_base >> 6;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r_blk << 5));

            for (size_t r = r_base; r < r_end; r += 4) {
                const size_t   r_idx0 = vtcm_start_row + r;
                const size_t   rg     = (r_idx0 & 63u) >> 2;
                const size_t   lane   = r_idx0 & 31u;
                const uint32_t da0    = da_ptr[0][lane + 0];
                const uint32_t da1    = da_ptr[0][lane + 1];
                const uint32_t da2    = da_ptr[0][lane + 2];
                const uint32_t da3    = da_ptr[0][lane + 3];

                const uint8_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * 2 * HTP_MM_HMX_TILE_SIZE + rg * 128;
                const float *   dw_base   = vtcm_weight_scales + c0_first * 32;

                const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
                __fp16 * out_row0 = (__fp16 *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
                __fp16 * out_row1 = (r + 1 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
                __fp16 * out_row2 = (r + 2 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[2].i1 * dst_nb1 + map_base[2].i2 * dst_nb2) : NULL;
                __fp16 * out_row3 = (r + 3 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[3].i1 * dst_nb1 + map_base[3].i2 * dst_nb2) : NULL;
                const bool aligned0 = hex_is_aligned(out_row0, 128);
                const bool aligned1 = out_row1 && hex_is_aligned(out_row1, 128);
                const bool aligned2 = out_row2 && hex_is_aligned(out_row2, 128);
                const bool aligned3 = out_row3 && hex_is_aligned(out_row3, 128);

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              NULL, 0, 0, n_rows, r, &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                              NULL, 0, 0, n_rows, r, &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                    hmx_store_vec_nt_or_u(out_row0 + c, aligned0, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (out_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    if (out_row2) hmx_store_vec_nt_or_u(out_row2 + c, aligned2, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                    if (out_row3) hmx_store_vec_nt_or_u(out_row3 + c, aligned3, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                    w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                              NULL, 0, 0, n_rows, r, &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                    if (valid_c > 32) {
                        w8a8_sm_eval_tile_quad_1g(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                                  NULL, 0, 0, n_rows, r, &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                    }

                    hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                    if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                    if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = vtcm_start_row + r_base;
        const size_t r_blk      = r_idx_base >> 5;
        const size_t r0         = r_idx_base >> 6;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r_blk << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 4) {
            const size_t r_idx0 = vtcm_start_row + r;
            const size_t rg     = (r_idx0 & 63u) >> 2;
            const size_t lane   = r_idx0 & 31u;

            const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
            __fp16 * out_row0 = (__fp16 *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
            __fp16 * out_row1 = (r + 1 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
            __fp16 * out_row2 = (r + 2 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[2].i1 * dst_nb1 + map_base[2].i2 * dst_nb2) : NULL;
            __fp16 * out_row3 = (r + 3 < n_rows) ? (__fp16 *) ((uint8_t *) dst + map_base[3].i1 * dst_nb1 + map_base[3].i2 * dst_nb2) : NULL;
            const bool aligned0 = hex_is_aligned(out_row0, 128);
            const bool aligned1 = out_row1 && hex_is_aligned(out_row1, 128);
            const bool aligned2 = out_row2 && hex_is_aligned(out_row2, 128);
            const bool aligned3 = out_row3 && hex_is_aligned(out_row3, 128);

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          NULL, c, 0, 0, n_rows, r, &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                          NULL, c + 32, 0, 0, n_rows, r, &v0_hi, &v1_hi, &v2_hi, &v3_hi);

                hmx_store_vec_nt_or_u(out_row0 + c, aligned0, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (out_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                if (out_row2) hmx_store_vec_nt_or_u(out_row2 + c, aligned2, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                if (out_row3) hmx_store_vec_nt_or_u(out_row3 + c, aligned3, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero(), v2_lo = Q6_V_vzero(), v3_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero(), v2_hi = Q6_V_vzero(), v3_hi = Q6_V_vzero();

                w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                          group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                          mc_pad, c0_glob, n_groups, da_ptr, sign_80,
                                          NULL, c, 0, 0, n_rows, r, &v0_lo, &v1_lo, &v2_lo, &v3_lo);
                if (valid_c > 32) {
                    w8a8_sm_eval_tile_quad_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                              group_stride, row_tile_stride, r0, r_blk, rg, lane,
                                              mc_pad, c0_glob + 1, n_groups, da_ptr, sign_80,
                                              NULL, c + 32, 0, 0, n_rows, r, &v0_hi, &v1_hi, &v2_hi, &v3_hi);
                }

                hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (out_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                if (out_row2) hvx_vec_store_u(out_row2 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v2_lo, v2_hi));
                if (out_row3) hvx_vec_store_u(out_row3 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v3_lo, v3_hi));
            }
        }
        r_base = r_end;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_w8a16_sm_to_fp32_scattered(
        float * restrict dst,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t global_start_row,
        uint32_t vtcm_start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_cols,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {

    const size_t total_col_tiles = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32  = mc_pad / 32;
    const size_t group_stride    = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first        = c_first / 32;

    const HVX_Vector sign_8000   = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c         = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned = (limit_c & ~31u);

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = vtcm_start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane     = (vtcm_start_row + r) & 31u;
                const size_t   r1       = lane >> 1;
                const uint32_t da0      = da_ptr[0][lane + 0];
                const uint32_t da1      = da_ptr[0][lane + 1];
                const bool     has_row1 = (r + 1 < n_rows);

                const uint16_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base   = vtcm_weight_scales + c0_first * 32;

                const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
                float * out_row0 = (float *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
                float * out_row1 = has_row1 ? (float *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
                const bool aligned0 = hex_is_aligned(out_row0, 128);
                const bool aligned1 = has_row1 && hex_is_aligned(out_row1, 128);

                for (size_t c = 0; c < limit_c_aligned; c += 32) {
                    HVX_Vector v0, v1 = Q6_V_vzero();
                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               NULL, 0, 0, has_row1, &v0, &v1);
                    hmx_store_vec_nt_or_u(out_row0 + c, aligned0, v0);
                    if (has_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, v1);
                }

                if (limit_c_aligned < limit_c) {
                    const size_t c           = limit_c_aligned;
                    const size_t valid_bytes = (limit_c - c) * sizeof(float);
                    HVX_Vector v0, v1 = Q6_V_vzero();
                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               NULL, 0, 0, has_row1, &v0, &v1);
                    hvx_vec_store_u(out_row0 + c, valid_bytes, v0);
                    if (has_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = vtcm_start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane     = (vtcm_start_row + r) & 31u;
            const size_t r1       = lane >> 1;
            const bool   has_row1 = (r + 1 < n_rows);

            const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
            float * out_row0 = (float *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
            float * out_row1 = has_row1 ? (float *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
            const bool aligned0 = hex_is_aligned(out_row0, 128);
            const bool aligned1 = has_row1 && hex_is_aligned(out_row1, 128);

            for (size_t c = 0; c < limit_c_aligned; c += 32) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0, v1 = Q6_V_vzero();
                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           NULL, c, 0, 0, has_row1, &v0, &v1);
                hmx_store_vec_nt_or_u(out_row0 + c, aligned0, v0);
                if (has_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, v1);
            }

            if (limit_c_aligned < limit_c) {
                const size_t c           = limit_c_aligned;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_bytes = (limit_c - c) * sizeof(float);
                HVX_Vector v0, v1 = Q6_V_vzero();
                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           NULL, c, 0, 0, has_row1, &v0, &v1);
                hvx_vec_store_u(out_row0 + c, valid_bytes, v0);
                if (has_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, v1);
            }
        }
        r_base = r_end;
    }
}

static __attribute__((noinline)) void transfer_output_chunk_w8a16_sm_to_fp16_scattered(
        __fp16 * restrict dst,
        const uint16_t * restrict vtcm_src,
        const float * restrict vtcm_act_scales,
        const float * restrict vtcm_weight_scales,
        uint32_t global_start_row,
        uint32_t vtcm_start_row,
        uint32_t n_rows,
        uint32_t mc_pad,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t total_n_cols,
        uint32_t n_groups,
        uint32_t dst_cols,
        const struct mmid_row_mapping * matrix_rows,
        uint32_t cur_a,
        uint32_t mapping_stride,
        size_t dst_nb1,
        size_t dst_nb2) {

    const size_t total_col_tiles   = hmx_ceil_div(total_n_cols, 32);
    const size_t n_row_tiles_32    = mc_pad / 32;
    const size_t group_stride      = n_row_tiles_32 * total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t row_tile_stride   = total_col_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t c0_first          = c_first / 32;

    const HVX_Vector sign_8000     = Q6_Vh_vsplat_R(0x8000);
    const size_t limit_c           = hex_smin(c_len, dst_cols);
    const size_t limit_c_aligned64 = (limit_c & ~63u);

    uint32_t da_buf[16][32] __attribute__((aligned(128)));
    uint32_t (*da_ptr)[32] = htp_da_buf_ptr(da_buf);

    if (n_groups <= 1) {
        for (size_t r_base = 0; r_base < n_rows; ) {
            const size_t r_idx_base = vtcm_start_row + r_base;
            const size_t r0         = r_idx_base >> 5;
            const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
            const size_t r_end      = r_base + r_step;

            *(HVX_Vector *) da_ptr[0] = hvx_vmem(vtcm_act_scales + (r0 << 5));

            for (size_t r = r_base; r < r_end; r += 2) {
                const size_t   lane     = (vtcm_start_row + r) & 31u;
                const size_t   r1       = lane >> 1;
                const uint32_t da0      = da_ptr[0][lane + 0];
                const uint32_t da1      = da_ptr[0][lane + 1];
                const bool     has_row1 = (r + 1 < n_rows);

                const uint16_t * tile_base = vtcm_src + r0 * row_tile_stride + c0_first * HTP_MM_HMX_TILE_N_ELMS + r1 * 64;
                const float *    dw_base   = vtcm_weight_scales + c0_first * 32;

                const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
                __fp16 * out_row0 = (__fp16 *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
                __fp16 * out_row1 = has_row1 ? (__fp16 *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
                const bool aligned0 = hex_is_aligned(out_row0, 128);
                const bool aligned1 = has_row1 && hex_is_aligned(out_row1, 128);

                for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               NULL, 0, 0, has_row1, &v0_lo, &v1_lo);
                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                               NULL, 0, 0, has_row1, &v0_hi, &v1_hi);

                    hmx_store_vec_nt_or_u(out_row0 + c, aligned0, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (has_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }

                if (limit_c_aligned64 < limit_c) {
                    const size_t c           = limit_c_aligned64;
                    const size_t valid_c     = limit_c - c;
                    const size_t valid_bytes = valid_c * sizeof(__fp16);

                    HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                    HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                    w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c, da0, da1, sign_8000,
                                               NULL, 0, 0, has_row1, &v0_lo, &v1_lo);
                    if (valid_c > 32) {
                        w8a16_sm_eval_tile_pair_1g(tile_base, dw_base, c + 32, da0, da1, sign_8000,
                                                   NULL, 0, 0, has_row1, &v0_hi, &v1_hi);
                    }

                    hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                    if (has_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
                }
            }
            r_base = r_end;
        }
        return;
    }

    for (size_t r_base = 0; r_base < n_rows; ) {
        const size_t r_idx_base = vtcm_start_row + r_base;
        const size_t r0         = r_idx_base >> 5;
        const size_t r_step     = hex_smin((size_t) n_rows - r_base, (size_t) (32u - (r_idx_base & 31u)));
        const size_t r_end      = r_base + r_step;

        if (n_groups <= 16u) {
            for (uint32_t g = 0; g < n_groups; ++g) {
                *(HVX_Vector *) da_ptr[g] = hvx_vmem(vtcm_act_scales + (size_t) g * mc_pad + (r0 << 5));
            }
        }

        for (size_t r = r_base; r < r_end; r += 2) {
            const size_t lane     = (vtcm_start_row + r) & 31u;
            const size_t r1       = lane >> 1;
            const bool   has_row1 = (r + 1 < n_rows);

            const struct mmid_row_mapping * map_base = matrix_rows + (size_t) cur_a * mapping_stride + global_start_row + r;
            __fp16 * out_row0 = (__fp16 *) ((uint8_t *) dst + map_base[0].i1 * dst_nb1 + map_base[0].i2 * dst_nb2);
            __fp16 * out_row1 = has_row1 ? (__fp16 *) ((uint8_t *) dst + map_base[1].i1 * dst_nb1 + map_base[1].i2 * dst_nb2) : NULL;
            const bool aligned0 = hex_is_aligned(out_row0, 128);
            const bool aligned1 = has_row1 && hex_is_aligned(out_row1, 128);

            for (size_t c = 0; c < limit_c_aligned64; c += 64) {
                const size_t c0_glob = c0_first + (c >> 5);
                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi, v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           NULL, c, 0, 0, has_row1, &v0_lo, &v1_lo);
                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                           NULL, c + 32, 0, 0, has_row1, &v0_hi, &v1_hi);

                hmx_store_vec_nt_or_u(out_row0 + c, aligned0, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (has_row1) hmx_store_vec_nt_or_u(out_row1 + c, aligned1, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
            }

            if (limit_c_aligned64 < limit_c) {
                const size_t c           = limit_c_aligned64;
                const size_t c0_glob     = c0_first + (c >> 5);
                const size_t valid_c     = limit_c - c;
                const size_t valid_bytes = valid_c * sizeof(__fp16);

                HVX_Vector v0_lo, v1_lo = Q6_V_vzero();
                HVX_Vector v0_hi = Q6_V_vzero(), v1_hi = Q6_V_vzero();

                w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                           group_stride, row_tile_stride, r0, r1, lane,
                                           mc_pad, c0_glob, n_groups, da_ptr, sign_8000,
                                           NULL, c, 0, 0, has_row1, &v0_lo, &v1_lo);
                if (valid_c > 32) {
                    w8a16_sm_eval_tile_pair_ng(vtcm_src, vtcm_act_scales, vtcm_weight_scales,
                                               group_stride, row_tile_stride, r0, r1, lane,
                                               mc_pad, c0_glob + 1, n_groups, da_ptr, sign_8000,
                                               NULL, c + 32, 0, 0, has_row1, &v0_hi, &v1_hi);
                }

                hvx_vec_store_u(out_row0 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v0_lo, v0_hi));
                if (has_row1) hvx_vec_store_u(out_row1 + c, valid_bytes, hmx_pack_f32_to_f16_clamped(v1_lo, v1_hi));
            }
        }
        r_base = r_end;
    }
}
