// Dynamic quantizers that produce tiled activations

static inline void quantize_block_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));

    float vmax0[32]  __attribute__((aligned(128)));
    float vmax1[32]  __attribute__((aligned(128)));
    float vmax2[32]  __attribute__((aligned(128)));
    float vmax3[32]  __attribute__((aligned(128)));
    int32_t sums[32] __attribute__((aligned(128)));

    hvx_vec_store_u(vmax0, 128, vmax0_sf);
    hvx_vec_store_u(vmax1, 128, vmax1_sf);
    hvx_vec_store_u(vmax2, 128, vmax2_sf);
    hvx_vec_store_u(vmax3, 128, vmax3_sf);
    hvx_vec_store_u(sums,  128, v_sums);

    float d0 = vmax0[0] / 127.0f;
    float d1 = vmax1[0] / 127.0f;
    float d2 = vmax2[0] / 127.0f;
    float d3 = vmax3[0] / 127.0f;

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        __fp16 scale_h, offset_h;
        if (b == 0) {
            scale_h  = (__fp16) d0;
            offset_h = (__fp16) (sums[0] * d0);
        } else if (b == 1) {
            scale_h  = (__fp16) d1;
            offset_h = (__fp16) (sums[8] * d1);
        } else if (b == 2) {
            scale_h  = (__fp16) d2;
            offset_h = (__fp16) (sums[16] * d2);
        } else {
            scale_h  = (__fp16) d3;
            offset_h = (__fp16) (sums[24] * d3);
        }

        HVX_Vector r_scale  = Q6_Vh_vsplat_R(*(int16_t *)&scale_h);
        HVX_Vector r_offset = Q6_Vh_vsplat_R(*(int16_t *)&offset_h);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_offset;
    }
}

static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vzero();

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vmax_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx01_hf));
    vmax_hf            = hvx_vec_reduce_max2_f16(hvx_vec_abs_f16(vx23_hf), vmax_hf);

    HVX_Vector vd_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd_hf   = Q6_Vhf_equals_Vqf16(vd_qf16);

    HVX_Vector vd_inv_hf = hvx_vec_inverse_f16(vd_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_Vector r_scale = hvx_vec_repl_f16(vd_hf);

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1152);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
    }
}

static void quantize_row_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1152;
        quantize_block_f32_q8_0_tiled(x + i * qk, y_block);
    }
}

static void quantize_row_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1280;
        quantize_block_f32_q8_1_tiled(x + i * qk, y_block);
    }
}

// Dot kernels & helpers that consume tiled activations

static inline HVX_Vector hvx_vec_mul_f16_f16_to_f32_lower32(HVX_Vector v1, HVX_Vector v2) {
#if __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#else
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

static inline HVX_Vector unpack_and_interleave_4bit(HVX_Vector v_a, HVX_Vector v_b, HVX_Vector mask_h4) {
    HVX_Vector v_W0 = Q6_V_vand_VV(v_a, mask_h4);
    HVX_Vector v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_Vector v_W2 = Q6_V_vand_VV(v_b, mask_h4);
    HVX_Vector v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_VectorPair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_VectorPair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    return Q6_V_lo_W(v0123_pair);
}

static inline HVX_VectorPair unpack_and_interleave_4bit_x2(HVX_Vector v_src, HVX_Vector mask_h4) {
    HVX_Vector v_lo = Q6_V_vand_VV(v_src, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_4bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_Vector accum_4bit_32x1_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_4bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_VectorPair accum_4bit_32x2_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_Vector accum_q8_0_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum = Q6_Vw_vrmpyacc_VwVbVb(v_sum, v_W, v_act[g]);
    }
    return v_sum;
}

static inline HVX_VectorPair accum_q8_0_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static void tiled_vec_dot_q4_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q4_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict vptr0 = (const HVX_Vector *) (tile_ptr + (kt + 0) * 640);
        const HVX_Vector * restrict v_act0_0 = (const HVX_Vector *) (y0_q + (kt + 0) * 1152);
        const HVX_Vector * restrict v_act1_0 = (const HVX_Vector *) (y1_q + (kt + 0) * 1152);

        const HVX_Vector * restrict vptr1 = (const HVX_Vector *) (tile_ptr + (kt + 1) * 640);
        const HVX_Vector * restrict v_act0_1 = (const HVX_Vector *) (y0_q + (kt + 1) * 1152);
        const HVX_Vector * restrict v_act1_1 = (const HVX_Vector *) (y1_q + (kt + 1) * 1152);

        HVX_VectorPair v_sums0 = accum_4bit_32x2(vptr0, v_act0_0, v_act1_0, i8);
        HVX_VectorPair v_sums1 = accum_4bit_32x2(vptr1, v_act0_1, v_act1_1, i8);

        HVX_Vector v_sum_c0_0 = Q6_V_lo_W(v_sums0);
        HVX_Vector v_sum_c1_0 = Q6_V_hi_W(v_sums0);
        HVX_Vector v_sum_c0_1 = Q6_V_lo_W(v_sums1);
        HVX_Vector v_sum_c1_1 = Q6_V_hi_W(v_sums1);

        HVX_Vector v_sum_sf_c0_0 = Q6_Vsf_equals_Vw(v_sum_c0_0);
        HVX_Vector v_sum_sf_c1_0 = Q6_Vsf_equals_Vw(v_sum_c1_0);
        HVX_Vector v_sum_sf_c0_1 = Q6_Vsf_equals_Vw(v_sum_c0_1);
        HVX_Vector v_sum_sf_c1_1 = Q6_Vsf_equals_Vw(v_sum_c1_1);

        HVX_Vector v_scale_w0 = vptr0[4];
        HVX_Vector v_scale_w1 = vptr1[4];
        HVX_Vector v_scale_a_c0_0 = v_act0_0[8];
        HVX_Vector v_scale_a_c1_0 = v_act1_0[8];
        HVX_Vector v_scale_a_c0_1 = v_act0_1[8];
        HVX_Vector v_scale_a_c1_1 = v_act1_1[8];

        HVX_Vector v_scale_comb_c0_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c0_0);
        HVX_Vector v_scale_comb_c1_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c1_0);
        HVX_Vector v_scale_comb_c0_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c0_1);
        HVX_Vector v_scale_comb_c1_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c1_1);

        HVX_Vector v_sum_scaled_c0_0 = hvx_vec_mul_f32_f32(v_sum_sf_c0_0, v_scale_comb_c0_0);
        HVX_Vector v_sum_scaled_c1_0 = hvx_vec_mul_f32_f32(v_sum_sf_c1_0, v_scale_comb_c1_0);
        HVX_Vector v_sum_scaled_c0_1 = hvx_vec_mul_f32_f32(v_sum_sf_c0_1, v_scale_comb_c0_1);
        HVX_Vector v_sum_scaled_c1_1 = hvx_vec_mul_f32_f32(v_sum_sf_c1_1, v_scale_comb_c1_1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_add_f32_f32(v_sum_scaled_c0_0, v_sum_scaled_c0_1));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_add_f32_f32(v_sum_scaled_c1_0, v_sum_scaled_c1_1));
    }

    for (; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void tiled_vec_dot_q4_1_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q4_1_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict vptr0 = (const HVX_Vector *) (tile_ptr + (kt + 0) * 640);
        const HVX_Vector * restrict v_act0_0 = (const HVX_Vector *) (y0_q + (kt + 0) * 1280);
        const HVX_Vector * restrict v_act1_0 = (const HVX_Vector *) (y1_q + (kt + 0) * 1280);

        const HVX_Vector * restrict vptr1 = (const HVX_Vector *) (tile_ptr + (kt + 1) * 640);
        const HVX_Vector * restrict v_act0_1 = (const HVX_Vector *) (y0_q + (kt + 1) * 1280);
        const HVX_Vector * restrict v_act1_1 = (const HVX_Vector *) (y1_q + (kt + 1) * 1280);

        HVX_VectorPair v_sums0 = accum_4bit_32x2(vptr0, v_act0_0, v_act1_0, Q6_V_vzero());
        HVX_VectorPair v_sums1 = accum_4bit_32x2(vptr1, v_act0_1, v_act1_1, Q6_V_vzero());

        HVX_Vector v_sum_c0_0 = Q6_V_lo_W(v_sums0);
        HVX_Vector v_sum_c1_0 = Q6_V_hi_W(v_sums0);
        HVX_Vector v_sum_c0_1 = Q6_V_lo_W(v_sums1);
        HVX_Vector v_sum_c1_1 = Q6_V_hi_W(v_sums1);

        HVX_Vector v_sum_sf_c0_0 = Q6_Vsf_equals_Vw(v_sum_c0_0);
        HVX_Vector v_sum_sf_c1_0 = Q6_Vsf_equals_Vw(v_sum_c1_0);
        HVX_Vector v_sum_sf_c0_1 = Q6_Vsf_equals_Vw(v_sum_c0_1);
        HVX_Vector v_sum_sf_c1_1 = Q6_Vsf_equals_Vw(v_sum_c1_1);

        HVX_Vector v_scale_offset0 = vptr0[4];
        HVX_VectorPair p_deal0 = Q6_W_vdeal_VVR(v_scale_offset0, v_scale_offset0, -2);
        HVX_Vector v_scale0 = Q6_V_lo_W(p_deal0);
        HVX_Vector v_offset0 = Q6_V_hi_W(p_deal0);

        HVX_Vector v_scale_offset1 = vptr1[4];
        HVX_VectorPair p_deal1 = Q6_W_vdeal_VVR(v_scale_offset1, v_scale_offset1, -2);
        HVX_Vector v_scale1 = Q6_V_lo_W(p_deal1);
        HVX_Vector v_offset1 = Q6_V_hi_W(p_deal1);

        HVX_Vector v_scale_a_c0_0 = v_act0_0[8];
        HVX_Vector v_sum_a_c0_0   = v_act0_0[9];
        HVX_Vector v_scale_a_c1_0 = v_act1_0[8];
        HVX_Vector v_sum_a_c1_0   = v_act1_0[9];

        HVX_Vector v_scale_a_c0_1 = v_act0_1[8];
        HVX_Vector v_sum_a_c0_1   = v_act0_1[9];
        HVX_Vector v_scale_a_c1_1 = v_act1_1[8];
        HVX_Vector v_sum_a_c1_1   = v_act1_1[9];

        HVX_Vector v_scale_comb_c0_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale0, v_scale_a_c0_0);
        HVX_Vector v_offset_comb_c0_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset0, v_sum_a_c0_0);
        HVX_Vector v_scale_comb_c1_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale0, v_scale_a_c1_0);
        HVX_Vector v_offset_comb_c1_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset0, v_sum_a_c1_0);

        HVX_Vector v_scale_comb_c0_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale1, v_scale_a_c0_1);
        HVX_Vector v_offset_comb_c0_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset1, v_sum_a_c0_1);
        HVX_Vector v_scale_comb_c1_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale1, v_scale_a_c1_1);
        HVX_Vector v_offset_comb_c1_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset1, v_sum_a_c1_1);

        HVX_Vector v_scaled_dot_c0_0 = hvx_vec_mul_f32_f32(v_sum_sf_c0_0, v_scale_comb_c0_0);
        HVX_Vector v_sum_scaled_c0_0 = hvx_vec_add_f32_f32(v_scaled_dot_c0_0, v_offset_comb_c0_0);

        HVX_Vector v_scaled_dot_c1_0 = hvx_vec_mul_f32_f32(v_sum_sf_c1_0, v_scale_comb_c1_0);
        HVX_Vector v_sum_scaled_c1_0 = hvx_vec_add_f32_f32(v_scaled_dot_c1_0, v_offset_comb_c1_0);

        HVX_Vector v_scaled_dot_c0_1 = hvx_vec_mul_f32_f32(v_sum_sf_c0_1, v_scale_comb_c0_1);
        HVX_Vector v_sum_scaled_c0_1 = hvx_vec_add_f32_f32(v_scaled_dot_c0_1, v_offset_comb_c0_1);

        HVX_Vector v_scaled_dot_c1_1 = hvx_vec_mul_f32_f32(v_sum_sf_c1_1, v_scale_comb_c1_1);
        HVX_Vector v_sum_scaled_c1_1 = hvx_vec_add_f32_f32(v_scaled_dot_c1_1, v_offset_comb_c1_1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_add_f32_f32(v_sum_scaled_c0_0, v_sum_scaled_c0_1));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_add_f32_f32(v_sum_scaled_c1_0, v_sum_scaled_c1_1));
    }

    for (; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void tiled_vec_dot_q8_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_q8_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict vptr0 = (const HVX_Vector *) (tile_ptr + (kt + 0) * 1152);
        const HVX_Vector * restrict v_act0_0 = (const HVX_Vector *) (y0_q + (kt + 0) * 1152);
        const HVX_Vector * restrict v_act1_0 = (const HVX_Vector *) (y1_q + (kt + 0) * 1152);

        const HVX_Vector * restrict vptr1 = (const HVX_Vector *) (tile_ptr + (kt + 1) * 1152);
        const HVX_Vector * restrict v_act0_1 = (const HVX_Vector *) (y0_q + (kt + 1) * 1152);
        const HVX_Vector * restrict v_act1_1 = (const HVX_Vector *) (y1_q + (kt + 1) * 1152);

        HVX_VectorPair v_sums0 = accum_q8_0_32x2(vptr0, v_act0_0, v_act1_0);
        HVX_VectorPair v_sums1 = accum_q8_0_32x2(vptr1, v_act0_1, v_act1_1);

        HVX_Vector v_sum_c0_0 = Q6_V_lo_W(v_sums0);
        HVX_Vector v_sum_c1_0 = Q6_V_hi_W(v_sums0);
        HVX_Vector v_sum_c0_1 = Q6_V_lo_W(v_sums1);
        HVX_Vector v_sum_c1_1 = Q6_V_hi_W(v_sums1);

        HVX_Vector v_sum_sf_c0_0 = Q6_Vsf_equals_Vw(v_sum_c0_0);
        HVX_Vector v_sum_sf_c1_0 = Q6_Vsf_equals_Vw(v_sum_c1_0);
        HVX_Vector v_sum_sf_c0_1 = Q6_Vsf_equals_Vw(v_sum_c0_1);
        HVX_Vector v_sum_sf_c1_1 = Q6_Vsf_equals_Vw(v_sum_c1_1);

        HVX_Vector v_scale_w0 = vptr0[8];
        HVX_Vector v_scale_w1 = vptr1[8];
        HVX_Vector v_scale_a_c0_0 = v_act0_0[8];
        HVX_Vector v_scale_a_c1_0 = v_act1_0[8];
        HVX_Vector v_scale_a_c0_1 = v_act0_1[8];
        HVX_Vector v_scale_a_c1_1 = v_act1_1[8];

        HVX_Vector v_scale_comb_c0_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c0_0);
        HVX_Vector v_scale_comb_c1_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c1_0);
        HVX_Vector v_scale_comb_c0_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c0_1);
        HVX_Vector v_scale_comb_c1_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c1_1);

        HVX_Vector v_sum_scaled_c0_0 = hvx_vec_mul_f32_f32(v_sum_sf_c0_0, v_scale_comb_c0_0);
        HVX_Vector v_sum_scaled_c1_0 = hvx_vec_mul_f32_f32(v_sum_sf_c1_0, v_scale_comb_c1_0);
        HVX_Vector v_sum_scaled_c0_1 = hvx_vec_mul_f32_f32(v_sum_sf_c0_1, v_scale_comb_c0_1);
        HVX_Vector v_sum_scaled_c1_1 = hvx_vec_mul_f32_f32(v_sum_sf_c1_1, v_scale_comb_c1_1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_add_f32_f32(v_sum_scaled_c0_0, v_sum_scaled_c0_1));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_add_f32_f32(v_sum_scaled_c1_0, v_sum_scaled_c1_1));
    }

    for (; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void tiled_vec_dot_iq4nl_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_iq4nl_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict vptr0 = (const HVX_Vector *) (tile_ptr + (kt + 0) * 640);
        const HVX_Vector * restrict v_act0_0 = (const HVX_Vector *) (y0_q + (kt + 0) * 1152);
        const HVX_Vector * restrict v_act1_0 = (const HVX_Vector *) (y1_q + (kt + 0) * 1152);

        const HVX_Vector * restrict vptr1 = (const HVX_Vector *) (tile_ptr + (kt + 1) * 640);
        const HVX_Vector * restrict v_act0_1 = (const HVX_Vector *) (y0_q + (kt + 1) * 1152);
        const HVX_Vector * restrict v_act1_1 = (const HVX_Vector *) (y1_q + (kt + 1) * 1152);

        HVX_VectorPair v_sums0 = accum_4bit_32x2_lut(vptr0, v_act0_0, v_act1_0, mask_h4, lut);
        HVX_VectorPair v_sums1 = accum_4bit_32x2_lut(vptr1, v_act0_1, v_act1_1, mask_h4, lut);

        HVX_Vector v_sum_c0_0 = Q6_V_lo_W(v_sums0);
        HVX_Vector v_sum_c1_0 = Q6_V_hi_W(v_sums0);
        HVX_Vector v_sum_c0_1 = Q6_V_lo_W(v_sums1);
        HVX_Vector v_sum_c1_1 = Q6_V_hi_W(v_sums1);

        HVX_Vector v_sum_sf_c0_0 = Q6_Vsf_equals_Vw(v_sum_c0_0);
        HVX_Vector v_sum_sf_c1_0 = Q6_Vsf_equals_Vw(v_sum_c1_0);
        HVX_Vector v_sum_sf_c0_1 = Q6_Vsf_equals_Vw(v_sum_c0_1);
        HVX_Vector v_sum_sf_c1_1 = Q6_Vsf_equals_Vw(v_sum_c1_1);

        HVX_Vector v_scale_w0 = vptr0[4];
        HVX_Vector v_scale_w1 = vptr1[4];
        HVX_Vector v_scale_a_c0_0 = v_act0_0[8];
        HVX_Vector v_scale_a_c1_0 = v_act1_0[8];
        HVX_Vector v_scale_a_c0_1 = v_act0_1[8];
        HVX_Vector v_scale_a_c1_1 = v_act1_1[8];

        HVX_Vector v_scale_comb_c0_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c0_0);
        HVX_Vector v_scale_comb_c1_0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w0, v_scale_a_c1_0);
        HVX_Vector v_scale_comb_c0_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c0_1);
        HVX_Vector v_scale_comb_c1_1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w1, v_scale_a_c1_1);

        HVX_Vector v_sum_scaled_c0_0 = hvx_vec_mul_f32_f32(v_sum_sf_c0_0, v_scale_comb_c0_0);
        HVX_Vector v_sum_scaled_c1_0 = hvx_vec_mul_f32_f32(v_sum_sf_c1_0, v_scale_comb_c1_0);
        HVX_Vector v_sum_scaled_c0_1 = hvx_vec_mul_f32_f32(v_sum_sf_c0_1, v_scale_comb_c0_1);
        HVX_Vector v_sum_scaled_c1_1 = hvx_vec_mul_f32_f32(v_sum_sf_c1_1, v_scale_comb_c1_1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_add_f32_f32(v_sum_scaled_c0_0, v_sum_scaled_c0_1));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_add_f32_f32(v_sum_scaled_c1_0, v_sum_scaled_c1_1));
    }

    for (; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void tiled_vec_dot_mxfp4_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_f16 = v_act[8];
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void tiled_vec_dot_mxfp4_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    uint32_t kt = 0;
    for (; kt + 1 < n_k_tiles; kt += 2) {
        const HVX_Vector * restrict vptr0 = (const HVX_Vector *) (tile_ptr + (kt + 0) * 640);
        const HVX_Vector * restrict v_act0_0 = (const HVX_Vector *) (y0_q + (kt + 0) * 1152);
        const HVX_Vector * restrict v_act1_0 = (const HVX_Vector *) (y1_q + (kt + 0) * 1152);

        const HVX_Vector * restrict vptr1 = (const HVX_Vector *) (tile_ptr + (kt + 1) * 640);
        const HVX_Vector * restrict v_act0_1 = (const HVX_Vector *) (y0_q + (kt + 1) * 1152);
        const HVX_Vector * restrict v_act1_1 = (const HVX_Vector *) (y1_q + (kt + 1) * 1152);

        HVX_VectorPair v_sums0 = accum_4bit_32x2_lut(vptr0, v_act0_0, v_act1_0, mask_h4, lut);
        HVX_VectorPair v_sums1 = accum_4bit_32x2_lut(vptr1, v_act0_1, v_act1_1, mask_h4, lut);

        HVX_Vector v_sum_c0_0 = Q6_V_lo_W(v_sums0);
        HVX_Vector v_sum_c1_0 = Q6_V_hi_W(v_sums0);
        HVX_Vector v_sum_c0_1 = Q6_V_lo_W(v_sums1);
        HVX_Vector v_sum_c1_1 = Q6_V_hi_W(v_sums1);

        HVX_Vector v_sum_sf_c0_0 = Q6_Vsf_equals_Vw(v_sum_c0_0);
        HVX_Vector v_sum_sf_c1_0 = Q6_Vsf_equals_Vw(v_sum_c1_0);
        HVX_Vector v_sum_sf_c0_1 = Q6_Vsf_equals_Vw(v_sum_c0_1);
        HVX_Vector v_sum_sf_c1_1 = Q6_Vsf_equals_Vw(v_sum_c1_1);

        HVX_Vector v_scale_w0 = hvx_vmem(tile_ptr + (kt + 0) * 640 + 512);
        HVX_Vector r0_d0 = Q6_V_vdelta_VV(v_scale_w0, expand);
        r0_d0 = Q6_V_vand_VV(r0_d0, e8m0_mask);
        HVX_Vector v_scale_w_f32_0 = Q6_Vw_vasl_VwR(r0_d0, 23);

        HVX_Vector v_scale_w1 = hvx_vmem(tile_ptr + (kt + 1) * 640 + 512);
        HVX_Vector r0_d1 = Q6_V_vdelta_VV(v_scale_w1, expand);
        r0_d1 = Q6_V_vand_VV(r0_d1, e8m0_mask);
        HVX_Vector v_scale_w_f32_1 = Q6_Vw_vasl_VwR(r0_d1, 23);

        HVX_Vector v_scale_a_c0_f16_0 = v_act0_0[8];
        HVX_Vector v_scale_a_c1_f16_0 = v_act1_0[8];
        HVX_Vector v_scale_a_c0_f16_1 = v_act0_1[8];
        HVX_Vector v_scale_a_c1_f16_1 = v_act1_1[8];

        HVX_VectorPair p_scale_a_c0_f32_0 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16_0);
        HVX_VectorPair p_scale_a_c1_f32_0 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16_0);
        HVX_VectorPair p_scale_a_c0_f32_1 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16_1);
        HVX_VectorPair p_scale_a_c1_f32_1 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16_1);

        HVX_Vector v_scale_a_c0_0 = Q6_V_lo_W(p_scale_a_c0_f32_0);
        HVX_Vector v_scale_a_c1_0 = Q6_V_lo_W(p_scale_a_c1_f32_0);
        HVX_Vector v_scale_a_c0_1 = Q6_V_lo_W(p_scale_a_c0_f32_1);
        HVX_Vector v_scale_a_c1_1 = Q6_V_lo_W(p_scale_a_c1_f32_1);

        HVX_Vector v_scale_comb_c0_0 = hvx_vec_mul_f32_f32(v_scale_w_f32_0, v_scale_a_c0_0);
        HVX_Vector v_scale_comb_c1_0 = hvx_vec_mul_f32_f32(v_scale_w_f32_0, v_scale_a_c1_0);
        HVX_Vector v_scale_comb_c0_1 = hvx_vec_mul_f32_f32(v_scale_w_f32_1, v_scale_a_c0_1);
        HVX_Vector v_scale_comb_c1_1 = hvx_vec_mul_f32_f32(v_scale_w_f32_1, v_scale_a_c1_1);

        HVX_Vector v_sum_scaled_c0_0 = hvx_vec_mul_f32_f32(v_sum_sf_c0_0, v_scale_comb_c0_0);
        HVX_Vector v_sum_scaled_c1_0 = hvx_vec_mul_f32_f32(v_sum_sf_c1_0, v_scale_comb_c1_0);
        HVX_Vector v_sum_scaled_c0_1 = hvx_vec_mul_f32_f32(v_sum_sf_c0_1, v_scale_comb_c0_1);
        HVX_Vector v_sum_scaled_c1_1 = hvx_vec_mul_f32_f32(v_sum_sf_c1_1, v_scale_comb_c1_1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_add_f32_f32(v_sum_scaled_c0_0, v_sum_scaled_c0_1));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_add_f32_f32(v_sum_scaled_c1_0, v_sum_scaled_c1_1));
    }

    for (; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_c0_f16 = v_act0[8];
        HVX_Vector v_scale_a_c1_f16 = v_act1[8];

        HVX_VectorPair p_scale_a_c0_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16);
        HVX_VectorPair p_scale_a_c1_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16);

        HVX_Vector v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_Vector v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    v_sum_float_c0 = hvx_vec_mul_f32_f32(v_sum_float_c0, hvx_vec_splat_f32(0.5f));
    v_sum_float_c1 = hvx_vec_mul_f32_f32(v_sum_float_c1, hvx_vec_splat_f32(0.5f));

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static inline void quantize_f32_q8_0_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));

    for (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
        hvx_copy_f32_aa(tmp_data, src_data, ne0);

        quantize_row_f32_q8_0_tiled((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_q8_1_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));

    for (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
        hvx_copy_f32_aa(tmp_data, src_data, ne0);

        quantize_row_f32_q8_1_tiled((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_q8_0_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const uint8_t * restrict src_ptr = (const uint8_t *) src + r * src_row_size + c * qk * sizeof(float);
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1152;

        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);

        if (c == nb - 1) {
            uint32_t active_elements = ne0 - c * qk;
            hvx_splat_f32_a(tmp_data, 0.0f, qk);
            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
        } else {
            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
        }

        quantize_block_f32_q8_0_tiled((float *) tmp_data, dst_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}

static inline void quantize_f32_q8_1_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const uint8_t * restrict src_ptr = (const uint8_t *) src + r * src_row_size + c * qk * sizeof(float);
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;

        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);

        if (c == nb - 1) {
            uint32_t active_elements = ne0 - c * qk;
            hvx_splat_f32_a(tmp_data, 0.0f, qk);
            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
        } else {
            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
        }

        quantize_block_f32_q8_1_tiled((float *) tmp_data, dst_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}
