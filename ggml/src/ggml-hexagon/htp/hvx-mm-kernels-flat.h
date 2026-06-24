// Dynamic quantizers that produce flat (non-tiled) activations

static inline void quantize_block_f32_q8_0_flat(
    float * restrict x,
    uint8_t * restrict y_quants,
    __fp16 * restrict y_scales,
    uint32_t block_idx
) {
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

    * (HVX_Vector *) (y_quants + block_idx * 128) = vx_i8;

    HVX_VectorPair vp1 = Q6_W_vshuff_VVR(vd23_hf, vd01_hf, -2);
    HVX_VectorPair vp2 = Q6_W_vshuff_VVR(Q6_V_hi_W(vp1), Q6_V_lo_W(vp1), -2);
    HVX_Vector v_scales = Q6_V_lo_W(vp2);
    hvx_vec_store_u(y_scales + block_idx * 4, 8, v_scales);
}

static inline void quantize_block_f32_q8_1_flat(
    float * restrict x,
    uint8_t * restrict y_quants,
    __fp16 * restrict y_scales,
    uint32_t block_idx
) {
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

    * (HVX_Vector *) (y_quants + block_idx * 128) = vx_i8;

    HVX_VectorPair vp1 = Q6_W_vshuff_VVR(vd23_hf, vd01_hf, -2);
    HVX_VectorPair vp2 = Q6_W_vshuff_VVR(Q6_V_hi_W(vp1), Q6_V_lo_W(vp1), -2);
    HVX_Vector v_scales = Q6_V_lo_W(vp2);

    HVX_VectorPair v_deal1 = Q6_W_vdeal_VVR(v_sums, v_sums, -4);
    HVX_Vector v_even1 = Q6_V_lo_W(v_deal1);
    HVX_VectorPair v_deal2 = Q6_W_vdeal_VVR(v_even1, v_even1, -4);
    HVX_Vector v_even2 = Q6_V_lo_W(v_deal2);
    HVX_VectorPair v_deal3 = Q6_W_vdeal_VVR(v_even2, v_even2, -4);
    HVX_Vector v_sums_shuffled = Q6_V_lo_W(v_deal3);

    HVX_Vector v_sums_sf = Q6_Vsf_equals_Vw(v_sums_shuffled);
    HVX_Vector v_sums_hf = hvx_vec_f32_to_f16(v_sums_sf, Q6_V_vzero());

    HVX_Vector v_prod = hvx_vec_mul_f16_f16(v_scales, v_sums_hf);

    HVX_VectorPair vp_scales = Q6_W_vshuff_VVR(v_prod, v_scales, -2);
    HVX_Vector v_final = Q6_V_lo_W(vp_scales);

    hvx_vec_store_u(y_scales + block_idx * 8, 16, v_final);
}

static inline void quantize_row_f32_q8_0_flat(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t quants_size = hex_round_up(k, 128);
    uint8_t * restrict y_quants = y;
    __fp16 * restrict y_scales = (__fp16 *) (y + quants_size);

    const uint32_t nb = (k + 127) / 128;
    for (uint32_t i = 0; i < nb; i++) {
        quantize_block_f32_q8_0_flat(x + i * 128, y_quants, y_scales, i);
    }
}

static inline void quantize_row_f32_q8_1_flat(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t quants_size = hex_round_up(k, 128);
    uint8_t * restrict y_quants = y;
    __fp16 * restrict y_scales = (__fp16 *) (y + quants_size);

    const uint32_t nb = (k + 127) / 128;
    for (uint32_t i = 0; i < nb; i++) {
        quantize_block_f32_q8_1_flat(x + i * 128, y_quants, y_scales, i);
    }
}

static inline void quantize_f32_q8_0_flat_kernel(
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

        quantize_row_f32_q8_0_flat((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_q8_1_flat_kernel(
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

        quantize_row_f32_q8_1_flat((float *) tmp_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_f32_flat_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    const size_t src_row_size = ne0 * sizeof(float);
    for (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

static inline void quantize_f32_f16_flat_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    const size_t src_row_size = ne0 * sizeof(float);
    for (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

static inline void quantize_f16_f16_flat_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (void) tmp_data;
    const size_t src_row_size = ne0 * sizeof(float);
    for (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

// Dot kernels that consume flat (non-tiled) activations

static void flat_vec_dot_q4_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx_i8 = * (const HVX_Vector *) (y_q + block_idx * 128);
        HVX_Vector v_act_raw = Q6_V_vror_VR(vx_i8, sub_idx * 32);

        HVX_Vector v_act_rep[8];
        v_act_rep[0] = Q6_V_vdelta_VV(v_act_raw, v_repl_ctrl);
        v_act_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4), v_repl_ctrl);
        v_act_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 8), v_repl_ctrl);
        v_act_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 12), v_repl_ctrl);
        v_act_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 16), v_repl_ctrl);
        v_act_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 20), v_repl_ctrl);
        v_act_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 24), v_repl_ctrl);
        v_act_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 28), v_repl_ctrl);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act_rep, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];

        __fp16 scale_a_val = y_scales[kt];
        HVX_Vector v_scale_a = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a_val));

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void flat_vec_dot_q4_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx0_i8 = * (const HVX_Vector *) (y0_q + block_idx * 128);
        HVX_Vector vx1_i8 = * (const HVX_Vector *) (y1_q + block_idx * 128);

        HVX_Vector v_act0_raw = Q6_V_vror_VR(vx0_i8, sub_idx * 32);
        HVX_Vector v_act1_raw = Q6_V_vror_VR(vx1_i8, sub_idx * 32);

        HVX_Vector v_act0_rep[8];
        v_act0_rep[0] = Q6_V_vdelta_VV(v_act0_raw, v_repl_ctrl);
        v_act0_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 4), v_repl_ctrl);
        v_act0_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 8), v_repl_ctrl);
        v_act0_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 12), v_repl_ctrl);
        v_act0_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 16), v_repl_ctrl);
        v_act0_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 20), v_repl_ctrl);
        v_act0_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 24), v_repl_ctrl);
        v_act0_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 28), v_repl_ctrl);

        HVX_Vector v_act1_rep[8];
        v_act1_rep[0] = Q6_V_vdelta_VV(v_act1_raw, v_repl_ctrl);
        v_act1_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 4), v_repl_ctrl);
        v_act1_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 8), v_repl_ctrl);
        v_act1_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 12), v_repl_ctrl);
        v_act1_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 16), v_repl_ctrl);
        v_act1_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 20), v_repl_ctrl);
        v_act1_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 24), v_repl_ctrl);
        v_act1_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 28), v_repl_ctrl);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0_rep, v_act1_rep, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];

        __fp16 scale_a0_val = y0_scales[kt];
        __fp16 scale_a1_val = y1_scales[kt];
        HVX_Vector v_scale_a0 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a0_val));
        HVX_Vector v_scale_a1 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a1_val));

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void flat_vec_dot_q4_1_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx_i8 = * (const HVX_Vector *) (y_q + block_idx * 128);
        HVX_Vector v_act_raw = Q6_V_vror_VR(vx_i8, sub_idx * 32);

        HVX_Vector v_act_rep[8];
        v_act_rep[0] = Q6_V_vdelta_VV(v_act_raw, v_repl_ctrl);
        v_act_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4), v_repl_ctrl);
        v_act_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 8), v_repl_ctrl);
        v_act_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 12), v_repl_ctrl);
        v_act_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 16), v_repl_ctrl);
        v_act_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 20), v_repl_ctrl);
        v_act_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 24), v_repl_ctrl);
        v_act_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 28), v_repl_ctrl);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act_rep, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        __fp16 scale_a_val = y_scales[kt * 2 + 0];
        __fp16 sum_a_val   = y_scales[kt * 2 + 1];
        HVX_Vector v_scale_a = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a_val));
        HVX_Vector v_sum_a   = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&sum_a_val));

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void flat_vec_dot_q4_1_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx0_i8 = * (const HVX_Vector *) (y0_q + block_idx * 128);
        HVX_Vector vx1_i8 = * (const HVX_Vector *) (y1_q + block_idx * 128);

        HVX_Vector v_act0_raw = Q6_V_vror_VR(vx0_i8, sub_idx * 32);
        HVX_Vector v_act1_raw = Q6_V_vror_VR(vx1_i8, sub_idx * 32);

        HVX_Vector v_act0_rep[8];
        v_act0_rep[0] = Q6_V_vdelta_VV(v_act0_raw, v_repl_ctrl);
        v_act0_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 4), v_repl_ctrl);
        v_act0_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 8), v_repl_ctrl);
        v_act0_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 12), v_repl_ctrl);
        v_act0_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 16), v_repl_ctrl);
        v_act0_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 20), v_repl_ctrl);
        v_act0_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 24), v_repl_ctrl);
        v_act0_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 28), v_repl_ctrl);

        HVX_Vector v_act1_rep[8];
        v_act1_rep[0] = Q6_V_vdelta_VV(v_act1_raw, v_repl_ctrl);
        v_act1_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 4), v_repl_ctrl);
        v_act1_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 8), v_repl_ctrl);
        v_act1_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 12), v_repl_ctrl);
        v_act1_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 16), v_repl_ctrl);
        v_act1_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 20), v_repl_ctrl);
        v_act1_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 24), v_repl_ctrl);
        v_act1_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 28), v_repl_ctrl);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0_rep, v_act1_rep, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        __fp16 scale_a0_val = y0_scales[kt * 2 + 0];
        __fp16 sum_a0_val   = y0_scales[kt * 2 + 1];
        __fp16 scale_a1_val = y1_scales[kt * 2 + 0];
        __fp16 sum_a1_val   = y1_scales[kt * 2 + 1];

        HVX_Vector v_scale_a0 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a0_val));
        HVX_Vector v_sum_a0   = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&sum_a0_val));
        HVX_Vector v_scale_a1 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a1_val));
        HVX_Vector v_sum_a1   = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&sum_a1_val));

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a1);

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

static void flat_vec_dot_q8_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx_i8 = * (const HVX_Vector *) (y_q + block_idx * 128);
        HVX_Vector v_act_raw = Q6_V_vror_VR(vx_i8, sub_idx * 32);

        HVX_Vector v_act_rep[8];
        v_act_rep[0] = Q6_V_vdelta_VV(v_act_raw, v_repl_ctrl);
        v_act_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4), v_repl_ctrl);
        v_act_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 8), v_repl_ctrl);
        v_act_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 12), v_repl_ctrl);
        v_act_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 16), v_repl_ctrl);
        v_act_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 20), v_repl_ctrl);
        v_act_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 24), v_repl_ctrl);
        v_act_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 28), v_repl_ctrl);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act_rep);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];

        __fp16 scale_a_val = y_scales[kt];
        HVX_Vector v_scale_a = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a_val));

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void flat_vec_dot_q8_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx0_i8 = * (const HVX_Vector *) (y0_q + block_idx * 128);
        HVX_Vector vx1_i8 = * (const HVX_Vector *) (y1_q + block_idx * 128);

        HVX_Vector v_act0_raw = Q6_V_vror_VR(vx0_i8, sub_idx * 32);
        HVX_Vector v_act1_raw = Q6_V_vror_VR(vx1_i8, sub_idx * 32);

        HVX_Vector v_act0_rep[8];
        v_act0_rep[0] = Q6_V_vdelta_VV(v_act0_raw, v_repl_ctrl);
        v_act0_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 4), v_repl_ctrl);
        v_act0_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 8), v_repl_ctrl);
        v_act0_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 12), v_repl_ctrl);
        v_act0_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 16), v_repl_ctrl);
        v_act0_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 20), v_repl_ctrl);
        v_act0_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 24), v_repl_ctrl);
        v_act0_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 28), v_repl_ctrl);

        HVX_Vector v_act1_rep[8];
        v_act1_rep[0] = Q6_V_vdelta_VV(v_act1_raw, v_repl_ctrl);
        v_act1_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 4), v_repl_ctrl);
        v_act1_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 8), v_repl_ctrl);
        v_act1_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 12), v_repl_ctrl);
        v_act1_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 16), v_repl_ctrl);
        v_act1_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 20), v_repl_ctrl);
        v_act1_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 24), v_repl_ctrl);
        v_act1_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 28), v_repl_ctrl);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0_rep, v_act1_rep);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[8];

        __fp16 scale_a0_val = y0_scales[kt];
        __fp16 scale_a1_val = y1_scales[kt];
        HVX_Vector v_scale_a0 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a0_val));
        HVX_Vector v_scale_a1 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a1_val));

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void flat_vec_dot_iq4nl_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx = * (const HVX_Vector *) (y_q + block_idx * 128);
        HVX_Vector v_act_raw = Q6_V_vror_VR(vx, sub_idx * 32);

        HVX_Vector v_act_rep[8];
        v_act_rep[0] = Q6_V_vdelta_VV(v_act_raw, v_repl_ctrl);
        v_act_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4), v_repl_ctrl);
        v_act_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 8), v_repl_ctrl);
        v_act_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 12), v_repl_ctrl);
        v_act_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 16), v_repl_ctrl);
        v_act_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 20), v_repl_ctrl);
        v_act_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 24), v_repl_ctrl);
        v_act_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 28), v_repl_ctrl);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act_rep, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];

        __fp16 scale_a_val = y_scales[kt];
        HVX_Vector v_scale_a = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a_val));

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void flat_vec_dot_iq4nl_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4        = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut            = *(const HVX_Vector *) kvalues_iq4nl_lut;

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx0 = * (const HVX_Vector *) (y0_q + block_idx * 128);
        HVX_Vector vx1 = * (const HVX_Vector *) (y1_q + block_idx * 128);

        HVX_Vector v_act0_raw = Q6_V_vror_VR(vx0, sub_idx * 32);
        HVX_Vector v_act1_raw = Q6_V_vror_VR(vx1, sub_idx * 32);

        HVX_Vector v_act0_rep[8];
        v_act0_rep[0] = Q6_V_vdelta_VV(v_act0_raw, v_repl_ctrl);
        v_act0_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 4), v_repl_ctrl);
        v_act0_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 8), v_repl_ctrl);
        v_act0_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 12), v_repl_ctrl);
        v_act0_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 16), v_repl_ctrl);
        v_act0_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 20), v_repl_ctrl);
        v_act0_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 24), v_repl_ctrl);
        v_act0_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 28), v_repl_ctrl);

        HVX_Vector v_act1_rep[8];
        v_act1_rep[0] = Q6_V_vdelta_VV(v_act1_raw, v_repl_ctrl);
        v_act1_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 4), v_repl_ctrl);
        v_act1_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 8), v_repl_ctrl);
        v_act1_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 12), v_repl_ctrl);
        v_act1_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 16), v_repl_ctrl);
        v_act1_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 20), v_repl_ctrl);
        v_act1_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 24), v_repl_ctrl);
        v_act1_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 28), v_repl_ctrl);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0_rep, v_act1_rep, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];

        __fp16 scale_a0_val = y0_scales[kt];
        __fp16 scale_a1_val = y1_scales[kt];
        HVX_Vector v_scale_a0 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a0_val));
        HVX_Vector v_scale_a1 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a1_val));

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
}

static void flat_vec_dot_mxfp4_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y_scales = (const __fp16 *) (y_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx = * (const HVX_Vector *) (y_q + block_idx * 128);
        HVX_Vector v_act_raw = Q6_V_vror_VR(vx, sub_idx * 32);

        HVX_Vector v_act_rep[8];
        v_act_rep[0] = Q6_V_vdelta_VV(v_act_raw, v_repl_ctrl);
        v_act_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 4), v_repl_ctrl);
        v_act_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 8), v_repl_ctrl);
        v_act_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 12), v_repl_ctrl);
        v_act_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 16), v_repl_ctrl);
        v_act_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 20), v_repl_ctrl);
        v_act_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 24), v_repl_ctrl);
        v_act_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act_raw, 28), v_repl_ctrl);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act_rep, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        __fp16 scale_a_val = y_scales[kt];
        HVX_Vector v_scale_a_f16 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a_val));
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
}

static void flat_vec_dot_mxfp4_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

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

    const uint32_t quants_size = hex_round_up(n, 128);
    const __fp16 * restrict y0_scales = (const __fp16 *) (y0_q + quants_size);
    const __fp16 * restrict y1_scales = (const __fp16 *) (y1_q + quants_size);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        uint32_t block_idx = kt / 4;
        uint32_t sub_idx = kt % 4;

        HVX_Vector vx0 = * (const HVX_Vector *) (y0_q + block_idx * 128);
        HVX_Vector vx1 = * (const HVX_Vector *) (y1_q + block_idx * 128);

        HVX_Vector v_act0_raw = Q6_V_vror_VR(vx0, sub_idx * 32);
        HVX_Vector v_act1_raw = Q6_V_vror_VR(vx1, sub_idx * 32);

        HVX_Vector v_act0_rep[8];
        v_act0_rep[0] = Q6_V_vdelta_VV(v_act0_raw, v_repl_ctrl);
        v_act0_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 4), v_repl_ctrl);
        v_act0_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 8), v_repl_ctrl);
        v_act0_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 12), v_repl_ctrl);
        v_act0_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 16), v_repl_ctrl);
        v_act0_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 20), v_repl_ctrl);
        v_act0_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 24), v_repl_ctrl);
        v_act0_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act0_raw, 28), v_repl_ctrl);

        HVX_Vector v_act1_rep[8];
        v_act1_rep[0] = Q6_V_vdelta_VV(v_act1_raw, v_repl_ctrl);
        v_act1_rep[1] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 4), v_repl_ctrl);
        v_act1_rep[2] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 8), v_repl_ctrl);
        v_act1_rep[3] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 12), v_repl_ctrl);
        v_act1_rep[4] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 16), v_repl_ctrl);
        v_act1_rep[5] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 20), v_repl_ctrl);
        v_act1_rep[6] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 24), v_repl_ctrl);
        v_act1_rep[7] = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act1_raw, 28), v_repl_ctrl);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0_rep, v_act1_rep, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        __fp16 scale_a0_val = y0_scales[kt];
        __fp16 scale_a1_val = y1_scales[kt];
        HVX_Vector v_scale_a0_f16 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a0_val));
        HVX_Vector v_scale_a1_f16 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *)&scale_a1_val));
        HVX_VectorPair p_scale_a0_f32 = hvx_vec_f16_to_f32(v_scale_a0_f16);
        HVX_VectorPair p_scale_a1_f32 = hvx_vec_f16_to_f32(v_scale_a1_f16);
        HVX_Vector v_scale_a0 = Q6_V_lo_W(p_scale_a0_f32);
        HVX_Vector v_scale_a1 = Q6_V_lo_W(p_scale_a1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a1);

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
