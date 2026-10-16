#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "htp-tensor.h"
#include "hvx-arith.h"
#include "hvx-base.h"
#include "hvx-reduce.h"
#include "hvx-repl.h"
#include "hvx-utils.h"

#ifndef GGML_COMMON_DECL_C
#define GGML_COMMON_DECL_C
#endif
#include "ggml-common.h"
#include "ggml-impl.h"

static inline void hvx_quantize_row_q8_0_f32(void * restrict dst_ptr, const float * restrict src_ptr, int n) {
    const int nb = n / QK8_0;
    block_q8_0 * dst = (block_q8_0 *) dst_ptr;
    HVX_Vector zero = Q6_V_vzero();

    int i = 0;
    for (; i + 3 < nb; i += 4) {
        HVX_Vector * vx = (HVX_Vector *) (src_ptr + i * QK8_0);

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

        hvx_vec_store_u(&dst[i + 0].d, 2, vd01_hf);
        hvx_vec_store_u(dst[i + 0].qs, 32, vx_i8);

        hvx_vec_store_u(&dst[i + 1].d, 2, Q6_V_vror_VR(vd01_hf, 64));
        hvx_vec_store_u(dst[i + 1].qs, 32, Q6_V_vror_VR(vx_i8, 32));

        hvx_vec_store_u(&dst[i + 2].d, 2, vd23_hf);
        hvx_vec_store_u(dst[i + 2].qs, 32, Q6_V_vror_VR(vx_i8, 64));

        hvx_vec_store_u(&dst[i + 3].d, 2, Q6_V_vror_VR(vd23_hf, 64));
        hvx_vec_store_u(dst[i + 3].qs, 32, Q6_V_vror_VR(vx_i8, 96));
    }

    for (; i < nb; i++) {
        const float * block_src = src_ptr + i * QK8_0;
        HVX_Vector vx = *(const HVX_UVector *) block_src;
        HVX_Vector v_abs = hvx_vec_abs_f32(vx);
        HVX_Vector v_max = hvx_vec_reduce_max_f32(v_abs);
        float amax = hvx_vec_get_f32(v_max);

        const float d = amax / 127.0f;
        const float id = d ? (1.0f / d) : 0.0f;
        dst[i].d = GGML_FP32_TO_FP16(d);

        HVX_Vector vid = hvx_vec_splat_f32(id);
        HVX_Vector v_scaled = hvx_vec_mul_f32_f32(vx, vid);
        HVX_Vector v_scaled_qf = Q6_Vqf32_vsub_VsfVsf(v_scaled, zero);
        HVX_Vector v_scaled_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(zero, v_scaled_qf)));
        HVX_Vector v_i16 = hvx_vec_i16_from_hf_rnd_sat(v_scaled_hf);
        HVX_Vector v_i8  = Q6_Vb_vpack_VhVh_sat(zero, v_i16);

        hvx_vec_store_u(dst[i].qs, 32, v_i8);
    }
}

static inline void hvx_dequantize_row_flat_q8_0_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK8_0;
    const block_q8_0 * src = (const block_q8_0 *) src_ptr;

    for (int i = 0; i < nb; i++) {
        HVX_Vector vd_f16     = Q6_Vh_vsplat_R(*(const int16_t *) &src[i].d);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(vd_f16);
        HVX_Vector vd         = Q6_V_lo_W(vp_f32);

        HVX_Vector vq_i8 = *(const HVX_UVector *) src[i].qs;

        HVX_VectorPair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_Vector     v_i16 = Q6_V_lo_W(p16);
        HVX_VectorPair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_Vector     v_i32 = Q6_V_lo_W(p32);

        HVX_Vector v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_Vector res   = hvx_vec_mul_f32_f32(v_f32, vd);

        float * block_dst = dst_ptr + i * QK8_0;
        hvx_vmem(block_dst) = res;
    }
}

static inline void hvx_dequantize_row_flat_q8_0_f16(__fp16 * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK8_0;
    const block_q8_0 * src = (const block_q8_0 *) src_ptr;

    for (int i = nb - 1; i >= 0; i--) {
        HVX_Vector vd_f16     = Q6_Vh_vsplat_R(*(const int16_t *) &src[i].d);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(vd_f16);
        HVX_Vector vd         = Q6_V_lo_W(vp_f32);

        HVX_Vector vq_i8 = *(const HVX_UVector *) src[i].qs;

        HVX_VectorPair p16   = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_Vector     v_i16 = Q6_V_lo_W(p16);
        HVX_VectorPair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_Vector     v_i32 = Q6_V_lo_W(p32);

        HVX_Vector v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_Vector res_f32 = hvx_vec_mul_f32_f32(v_f32, vd);

        HVX_Vector res_f16 = hvx_vec_f32_to_f16(res_f32, Q6_V_vzero());

        __fp16 * block_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(block_dst, QK8_0 * sizeof(__fp16), res_f16);
    }
}

static inline void hvx_dequantize_row_flat_f16_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;
    const _Float16 * src = (const _Float16 *) src_ptr;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v_f16 = *(const HVX_UVector *) (src + i * VLEN_FP32);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(v_f16);
        HVX_Vector res = Q6_V_lo_W(vp_f32);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    if (nloe) {
        HVX_Vector v_f16 = *(const HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(v_f16);
        HVX_Vector res = Q6_V_lo_W(vp_f32);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * sizeof(float), res);
    }
}

static inline void hvx_dequantize_row_flat_bf16_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;
    const uint16_t * src = (const uint16_t *) src_ptr;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v_bf16 = *(const HVX_UVector *) (src + i * VLEN_FP32);
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vmem(dst_ptr + i * VLEN_FP32) = res;
    }

    if (nloe) {
        HVX_Vector v_bf16 = *(const HVX_UVector *) (src + nvec * VLEN_FP32);
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector res = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);

        hvx_vec_store_a(dst_ptr + nvec * VLEN_FP32, nloe * sizeof(float), res);
    }
}

static inline void hvx_dequantize_row_flat_q4_0_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK4_0;
    const block_q4_0 * src = (const block_q4_0 *) src_ptr;

    const HVX_VectorPred q16 = Q6_Q_vsetq_R(16);
    const HVX_Vector v_eight = Q6_Vb_vsplat_R(8);
    const HVX_Vector v_zero  = Q6_V_vzero();

    for (int i = 0; i < nb; i++) {
        HVX_Vector vd_f16     = Q6_Vh_vsplat_R(*(const int16_t *) &src[i].d);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(vd_f16);
        HVX_Vector vd         = Q6_V_lo_W(vp_f32);

        HVX_Vector vq = *(const HVX_UVector *) src[i].qs;
        HVX_Vector v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

        HVX_Vector v_lo_s8 = Q6_Vb_vsub_VbVb(v_lo, v_eight);
        HVX_Vector v_hi_s8 = Q6_Vb_vsub_VbVb(v_hi, v_eight);

        HVX_Vector v32 = Q6_V_vmux_QVV(q16, v_lo_s8, Q6_V_vlalign_VVR(v_hi_s8, v_zero, 16));

        HVX_VectorPair p16   = Q6_Wh_vunpack_Vb(v32);
        HVX_Vector     v_i16 = Q6_V_lo_W(p16);
        HVX_VectorPair p32   = Q6_Ww_vunpack_Vh(v_i16);
        HVX_Vector     v_i32 = Q6_V_lo_W(p32);

        HVX_Vector v_f32 = Q6_Vsf_equals_Vw(v_i32);
        HVX_Vector res   = hvx_vec_mul_f32_f32(v_f32, vd);

        hvx_vmem(dst_ptr + i * QK4_0) = res;
    }
}

static inline void hvx_dequantize_row_flat_q4_k_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK_K;
    const block_q4_K * src = (const block_q4_K *) src_ptr;

    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(src[i].d);
        const float min = GGML_FP16_TO_FP32(src[i].dmin);
        const uint8_t * sc = src[i].scales;

        float * y = dst_ptr + i * QK_K;

        for (int step = 0; step < 4; ++step) {
            uint8_t sc1, m1_u, sc2, m2_u;
            if (step < 2) {
                sc1  = sc[2 * step + 0] & 63;
                m1_u = sc[2 * step + 4] & 63;
                sc2  = sc[2 * step + 1] & 63;
                m2_u = sc[2 * step + 5] & 63;
            } else {
                const int s = step - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_Vector vq = *(const HVX_UVector *) (src[i].qs + step * 32);
            HVX_Vector v_lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
            HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

            HVX_VectorPair p16_lo = Q6_Wuh_vunpack_Vub(v_lo);
            HVX_VectorPair p32_lo = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_lo));
            HVX_Vector     v_i32_lo = Q6_V_lo_W(p32_lo);
            HVX_Vector     v_f32_lo = Q6_Vsf_equals_Vw(v_i32_lo);

            HVX_Vector vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_Vector vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_Vector v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_lo, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_VectorPair p16_hi = Q6_Wuh_vunpack_Vub(v_hi);
            HVX_VectorPair p32_hi = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_hi));
            HVX_Vector     v_i32_hi = Q6_V_lo_W(p32_hi);
            HVX_Vector     v_f32_hi = Q6_Vsf_equals_Vw(v_i32_hi);

            HVX_Vector vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_Vector vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_Vector v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_hi, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

static inline void hvx_dequantize_row_flat_q5_k_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK_K;
    const block_q5_K * src = (const block_q5_K *) src_ptr;

    for (int i = 0; i < nb; i++) {
        const float d   = GGML_FP16_TO_FP32(src[i].d);
        const float min = GGML_FP16_TO_FP32(src[i].dmin);
        const uint8_t * sc = src[i].scales;

        HVX_Vector v_qh = *(const HVX_UVector *) src[i].qh;
        float * y = dst_ptr + i * QK_K;

        for (int step = 0; step < 4; ++step) {
            uint8_t sc1, m1_u, sc2, m2_u;
            if (step < 2) {
                sc1  = sc[2 * step + 0] & 63;
                m1_u = sc[2 * step + 4] & 63;
                sc2  = sc[2 * step + 1] & 63;
                m2_u = sc[2 * step + 5] & 63;
            } else {
                const int s = step - 2;
                sc1  = (sc[8 + 2 * s + 0] & 0xF) | ((sc[2 * s + 0] >> 6) << 4);
                m1_u = (sc[8 + 2 * s + 0] >>  4) | ((sc[4 + 2 * s + 0] >> 6) << 4);
                sc2  = (sc[8 + 2 * s + 1] & 0xF) | ((sc[2 * s + 1] >> 6) << 4);
                m2_u = (sc[8 + 2 * s + 1] >>  4) | ((sc[4 + 2 * s + 1] >> 6) << 4);
            }

            HVX_Vector v_ql = *(const HVX_UVector *) (src[i].qs + step * 32);
            HVX_Vector v_lo = Q6_V_vand_VV(v_ql, Q6_Vb_vsplat_R(0x0F));
            HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_ql, 4);

            HVX_Vector v_h1 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * step), Q6_Vb_vsplat_R(1)), 4);
            HVX_Vector v_val1 = Q6_V_vor_VV(v_lo, v_h1);

            HVX_VectorPair p16_1 = Q6_Wuh_vunpack_Vub(v_val1);
            HVX_VectorPair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_Vector     v_i32_1 = Q6_V_lo_W(p32_1);
            HVX_Vector     v_f32_1 = Q6_Vsf_equals_Vw(v_i32_1);

            HVX_Vector vd1 = hvx_vec_splat_f32(d * sc1);
            HVX_Vector vm1 = hvx_vec_splat_f32(min * m1_u);
            HVX_Vector v_out1 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_1, vd1), vm1);
            hvx_vmem(y + 0) = v_out1;

            HVX_Vector v_h2 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * step + 1), Q6_Vb_vsplat_R(1)), 4);
            HVX_Vector v_val2 = Q6_V_vor_VV(v_hi, v_h2);

            HVX_VectorPair p16_2 = Q6_Wuh_vunpack_Vub(v_val2);
            HVX_VectorPair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_Vector     v_i32_2 = Q6_V_lo_W(p32_2);
            HVX_Vector     v_f32_2 = Q6_Vsf_equals_Vw(v_i32_2);

            HVX_Vector vd2 = hvx_vec_splat_f32(d * sc2);
            HVX_Vector vm2 = hvx_vec_splat_f32(min * m2_u);
            HVX_Vector v_out2 = hvx_vec_sub_f32_f32(hvx_vec_mul_f32_f32(v_f32_2, vd2), vm2);
            hvx_vmem(y + 32) = v_out2;

            y += 64;
        }
    }
}

static inline void hvx_dequantize_row_flat_q6_k_f32(float * restrict dst_ptr, const void * restrict src_ptr, uint32_t n) {
    const int nb = n / QK_K;
    const block_q6_K * src = (const block_q6_K *) src_ptr;

    const HVX_VectorPred q64 = Q6_Q_vsetq_R(64);
    const HVX_Vector v32_b   = Q6_Vb_vsplat_R(32);
    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(src[i].d);

        const uint8_t * restrict ql = src[i].ql;
        const uint8_t * restrict qh = src[i].qh;
        const int8_t  * restrict sc = src[i].scales;

        float * y = dst_ptr + i * QK_K;

        for (int half = 0; half < 2; ++half) {
            float s[8];
            for (int j = 0; j < 8; ++j) {
                s[j] = d * sc[j];
            }

            HVX_Vector vscale1 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[0]), hvx_vec_splat_f32(s[1]));
            HVX_Vector vscale2 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[2]), hvx_vec_splat_f32(s[3]));
            HVX_Vector vscale3 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[4]), hvx_vec_splat_f32(s[5]));
            HVX_Vector vscale4 = Q6_V_vmux_QVV(q64, hvx_vec_splat_f32(s[6]), hvx_vec_splat_f32(s[7]));

            HVX_Vector v_ql0 = *(const HVX_UVector *) (ql);
            HVX_Vector v_ql1 = *(const HVX_UVector *) (ql + 32);
            HVX_Vector v_qh  = *(const HVX_UVector *) (qh);

            HVX_Vector ql0_lo = Q6_V_vand_VV(v_ql0, mask_0f);
            HVX_Vector qh_0   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_qh, mask_03), 4);
            HVX_Vector v_q1   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_lo, qh_0), v32_b);

            HVX_Vector ql1_lo = Q6_V_vand_VV(v_ql1, mask_0f);
            HVX_Vector qh_1   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2), mask_03), 4);
            HVX_Vector v_q2   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_lo, qh_1), v32_b);

            HVX_Vector ql0_hi = Q6_Vub_vlsr_VubR(v_ql0, 4);
            HVX_Vector qh_2   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 4), mask_03), 4);
            HVX_Vector v_q3   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql0_hi, qh_2), v32_b);

            HVX_Vector ql1_hi = Q6_Vub_vlsr_VubR(v_ql1, 4);
            HVX_Vector qh_3   = Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 6), mask_03), 4);
            HVX_Vector v_q4   = Q6_Vb_vsub_VbVb(Q6_V_vor_VV(ql1_hi, qh_3), v32_b);

            HVX_VectorPair p16_1 = Q6_Wh_vunpack_Vb(v_q1);
            HVX_VectorPair p32_1 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_1));
            HVX_Vector     v_f32_1 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_1));
            hvx_vmem(y + 0) = hvx_vec_mul_f32_f32(v_f32_1, vscale1);

            HVX_VectorPair p16_2 = Q6_Wh_vunpack_Vb(v_q2);
            HVX_VectorPair p32_2 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_2));
            HVX_Vector     v_f32_2 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_2));
            hvx_vmem(y + 32) = hvx_vec_mul_f32_f32(v_f32_2, vscale2);

            HVX_VectorPair p16_3 = Q6_Wh_vunpack_Vb(v_q3);
            HVX_VectorPair p32_3 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_3));
            HVX_Vector     v_f32_3 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_3));
            hvx_vmem(y + 64) = hvx_vec_mul_f32_f32(v_f32_3, vscale3);

            HVX_VectorPair p16_4 = Q6_Wh_vunpack_Vb(v_q4);
            HVX_VectorPair p32_4 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16_4));
            HVX_Vector     v_f32_4 = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32_4));
            hvx_vmem(y + 96) = hvx_vec_mul_f32_f32(v_f32_4, vscale4);

            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// bytes row, row + 32, row + 64 and row + 96 of v, to bytes 0..3
static inline HVX_Vector hvx_dequantize_gather4(HVX_Vector v, uint32_t row) {
    v = Q6_V_vror_VR(v, row);
    HVX_Vector packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(1), v, Q6_V_vror_VR(v, 31));
    packed = Q6_V_vmux_QVV(Q6_Q_vsetq_R(2), packed, Q6_V_vror_VR(v, 62));
    return Q6_V_vmux_QVV(Q6_Q_vsetq_R(3), packed, Q6_V_vror_VR(v, 93));
}

static inline void hvx_dequantize_row_tiled_q8_0_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const HVX_VectorPred first2 = Q6_Q_vsetq_R(2);
    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        HVX_Vector vq = Q6_V_vzero();
        for (int group = 7; group >= 0; --group) {
            const HVX_Vector v = Q6_V_vror_VR(hvx_vmem(tile + group * VLEN), 2 * row);
            const HVX_Vector packed = Q6_V_vmux_QVV(first2, v, Q6_V_vror_VR(v, 62));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        const HVX_Vector scales = hvx_vmem(tile + 1024);
        const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * row));
        const HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hf));
        const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq);
        const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_Vector *) dst = values;
    }
}

static inline void hvx_dequantize_row_tiled_q4_0_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        HVX_Vector vq = Q6_V_vzero();
        for (int group = 3; group >= 0; --group) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + group * VLEN), row), Q6_V_vror_VR(vq, VLEN - 4));
        }

        const HVX_Vector lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        const HVX_Vector hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));

        const HVX_Vector scales = hvx_vmem(tile + 512);
        const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * row));
        const HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hf));
        const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq);
        const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_Vector *) dst = values;
    }
}

static inline void hvx_dequantize_row_tiled_q4_k_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        HVX_Vector vq = Q6_V_vzero();
        for (int group = 3; group >= 0; --group) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + group * VLEN), row), Q6_V_vror_VR(vq, VLEN - 4));
        }

        const HVX_Vector lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
        const HVX_Vector hi = Q6_Vub_vlsr_VubR(vq, 4);
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        const HVX_Vector scales = hvx_vmem(tile + 512);
        const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * row));
        const HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hf));
        const HVX_Vector offset_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * row + 2));
        const HVX_Vector offset = Q6_V_lo_W(hvx_vec_f16_to_f32(offset_hf));

        const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq);
        const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_Vector *) dst = hvx_vec_add_f32_f32(values, offset);
    }
}

static inline void hvx_dequantize_row_tiled_q5_k_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);
    const HVX_Vector one = Q6_Vb_vsplat_R(1);

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        const HVX_Vector p5 = hvx_dequantize_gather4(hvx_vmem(tile + 640), row);
        HVX_Vector vq = Q6_V_vzero();
        HVX_Vector vh = Q6_V_vzero();
        for (int group = 3; group >= 0; --group) {
            vq = Q6_V_vmux_QVV(first4, hvx_dequantize_gather4(hvx_vmem(tile + group * VLEN), row), Q6_V_vror_VR(vq, VLEN - 4));
            vh = Q6_V_vmux_QVV(first4, Q6_Vub_vlsr_VubR(p5, 2 * group), Q6_V_vror_VR(vh, VLEN - 4));
        }

        const HVX_Vector lo = Q6_V_vor_VV(Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F)), Q6_Vw_vasl_VwR(Q6_V_vand_VV(vh, one), 4));
        const HVX_Vector hi = Q6_V_vor_VV(Q6_Vub_vlsr_VubR(vq, 4), Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vub_vlsr_VubR(vh, 1), one), 4));
        vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));

        const HVX_Vector scales = hvx_vmem(tile + 512);
        const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * row));
        const HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hf));
        const HVX_Vector offset_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * row + 2));
        const HVX_Vector offset = Q6_V_lo_W(hvx_vec_f16_to_f32(offset_hf));

        const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq);
        const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
        *(HVX_Vector *) dst = hvx_vec_add_f32_f32(values, offset);
    }
}

static inline void hvx_dequantize_row_tiled_q6_k_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);
    const HVX_VectorPred first16 = Q6_Q_vsetq_R(16 * sizeof(float));
    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        HVX_Vector vq = Q6_V_vzero();
        for (int group = 7; group >= 0; --group) {
            const HVX_Vector lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (group >> 1) * VLEN), 4 * row);
            const HVX_Vector hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (group >> 2) * VLEN), 4 * row);
            const HVX_Vector lo = (group & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, mask_0f);
            const HVX_Vector hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (group & 3));
            const HVX_Vector packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, mask_03), 4));
            vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
        }

        const HVX_Vector scales = hvx_vmem(tile + 768);
        const HVX_Vector scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * row));
        const HVX_Vector scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * row));
        const HVX_Vector scale_lo = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_lo_hf));
        const HVX_Vector scale_hi = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hi_hf));
        const HVX_Vector scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
        const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
        const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
        *(HVX_Vector *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
    }
}

static inline HVX_Vector hvx_dequantize_tiled_f16_gather4(const uint8_t * p, uint32_t r_byte) {
    HVX_Vector w0  = Q6_V_vror_VR(hvx_vmem(p + 0 * VLEN), r_byte);
    HVX_Vector w1  = Q6_V_vror_VR(hvx_vmem(p + 1 * VLEN), r_byte);
    HVX_Vector w01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, 4));
    HVX_Vector w2  = Q6_V_vror_VR(hvx_vmem(p + 2 * VLEN), r_byte);
    HVX_Vector w3  = Q6_V_vror_VR(hvx_vmem(p + 3 * VLEN), r_byte);
    HVX_Vector w23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, 4));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(w23, w01, 8));
}

static inline void hvx_dequantize_row_tiled_f16_f32(float * dst_spad, const uint8_t * src_spad, uint32_t n_k_tiles, uint32_t tile_stride, uint32_t row) {
    const uint32_t r_byte = 4 * row;

    for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        const uint8_t * tile = src_spad + k_tile * tile_stride;
        float * dst = dst_spad + k_tile * HTP_TENSOR_TILE_N_COLS;

        HVX_Vector g0 = hvx_dequantize_tiled_f16_gather4(tile + 0 * VLEN, r_byte);
        HVX_Vector g1 = hvx_dequantize_tiled_f16_gather4(tile + 4 * VLEN, r_byte);
        HVX_Vector g01 = Q6_V_lo_W(Q6_W_vshuff_VVR(g1, g0, 16));

        HVX_Vector g2 = hvx_dequantize_tiled_f16_gather4(tile + 8 * VLEN, r_byte);
        HVX_Vector g3 = hvx_dequantize_tiled_f16_gather4(tile + 12 * VLEN, r_byte);
        HVX_Vector g23 = Q6_V_lo_W(Q6_W_vshuff_VVR(g3, g2, 16));

        HVX_Vector g = Q6_V_lo_W(Q6_W_vshuff_VVR(g23, g01, 32));
        *(HVX_Vector *) dst = Q6_V_lo_W(hvx_vec_f16_to_f32(g));
    }
}

#endif // HVX_QUANT_H
