#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <math.h>
#include <stdint.h>
#include <string.h>

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

        const uint16_t * s16_01 = (const uint16_t *) &vd01_hf;
        const uint16_t * s16_23 = (const uint16_t *) &vd23_hf;
        dst[i + 0].d = s16_01[0];
        dst[i + 1].d = s16_01[1];
        dst[i + 2].d = s16_23[0];
        dst[i + 3].d = s16_23[1];

        const uint64_t * q64 = (const uint64_t *) &vx_i8;
        uint64_t * d0 = (uint64_t *) dst[i + 0].qs;
        uint64_t * d1 = (uint64_t *) dst[i + 1].qs;
        uint64_t * d2 = (uint64_t *) dst[i + 2].qs;
        uint64_t * d3 = (uint64_t *) dst[i + 3].qs;

        d0[0] = q64[0];  d0[1] = q64[1];  d0[2] = q64[2];  d0[3] = q64[3];
        d1[0] = q64[4];  d1[1] = q64[5];  d1[2] = q64[6];  d1[3] = q64[7];
        d2[0] = q64[8];  d2[1] = q64[9];  d2[2] = q64[10]; d2[3] = q64[11];
        d3[0] = q64[12]; d3[1] = q64[13]; d3[2] = q64[14]; d3[3] = q64[15];
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

        const uint64_t * q64 = (const uint64_t *) &v_i8;
        uint64_t * d_qs = (uint64_t *) dst[i].qs;
        d_qs[0] = q64[0]; d_qs[1] = q64[1]; d_qs[2] = q64[2]; d_qs[3] = q64[3];
    }
}

static inline void hvx_dequantize_row_q8_0_f32(float * restrict dst_ptr, const void * restrict src_ptr, int n) {
    const int nb = n / QK8_0;
    const block_q8_0 * src = (const block_q8_0 *) src_ptr;

    int i = 0;
    for (; i + 3 < nb; i += 4) {
        HVX_Vector v_scales_raw = Q6_V_vzero();
        int16_t * s_ptr = (int16_t *) &v_scales_raw;
        s_ptr[0] = (int16_t) src[i+0].d;
        s_ptr[1] = (int16_t) src[i+1].d;
        s_ptr[2] = (int16_t) src[i+2].d;
        s_ptr[3] = (int16_t) src[i+3].d;

        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(v_scales_raw);
        HVX_Vector v_scales_f32 = Q6_V_lo_W(vp_f32);

        HVX_Vector vd0 = hvx_vec_repl_f32(v_scales_f32);
        HVX_Vector vd1 = hvx_vec_repl_f32(Q6_V_vror_VR(v_scales_f32, 4));
        HVX_Vector vd2 = hvx_vec_repl_f32(Q6_V_vror_VR(v_scales_f32, 8));
        HVX_Vector vd3 = hvx_vec_repl_f32(Q6_V_vror_VR(v_scales_f32, 12));

        for (int b = 0; b < 4; b++) {
            HVX_Vector vd = (b == 0) ? vd0 : (b == 1) ? vd1 : (b == 2) ? vd2 : vd3;

            HVX_Vector vq_i8 = *(const HVX_UVector *) src[i + b].qs;

            HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq_i8);
            HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

            HVX_Vector v_f32_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32));
            HVX_Vector v_f32_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(p32));

            HVX_Vector res_lo = hvx_vec_mul_f32_f32(v_f32_lo, vd);
            HVX_Vector res_hi = hvx_vec_mul_f32_f32(v_f32_hi, vd);

            float * block_dst = dst_ptr + (i + b) * QK8_0;
            hvx_vec_store_u(block_dst,      64, res_lo);
            hvx_vec_store_u(block_dst + 16, 64, res_hi);
        }
    }

    for (; i < nb; i++) {
        HVX_Vector vd_f16 = hvx_vec_repl_f16(Q6_Vh_vsplat_R(*(const int16_t *) &src[i].d));
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(vd_f16);
        HVX_Vector vd = Q6_V_lo_W(vp_f32);

        HVX_Vector vq_i8 = *(const HVX_UVector *) src[i].qs;

        HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq_i8);
        HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));

        HVX_Vector v_f32_lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(p32));
        HVX_Vector v_f32_hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(p32));

        HVX_Vector res_lo = hvx_vec_mul_f32_f32(v_f32_lo, vd);
        HVX_Vector res_hi = hvx_vec_mul_f32_f32(v_f32_hi, vd);

        float * block_dst = dst_ptr + i * QK8_0;
        hvx_vec_store_u(block_dst,      64, res_lo);
        hvx_vec_store_u(block_dst + 16, 64, res_hi);
    }
}

#endif // HVX_QUANT_H
