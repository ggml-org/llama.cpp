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

static inline void hvx_dequantize_row_q8_0_f32(float * restrict dst_ptr, const void * restrict src_ptr, int n) {
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

static inline void hvx_dequantize_row_q8_0_f16(__fp16 * restrict dst_ptr, const void * restrict src_ptr, int n) {
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

static inline void hvx_dequantize_row_f16_f32(float * restrict dst_ptr, const void * restrict src_ptr, int n) {
    const int nb = n / 32;
    const _Float16 * src = (const _Float16 *) src_ptr;

    for (int i = 0; i < nb; i++) {
        HVX_Vector v_f16 = *(const HVX_UVector *) (src + i * 32);
        HVX_VectorPair vp_f32 = hvx_vec_f16_to_f32(v_f16);
        HVX_Vector res = Q6_V_lo_W(vp_f32);

        float * block_dst = dst_ptr + i * 32;
        hvx_vmem(block_dst) = res;
    }
}



#endif // HVX_QUANT_H
