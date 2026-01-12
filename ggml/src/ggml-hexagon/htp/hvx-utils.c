#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hvx-utils.h"

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    if (0 == hex_is_aligned((void *) src, VLEN)) {
        FARF(HIGH, "hvx_sum_of_squares_f32: unaligned address in hvx op, possibly slower execution\n");
    }

    assert((1 == hex_is_aligned((void *) src, VLEN)) || (0 == num_elems_whole));

    HVX_Vector * restrict vec_in1 = (HVX_Vector *) src;

    HVX_Vector sum_vec_acc = Q6_V_vsplat_R(0x00000000);
    HVX_Vector zero_vec    = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(4)
    for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(*vec_in1, *vec_in1);
        sum_vec_acc  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_vec_acc, v);
        vec_in1++;
    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;

        HVX_Vector vec_left = *(HVX_UVector *) srcf;

        HVX_Vector vec_left_sq = Q6_Vqf32_vmpy_VsfVsf(vec_left, vec_left);
        HVX_Vector vec_tmp     = Q6_V_valign_VVR(vec_left_sq, zero_vec, left_over * SIZEOF_FP32);

        sum_vec_acc = Q6_Vqf32_vadd_Vqf32Vqf32(sum_vec_acc, vec_tmp);
    }

    HVX_Vector v = hvx_vec_qf32_reduce_sum(sum_vec_acc);
    return hvx_vec_get_fp32(Q6_Vsf_equals_Vqf32(v));
}

void hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;
    int unalign_address = 0;
    if ((0 == hex_is_aligned((void *) src, VLEN)) || (0 == hex_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_min_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
        unalign_address = 1;
    }

    const float * src_f = (const float *) src;

    HVX_Vector vec_min = hvx_vec_splat_fp32(val);

    if(unalign_address == 0){
        HVX_Vector * restrict vec_in  = (HVX_Vector *) src;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector min_clamp    = Q6_Vsf_vmin_VsfVsf(vec_min, *vec_in++);
            *vec_out++ = (min_clamp);
        }
    }else{
        HVX_UVector * restrict vec_in  = (HVX_Vector *) src;
        HVX_UVector * restrict vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector min_clamp     = Q6_Vsf_vmin_VsfVsf(vec_min, *vec_in++);
            *vec_out++ = (min_clamp);
        }
    }

    if (left_over > 0 ) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_UVector in = *(HVX_UVector *) srcf;

        HVX_UVector min_clamp = Q6_Vsf_vmin_VsfVsf(vec_min, in);

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, (min_clamp));
    }
}

void hvx_clamp_scalar_f32(const uint8_t * restrict src,
                          const float limit_left,
                          const float limit_right,
                          uint8_t * restrict dst,
                          const int num_elems) {
    size_t left_over       = num_elems & (VLEN_FP32 - 1);
    size_t num_elems_whole = num_elems - left_over;

    int unalign_address = 0;
    if ((0 == hex_is_aligned((void *) src, VLEN)) || (0 == hex_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_clamp_scalar_f32: unaligned address in hvx op, possibly slower execution\n");
        unalign_address = 1;
    }

    HVX_Vector range_left  = hvx_vec_splat_fp32(limit_left);
    HVX_Vector range_right = hvx_vec_splat_fp32(limit_right);

    if(unalign_address == 0){
        HVX_Vector * restrict vec_in  = (HVX_Vector *) src;
        HVX_Vector * restrict vec_out = (HVX_Vector *) dst;



        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in_vec = *vec_in++;
            HVX_Vector temp_v = in_vec;

            HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, range_right);
            HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(range_left, in_vec);

            in_vec = Q6_V_vmux_QVV(pred_cap_right, range_right, temp_v);
            in_vec = Q6_V_vmux_QVV(pred_cap_left, range_left, in_vec);

            *vec_out++ = in_vec;
        }

    }else{

        HVX_UVector * restrict vec_in  = (HVX_UVector *) src;
        HVX_UVector * restrict vec_out = (HVX_UVector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in_vec = *vec_in++;
            HVX_Vector temp_v = in_vec;

            HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, range_right);
            HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(range_left, in_vec);

            in_vec = Q6_V_vmux_QVV(pred_cap_right, range_right, temp_v);
            in_vec = Q6_V_vmux_QVV(pred_cap_left, range_left, in_vec);

            *vec_out++ = in_vec;
        }

    }

    if (left_over > 0) {
        const float * srcf = (const float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in_vec = *(HVX_UVector *) srcf;

        HVX_Vector temp_v = in_vec;

        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, range_right);
        HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(range_left, in_vec);

        in_vec = Q6_V_vmux_QVV(pred_cap_right, range_right, temp_v);
        in_vec = Q6_V_vmux_QVV(pred_cap_left, range_left, in_vec);

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, in_vec);
    }
}
