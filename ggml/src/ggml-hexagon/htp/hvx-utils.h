#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils.h"

#include "hvx-types.h"
#include "hvx-copy.h"
#include "hvx-scale.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"
#include "hvx-base.h"
#include "hvx-reduce.h"



#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_fp32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_fp32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

static inline HVX_Vector hvx_vec_rsqrt_fp32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>2)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    return Q6_Vsf_equals_Vqf32(temp);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_fp32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_fp32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_fp32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_fp32(2.0f);
    HVX_Vector one = hvx_vec_splat_fp32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1
    HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_fp32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

static inline void hvx_fast_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems) {
    int step_of_1 = num_elems >> 5;
    int remaining = num_elems - step_of_1 * VLEN_FP32;

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        v_dst[i] = hvx_vec_fast_sigmoid_fp32_guard(v_src[i], one, max_exp, min_exp);
    }

    if (remaining > 0) {
        const float * srcf = ((const float *) src) + step_of_1* VLEN_FP32;
        float *       dstf = (float *) dst + step_of_1*VLEN_FP32;

        HVX_Vector in  = *(HVX_UVector *) srcf;
        HVX_Vector out = hvx_vec_fast_sigmoid_fp32_guard(in, one, max_exp, min_exp);
        hvx_vec_store_u((void *) dstf, remaining * SIZEOF_FP32, out);
    }
}

static inline void hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems){
    int step_of_1 = num_elems >> 5;  // divby 32, because 32 float = 128 bytes per HVX vector
    int leftover = num_elems - (step_of_1 * VLEN_FP32);

    int32_t leftover_size = leftover * sizeof(float);

    static const float kMinExp = -87.f;  // 0
    static const float kMaxExp = 87.f;   // 1

    const HVX_Vector one     = hvx_vec_splat_fp32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_fp32(kMaxExp);
    const HVX_Vector min_exp = hvx_vec_splat_fp32(kMinExp);

    const float *input = (float *)src;
    float *output = (float *)dst;

    HVX_Vector *  input_v_ptr  = (HVX_Vector *) input;
    HVX_UVector * output_v_ptr = (HVX_UVector *) output;

    HVX_Vector slinep;
    HVX_Vector slinec;
    HVX_Vector sline;

    slinep = *input_v_ptr++;
    #pragma unroll(4)
    for (int i = step_of_1 - 1; i > 0; i--) {
        slinec                              = *input_v_ptr++;
        sline                               = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        slinep                              = slinec;
    }

    if (step_of_1 > 0) {
        slinec = hex_is_aligned(input_v_ptr, 128) && leftover == 0 ? slinep : *input_v_ptr++;
        sline  = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        *((HVX_UVector *) (output_v_ptr++)) = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        slinep = slinec;
    }
    if (leftover > 0) {
        slinec = (hex_is_one_chunk(input_v_ptr, leftover_size, 128) ? slinep : *input_v_ptr++);

        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

        HVX_Vector sout = hvx_vec_fast_sigmoid_fp32_guard(sline, one, max_exp, min_exp);
        hvx_vec_store_u(output_v_ptr, leftover_size, sout);
    }
}

float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_mul_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_mul_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_mul_mul_f32_opt(const uint8_t * restrict src0,
                          const uint8_t * restrict src1,
                          const uint8_t * restrict src2,
                          uint8_t * restrict dst,
                          const int num_elems);
void  hvx_mul_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_add_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_add_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_add_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_sub_f32(const uint8_t * restrict src0,
                  const uint8_t * restrict src1,
                  uint8_t * restrict dst,
                  const int num_elems);
void  hvx_sub_f32_opt(const uint8_t * restrict src0,
                      const uint8_t * restrict src1,
                      uint8_t * restrict dst,
                      const int num_elems);
void  hvx_sub_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_sigmoid_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems);
float hvx_self_max_f32(const uint8_t * restrict src, const int num_elems);
float hvx_self_sum_f32(const uint8_t * restrict src, const int num_elems);
void  hvx_min_scalar_f32(const uint8_t * restrict src, const float val, uint8_t * restrict dst, const int num_elems);
void  hvx_clamp_scalar_f32(const uint8_t * restrict src,
                           const float limit_left,
                           const float limit_right,
                           uint8_t * restrict dst,
                           const int num_elems);

#endif /* HVX_UTILS_H */
