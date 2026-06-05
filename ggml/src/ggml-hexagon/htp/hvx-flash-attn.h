#ifndef HVX_FLASH_ATTN_H
#define HVX_FLASH_ATTN_H

#include <math.h>
#include "hvx-utils.h"

// Scalar helper to compute a single ALiBi slope.
static inline float alibi_slope(uint32_t h, uint32_t n_head_log2, float m0, float m1) {
    return (h < n_head_log2) ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1);
}

// Vectorized helper to compute 32 ALiBi slopes starting from (kv_head * G).
static inline HVX_Vector hvx_alibi_slopes(
    uint32_t kv_head,
    uint32_t G,
    uint32_t n_head_log2,
    float m0,
    float m1
) {
    static const float ramp_32[32] __attribute__((aligned(128))) = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f
    };
    HVX_Vector v_ramp = hvx_vmem(ramp_32);
    HVX_Vector v_h_base = hvx_vec_splat_f32((float)(kv_head * G));
    HVX_Vector v_h = hvx_vec_add_f32_f32(v_h_base, v_ramp);

    // Compute exponent_m0: h + 1
    HVX_Vector v_exp_m0 = hvx_vec_add_f32_f32(v_h, hvx_vec_splat_f32(1.0f));

    // Compute exponent_m1: 2 * (h - n_head_log2) + 1
    HVX_Vector v_n_head_log2 = hvx_vec_splat_f32((float)n_head_log2);
    HVX_Vector v_h_minus = hvx_vec_sub_f32_f32(v_h, v_n_head_log2);
    HVX_Vector v_exp_m1 = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(hvx_vec_splat_f32(2.0f), v_h_minus), hvx_vec_splat_f32(1.0f));

    // Compute powers
    HVX_Vector v_pow_m0 = hvx_vec_pow_const_base_f32(m0, v_exp_m0);
    HVX_Vector v_pow_m1 = hvx_vec_pow_const_base_f32(m1, v_exp_m1);

    // Select based on h < n_head_log2
    HVX_VectorPred p_cond = Q6_Q_vcmp_gt_VsfVsf(v_n_head_log2, v_h); // v_n_head_log2 > v_h <=> h < n_head_log2
    return Q6_V_vmux_QVV(p_cond, v_pow_m0, v_pow_m1);
}

#endif /* HVX_FLASH_ATTN_H */
