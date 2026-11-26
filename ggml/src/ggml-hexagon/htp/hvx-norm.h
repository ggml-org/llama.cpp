#ifndef HVX_NORM_H
#define HVX_NORM_H

#include <stdint.h>
#include "hvx-base.h"
#include "hvx-reduce.h"
#include "hvx-inverse.h"
#include "hvx-sqrt.h"
#include "hvx-repl.h"

static inline void hvx_fast_rms_norm_f32(const uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         const int num_elems,
                                         float     epsilon,
                                         float     scale) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale full vectors
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));
    if (scale != 0.0f) {
        scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, hvx_vec_splat_f32(scale)));
    }

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static inline void hvx_fast_rms_norm_mul_f32(const uint8_t * restrict src,
                                             const uint8_t * restrict weight,
                                             uint8_t * restrict dst,
                                             const int num_elems,
                                             float     epsilon,
                                             float     scale) {
    const HVX_Vector * restrict v_src    = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale and multiply
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));
    if (scale != 0.0f) {
        scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, hvx_vec_splat_f32(scale)));
    }

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(result);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_Vector res_v = Q6_Vsf_equals_Vqf32(result);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

static inline void hvx_fast_rms_norm_mul_add_f32(const uint8_t * restrict src,
                                                 const uint8_t * restrict weight,
                                                 const uint8_t * restrict add,
                                                 uint8_t * restrict dst,
                                                 const int num_elems,
                                                 float     epsilon,
                                                 float     scale) {
    const HVX_Vector * restrict v_src    = (const HVX_Vector *) src;
    const HVX_Vector * restrict v_weight = (const HVX_Vector *) weight;
    const HVX_Vector * restrict v_add    = (const HVX_Vector *) add;
    HVX_Vector * restrict v_dst          = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;
    const int nloe = num_elems % VLEN_FP32;

    HVX_Vector sum_v     = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));
    if (scale != 0.0f) {
        scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, hvx_vec_splat_f32(scale)));
    }

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]));
        v_dst[i] = hvx_vec_add_f32_f32(v4, v_add[i]);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]));
        HVX_Vector res_v = hvx_vec_add_f32_f32(v4, v_add[nvec]);

        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

static inline void hvx_fast_norm_f32(const uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     const int num_elems,
                                     float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_Vector sum_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector sum_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, v2);
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq_v, v2);
        sum_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    sum_sq_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_sq_v));
    sum_x_v  = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_x_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(sum_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(sum_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, mean_x_b);
        HVX_Vector v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static inline void hvx_fast_l2_norm_f32(const uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        const int num_elems,
                                        float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_v = hvx_vec_splat_f32(0.0f);

    const int nvec = num_elems / VLEN_FP32;
    const int nloe = num_elems % VLEN_FP32;

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_Vector sum_sf    = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));
    HVX_Vector rsqrt_v   = hvx_vec_rsqrt_f32(sum_sf);              // 1/sqrt(sum)
    HVX_Vector sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_Vector denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_Vector scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

static inline void hvx_fast_rms_norm_f16(const uint8_t * restrict src,
                                         uint8_t * restrict dst,
                                         const int num_elems,
                                         float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;    // number of full f16 vectors
    const int nloe = num_elems % VLEN_FP16;    // leftover elements

    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

#if __HVX_ARCH__ >= 79
    HVX_VectorPair sum_p = Q6_W_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    HVX_Vector sum_v = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p), Q6_V_hi_W(sum_p));
#else
    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    HVX_Vector sum_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum0, sum1));
#endif

    sum_v = hvx_vec_reduce_sum_f32(sum_v);

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16_shuff(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16_shuff(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

static inline void hvx_fast_norm_f16(const uint8_t * restrict src,
                                     uint8_t * restrict dst,
                                     const int num_elems,
                                     float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;
    const int nloe = num_elems % VLEN_FP16;

    const HVX_Vector one_hf = hvx_vec_splat_f16(1.0);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

#if __HVX_ARCH__ >= 79
    HVX_VectorPair sum_sq_p = Q6_W_vzero();
    HVX_VectorPair sum_x_p  = Q6_W_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        sum_sq_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_sq_p, v, v);
        sum_x_p  = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_x_p,  v, one_hf);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        sum_sq_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_sq_p, v, v);
        sum_x_p  = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_x_p,  v, one_hf);
    }

    HVX_Vector sum_sq_v = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_sq_p), Q6_V_hi_W(sum_sq_p));
    HVX_Vector sum_x_v  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_x_p),  Q6_V_hi_W(sum_x_p));
#else
    HVX_Vector sum_sq0 = Q6_V_vzero();
    HVX_Vector sum_sq1 = Q6_V_vzero();
    HVX_Vector sum_x0  = Q6_V_vzero();
    HVX_Vector sum_x1  = Q6_V_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        HVX_VectorPair xp = Q6_Wqf32_vmpy_VhfVhf(v, one_hf);
        sum_sq0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq0, Q6_V_lo_W(sq));
        sum_sq1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq1, Q6_V_hi_W(sq));
        sum_x0  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x0,  Q6_V_lo_W(xp));
        sum_x1  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x1,  Q6_V_hi_W(xp));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        HVX_VectorPair xp = Q6_Wqf32_vmpy_VhfVhf(v, one_hf);
        sum_sq0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq0, Q6_V_lo_W(sq));
        sum_sq1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq1, Q6_V_hi_W(sq));
        sum_x0  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x0,  Q6_V_lo_W(xp));
        sum_x1  = Q6_Vqf32_vadd_Vqf32Vqf32(sum_x1,  Q6_V_hi_W(xp));
    }

    HVX_Vector sum_sq_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_sq0, sum_sq1));
    HVX_Vector sum_x_v  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_x0,  sum_x1));
#endif

    sum_sq_v = hvx_vec_reduce_sum_f32(sum_sq_v);
    sum_x_v  = hvx_vec_reduce_sum_f32(sum_x_v);

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_sq_v      = Q6_Vqf32_vmpy_VsfVsf(sum_sq_v, denom_v);
    HVX_Vector mean_x_v       = Q6_Vqf32_vmpy_VsfVsf(sum_x_v,  denom_v);
    HVX_Vector mean_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(mean_x_v), Q6_Vsf_equals_Vqf32(mean_x_v));
    HVX_Vector var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(mean_sq_v, mean_x_sq_v);
    HVX_Vector var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_Vector mean_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(mean_x_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v_src[i]);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16_shuff(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v1);
        HVX_Vector d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), mean_x_b);
        HVX_Vector d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), mean_x_b);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16_shuff(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

static inline void hvx_fast_l2_norm_f16(const uint8_t * restrict src,
                                        uint8_t * restrict dst,
                                        const int num_elems,
                                        float     epsilon) {

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP16;
    const int nloe = num_elems % VLEN_FP16;

#if __HVX_ARCH__ >= 79
    HVX_VectorPair sum_p = Q6_W_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        sum_p = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, v, v);
    }

    HVX_Vector sum_sf = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p), Q6_V_hi_W(sum_p));
#else
    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v = v_src[i];
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair sq = Q6_Wqf32_vmpy_VhfVhf(v, v);
        sum0 = Q6_Vqf32_vadd_Vqf32Vqf32(sum0, Q6_V_lo_W(sq));
        sum1 = Q6_Vqf32_vadd_Vqf32Vqf32(sum1, Q6_V_hi_W(sq));
    }

    HVX_Vector sum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum0, sum1));
#endif

    sum_sf               = hvx_vec_reduce_sum_f32(sum_sf);
    HVX_Vector eps2_v    = hvx_vec_splat_f32(epsilon * epsilon);
    HVX_Vector scale_v   = hvx_vec_rsqrt_f32(Q6_Vsf_vmax_VsfVsf(sum_sf, eps2_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v_src[i]);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_to_f16_shuff(r0, r1);
    }

    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(v1);
        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_Vector result = hvx_vec_f32_to_f16_shuff(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, result);
    }
}

static __attribute__((noinline, unused)) void hvx_rms_norm_mul_f16_w32_ne128_bcast_rows(
        const uint8_t * restrict src,
        const uint8_t * restrict weight,
        uint8_t * restrict       dst,
        uint32_t                 num_rows,
        size_t                   src_stride,
        size_t                   dst_stride,
        float                    epsilon,
        float                    scale) {
    const HVX_Vector * restrict v_w = (const HVX_Vector *) weight;
    const HVX_VectorPair wp0 = Q6_W_vdeal_VVR(v_w[1], v_w[0], -4);
    const HVX_VectorPair wp1 = Q6_W_vdeal_VVR(v_w[3], v_w[2], -4);
    const HVX_Vector w0 = Q6_V_lo_W(wp0);
    const HVX_Vector w1 = Q6_V_hi_W(wp0);
    const HVX_Vector w2 = Q6_V_lo_W(wp1);
    const HVX_Vector w3 = Q6_V_hi_W(wp1);

    const HVX_Vector v_one_f16 = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector denom_v   = hvx_vec_splat_f32(1.0f / 128.0f);
    const HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    const HVX_Vector scale_opt = hvx_vec_splat_f32(scale);
    const bool       has_scale = (scale != 0.0f);

    #pragma nounroll
    for (uint32_t ir = 0; ir < num_rows; ++ir) {
        const HVX_Vector * restrict v_src = (const HVX_Vector *) (src + ir * src_stride);
        HVX_Vector * restrict       v_dst = (HVX_Vector *) (dst + ir * dst_stride);

        const HVX_Vector     s0_hf = v_src[0];
        const HVX_Vector     s1_hf = v_src[1];
        const HVX_VectorPair p0    = hvx_vec_mpy_f32_f16(s0_hf, v_one_f16);
        const HVX_VectorPair p1    = hvx_vec_mpy_f32_f16(s1_hf, v_one_f16);
        const HVX_Vector     x0    = Q6_V_lo_W(p0);
        const HVX_Vector     x1    = Q6_V_hi_W(p0);
        const HVX_Vector     x2    = Q6_V_lo_W(p1);
        const HVX_Vector     x3    = Q6_V_hi_W(p1);

#if __HVX_ARCH__ >= 79
        HVX_VectorPair sum_p = Q6_Wsf_vmpy_VhfVhf(s0_hf, s0_hf);
        sum_p                = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, s1_hf, s1_hf);
        HVX_Vector sum_v     = hvx_vec_reduce_sum_f32(Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p), Q6_V_hi_W(sum_p)));
#else
        HVX_Vector sum01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, x0), Q6_Vqf32_vmpy_VsfVsf(x1, x1));
        HVX_Vector sum23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, x2), Q6_Vqf32_vmpy_VsfVsf(x3, x3));
        HVX_Vector sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum01, sum23)));
#endif

        HVX_Vector mean_eps = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v), epsilon_v);
        HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_eps));
        if (has_scale) {
            scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, scale_opt));
        }

        HVX_Vector s0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, scale_v));
        HVX_Vector s1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x1, scale_v));
        HVX_Vector s2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, scale_v));
        HVX_Vector s3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x3, scale_v));

        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, w0));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, w1));
        HVX_Vector r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s2, w2));
        HVX_Vector r3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s3, w3));

        v_dst[0] = hvx_vec_f32_to_f16_shuff(r0, r1);
        v_dst[1] = hvx_vec_f32_to_f16_shuff(r2, r3);
    }
}

static __attribute__((noinline, unused)) void hvx_rms_norm_mul_f16_w32_ne256_bcast_rows(
        const uint8_t * restrict src,
        const uint8_t * restrict weight,
        uint8_t * restrict       dst,
        uint32_t                 num_rows,
        size_t                   src_stride,
        size_t                   dst_stride,
        float                    epsilon,
        float                    scale) {
    const HVX_Vector * restrict v_w = (const HVX_Vector *) weight;
    const HVX_VectorPair wp0 = Q6_W_vdeal_VVR(v_w[1], v_w[0], -4);
    const HVX_VectorPair wp1 = Q6_W_vdeal_VVR(v_w[3], v_w[2], -4);
    const HVX_VectorPair wp2 = Q6_W_vdeal_VVR(v_w[5], v_w[4], -4);
    const HVX_VectorPair wp3 = Q6_W_vdeal_VVR(v_w[7], v_w[6], -4);
    const HVX_Vector w0 = Q6_V_lo_W(wp0);
    const HVX_Vector w1 = Q6_V_hi_W(wp0);
    const HVX_Vector w2 = Q6_V_lo_W(wp1);
    const HVX_Vector w3 = Q6_V_hi_W(wp1);
    const HVX_Vector w4 = Q6_V_lo_W(wp2);
    const HVX_Vector w5 = Q6_V_hi_W(wp2);
    const HVX_Vector w6 = Q6_V_lo_W(wp3);
    const HVX_Vector w7 = Q6_V_hi_W(wp3);

    const HVX_Vector v_one_f16 = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector denom_v   = hvx_vec_splat_f32(1.0f / 256.0f);
    const HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);
    const HVX_Vector scale_opt = hvx_vec_splat_f32(scale);
    const bool       has_scale = (scale != 0.0f);

    #pragma nounroll
    for (uint32_t ir = 0; ir < num_rows; ++ir) {
        const HVX_Vector * restrict v_src = (const HVX_Vector *) (src + ir * src_stride);
        HVX_Vector * restrict       v_dst = (HVX_Vector *) (dst + ir * dst_stride);

        const HVX_Vector     s0_hf = v_src[0];
        const HVX_Vector     s1_hf = v_src[1];
        const HVX_Vector     s2_hf = v_src[2];
        const HVX_Vector     s3_hf = v_src[3];
        const HVX_VectorPair p0    = hvx_vec_mpy_f32_f16(s0_hf, v_one_f16);
        const HVX_VectorPair p1    = hvx_vec_mpy_f32_f16(s1_hf, v_one_f16);
        const HVX_VectorPair p2    = hvx_vec_mpy_f32_f16(s2_hf, v_one_f16);
        const HVX_VectorPair p3    = hvx_vec_mpy_f32_f16(s3_hf, v_one_f16);

        const HVX_Vector x0 = Q6_V_lo_W(p0);
        const HVX_Vector x1 = Q6_V_hi_W(p0);
        const HVX_Vector x2 = Q6_V_lo_W(p1);
        const HVX_Vector x3 = Q6_V_hi_W(p1);
        const HVX_Vector x4 = Q6_V_lo_W(p2);
        const HVX_Vector x5 = Q6_V_hi_W(p2);
        const HVX_Vector x6 = Q6_V_lo_W(p3);
        const HVX_Vector x7 = Q6_V_hi_W(p3);

#if __HVX_ARCH__ >= 79
        HVX_VectorPair sum_p = Q6_Wsf_vmpy_VhfVhf(s0_hf, s0_hf);
        sum_p                = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, s1_hf, s1_hf);
        sum_p                = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, s2_hf, s2_hf);
        sum_p                = Q6_Wsf_vmpyacc_WsfVhfVhf(sum_p, s3_hf, s3_hf);
        HVX_Vector sum_v     = hvx_vec_reduce_sum_f32(Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p), Q6_V_hi_W(sum_p)));
#else
        HVX_Vector s01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, x0), Q6_Vqf32_vmpy_VsfVsf(x1, x1));
        HVX_Vector s23 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, x2), Q6_Vqf32_vmpy_VsfVsf(x3, x3));
        HVX_Vector s45 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x4, x4), Q6_Vqf32_vmpy_VsfVsf(x5, x5));
        HVX_Vector s67 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x6, x6), Q6_Vqf32_vmpy_VsfVsf(x7, x7));
        HVX_Vector sum_qf = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(s01, s23),
                                                     Q6_Vqf32_vadd_Vqf32Vqf32(s45, s67));
        HVX_Vector sum_v  = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_qf));
#endif

        HVX_Vector mean_eps = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v), epsilon_v);
        HVX_Vector scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_eps));
        if (has_scale) {
            scale_v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale_v, scale_opt));
        }

        HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, scale_v)), w0));
        HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x1, scale_v)), w1));
        HVX_Vector r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, scale_v)), w2));
        HVX_Vector r3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x3, scale_v)), w3));
        HVX_Vector r4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x4, scale_v)), w4));
        HVX_Vector r5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x5, scale_v)), w5));
        HVX_Vector r6 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x6, scale_v)), w6));
        HVX_Vector r7 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x7, scale_v)), w7));

        v_dst[0] = hvx_vec_f32_to_f16_shuff(r0, r1);
        v_dst[1] = hvx_vec_f32_to_f16_shuff(r2, r3);
        v_dst[2] = hvx_vec_f32_to_f16_shuff(r4, r5);
        v_dst[3] = hvx_vec_f32_to_f16_shuff(r6, r7);
    }
}

#endif // HVX_NORM_H
