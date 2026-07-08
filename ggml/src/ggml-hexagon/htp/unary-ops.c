#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hvx-exp.h"
#include "hvx-sigmoid.h"
#include "hvx-utils.h"
#include "unary-ops.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-vtcm.h"
#include "hex-profile.h"

struct htp_unary_context {
    struct htp_ops_context * octx;

    // Precomputed values
    const uint8_t *           data_src0;
    const uint8_t *           data_src1;            // weight/scale tensor for RMS_NORM_MUL
    uint8_t *                 data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    src1_data_row_size;
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_vtcm_half_size;
    size_t                    src1_vtcm_half_size;
    size_t                    dst_vtcm_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  nc;
    uint32_t                  col_tile;            // wide-row mode
    bool                      broadcast_weight;

    // Direct VTCM layout pointers and sizes
    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_size_per_thread;
    size_t                    vtcm_src1_size_per_thread;
    size_t                    vtcm_dst_size_per_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
static inline size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      size_t nb1, size_t nb2, size_t nb3) {
    const uint32_t i1 = ir % ne1;
    const uint32_t i2 = (ir / ne1) % ne2;
    const uint32_t i3 = ir / (ne1 * ne2);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}
// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t src_ne1,
                                        uint32_t dst_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!src_contig) {
        const uint32_t src_slice_end = (ir / src_ne1 + 1) * src_ne1;
        limit = MIN(limit, src_slice_end - ir);
    }

    if (!dst_contig) {
        const uint32_t dst_slice_end = (ir / dst_ne1 + 1) * dst_ne1;
        limit = MIN(limit, dst_slice_end - ir);
    }

    return limit;
}

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

static void hvx_fast_rms_norm_f32(const uint8_t * restrict src,
                                  uint8_t * restrict dst,
                                  uint8_t * restrict pad,
                                  const int num_elems,
                                  float     epsilon) {
    (void)pad;

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

static void hvx_fast_rms_norm_mul_f32(const uint8_t * restrict src,
                                      const uint8_t * restrict weight,
                                      uint8_t * restrict dst,
                                      const int num_elems,
                                      float     epsilon) {
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

static void hvx_fast_norm_f32(const uint8_t * restrict src,
                                  uint8_t * restrict dst,
                                  uint8_t * restrict pad,
                                  const int num_elems,
                                  float     epsilon) {
    (void)pad;

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

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    const uint32_t ne0 = uctx->nc;                                    \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    const size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

static void scale_f32(const float * restrict src,
                      float * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0, scale, bias);
    }
}

static void rms_norm_f32(const float * restrict src,
                         float * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, NULL, ne0, epsilon);
    }
}

static void rms_norm_mul_f32(const float * restrict src,
                             const float * restrict weight,
                             float * restrict dst,
                             const uint32_t num_rows,
                             const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        const uint8_t * restrict w_local   = (const uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

static void norm_f32(const float * restrict src,
                     float * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, NULL, ne0, epsilon);
    }
}

static void sqr_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void sqrt_f32(const float * restrict src,
                     float * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, ne0);
    }
}

static void neg_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

static void exp_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

static void sigmoid_f32(const float * restrict src,
                        float * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

static void tri_f32(const float * restrict src,
                    float * restrict dst,
                    const uint32_t num_rows,
                    const uint32_t ir,
                    const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    const int32_t ttype = op_params[0];
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    const uint32_t nvec  = ne0 / VLEN_FP32;
    const uint32_t nloe  = ne0 % VLEN_FP32;

    const uint32_t ne01 = uctx->octx->src[0]->ne[1];

    for (uint32_t b = 0; b < num_rows; b++) {
        const uint32_t abs_row = ir + b;
        const uint32_t i01     = abs_row % ne01;

        const HVX_Vector * restrict v_src = (const HVX_Vector *) ((const uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        int      keep_left;
        switch (ttype) {
            case 0: boundary = i01;     keep_left = 0; break;  // keep col >= row
            case 1: boundary = i01 + 1; keep_left = 0; break;  // keep col > row
            case 2: boundary = i01 + 1; keep_left = 1; break;  // keep col <= row
            case 3: boundary = i01;     keep_left = 1; break;  // keep col < row
            default: boundary = 0; keep_left = 0; break;
        }
        if (boundary > ne0) boundary = ne0;

        // Full HVX vectors — each starts at a 128-byte aligned offset
        for (uint32_t i = 0; i < nvec; i++) {
            const uint32_t vec_start = i * VLEN_FP32;
            const uint32_t vec_end   = vec_start + VLEN_FP32;
            if (keep_left) {
                if (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } else if (vec_start >= boundary) {
                    v_dst[i] = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } else {
                if (vec_end <= boundary) {
                    v_dst[i] = zero;
                } else if (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * sizeof(float));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        if (nloe > 0) {
            const uint32_t abs_start = nvec * VLEN_FP32;
            const uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            if (keep_left) {
                if (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } else if (abs_start >= boundary) {
                    tail_val = zero;
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } else {
                if (abs_end <= boundary) {
                    tail_val = zero;
                } else if (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } else {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
        }
    }
}

static void softplus_f32(const float * restrict src,
                         float * restrict dst,
                         const uint32_t num_rows,
                         const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    // softplus(x) = log(1 + exp(x))
    // Match CPU reference: ggml_compute_softplus_f32() in ggml-impl.h
    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * src0_row_size_aligned));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * dst_row_size_aligned));

        for (uint32_t i = 0; i < ne0; i++) {
            float x = src_f[i];
            // For x > 20: softplus(x) ≈ x (avoids exp overflow)
            dst_f[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
        }
    }
}

// --- L2_NORM HVX kernel ---
// Computes y[i] = x[i] / fmax(sqrt(sum(x[j]^2)), epsilon) for each row.
// scale = 1/fmax(sqrt(sum), epsilon) is computed entirely in HVX registers
// using rsqrt + inverse to avoid scalar extraction.
static void hvx_fast_l2_norm_f32(const uint8_t * restrict src,
                                 uint8_t * restrict dst,
                                 uint8_t * restrict pad,
                                 const int num_elems,
                                 float     epsilon) {
    (void)pad;

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

static void l2_norm_f32(const float * restrict src,
                        float * restrict dst,
                        const uint32_t num_rows,
                        const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * src0_row_size_aligned));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * dst_row_size_aligned));

        hvx_fast_l2_norm_f32((const uint8_t *)src_f, (uint8_t *)dst_f, NULL, ne0, epsilon);
    }
}

static void tanh_f32(const float * restrict src,
                     float * restrict dst,
                     const uint32_t num_rows,
                     const struct htp_unary_context * uctx) {
    htp_unary_op_preamble;

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

#define DEFINE_UNARY_TASK(NAME, IS_RMS_NORM_MUL, IS_TRI, CORE_EXPR)                                                 \
static void unary_task_f32_##NAME(unsigned int nth, unsigned int ith, void * data) {                                \
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;                                \
    struct htp_ops_context * octx = uctx->octx;                                                                     \
    const struct htp_tensor * src = octx->src[0];                                                                   \
    const struct htp_tensor * dst = octx->dst;                                                                      \
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;                                       \
                                                                                                                    \
    htp_unary_preamble;                                                                                             \
                                                                                                                    \
    int32_t *    op_params = octx->op_params;                                                                       \
    uint32_t     src0_nrows_per_thread = uctx->src0_nrows_per_thread;                                               \
                                                                                                                    \
    const size_t src0_data_row_size = uctx->src0_data_row_size;                                                     \
    const size_t dst_data_row_size  = uctx->dst_data_row_size;                                                      \
                                                                                                                    \
    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;                                               \
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;                                                \
                                                                                                                    \
    const uint32_t src0_nrows = uctx->src0_nrows;                                                                   \
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;                                                    \
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                        \
                                                                                                                    \
    if (src0_start_row >= src0_end_row) {                                                                           \
        return;                                                                                                     \
    }                                                                                                               \
                                                                                                                    \
    const uint8_t * restrict data_src = uctx->data_src0;                                                            \
    const uint8_t * restrict data_src1 = uctx->data_src1;                                                           \
    uint8_t * restrict       data_dst = uctx->data_dst;                                                             \
                                                                                                                    \
    const struct htp_tensor * src1 = (IS_RMS_NORM_MUL) ? octx->src[1] : NULL;                                       \
    const uint32_t nb11 = src1 ? src1->nb[1] : 0;                                                                   \
    const uint32_t nb12 = src1 ? src1->nb[2] : 0;                                                                   \
    const uint32_t nb13 = src1 ? src1->nb[3] : 0;                                                                   \
    const bool src1_contig = src1 ? ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12)) : false;       \
                                                                                                                    \
    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);                           \
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : NULL;\
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);                             \
                                                                                                                    \
    size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;                                                         \
    size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;                                                         \
    size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;                                                          \
                                                                                                                    \
    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&                                                       \
                             (nb03 == (size_t)ne02 * nb02);                                                         \
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&                                                       \
                             (nb3  == (size_t)ne2  * nb2);                                                          \
    const uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);                   \
    const uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);                    \
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);                                                      \
    if (BLOCK == 0) {                                                                                               \
        FARF(ERROR, "unary-f32 : current VTCM reservation %zu is too small, needed at least %zu\n",                 \
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);                                               \
        return;                                                                                                     \
    }                                                                                                               \
                                                                                                                    \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                    \
                                                                                                                    \
    if ((IS_RMS_NORM_MUL) && uctx->broadcast_weight) {                                                              \
        dma_queue_push(dma_queue, dma_make_ptr(src1_vtcm_data, data_src1),                                          \
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);                                \
        dma_queue_flush(dma_queue);                                                                                 \
    }                                                                                                               \
                                                                                                                    \
    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {               \
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);  \
                                                                                                                    \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),                                \
            nb1, dst_row_size_aligned, dst_data_row_size, 0);                                                       \
                                                                                                                    \
        const size_t src0_off = src0_contig ? (ir * nb01) : unary_row_offset(ir, ne01, ne02, nb01, nb02, nb03);     \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),                   \
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);                                           \
                                                                                                                    \
        if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                         \
            const size_t src1_off = src1_contig ? (ir * nb11) : unary_row_offset(ir, ne01, ne02, nb11, nb12, nb13); \
            dma_queue_push(dma_queue,                                                                               \
                dma_make_ptr(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),              \
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);                           \
        }                                                                                                           \
                                                                                                                    \
        ir += block_size;                                                                                           \
    }                                                                                                               \
                                                                                                                    \
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {                                                       \
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);  \
                                                                                                                    \
        float * dst_vtcm  = (float *) dma_queue_pop(dma_queue).src;                                                 \
        float * src0_vtcm = (float *) dma_queue_pop(dma_queue).dst;                                                 \
        float * src1_vtcm = NULL;                                                                                   \
        if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                         \
            src1_vtcm = (float *) dma_queue_pop(dma_queue).dst;                                                     \
        }                                                                                                           \
                                                                                                                    \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
        CORE_EXPR;                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                       \
                                                                                                                    \
        const size_t dst_off = dst_contig ? (ir * nb1) : unary_row_offset(ir, ne1, ne2, nb1, nb2, nb3);             \
        dma_queue_push(dma_queue,                                                                                   \
            dma_make_ptr(data_dst + dst_off, dst_vtcm),                                                             \
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);                                              \
                                                                                                                    \
        const uint32_t next_ir = ir + block_size;                                                                   \
        if (next_ir < src0_end_row) {                                                                               \
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1); \
            const uint32_t pref_ir = next_ir + next_block_size;                                                     \
            if (pref_ir < src0_end_row) {                                                                           \
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);   \
                const size_t src0_pref_off = src0_contig ? (pref_ir * nb01) : unary_row_offset(pref_ir, ne01, ne02, nb01, nb02, nb03); \
                dma_queue_push(dma_queue,                                                                           \
                    dma_make_ptr(src0_vtcm, data_src + src0_pref_off),                                              \
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);                              \
                                                                                                                    \
                if ((IS_RMS_NORM_MUL) && !uctx->broadcast_weight) {                                                 \
                    const size_t src1_pref_off = src1_contig ? (pref_ir * nb11) : unary_row_offset(pref_ir, ne01, ne02, nb11, nb12, nb13); \
                    dma_queue_push(dma_queue,                                                                       \
                        dma_make_ptr(src1_vtcm, data_src1 + src1_pref_off),                                         \
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);              \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        ir += block_size;                                                                                           \
    }                                                                                                               \
                                                                                                                    \
    dma_queue_flush(dma_queue);                                                                                     \
}

DEFINE_UNARY_TASK(norm,           false, false, norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(rms_norm,       false, false, rms_norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(rms_norm_mul,   true,  false, rms_norm_mul_f32(src0_vtcm, uctx->broadcast_weight ? (const float *) src1_vtcm_data : src1_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(scale,          false, false, scale_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(sqr,            false, false, sqr_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(sqrt,           false, false, sqrt_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_neg,      false, false, neg_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_exp,      false, false, exp_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_sigmoid,  false, false, sigmoid_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_softplus, false, false, softplus_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(unary_tanh,     false, false, tanh_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(l2_norm,        false, false, l2_norm_f32(src0_vtcm, dst_vtcm, block_size, uctx))
DEFINE_UNARY_TASK(tri,            false, true,  tri_f32(src0_vtcm, dst_vtcm, block_size, ir, uctx))

// Apply a pointwise unary op to one column tile that is already in VTCM.
static inline void unary_apply_tile_f32(int htp_op, const uint8_t * restrict src, uint8_t * restrict dst,
                                        uint32_t tile_elems, int32_t * op_params) {
    switch (htp_op) {
        case HTP_OP_SCALE: {
            float scale = 0.f, bias = 0.f;
            memcpy(&scale, &op_params[0], sizeof(float));
            memcpy(&bias,  &op_params[1], sizeof(float));
            hvx_scale_offset_f32_aa(dst, src, tile_elems, scale, bias);
            break;
        }
        case HTP_OP_SQR:            hvx_sqr_f32_aa(dst, src, tile_elems); break;
        case HTP_OP_SQRT:           hvx_sqrt_f32_aa(dst, src, tile_elems); break;
        case HTP_OP_UNARY_NEG:      hvx_scale_f32_aa(dst, src, tile_elems, -1.0f); break;
        case HTP_OP_UNARY_EXP:      hvx_exp_f32(dst, src, tile_elems, false); break;
        case HTP_OP_UNARY_SIGMOID:  hvx_sigmoid_f32_aa(dst, src, tile_elems); break;
        case HTP_OP_UNARY_TANH:     hvx_tanh_f32_aa(dst, src, tile_elems); break;
        case HTP_OP_UNARY_SOFTPLUS: {
            const float * restrict sf = (const float *) src;
            float * restrict df       = (float *) dst;
            for (uint32_t i = 0; i < tile_elems; i++) {
                float x = sf[i];
                df[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
            }
            break;
        }
        default: break;
    }
}

// Triangular mask applied to one column tile. Boundary is an absolute column index, so
// each vector compares against its absolute column position (col_start + i*VLEN_FP32).
static inline void tri_apply_tile_f32(const uint8_t * restrict src, uint8_t * restrict dst,
                                      uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                                      uint32_t ne0, int32_t ttype) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    const HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    int      keep_left;
    switch (ttype) {
        case 0: boundary = i01;     keep_left = 0; break;
        case 1: boundary = i01 + 1; keep_left = 0; break;
        case 2: boundary = i01 + 1; keep_left = 1; break;
        case 3: boundary = i01;     keep_left = 1; break;
        default: boundary = 0; keep_left = 0; break;
    }
    if (boundary > ne0) boundary = ne0;

    const uint32_t nvec = tile_elems / VLEN_FP32;
    const uint32_t nloe = tile_elems % VLEN_FP32;

    for (uint32_t i = 0; i < nvec; i++) {
        const uint32_t abs_start = col_start + i * VLEN_FP32;
        const uint32_t abs_end   = abs_start + VLEN_FP32;
        if (keep_left) {
            if (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } else if (abs_start >= boundary) {
                v_dst[i] = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } else {
            if (abs_end <= boundary) {
                v_dst[i] = zero;
            } else if (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    if (nloe > 0) {
        const uint32_t abs_start = col_start + nvec * VLEN_FP32;
        const uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        if (keep_left) {
            if (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } else if (abs_start >= boundary) {
                tail_val = zero;
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } else {
            if (abs_end <= boundary) {
                tail_val = zero;
            } else if (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } else {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * sizeof(float));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * sizeof(float), tail_val);
    }
}

// Wide-row mode: a single pointwise row is too large to fit double-buffered in VTCM, so
// each row is processed as a sequence of column tiles, in one double-buffered pass.
static void unary_task_f32_wide_row_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    struct htp_thread_trace * tr = octx->ctx ? &octx->ctx->trace[ith] : NULL;

    htp_unary_preamble;

    const int      htp_op    = octx->op;
    int32_t *      op_params = octx->op_params;
    const uint32_t col_tile  = uctx->col_tile;

    const uint32_t src0_nrows     = uctx->src0_nrows;
    const uint32_t src0_start_row = uctx->src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + uctx->src0_nrows_per_thread, src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src  = uctx->data_src0;
    uint8_t * restrict       data_dst  = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    const size_t src0_half = uctx->src0_vtcm_half_size;
    const size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dmaq = octx->ctx->dma[ith];

    const uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    const int32_t  tri_ttype     = (htp_op == HTP_OP_TRI) ? op_params[0] : 0;

    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    // Single-pass pointwise pipeline, flattened over all (row, tile) pairs so tiles
    // stream continuously across row boundaries. 
    const uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    for (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        const uint32_t row  = src0_start_row + t / tiles_per_row;
        const uint32_t col  = (t % tiles_per_row) * col_tile;
        const uint32_t tw   = MIN(col_tile, ne0 - col);
        const size_t   tb   = (size_t) tw * sizeof(float);
        const size_t   soff = (src0_contig ? (row * nb01) : unary_row_offset(row, ne01, ne02, nb01, nb02, nb03)) + (size_t) col * sizeof(float);

        dma_queue_push(dmaq, dma_make_ptr(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_push(dmaq, dma_make_ptr(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;
    const struct fastdiv_values * div_ne01 = &kparams->div_ne01;
    const struct fastdiv_values * div_tpr  = &kparams->div_tpr;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    for (uint32_t t = 0; t < total_tiles; t++) {
        uint8_t * dst_vtcm = (uint8_t *) dma_queue_pop(dmaq).src;
        uint8_t * src_vtcm = (uint8_t *) dma_queue_pop(dmaq).dst;

        const uint32_t tw  = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        if (htp_op == HTP_OP_TRI) {
            tri_apply_tile_f32(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        } else {
            unary_apply_tile_f32(htp_op, src_vtcm, dst_vtcm, tw, op_params);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        const size_t doff = (dst_contig ? (row * nb1) : unary_row_offset(row, ne1, ne2, nb1, nb2, nb3)) + (size_t) col * sizeof(float);
        const size_t tb   = (size_t) tw * sizeof(float);
        dma_queue_push(dmaq, dma_make_ptr(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        const uint32_t pt = t + 2;
        if (pt < total_tiles) {
            const uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            const size_t   ptb  = (size_t) ptw * sizeof(float);
            const size_t   psoff = (src0_contig ? (prow * nb01) : unary_row_offset(prow, ne01, ne02, nb01, nb02, nb03)) + (size_t) pcol * sizeof(float);
            dma_queue_push(dmaq, dma_make_ptr(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        if (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            if (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        if (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dmaq);

    uint64_t t2 = HAP_perf_get_qtimer_count();
    FARF(HIGH, "unary-f32-wide %d/%d: %ux%ux%ux%u (%u:%u) col_tile %u usec %u\n", ith, nth, src->ne[0],
         src->ne[1], src->ne[2], src->ne[3], src0_start_row, src0_end_row, col_tile,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_unary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_NORM:
            op_type = "norm-f32";
            break;
        case HTP_OP_RMS_NORM:
            op_type = "rmsnorm-f32";
            break;
        case HTP_OP_RMS_NORM_MUL:
            op_type = "rmsnorm-mul-f32";
            break;
        case HTP_OP_SCALE:
            op_type = "scale-f32";
            break;
        case HTP_OP_SQR:
            op_type = "sqr-f32";
            break;
        case HTP_OP_SQRT:
            op_type = "sqrt-f32";
            break;
        case HTP_OP_UNARY_NEG:
            op_type = "neg-f32";
            break;
        case HTP_OP_UNARY_EXP:
            op_type = "exp-f32";
            break;
        case HTP_OP_UNARY_SIGMOID:
            op_type = "sigmoid-f32";
            break;
        case HTP_OP_UNARY_SOFTPLUS:
            op_type = "softplus-f32";
            break;
        case HTP_OP_UNARY_TANH:
            op_type = "tanh-f32";
            break;
        case HTP_OP_L2_NORM:
            op_type = "l2norm-f32";
            break;
        case HTP_OP_TRI:
            op_type = "tri-f32";
            break;

        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_unary_kernel_params * kparams = (const struct htp_unary_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = kparams->n_threads;

    const size_t src0_data_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_data_row_size  = dst->ne[0]  * sizeof(float);

    const size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    const uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    const struct htp_tensor * src1 = NULL;

    if (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * sizeof(float);
    }

    if (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
        struct htp_unary_context uctx = {
            .octx                  = octx,
            .src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads,
            .src0_nrows            = src0_nrows,

            .data_src0             = (const uint8_t *)src0->data,
            .data_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? (const uint8_t *)src1->data : NULL,
            .data_dst              = (uint8_t *)dst->data,

            .src0_data_row_size    = src0_data_row_size,
            .src1_data_row_size    = src1_data_row_size,
            .dst_data_row_size     = dst_data_row_size,

            .src0_row_size_aligned = src0_row_size_aligned,
            .src1_row_size_aligned = src1_row_size_aligned,
            .dst_row_size_aligned  = dst_row_size_aligned,

            .src0_vtcm_half_size   = kparams->vtcm_src0_size_per_thread / 2,
            .src1_vtcm_half_size   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_size_per_thread / (broadcast_weight ? 1 : 2)) : 0,
            .dst_vtcm_half_size    = kparams->vtcm_dst_size_per_thread / 2,

            .block                 = kparams->block,
            .nc                    = src0->ne[0],
            .col_tile              = (uint32_t) kparams->col_tile,
            .broadcast_weight      = broadcast_weight,

            .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, 0),
            .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, kparams->vtcm_src0_size, kparams->vtcm_src1_size > 0),
            .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_size + kparams->vtcm_src1_size),

            .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
            .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
            .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,
        };

        FARF(HIGH, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "wide-row" : "row-block", col_tile);

        worker_callback_t task_func = NULL;
        if (col_tile) {
            task_func = unary_task_f32_wide_row_per_thread;
        } else {
            switch (octx->op) {
                case HTP_OP_NORM:            task_func = unary_task_f32_norm; break;
                case HTP_OP_RMS_NORM:        task_func = unary_task_f32_rms_norm; break;
                case HTP_OP_RMS_NORM_MUL:    task_func = unary_task_f32_rms_norm_mul; break;
                case HTP_OP_SCALE:           task_func = unary_task_f32_scale; break;
                case HTP_OP_SQR:             task_func = unary_task_f32_sqr; break;
                case HTP_OP_SQRT:            task_func = unary_task_f32_sqrt; break;
                case HTP_OP_UNARY_NEG:       task_func = unary_task_f32_unary_neg; break;
                case HTP_OP_UNARY_EXP:       task_func = unary_task_f32_unary_exp; break;
                case HTP_OP_UNARY_SIGMOID:   task_func = unary_task_f32_unary_sigmoid; break;
                case HTP_OP_UNARY_SOFTPLUS:  task_func = unary_task_f32_unary_softplus; break;
                case HTP_OP_UNARY_TANH:      task_func = unary_task_f32_unary_tanh; break;
                case HTP_OP_L2_NORM:         task_func = unary_task_f32_l2_norm; break;
                case HTP_OP_TRI:             task_func = unary_task_f32_tri; break;
                default:                     break;
            }
        }

        if (task_func) {
            worker_pool_run_func(octx->ctx->worker_pool, task_func, &uctx, n_threads);
        } else {
            FARF(ERROR, "execute_op_unary_f32: task function is NULL for op %d\n", octx->op);
            err = HTP_STATUS_NO_SUPPORT;
        }
    }

    return err;
}

int op_tri(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            err = execute_op_unary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            err = execute_op_unary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
