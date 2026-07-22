#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"

#define htp_act_preamble                                 \
    const struct htp_tensor * src0 = actx->octx->src[0]; \
    const struct htp_tensor * src1 = actx->octx->src[1]; \
    const struct htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    const uint32_t ne00 = src0->ne[0];                   \
    const uint32_t ne01 = src0->ne[1];                   \
    const uint32_t ne02 = src0->ne[2];                   \
    const uint32_t ne03 = src0->ne[3];                   \
                                                         \
    const uint32_t nb00 = src0->nb[0];                   \
    const uint32_t nb01 = src0->nb[1];                   \
    const uint32_t nb02 = src0->nb[2];                   \
    const uint32_t nb03 = src0->nb[3];                   \
                                                         \
    const uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    const uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    const uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    const uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    const uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    const uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    const uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    const uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    const uint32_t ne0 = dst->ne[0];                     \
    const uint32_t ne1 = dst->ne[1];                     \
    const uint32_t ne2 = dst->ne[2];                     \
    const uint32_t ne3 = dst->ne[3];                     \
                                                         \
    const uint32_t nb0 = dst->nb[0];                     \
    const uint32_t nb1 = dst->nb[1];                     \
    const uint32_t nb2 = dst->nb[2];                     \
    const uint32_t nb3 = dst->nb[3];

struct htp_act_context {
    struct htp_ops_context * octx;

    // Precomputed values
    const uint8_t *          data_src0;
    const uint8_t *          data_src1;
    uint8_t *                data_dst;

    size_t                   src0_row_size;
    size_t                   src1_row_size;
    size_t                   dst_row_size;

    size_t                   src0_row_stride;
    size_t                   src1_row_stride;

    size_t                   src0_row_size_aligned;
    size_t                   src1_row_size_aligned;
    size_t                   dst_row_size_aligned;

    size_t                   src0_spad_half_size;
    size_t                   src1_spad_half_size;
    size_t                   dst_spad_half_size;

    uint32_t                 block;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_per_thread;
    int                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_size_per_thread;
    size_t                   vtcm_src1_size_per_thread;
    size_t                   vtcm_dst_size_per_thread;
};

struct htp_act_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    uint32_t vtcm_row_per_thread;
};

static inline void htp_act_vtcm_layout_build(struct htp_act_vtcm_layout * L,
                                             size_t                       src0_row_size_aligned,
                                             size_t                       src1_row_size_aligned,
                                             size_t                       dst_row_size_aligned,
                                             uint32_t                     n_threads,
                                             size_t                       vtcm_size) {
    const size_t   spad_size_per_row   = src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned;
    const uint32_t vtcm_row_per_thread = (uint32_t) (vtcm_size / (n_threads * spad_size_per_row));

    L->vtcm_row_per_thread = vtcm_row_per_thread;

    L->src0_bytes_per_thread = src0_row_size_aligned * vtcm_row_per_thread;
    L->src1_bytes_per_thread = src1_row_size_aligned * vtcm_row_per_thread;
    L->dst_bytes_per_thread  = dst_row_size_aligned * vtcm_row_per_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_bytes_per_thread * n_threads;
    L->off_dst  = L->off_src1 + L->src1_bytes_per_thread * n_threads;

    L->total_bytes = L->off_dst + L->dst_bytes_per_thread * n_threads;
}

#define htp_glu_op_preamble                                            \
    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;  \
    const size_t src1_row_size_aligned = actx->src1_row_size_aligned;  \
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;   \
    const int    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
static void swiglu_f32(const float * restrict src0,
                       const float * restrict src1,
                       float * restrict dst,
                       const uint32_t num_rows,
                       const struct htp_act_context * actx) {
    htp_glu_op_preamble;

    for (uint32_t ib = 0; ib < num_rows; ib++) {
        const uint8_t * restrict src0_ptr = (const uint8_t *) src0 + (ib * src0_row_size_aligned);
        const uint8_t * restrict src1_ptr = (const uint8_t *) src1 + (ib * src1_row_size_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
static void swiglu_oai_f32(const float * restrict src0,
                           const float * restrict src1,
                           float * restrict dst,
                           const uint32_t num_rows,
                           const struct htp_act_context * actx) {
    htp_glu_op_preamble;
    const float alpha = ((const float *) (actx->octx->op_params))[2];
    const float limit = ((const float *) (actx->octx->op_params))[3];

    for (uint32_t ib = 0; ib < num_rows; ib++) {
        const uint8_t * restrict src0_ptr = (const uint8_t *) src0 + (ib * src0_row_size_aligned);
        const uint8_t * restrict src1_ptr = (const uint8_t *) src1 + (ib * src1_row_size_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_size_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, limit, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -limit, limit, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_add_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

static const float GELU_COEF_A     = 0.044715f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

static inline void hvx_geglu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) {
    assert((unsigned long) dst  % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);

    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;
    const HVX_Vector * restrict vsrc1 = (const HVX_Vector *) src1;

    const uint32_t epv  = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;

    const float GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    const HVX_Vector v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    const HVX_Vector v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    const HVX_Vector v_half              = hvx_vec_splat_f32(0.5f);
    const HVX_Vector v_one               = hvx_vec_splat_f32(1.0f);
    const HVX_Vector v_two               = hvx_vec_splat_f32(2.0f);

    // Hoisted fast sigmoid / inverse constants to avoid loop-internal overhead
    const HVX_Vector v_log2f             = Q6_V_vsplat_R(FAST_SIGMOID_LOG2F);
    const HVX_Vector v_c1                = Q6_V_vsplat_R(FAST_SIGMOID_C1);
    const HVX_Vector v_c2                = Q6_V_vsplat_R(FAST_SIGMOID_C2);
    const HVX_Vector v_inv_aprox         = Q6_V_vsplat_R(0x7EEEEBB3);
    const HVX_Vector v_max_exp           = hvx_vec_splat_f32(87.0f);
    const HVX_Vector v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    for (; i < nvec; i++) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_Vector coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_add_f32_f32(coef, v_sqrt_2_pi);
        HVX_Vector inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner
        HVX_Vector y2 = hvx_vec_mul_f32_f32(inner, v_two);

        // Sigmoid guard check predicates
        HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(v_max_exp, y2);
        HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(y2, v_min_exp);

        // Fast sigmoid approximation
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(y2, v_log2f);
        v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), v_half);

        HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
        HVX_Vector x_sig  = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
        HVX_Vector xx_sig = Q6_Vqf32_vmpy_Vqf32Vqf32(x_sig, x_sig);

        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx_sig), v_c2);
        v1 = Q6_Vqf32_vadd_Vqf32Vsf(v1, v_log2f);

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x_sig), v_c1);
        v2 = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx_sig);
        v2 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x_sig);

        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
        v3 = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

        HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
        HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

        // Fast division (Newton-Raphson with 2 iterations)
        HVX_Vector i_sf = Q6_Vw_vsub_VwVw(v_inv_aprox, v5);
        HVX_Vector r_qf = Q6_Vqf32_vmpy_VsfVsf(
            i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_two, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
        r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
            r_qf, Q6_Vqf32_vsub_VsfVsf(v_two, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
        HVX_Vector res_inv = Q6_Vsf_equals_Vqf32(r_qf);

        HVX_Vector sig2y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, res_inv));

        // Sigmoid guards
        sig2y = Q6_V_vmux_QVV(pred_max, sig2y, v_one);
        sig2y = Q6_V_vmux_QVV(pred_min, sig2y, Q6_V_vzero());

        // tanh(inner) = 2 * sigmoid(2 * inner) - 1
        HVX_Vector tanh_val = hvx_vec_mul_f32_f32(sig2y, v_two);
        tanh_val = hvx_vec_sub_f32_f32(tanh_val, v_one);

        HVX_Vector tanh_plus_one = hvx_vec_add_f32_f32(tanh_val, v_one);
        HVX_Vector half_x = hvx_vec_mul_f32_f32(x, v_half);
        HVX_Vector gelu_x = hvx_vec_mul_f32_f32(half_x, tanh_plus_one);

        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    if (nloe) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_Vector coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_add_f32_f32(coef, v_sqrt_2_pi);
        HVX_Vector inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_Vector y2 = hvx_vec_mul_f32_f32(inner, v_two);

        HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(v_max_exp, y2);
        HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(y2, v_min_exp);

        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(y2, v_log2f);
        v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), v_half);

        HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
        HVX_Vector x_sig  = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
        HVX_Vector xx_sig = Q6_Vqf32_vmpy_Vqf32Vqf32(x_sig, x_sig);

        HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx_sig), v_c2);
        v1 = Q6_Vqf32_vadd_Vqf32Vsf(v1, v_log2f);

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x_sig), v_c1);
        v2 = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx_sig);
        v2 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x_sig);

        HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
        v3 = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

        HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
        HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

        HVX_Vector i_sf = Q6_Vw_vsub_VwVw(v_inv_aprox, v5);
        HVX_Vector r_qf = Q6_Vqf32_vmpy_VsfVsf(
            i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v_two, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
        r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
            r_qf, Q6_Vqf32_vsub_VsfVsf(v_two, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
        HVX_Vector res_inv = Q6_Vsf_equals_Vqf32(r_qf);

        HVX_Vector sig2y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, res_inv));

        sig2y = Q6_V_vmux_QVV(pred_max, sig2y, v_one);
        sig2y = Q6_V_vmux_QVV(pred_min, sig2y, Q6_V_vzero());

        HVX_Vector tanh_val = hvx_vec_mul_f32_f32(sig2y, v_two);
        tanh_val = hvx_vec_sub_f32_f32(tanh_val, v_one);

        HVX_Vector tanh_plus_one = hvx_vec_add_f32_f32(tanh_val, v_one);
        HVX_Vector half_x = hvx_vec_mul_f32_f32(x, v_half);
        HVX_Vector gelu_x = hvx_vec_mul_f32_f32(half_x, tanh_plus_one);

        HVX_Vector res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((void *) &vdst[i], nloe * sizeof(float), res);
    }
}

// geglu(x, g) = gelu(x) * g
static void geglu_f32(const float * restrict src0,
                      const float * restrict src1,
                      float * restrict dst,
                      const uint32_t num_rows,
                      const struct htp_act_context * actx) {
    htp_glu_op_preamble;

    for (uint32_t ib = 0; ib < num_rows; ib++) {
        const uint8_t * restrict src0_ptr = (const uint8_t *) src0 + (ib * src0_row_size_aligned);
        const uint8_t * restrict src1_ptr = (const uint8_t *) src1 + (ib * src1_row_size_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_size_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#define DEFINE_GLU_PER_THREAD(NAME, OP_STR, CORE_EXPR)                                                                 \
    static void glu_##NAME##_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {                         \
        struct htp_act_context * actx = (struct htp_act_context *) data;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        struct htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : NULL;                          \
                                                                                                                       \
        size_t src0_row_size = actx->src0_row_size;                                                                    \
        size_t src1_row_size = actx->src1_row_size;                                                                    \
        size_t dst_row_size  = actx->dst_row_size;                                                                     \
                                                                                                                       \
        size_t src0_row_stride = actx->src0_row_stride;                                                                \
        size_t src1_row_stride = actx->src1_row_stride;                                                                \
                                                                                                                       \
        const uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;                                            \
                                                                                                                       \
        const uint32_t src0_start_row = src0_nrows_per_thread * ith;                                                   \
        const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                       \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        if (src0_start_row >= src0_end_row) {                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        const uint8_t * restrict data_src0 = actx->data_src0;                                                          \
        const uint8_t * restrict data_src1 = actx->data_src1;                                                          \
        uint8_t * restrict data_dst        = actx->data_dst;                                                           \
                                                                                                                       \
        const size_t src0_row_size_aligned = actx->src0_row_size_aligned;                                              \
        const size_t src1_row_size_aligned = actx->src1_row_size_aligned;                                              \
        const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;                                               \
                                                                                                                       \
        uint8_t * restrict src0_spad_data = actx->vtcm_src0 + (ith * actx->vtcm_src0_size_per_thread);                 \
        uint8_t * restrict src1_spad_data = actx->vtcm_src1 + (ith * actx->vtcm_src1_size_per_thread);                 \
        uint8_t * restrict dst_spad_data  = actx->vtcm_dst  + (ith * actx->vtcm_dst_size_per_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_size = actx->src0_spad_half_size;                                                        \
        size_t src1_spad_half_size = actx->src1_spad_half_size;                                                        \
        size_t dst_spad_half_size  = actx->dst_spad_half_size;                                                         \
                                                                                                                       \
        const int BLOCK = actx->block;                                                                                 \
        if (BLOCK == 0) {                                                                                              \
            FARF(ERROR,                                                                                                \
                 OP_STR                                                                                                \
                 " : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",      \
                 actx->vtcm_src0_size_per_thread, src0_row_size_aligned);                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_queue * dma_queue = actx->octx->ctx->dma[ith];                                                             \
                                                                                                                       \
        /* See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379 */                 \
        for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) { \
            const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_queue_push_vtcm_to_ddr(dma_queue,                                                                      \
                                       dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),        \
                                       dst_row_size, dst_row_size_aligned, 0);                                         \
                                                                                                                       \
            dma_queue_push(                                                                                            \
                dma_queue,                                                                                             \
                dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_stride)),   \
                src0_row_size_aligned, src0_row_stride, src0_row_size, block_size);                                    \
            dma_queue_push(                                                                                            \
                dma_queue,                                                                                             \
                dma_make_ptr(src1_spad_data + (spad_idx * src1_spad_half_size), data_src1 + (ir * src1_row_stride)),   \
                src1_row_size_aligned, src1_row_stride, src1_row_size, block_size);                                    \
        }                                                                                                              \
                                                                                                                       \
        for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {                                           \
            const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);                                                 \
                                                                                                                       \
            float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;                                                \
            float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;                                                \
            float * src1_spad = (float *) dma_queue_pop(dma_queue).dst;                                                \
                                                                                                                       \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            CORE_EXPR;                                                                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad),              \
                                       dst_row_size, dst_row_size_aligned, block_size);                                \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            const uint32_t pref_block = (ir + BLOCK * 2);                                                              \
            if (pref_block < src0_end_row) {                                                                           \
                const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);                                \
                dma_queue_push(dma_queue, dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_stride)),         \
                               src0_row_size_aligned, src0_row_stride, src0_row_size, pref_block_size);                \
                dma_queue_push(dma_queue, dma_make_ptr(src1_spad, data_src1 + (pref_block * src1_row_stride)),         \
                               src1_row_size_aligned, src1_row_stride, src1_row_size, pref_block_size);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_queue_flush(dma_queue);                                                                                    \
                                                                                                                       \
    }

DEFINE_GLU_PER_THREAD(swiglu, "swiglu-f32", swiglu_f32(src0_spad, src1_spad, dst_spad, block_size, actx))
DEFINE_GLU_PER_THREAD(swiglu_oai, "swiglu-oai-f32", swiglu_oai_f32(src0_spad, src1_spad, dst_spad, block_size, actx))
DEFINE_GLU_PER_THREAD(geglu, "geglu-f32", geglu_f32(src0_spad, src1_spad, dst_spad, block_size, actx))

static int execute_op_activations_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_callback_t act_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_GLU_SWIGLU:
            act_op_func = (worker_callback_t)glu_swiglu_f32_per_thread;
            op_type     = "swiglu-f32";
            break;

        case HTP_OP_GLU_SWIGLU_OAI:
            act_op_func = (worker_callback_t)glu_swiglu_oai_f32_per_thread;
            op_type     = "swiglu-oai-f32";
            break;

        case HTP_OP_GLU_GEGLU:
            act_op_func = (worker_callback_t)glu_geglu_f32_per_thread;
            op_type     = "geglu-f32";
            break;
        default:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = MIN(octx->n_threads, src0_nrows);

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    const size_t nc_bytes    = dst->ne[0] * SIZEOF_FP32;
    const size_t src0_row_size = nc_bytes;
    const size_t src1_row_size = nc_bytes;
    const size_t dst_row_size  = nc_bytes;
    const size_t src0_row_stride = src0->nb[1];
    const size_t src1_row_stride = src1 ? src1->nb[1] : src0->nb[1];

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);

    struct htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_row_size_aligned, src1_row_size_aligned, dst_row_size_aligned, n_threads,
                              octx->ctx->vtcm_size);

    // Make sure the reserved vtcm size is sufficient
    if (L.vtcm_row_per_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_size,
             (src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned) * n_threads);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (src1) {
        FARF(HIGH, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_bytes_per_thread * n_threads,
             L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    } else {
        FARF(HIGH, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_bytes_per_thread * n_threads, L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    }

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        return HTP_STATUS_OK;
    }

    // Prepare context
    struct htp_act_context actx;
    actx.octx = octx;

    actx.src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads;

    actx.src0_row_size = src0_row_size;
    actx.src1_row_size = src1_row_size;
    actx.dst_row_size  = dst_row_size;

    actx.src0_row_size_aligned = src0_row_size_aligned;
    actx.src1_row_size_aligned = src1_row_size_aligned;
    actx.dst_row_size_aligned  = dst_row_size_aligned;

    actx.src0_row_stride = src0_row_stride;
    actx.src1_row_stride = src1_row_stride;

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    actx.vtcm_src0_size_per_thread = L.src0_bytes_per_thread;
    actx.vtcm_src1_size_per_thread = L.src1_bytes_per_thread;
    actx.vtcm_dst_size_per_thread  = L.dst_bytes_per_thread;

    actx.src0_spad_half_size = L.src0_bytes_per_thread / 2;
    actx.src1_spad_half_size = L.src1_bytes_per_thread / 2;
    actx.dst_spad_half_size  = L.dst_bytes_per_thread / 2;

    actx.block = actx.src0_spad_half_size / actx.src0_row_size_aligned;
    actx.src0_nrows = src0_nrows;

    actx.nc = dst->ne[0];

    // Pointers and GLU logic
    const uint8_t * data_src0 = (const uint8_t *) src0->data;
    const uint8_t * data_src1 = src1 ? (const uint8_t *) src1->data : NULL;

    if (!src1 && (octx->op == HTP_OP_GLU_SWIGLU || octx->op == HTP_OP_GLU_SWIGLU_OAI || octx->op == HTP_OP_GLU_GEGLU)) {
         const int32_t swapped = octx->op_params[1];
         data_src1 = data_src0;
         actx.src1_row_size = actx.src0_row_size;

         size_t nc_in_bytes = actx.nc * SIZEOF_FP32;
         if (swapped) {
             data_src0 += nc_in_bytes;
         } else {
             data_src1 += nc_in_bytes;
         }
    }

    actx.data_src0 = data_src0;
    actx.data_src1 = data_src1;
    actx.data_dst  = (uint8_t *) dst->data;

    worker_pool_run_func(octx->ctx->worker_pool, act_op_func, &actx, n_threads);
    return HTP_STATUS_OK;
}

int op_activations(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            return execute_op_activations_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
