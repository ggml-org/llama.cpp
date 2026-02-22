#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#define htp_softmax_preamble3                              \
    const uint32_t ne00 = src0->ne[0];                     \
    const uint32_t ne01 = src0->ne[1];                     \
    const uint32_t ne02 = src0->ne[2];                     \
    const uint32_t ne03 = src0->ne[3];                     \
                                                           \
    const uint32_t nb00 = src0->nb[0];                     \
    const uint32_t nb01 = src0->nb[1];                     \
    const uint32_t nb02 = src0->nb[2];                     \
    const uint32_t nb03 = src0->nb[3];                     \
                                                           \
    const uint32_t ne10 = (src1->ne[0]) ? src1->ne[0] : 1; \
    const uint32_t ne11 = (src1->ne[0]) ? src1->ne[1] : 1; \
    const uint32_t ne12 = (src1->ne[0]) ? src1->ne[2] : 1; \
    const uint32_t ne13 = (src1->ne[0]) ? src1->ne[3] : 1; \
                                                           \
    const uint32_t nb10 = (src1->ne[0]) ? src1->nb[0] : 1; \
    const uint32_t nb11 = (src1->ne[0]) ? src1->nb[1] : 1; \
    const uint32_t nb12 = (src1->ne[0]) ? src1->nb[2] : 1; \
    const uint32_t nb13 = (src1->ne[0]) ? src1->nb[3] : 1; \
                                                           \
    const uint32_t ne0 = dst->ne[0];                       \
    const uint32_t ne1 = dst->ne[1];                       \
    const uint32_t ne2 = dst->ne[2];                       \
    const uint32_t ne3 = dst->ne[3];                       \
                                                           \
    const uint32_t nb0 = dst->nb[0];                       \
    const uint32_t nb1 = dst->nb[1];                       \
    const uint32_t nb2 = dst->nb[2];                       \
    const uint32_t nb3 = dst->nb[3];

struct htp_softmax_context {
    bool     use_f16;
    bool     use_src1;
    uint32_t n_head;
    uint32_t n_head_log2;

    float scale;
    float max_bias;
    float m0;
    float m1;

    uint32_t src0_nrows_per_thread;
    struct fastdiv_values fastdiv_ne01;
    struct fastdiv_values fastdiv_ne02;
    struct fastdiv_values fastdiv_ne12; // For mask broadcasting
    struct fastdiv_values fastdiv_ne13; // For mask broadcasting
    size_t spad_stride;

    struct htp_ops_context * octx;
};

static void init_softmax_ctx(struct htp_softmax_context * smctx, struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;

    memset(smctx, 0, sizeof(struct htp_softmax_context));

    memcpy(&smctx->scale, (float *) octx->op_params, sizeof(float));
    memcpy(&smctx->max_bias, (float *) octx->op_params + 1, sizeof(float));

    smctx->n_head      = src0->ne[2];
    smctx->n_head_log2 = 1u << (uint32_t) floor(log2(smctx->n_head));

    smctx->m0 = powf(2.0f, -(smctx->max_bias) / smctx->n_head_log2);
    smctx->m1 = powf(2.0f, -(smctx->max_bias / 2.0f) / smctx->n_head_log2);

    smctx->use_src1 = (src1->ne[0] != 0);
    smctx->use_f16  = (src1->ne[0] != 0) && (src1->type == HTP_TYPE_F16);

    smctx->octx = octx;

    // Initialize fastdiv values
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];

    if (ne01 > 0) smctx->fastdiv_ne01 = init_fastdiv_values(ne01);
    if (ne02 > 0) smctx->fastdiv_ne02 = init_fastdiv_values(ne02);

    const uint32_t ne12 = (src1->ne[0]) ? src1->ne[2] : 1;
    const uint32_t ne13 = (src1->ne[0]) ? src1->ne[3] : 1;

    if (ne12 > 0) smctx->fastdiv_ne12 = init_fastdiv_values(ne12);
    if (ne13 > 0) smctx->fastdiv_ne13 = init_fastdiv_values(ne13);
}

static void hvx_fast_softmax_prep_f32(const uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      const int num_elems,
                                      float     scale,
                                      const uint8_t * restrict mask,
                                      float slope) {
    const uint8_t * restrict src_curr  = src;
    uint8_t * restrict dst_curr        = dst;
    const uint8_t * restrict mask_curr = mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = *(HVX_Vector *) src_curr;

        HVX_Vector v3 = *(HVX_Vector *) mask_curr;

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);

        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);

        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        *(HVX_Vector *) dst_curr = Q6_Vsf_equals_Vqf32(v5);

        src_curr += VLEN;
        dst_curr += VLEN;
        mask_curr += VLEN;
    }
}

static void hvx_fast_softmax_f32(const uint8_t * restrict src,
                                 uint8_t * restrict dst,
                                 uint8_t * restrict pad,
                                 const int num_elems) {
    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_pad       = (HVX_Vector *) pad;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);
    HVX_Vector max_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    HVX_Vector zero_v  = Q6_V_vzero();
    HVX_Vector one_v   = hvx_vec_splat_f32(1.0);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec); // replicated over all lanes

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_pad[i] = v3;
    }

    sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec)); // replicated over all lanes

    HVX_VectorPred pos_sum   = Q6_Q_vcmp_gt_VwVw(sum_vec, zero_v);
    HVX_Vector     v4        = hvx_vec_inverse_f32(sum_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_sum, v4, one_v);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_pad[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }
}

static float hvx_softmax_f32(const uint8_t * restrict src,
                             uint8_t * restrict dst,
                             uint8_t * restrict spad,
                             const int   num_elems,
                             const float max) {
    hvx_sub_scalar_f32(spad, src, max, num_elems);

    hvx_exp_f32(spad, dst, num_elems, false);

    float sum = hvx_reduce_sum_f32(dst, num_elems);

    return sum;
}

static void softmax_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_softmax_context * smctx = (struct htp_softmax_context *) data;
    struct htp_ops_context * octx = smctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    htp_softmax_preamble3;

    const uint32_t src0_nrows            = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    int is_aligned = 1;
    int opt_path   = 0;
    if (!hex_is_aligned((void *) src0->data, VLEN) || !hex_is_aligned((void *) dst->data, VLEN)) {
        is_aligned = 0;
        FARF(HIGH, "softmax-f32: unaligned addresses in elementwise op, possibly slower execution\n");
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    uint8_t * src0_spad_data = octx->src0_spad.data + (ith * smctx->spad_stride);
    uint8_t * src1_spad_data = octx->src1_spad.data + (ith * smctx->spad_stride);
    uint8_t * dst_spad_data  = octx->dst_spad.data + (ith * smctx->spad_stride);

    float * wp0 = (float *) src0_spad_data;
    float * wp1 = (float *) src1_spad_data;
    float * wp2 = (float *) dst_spad_data;

    uint32_t prev_i2 = (uint32_t)-1;
    float slope = 1.0f;

    for (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        uint32_t i1 = fastmodulo(r, ne01, &smctx->fastdiv_ne01);
        uint32_t r_div_ne01 = fastdiv(r, &smctx->fastdiv_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, &smctx->fastdiv_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, &smctx->fastdiv_ne02);

        // Map to original logic indices
        // i01 = i1
        // i02 = i2
        // i03 = i3

        const uint32_t i11 = i1;
        // const uint32_t i12 = i2 % ne12;
        // const uint32_t i13 = i3 % ne13;

        uint32_t i12, i13;
        if (ne12 == ne02) {
             i12 = i2;
        } else {
             i12 = fastmodulo(i2, ne12, &smctx->fastdiv_ne12);
        }

        if (ne13 == ne03) {
             i13 = i3;
        } else {
             i13 = fastmodulo(i3, ne13, &smctx->fastdiv_ne13);
        }

        // ALiBi
        if (i2 != prev_i2) {
            const uint32_t h = i2;  // head

            slope = (smctx->max_bias > 0.0f) ?
                        h < smctx->n_head_log2 ?
                        powf(smctx->m0, h + 1) :
                        powf(smctx->m1, 2 * (h - smctx->n_head_log2) + 1) :
                        1.0f;
            prev_i2 = i2;
        }

        float * sp = (float *) ((char *) octx->src0.data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float * dp = (float *) ((char *) octx->dst.data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // broadcast the mask across rows
        __fp16 * mp_f16 = (smctx->use_src1) ?
                              (__fp16 *) ((char *) octx->src1.data + i11 * nb11 + i12 * nb12 + i13 * nb13) :
                              NULL;
        float *  mp_f32 = (smctx->use_src1) ?
                              (float *) ((char *) octx->src1.data + i11 * nb11 + i12 * nb12 + i13 * nb13) :
                              NULL;

        if ((1 == opt_path) && (mp_f32) && !(smctx->use_f16)) {
            hvx_fast_softmax_prep_f32((const uint8_t *) sp, (uint8_t *) wp0, ne00, smctx->scale,
                                      (const uint8_t *) mp_f32, slope);
        } else {
            hvx_scale_f32((uint8_t *) wp0, (const uint8_t *) sp, ne00, smctx->scale);
            if (mp_f32) {
                if (smctx->use_f16) {
                    for (int i = 0; i < ne00; ++i) {
                        wp0[i] += slope * (float) mp_f16[i];
                    }
                } else {
                    for (int i = 0; i < ne00; ++i) {
                        wp0[i] += slope * mp_f32[i];
                    }
                }
            }
        }

        if (1 == opt_path) {
            hvx_fast_softmax_f32((const uint8_t *) wp0, (uint8_t *) dp, (uint8_t *) wp1, ne00);
        } else {
            float max = hvx_reduce_max_f32((const uint8_t *) wp0, ne00);
            float sum = hvx_softmax_f32((const uint8_t *) wp0, (uint8_t *) wp2, (uint8_t *) wp1, ne00, max);
            sum       = sum > 0.0 ? (1.0 / sum) : 1;
            hvx_scale_f32((uint8_t *) dp, (const uint8_t *) wp2, ne00, sum);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "softmax-f32 %d/%d/%d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         smctx->use_f16, opt_path, ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13,
         ne0, ne1, ne2, ne3, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_softmax_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    struct htp_softmax_context smctx;
    const char * op_type = "softmax-f32";

    switch (octx->op) {
        case HTP_OP_SOFTMAX:
            init_softmax_ctx(&smctx, octx);
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads = octx->n_threads;

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src0_row_size;
    const size_t dst_row_size  = dst->nb[1];

    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size
    octx->dst_spad.size  = hex_round_up(dst_row_size, 128) * n_threads;
    octx->src0_spad.size = hex_round_up(src0_row_size, 128) * n_threads;
    octx->src1_spad.size = hex_round_up(src1_row_size, 128) * n_threads;

    // Use stride for calculating offset
    smctx.spad_stride = hex_round_up(src0_row_size, 128);

    size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

    if (src1->ne[0]) {
        FARF(HIGH,
             "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], octx->src0_spad.size, octx->src1_spad.size,
             octx->dst_spad.size);
    } else {
        FARF(HIGH, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);
    }

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs             = MIN(n_threads, src0_nrows);
        smctx.src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, softmax_job_f32, &smctx, n_jobs);
    }

    return err;
}

int op_softmax(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
