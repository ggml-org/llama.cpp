#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

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
    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    HVX_Vector sum_v     = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_fp32(epsilon);

    int step_of_1 = num_elems >> 5;
    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v         = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    HVX_Vector reduced_sum = hvx_vec_qf32_reduce_sum(sum_v);
    sum_v                  = hvx_vec_repl4(Q6_Vsf_equals_Vqf32(reduced_sum));

    HVX_Vector t_v            = hvx_vec_splat_fp32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_fp32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    HVX_Vector scale_v = hvx_vec_rsqrt_fp32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }
}

static void scale_htp_f32(const float * restrict src,
                          float * restrict dst,
                          uint8_t * restrict spad,
                          const uint32_t num_rows,
                          const uint32_t row_elems,
                          const size_t   row_size,
                          int32_t *      op_params,
                          int            opt_path) {
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_local = src + (ir * row_elems);
        float * restrict dst_local       = dst + (ir * row_elems);

        if (ir + 1 < num_rows) {
            htp_l2fetch(src_local + row_elems, 1, row_size, row_size);
        }

        hvx_scale_offset_f32((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems, scale, bias);
    }
}

static void rms_norm_htp_f32(const float * restrict src,
                             float * restrict dst,
                             uint8_t * restrict spad,
                             const uint32_t num_rows,
                             const uint32_t row_elems,
                             const size_t   row_size,
                             int32_t *      op_params,
                             int            opt_path) {
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_local = src + (ir * row_elems);
        float * restrict dst_local       = dst + (ir * row_elems);

        if (ir + 1 < num_rows) {
            htp_l2fetch(src_local + row_elems, 1, row_size, row_size);
        }

        if (1 == opt_path) {
            hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, spad, row_elems, epsilon);
        } else {
            float sum = hvx_sum_of_squares_f32((const uint8_t *) src_local, row_elems);

            const float mean  = sum / row_elems;
            const float scale = 1.0f / sqrtf(mean + epsilon);

            hvx_scale_f32((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems, scale);
        }
    }
}

static void unary_job_f32_per_thread(const struct htp_tensor * src,
                                     struct htp_tensor *       dst,
                                     uint8_t *                 spad,
                                     int                       htp_op,
                                     int32_t *                 op_params,
                                     uint32_t                  nth,
                                     uint32_t                  ith,
                                     uint32_t                  src0_nrows_per_thread) {
    htp_unary_preamble;

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows

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
    if ((0 == htp_is_aligned((void *) src->data, VLEN)) || (0 == htp_is_aligned((void *) dst->data, VLEN))) {
        is_aligned = 0;
        FARF(HIGH, "unary-f32: unaligned addresses in unary op, possibly slower execution\n");
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    const uint8_t * restrict data_src = (const uint8_t *) src->data;
    uint8_t * restrict data_dst       = (uint8_t *) dst->data;

    const float * restrict src_th = (float *) (data_src + (src0_start_row * src0_row_size));
    float * restrict dst_th       = (float *) (data_dst + (src0_start_row * dst_row_size));
    uint8_t * restrict spad_th    = (uint8_t *) spad + (ith * nb01);

    switch (htp_op) {
        case HTP_OP_RMS_NORM:
            rms_norm_htp_f32(src_th, dst_th, spad_th, src0_end_row - src0_start_row, ne0, nb1, op_params, opt_path);
            break;
        case HTP_OP_SCALE:
            scale_htp_f32(src_th, dst_th, spad_th, src0_end_row - src0_start_row, ne0, nb1, op_params, opt_path);
            break;

        default:
            break;
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "unary-f32 %d/%d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, opt_path, src->ne[0],
         src->ne[1], src->ne[2], src->ne[3], src0_start_row, src0_end_row, dst->ne[0], dst->ne[1], dst->ne[2],
         dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void unary_job_dispatcher_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;

    unary_job_f32_per_thread(&octx->src0, &octx->dst, octx->src0_spad.data, octx->op, octx->op_params, n, i,
                             octx->src0_nrows_per_thread);
}

static int execute_op_unary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    struct htp_tensor *       dst  = &octx->dst;

    worker_callback_t unary_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_RMS_NORM:
            unary_op_func = unary_job_dispatcher_f32;
            op_type       = "rmsnorm-f32";
            break;
        case HTP_OP_SCALE:
            unary_op_func = unary_job_dispatcher_f32;
            op_type       = "scale-f32";
            break;

        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const int      n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // VTCM scratchpads for all tensors
    octx->dst_spad.size  = htp_round_up(dst_row_size, 128) * n_threads;
    octx->src0_spad.size = htp_round_up(src0_row_size, 128) * n_threads;

    size_t spad_size = octx->src0_spad.size + octx->dst_spad.size;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

        worker_pool_run_func(octx->ctx->worker_pool, unary_op_func, octx, n_jobs);
    }

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_unary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
