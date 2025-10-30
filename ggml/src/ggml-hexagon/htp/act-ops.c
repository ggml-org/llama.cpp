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

#define htp_act_preamble3              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

#define htp_act_preamble2              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

static void glu_swiglu_fp32_per_thread(const struct htp_tensor * src0,
                                       const struct htp_tensor * src1,
                                       struct htp_tensor *       dst,
                                       const int32_t *           op_params,
                                       struct htp_spad *         src0_spad,
                                       struct htp_spad *         src1_spad,
                                       struct htp_spad *         dst_spad,
                                       uint32_t                  nth,
                                       uint32_t                  ith,
                                       uint32_t                  src0_nrows_per_thread) {
    htp_act_preamble3;

    size_t src0_row_size = nb01;
    size_t src1_row_size = nb11;
    size_t dst_row_size  = nb1;

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
    if (!htp_is_aligned((void *) src0->data, VLEN) || !htp_is_aligned((void *) dst->data, VLEN)) {
        is_aligned = 0;
        FARF(HIGH, "swiglu-f32: unaligned addresses in elementwise op, possibly slower execution\n");
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    bool src1_valid = src1->ne[0];
    if (!src1_valid) {
        data_src1     = data_src0;
        src1_row_size = src0_row_size;
    }

    uint8_t * restrict src0_spad_data = src0_spad->data + (ith * src0_row_size);
    uint8_t * restrict src1_spad_data = src1_spad->data + (ith * src1_row_size);
    uint8_t * restrict dst_spad_data  = dst_spad->data + (ith * dst_row_size);

    const int32_t swapped = op_params[1];

    const int nc = (src1_valid) ? ne0 : ne0 / 2;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        const float * restrict src0 = (float *) (data_src0 + (ir * src0_row_size));
        const float * restrict src1 = (float *) (data_src1 + (ir * src1_row_size));
        float * restrict dst        = (float *) (data_dst + (ir * dst_row_size));

        if (ir + 1 < src0_end_row) {
            htp_l2fetch(src0 + src0_row_size, 1, src0_row_size, src0_row_size);
        }

        if (!src1_valid) {
            src0 += swapped ? nc : 0;
            src1 += swapped ? 0 : nc;
        }

        if (1 == opt_path) {
            hvx_fast_sigmoid_f32((const uint8_t *) src0, (uint8_t *) src0_spad_data, nc);
            hvx_mul_mul_f32_opt((const uint8_t *) src0, (const uint8_t *) src0_spad_data, (const uint8_t *) src1,
                                (uint8_t *) dst, nc);
        } else {
            hvx_exp_f32((const uint8_t *) src0, src0_spad_data, nc, true);
            hvx_add_scalar_f32(src0_spad_data, 1.0, src1_spad_data, nc);
            hvx_inverse_f32(src1_spad_data, src0_spad_data, nc);

            hvx_mul_f32((const uint8_t *) src0, src0_spad_data, dst_spad_data, nc);
            hvx_mul_f32(dst_spad_data, (const uint8_t *) src1, (uint8_t *) dst, nc);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "swiglu-f32 %d/%d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth, opt_path,
         ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void glu_swiglu_oai_fp32_per_thread(const struct htp_tensor * src0,
                                           const struct htp_tensor * src1,
                                           struct htp_tensor *       dst,
                                           const int32_t *           op_params,
                                           struct htp_spad *         src0_spad,
                                           struct htp_spad *         src1_spad,
                                           struct htp_spad *         dst_spad,
                                           uint32_t                  nth,
                                           uint32_t                  ith,
                                           uint32_t                  src0_nrows_per_thread) {
    htp_act_preamble3;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    if (!htp_is_aligned((void *) src0->data, VLEN) || !htp_is_aligned((void *) dst->data, VLEN)) {
        FARF(HIGH, "act-f32: unaligned addresses in activations op, possibly slower execution\n");
    }

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    const uint8_t * restrict data_src1 = (const uint8_t *) src1->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    bool src1_valid = src1->ne[0];
    if (!src1_valid) {
        data_src1 = data_src0;
    }

    uint8_t * restrict src0_spad_data = src0_spad->data + (ith * src0_row_size);
    uint8_t * restrict src1_spad_data = src1_spad->data + (ith * src1_row_size);
    uint8_t * restrict dst_spad_data  = dst_spad->data + (ith * dst_row_size);

    const int32_t swapped = op_params[1];
    const float   alpha   = ((const float *) (op_params))[2];
    const float   limit   = ((const float *) (op_params))[3];

    const int nc = (src1_valid) ? ne0 : ne0 / 2;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        const float * restrict src0 = (float *) (data_src0 + (ir * src0_row_size));
        const float * restrict src1 = (float *) (data_src1 + (ir * src1_row_size));
        float * restrict dst        = (float *) (data_dst + (ir * dst_row_size));

        if (ir + 1 < src0_end_row) {
            htp_l2fetch(src0 + src0_row_size, 1, src0_row_size, src0_row_size);
        }

        if (!src1) {
            src0 += swapped ? nc : 0;
            src1 += swapped ? 0 : nc;
        }

        // x (src0_spad_data) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((const uint8_t *) src0, limit, src0_spad_data, nc);
        // y1 (src1_spad_data) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((const uint8_t *) src1, limit, limit, src1_spad_data, nc);
        // y (src1_spad_data)  = y1 + 1.f
        hvx_add_scalar_f32(src1_spad_data, 1.0, src1_spad_data, nc);
        // x1 (dst_spad_data) = alpha * (x)
        hvx_mul_scalar_f32(src0_spad_data, alpha, dst_spad_data, nc);
        // x2 (dst_spad_data) = expf(-x1)
        hvx_exp_f32(dst_spad_data, dst_spad_data, nc, true);
        // x3 (dst_spad_data) = x2 + 1.f
        hvx_add_scalar_f32(dst_spad_data, 1.0, dst_spad_data, nc);
        // x4 (dst_spad_data) = 1 / x3
        hvx_inverse_f32(dst_spad_data, dst_spad_data, nc);
        // out_glu(dst_spad_data) = x * x4
        hvx_mul_f32(src0_spad_data, dst_spad_data, dst_spad_data, nc);
        // out = out_glu * (y + 1.f);
        hvx_mul_f32(dst_spad_data, src1_spad_data, (uint8_t *) dst, nc);
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "swiglu-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1], src1->ne[2],
         src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void unary_silu_fp32_per_thread(const struct htp_tensor * src0,
                                       struct htp_tensor *       dst,
                                       const int32_t *           op_params,
                                       struct htp_spad *         src0_spad,
                                       struct htp_spad *         dst_spad,
                                       uint32_t                  nth,
                                       uint32_t                  ith,
                                       uint32_t                  src0_nrows_per_thread) {
    htp_act_preamble2;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    int is_aligned = 1;
    int opt_path   = 0;
    if (!htp_is_aligned((void *) src0->data, VLEN) || !htp_is_aligned((void *) dst->data, VLEN)) {
        is_aligned = 0;
        FARF(HIGH, "silu-f32: unaligned addresses in elementwise op, possibly slower execution\n");
    }
    if ((1 == is_aligned) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    const uint8_t * restrict data_src0 = (const uint8_t *) src0->data;
    uint8_t * restrict data_dst        = (uint8_t *) dst->data;

    uint8_t * restrict src0_spad_data = src0_spad->data + (ith * src0_row_size);
    uint8_t * restrict dst_spad_data  = dst_spad->data + (ith * dst_row_size);

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        const float * restrict src0 = (float *) (data_src0 + (ir * src0_row_size));
        float * restrict dst        = (float *) (data_dst + (ir * dst_row_size));

        if (ir + 1 < src0_end_row) {
            htp_l2fetch(src0 + src0_row_size, 1, src0_row_size, src0_row_size);
        }

        if (1 == opt_path) {
            hvx_fast_sigmoid_f32((const uint8_t *) src0, (uint8_t *) src0_spad_data, ne0);
            hvx_mul_f32_opt((const uint8_t *) src0, src0_spad_data, (uint8_t *) dst, ne0);
        } else {
            hvx_exp_f32((const uint8_t *) src0, src0_spad_data, ne0, true);
            hvx_add_scalar_f32(src0_spad_data, 1.0, dst_spad_data, ne0);
            hvx_inverse_f32(dst_spad_data, src0_spad_data, ne0);

            hvx_mul_f32((const uint8_t *) src0, src0_spad_data, (uint8_t *) dst, ne0);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "silu-f32 %d/%d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, opt_path, ne00, ne01, ne02,
         ne03, src0_start_row, src0_end_row, ne0, ne1, ne2, ne3, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void unary_silu_fp32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    unary_silu_fp32_per_thread(&octx->src0, &octx->dst, octx->op_params, &octx->src0_spad, &octx->dst_spad, n, i,
                               octx->src0_nrows_per_thread);
}

static void glu_swiglu_fp32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    glu_swiglu_fp32_per_thread(&octx->src0, &octx->src1, &octx->dst, octx->op_params, &octx->src0_spad,
                               &octx->src1_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread);
}

static void glu_swiglu_oai_fp32(unsigned int n, unsigned int i, void * data) {
    struct htp_ops_context * octx = (struct htp_ops_context *) data;
    glu_swiglu_oai_fp32_per_thread(&octx->src0, &octx->src1, &octx->dst, octx->op_params, &octx->src0_spad,
                                   &octx->src1_spad, &octx->dst_spad, n, i, octx->src0_nrows_per_thread);
}

static int execute_op_activations_fp32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    if (((src0->ne[0] * SIZEOF_FP32) != src0->nb[1]) || ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1])) {
        FARF(ERROR, "Non-contiguous tensors are not supported at this time \n");
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_callback_t act_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_UNARY_SILU:
            act_op_func = unary_silu_fp32;
            op_type     = "silu-f32";
            break;

        case HTP_OP_GLU_SWIGLU:
            act_op_func = glu_swiglu_fp32;
            op_type     = "swiglu-f32";
            break;

        case HTP_OP_GLU_SWIGLU_OAI:
            act_op_func = glu_swiglu_oai_fp32;
            op_type     = "swiglu-oai-f32";
            break;

        default:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t src1_row_size = src1->ne[0] ? src1->nb[1] : src0->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size
    octx->dst_spad.size  = htp_round_up(dst_row_size, 128) * octx->n_threads;
    octx->src0_spad.size = htp_round_up(src0_row_size, 128) * octx->n_threads;
    octx->src1_spad.size = htp_round_up(src1_row_size, 128) * octx->n_threads;

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
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);

        octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, act_op_func, octx, n_jobs);
    }

    return err;
}

int op_activations(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_activations_fp32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
