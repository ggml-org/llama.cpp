#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "fill-ops.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                          \
    const struct htp_tensor * dst = octx->dst; \
                                               \
    const uint32_t ne0 = dst->ne[0];           \
    const uint32_t ne1 = dst->ne[1];           \
    const uint32_t ne2 = dst->ne[2];           \
    const uint32_t ne3 = dst->ne[3];           \
                                               \
    const uint32_t nb1 = dst->nb[1];           \
    const uint32_t nb2 = dst->nb[2];           \
    const uint32_t nb3 = dst->nb[3];

struct htp_fill_context {
    struct htp_ops_context * octx;
    const struct htp_fill_kernel_params * kparams;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

static void fill_thread_contiguous(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
    struct htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const uint32_t elem_start = ir0 * ne0;
    const uint32_t elem_end   = ir1 * ne0;
    uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
    hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(HIGH, "fill-contig %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

static void fill_thread_strided(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
    struct htp_ops_context        * octx = fctx->octx;
    const struct htp_fill_kernel_params * kparams = fctx->kparams;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->row_start + fctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t i3  = fastdiv(ir, &kparams->div_ne1_ne2);
        const uint32_t rem = ir - i3 * (ne1 * ne2);
        const uint32_t i2  = fastdiv(rem, &kparams->div_ne1);
        const uint32_t i1  = rem - i2 * ne1;
        uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
        hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(HIGH, "fill-strided %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

int op_fill(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;
    const struct htp_fill_kernel_params * kparams =
        (const struct htp_fill_kernel_params *) octx->kernel_params;

    if (htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? kparams->rows_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, MIN(kparams->n_threads, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = octx->n_threads;

    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u ktype=%u\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, kparams->kernel_type);

    float val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], sizeof(float));

    struct htp_fill_context fctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows       = nrows,
        .row_start        = row_start,
        .elem_size        = kparams->elem_size,
    };

    switch (dst->type) {
    case HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        break;
    case HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_func_t worker = (kparams->kernel_type == HTP_FILL_KERNEL_CONTIGUOUS)
        ? fill_thread_contiguous
        : fill_thread_strided;

    work_queue_run(octx->ctx->work_queue, worker, &fctx, n_threads);

    return HTP_STATUS_OK;
}
