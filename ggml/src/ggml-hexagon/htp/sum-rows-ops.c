#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#define sum_rows_preamble                         \
    const struct htp_tensor *src0 = octx->src[0]; \
    const struct htp_tensor *dst  = octx->dst;    \
                                                  \
    const uint32_t ne00 = src0->ne[0];            \
    const uint32_t ne01 = src0->ne[1];            \
    const uint32_t ne02 = src0->ne[2];            \
    const uint32_t ne03 = src0->ne[3];            \
                                                  \
    const uint32_t nb00 = src0->nb[0];            \
    const uint32_t nb01 = src0->nb[1];            \
    const uint32_t nb02 = src0->nb[2];            \
    const uint32_t nb03 = src0->nb[3];            \
                                                  \
    const uint32_t  ne0 = dst->ne[0];             \
    const uint32_t  ne1 = dst->ne[1];             \
    const uint32_t  ne2 = dst->ne[2];             \
    const uint32_t  ne3 = dst->ne[3];             \
                                                  \
    const uint32_t  nb0 = dst->nb[0];             \
    const uint32_t  nb1 = dst->nb[1];             \
    const uint32_t  nb2 = dst->nb[2];             \
    const uint32_t  nb3 = dst->nb[3];             \

struct sum_rows_context {
    struct htp_ops_context * octx;
    const uint8_t * src_data;
    uint8_t       * dst_data;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    bool            opt_path;
};

static void sum_rows_thread_f32(unsigned int nth, unsigned int ith, void *data) {
    const struct sum_rows_context * smctx = (const struct sum_rows_context *) data;

    const uint32_t rows_per_thread = smctx->rows_per_thread;
    const uint32_t total_rows      = smctx->total_rows;

    const uint32_t start_row = rows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + rows_per_thread, total_rows);

    if (start_row >= end_row) {
        return;
    }

    const size_t   src_stride = smctx->src_stride;
    const size_t   dst_stride = smctx->dst_stride;
    const uint32_t ne00       = smctx->ne00;
    const bool     opt_path   = smctx->opt_path;

    const float * restrict src_th = (const float *) (smctx->src_data + (start_row * src_stride));
    float       * restrict dst_th = (float *)       (smctx->dst_data + (start_row * dst_stride));

    // Calculate actual number of rows for this thread
    const uint32_t n_rows = end_row - start_row;

    struct htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);

    for (uint32_t ir = 0; ir < n_rows; ir++) {
        const float * restrict src_local = src_th + (ir * (src_stride / sizeof(float)));

        if (ir + 1 < n_rows) {
            hex_l2fetch(src_local + (src_stride / sizeof(float)), src_stride, src_stride, 1);
        }

        if (opt_path) {
            dst_th[ir] = hvx_reduce_sum_f32_a((const uint8_t *) src_local, ne00);
        } else {
            dst_th[ir] = hvx_reduce_sum_f32((const uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

int op_sum_rows(struct htp_ops_context * octx) {
    sum_rows_preamble;

    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t src0_nrows      = ne01 * ne02 * ne03;
    const size_t dst_data_row_size = dst->ne[0] * sizeof(float);

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, sizeof(float), (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;
    const uint32_t rows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    bool opt_path = false;
    if ((0 == hex_is_aligned((void *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = true;
    }

    struct sum_rows_context smctx = {
        .octx            = octx,
        .src_data        = (const uint8_t *) src0->data + row_start * nb01,
        .dst_data        = (uint8_t *) dst->data + row_start * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .rows_per_thread = rows_per_thread,
        .total_rows      = nrows,
        .opt_path        = opt_path,
    };

    work_queue_run(octx->ctx->work_queue, sum_rows_thread_f32, &smctx, n_threads);

    return HTP_STATUS_OK;
}
