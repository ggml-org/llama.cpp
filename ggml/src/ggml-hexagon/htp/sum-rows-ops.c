#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "dma-queue.h"
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
        const float * restrict src_local = (const float *) ((const uint8_t *) src_th + ir * src_stride);
        float       * restrict dst_local = (float *)       ((uint8_t *)       dst_th + ir * dst_stride);

        if (ir + 1 < n_rows) {
            hex_l2fetch((const uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        if (opt_path) {
            *dst_local = hvx_reduce_sum_f32_a((const uint8_t *) src_local, ne00);
        } else {
            *dst_local = hvx_reduce_sum_f32((const uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_row);
}

int op_sum_rows(struct htp_ops_context * octx) {
    sum_rows_preamble;

    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
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

struct sum_context {
    struct htp_ops_context * octx;
    const float            * src_data;
    float                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 total_elems;
    uint32_t                 elems_per_thread;
};

static void sum_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct sum_context * sctx = (struct sum_context *) data;
    const uint32_t start = sctx->elems_per_thread * ith;
    const uint32_t end   = MIN(start + sctx->elems_per_thread, sctx->total_elems);

    if (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        return;
    }

    const uint32_t n = end - start;
    const float * src = sctx->src_data + start;

    struct htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((const void *) src, n * sizeof(float));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((const uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

int op_sum(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    if (total_elems == 0) {
        ((float *) dst->data)[0] = 0.0f;
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = (total_elems >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
    const uint32_t raw_chunk = (total_elems + n_threads - 1) / n_threads;
    const uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

    struct sum_context sctx = {
        .octx             = octx,
        .src_data         = (const float *) src0->data,
        .total_elems      = total_elems,
        .elems_per_thread = elems_per_thread,
    };

    work_queue_run(octx->ctx->work_queue, sum_thread_f32, &sctx, n_threads);

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_threads; i++) {
        sum += sctx.partial_sums[i];
    }
    ((float *) dst->data)[0] = sum;

    return HTP_STATUS_OK;
}

static inline void argmax_slice_f32(
    const float * restrict src,
    uint32_t n,
    uint32_t offset,
    float * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

struct argmax_context {
    struct htp_ops_context * octx;
    const float            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_thread;
    uint32_t                 elems_per_thread;

    float                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

static void argmax_thread_single_row(unsigned int nth, unsigned int ith, void * data) {
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t start = actx->elems_per_thread * ith;
    const uint32_t end   = MIN(start + actx->elems_per_thread, actx->ne00);

    if (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        return;
    }

    const uint32_t n = end - start;
    const float * src = actx->src_data + start;

    struct htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((const void *) src, n * sizeof(float));

    float max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

static void argmax_thread_multi_row(unsigned int nth, unsigned int ith, void * data) {
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t r0 = actx->row_start + actx->rows_per_thread * ith;
    const uint32_t r1 = MIN(r0 + actx->rows_per_thread, actx->row_start + actx->nrows);

    if (r0 >= r1) {
        return;
    }

    struct htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    for (uint32_t r = r0; r < r1; r++) {
        const float * src_row = (const float *) ((const uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((const void *) src_row, actx->ne00 * sizeof(float));

        float max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

int op_argmax(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t ne00 = src0->ne[0];
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (ne00 == 0 || src0_nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (src0_nrows == 1) {
        if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
            return HTP_STATUS_OK;
        }

        if (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            return HTP_STATUS_OK;
        }

        const uint32_t n_threads = (ne00 >= 1024) ? MIN(octx->n_threads, HTP_MAX_NTHREADS) : 1;
        const uint32_t raw_chunk = (ne00 + n_threads - 1) / n_threads;
        const uint32_t elems_per_thread = hex_round_up(raw_chunk, 32);

        struct argmax_context actx = {
            .octx             = octx,
            .src_data         = (const float *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * sizeof(float)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_thread  = 1,
            .elems_per_thread = elems_per_thread,
        };

        work_queue_run(octx->ctx->work_queue, argmax_thread_single_row, &actx, n_threads);

        float best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        for (uint32_t i = 1; i < n_threads; i++) {
            if (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        return HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, sizeof(int32_t));
        const uint32_t elems_per_chunk = can_split ? 32 : 0;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, nrows);
    const uint32_t rows_per_thread = (nrows + n_threads - 1) / n_threads;

    struct argmax_context actx = {
        .octx             = octx,
        .src_data         = (const float *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * sizeof(float)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_thread  = rows_per_thread,
        .elems_per_thread = 0,
    };

    work_queue_run(octx->ctx->work_queue, argmax_thread_multi_row, &actx, n_threads);
    return HTP_STATUS_OK;
}
