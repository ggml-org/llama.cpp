#include <HAP_farf.h>
#include <HAP_perf.h>

#include "dma-queue.h"
#include "dma-pipeline.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "pad-ops.h"

// Circular wrap: maps any integer x into [0, n)
static inline uint32_t wrap_around(int32_t x, uint32_t n, const struct fastdiv_values * div_n) {
    if (x >= 0) {
        const uint32_t ux = (uint32_t) x;
        return ux < n ? ux : fastmodulo(ux, n, div_n);
    }
    const uint32_t ux  = (uint32_t) (-x);
    const uint32_t rem = ux < n ? ux : fastmodulo(ux, n, div_n);
    return rem ? (n - rem) : 0;
}

// Decompose a flat dst row index into (i1, i2, i3)
static inline void pad_decompose_row(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     const struct htp_pad_kernel_params * kparams,
                                     uint32_t * i1, uint32_t * i2, uint32_t * i3) {
    *i3 = fastdiv(ir, &kparams->div_ne1_ne2);
    const uint32_t rem = ir - (*i3) * (ne1 * ne2);
    *i2 = fastdiv(rem, &kparams->div_ne1);
    *i1 = rem - (*i2) * ne1;
}

// Return non-zero if row (i1,i2,i3) falls in the non-padded interior
static inline int pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                  int32_t lp1, int32_t rp1, uint32_t ne1,
                                  int32_t lp2, int32_t rp2, uint32_t ne2,
                                  int32_t lp3, int32_t rp3, uint32_t ne3) {
    return ((int32_t) i1 >= lp1 && (int32_t) i1 < (int32_t) ne1 - rp1) &&
           ((int32_t) i2 >= lp2 && (int32_t) i2 < (int32_t) ne2 - rp2) &&
           ((int32_t) i3 >= lp3 && (int32_t) i3 < (int32_t) ne3 - rp3);
}

static inline dma_addr_t pad_src_row_data(const struct htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    return src->data
        + (size_t) (i1 - (uint32_t) lp1) * src->nb[1]
        + (size_t) (i2 - (uint32_t) lp2) * src->nb[2]
        + (size_t) (i3 - (uint32_t) lp3) * src->nb[3];
}

static inline dma_addr_t pad_circ_src_row_data(const struct htp_tensor * src,
                                               const struct htp_pad_kernel_params * kparams,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    return src->data
        + (size_t) wrap_around((int32_t) i1 - lp1, src->ne[1], &kparams->div_ne01) * src->nb[1]
        + (size_t) wrap_around((int32_t) i2 - lp2, src->ne[2], &kparams->div_ne02) * src->nb[2]
        + (size_t) wrap_around((int32_t) i3 - lp3, src->ne[3], &kparams->div_ne03) * src->nb[3];
}

struct htp_pad_context {
    struct htp_ops_context *             octx;
    const struct htp_pad_kernel_params * kparams;
    uint8_t *                            vtcm_src0;
    uint8_t *                            vtcm_dst;
    size_t                               src0_bytes_per_hvx;
    size_t                               dst_bytes_per_hvx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_per_hvx;
    uint32_t total_dst_rows;
    uint32_t row_start;
    uint32_t chunk_elems;

    size_t   type_size;
    size_t   chunk_bytes_aligned;
};

struct pad_pipe_state {
    const struct htp_tensor *            src;
    const struct htp_tensor *            dst;
    const struct htp_pad_kernel_params * kparams;
    size_t                               type_size;
    size_t                               chunk_bytes_aligned;
    int32_t                              lp0, rp0;
    int32_t                              lp1, rp1;
    int32_t                              lp2, rp2;
    int32_t                              lp3, rp3;
    uint32_t                             ne00;
    uint32_t                             ne0;
    uint32_t                             ne1;
    uint32_t                             ne2;
    uint32_t                             ne3;
    uint32_t                             nb1;
    uint32_t                             nb2;
    uint32_t                             nb3;
    uint32_t                             chunk_elems;
    uint32_t                             row;
    uint32_t                             row_end;
    uint32_t                             off;
};

// ---------------------------------------------------------------------------
// Shared task iteration and output writeback
// ---------------------------------------------------------------------------

static bool pad_next_task(void * ctx, struct dma_pipe_task * task) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;
    if (s->row >= s->row_end) {
        return false;
    }

    const uint32_t rem = s->ne0 - s->off;
    const uint32_t len = MIN(s->chunk_elems, rem);

    task->row = s->row;
    task->off = s->off;
    task->len = len;
    task->aux = 0;

    s->off += len;
    if (s->off >= s->ne0) {
        s->off = 0;
        s->row++;
    }
    return true;
}

static bool pad_const_next_task(void * ctx, struct dma_pipe_task * task) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;
    if (!pad_next_task(ctx, task)) {
        return false;
    }

    uint32_t i1, i2, i3;
    pad_decompose_row(task->row, s->ne1, s->ne2, s->kparams, &i1, &i2, &i3);
    task->aux = (uint32_t) pad_is_interior(i1, i2, i3,
                                           s->lp1, s->rp1, s->ne1,
                                           s->lp2, s->rp2, s->ne2,
                                           s->lp3, s->rp3, s->ne3);
    return true;
}

static bool pad_circ_next_task(void * ctx, struct dma_pipe_task * task) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;
    if (s->row >= s->row_end) {
        return false;
    }

    const uint32_t lp0  = (uint32_t) s->lp0;
    const uint32_t ne00 = s->ne00;
    uint32_t sx;
    if (s->off < lp0) {
        sx = ne00 - lp0 + s->off;
    } else if (s->off < lp0 + ne00) {
        sx = s->off - lp0;
    } else {
        sx = s->off - lp0 - ne00;
    }

    // Split at ne00 wrap boundary so each task is one contiguous DMA read
    const uint32_t rem = s->ne0 - s->off;
    const uint32_t len = MIN(MIN(s->chunk_elems, rem), ne00 - sx);

    task->row = s->row;
    task->off = s->off;
    task->len = len;
    task->aux = sx;

    s->off += len;
    if (s->off >= s->ne0) {
        s->off = 0;
        s->row++;
    }
    return true;
}

static void pad_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;

    uint32_t i1, i2, i3;
    pad_decompose_row(task->row, s->ne1, s->ne2, s->kparams, &i1, &i2, &i3);

    const size_t     bytes = (size_t) task->len * s->type_size;
    const dma_addr_t dst   = s->dst->data
        + (size_t) i1 * s->nb1 + (size_t) i2 * s->nb2 + (size_t) i3 * s->nb3
        + (size_t) task->off * s->type_size;

    dma_queue_push(dma_q, dma_make_data(dst, out_spad), bytes, s->chunk_bytes_aligned, bytes, 1);
}

// ---------------------------------------------------------------------------
// Constant (zero) PAD pipeline callbacks
// ---------------------------------------------------------------------------

static void pad_const_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;

    if (!task->aux) {
        dma_queue_push(dma_q, dma_make_data(in_spad, 0), 0, 0, 0, 0);
        return;
    }

    const uint32_t ov_start = MAX(task->off, (uint32_t) s->lp0);
    const uint32_t ov_end   = MIN(task->off + task->len, (uint32_t) s->lp0 + s->ne00);
    if (ov_start >= ov_end) {
        dma_queue_push(dma_q, dma_make_data(in_spad, 0), 0, 0, 0, 0);
        return;
    }

    uint32_t i1, i2, i3;
    pad_decompose_row(task->row, s->ne1, s->ne2, s->kparams, &i1, &i2, &i3);

    const uint32_t   copy_len = ov_end - ov_start;
    const uint32_t   src_off  = ov_start - (uint32_t) s->lp0;
    const size_t     bytes    = (size_t) copy_len * s->type_size;
    const dma_addr_t src_addr = pad_src_row_data(s->src, i1, i2, i3, s->lp1, s->lp2, s->lp3)
        + (size_t) src_off * s->type_size;

    dma_queue_push(dma_q, dma_make_data(in_spad, src_addr), s->chunk_bytes_aligned, bytes, bytes, 1);
}

static void __attribute__((noinline)) pad_const_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;

    const uint8_t * src_spad    = (const uint8_t *) in_spad;
    uint8_t *       dst_spad    = (uint8_t *) out_spad;
    const uint32_t  total_bytes = task->len * (uint32_t) s->type_size;

    if (!task->aux) {
        hvx_splat_u8_a(dst_spad, 0, total_bytes);
        return;
    }

    const uint32_t ov_start = MAX(task->off, (uint32_t) s->lp0);
    const uint32_t ov_end   = MIN(task->off + task->len, (uint32_t) s->lp0 + s->ne00);
    if (ov_start >= ov_end) {
        hvx_splat_u8_a(dst_spad, 0, total_bytes);
        return;
    }

    const uint32_t copy_len   = ov_end - ov_start;
    const uint32_t copy_bytes = copy_len * (uint32_t) s->type_size;
    const uint32_t dst_off    = ov_start - task->off;

    if (copy_len < task->len) {
        hvx_splat_u8_a(dst_spad, 0, total_bytes);
    }

    uint8_t * dst_interior = dst_spad + (size_t) dst_off * s->type_size;
    if (((uintptr_t) dst_interior & (VLEN - 1)) == 0) {
        hvx_copy_aa(dst_interior, src_spad, copy_bytes, 1);
    } else {
        hvx_copy_ua(dst_interior, src_spad, copy_bytes, 1);
    }
}

// ---------------------------------------------------------------------------
// Circular PAD pipeline callbacks
// ---------------------------------------------------------------------------

static void pad_circ_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;

    uint32_t i1, i2, i3;
    pad_decompose_row(task->row, s->ne1, s->ne2, s->kparams, &i1, &i2, &i3);

    const dma_addr_t src_row = pad_circ_src_row_data(s->src, s->kparams, i1, i2, i3, s->lp1, s->lp2, s->lp3);
    const size_t     bytes   = (size_t) task->len * s->type_size;

    dma_queue_push(dma_q,
                   dma_make_data(in_spad, src_row + (size_t) task->aux * s->type_size),
                   s->chunk_bytes_aligned, bytes, bytes, 1);
}

static void __attribute__((noinline)) pad_circ_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct pad_pipe_state * s = (struct pad_pipe_state *) ctx;

    hvx_copy_aa((uint8_t *) out_spad, (const uint8_t *) in_spad, task->len * (uint32_t) s->type_size, 1);
}

// ---------------------------------------------------------------------------
// Per thread workers
// ---------------------------------------------------------------------------

static void pad_state_init(struct pad_pipe_state * s, const struct htp_pad_context * pctx,
                           uint32_t row_start, uint32_t row_end) {
    struct htp_ops_context *  octx = pctx->octx;
    const struct htp_tensor * src  = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    s->src                 = src;
    s->dst                 = dst;
    s->kparams             = pctx->kparams;
    s->type_size           = pctx->type_size;
    s->chunk_bytes_aligned = pctx->chunk_bytes_aligned;
    s->lp0 = pctx->lp0; s->rp0 = pctx->rp0;
    s->lp1 = pctx->lp1; s->rp1 = pctx->rp1;
    s->lp2 = pctx->lp2; s->rp2 = pctx->rp2;
    s->lp3 = pctx->lp3; s->rp3 = pctx->rp3;
    s->ne00                = src->ne[0];
    s->ne0                 = dst->ne[0];
    s->ne1                 = dst->ne[1];
    s->ne2                 = dst->ne[2];
    s->ne3                 = dst->ne[3];
    s->nb1                 = dst->nb[1];
    s->nb2                 = dst->nb[2];
    s->nb3                 = dst->nb[3];
    s->chunk_elems         = pctx->chunk_elems;
    s->row                 = row_start;
    s->row_end             = row_end;
    s->off                 = 0;
}

static void pad_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;

    const uint32_t row_start = pctx->row_start + pctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + pctx->nrows_per_hvx, pctx->row_start + pctx->total_dst_rows);

    if (row_start >= row_end) {
        return;
    }

    uint8_t * src_spad = pctx->vtcm_src0 + ith * pctx->src0_bytes_per_hvx;
    uint8_t * dst_spad = pctx->vtcm_dst  + ith * pctx->dst_bytes_per_hvx;
    void *    in_buf[2]  = { src_spad, src_spad + pctx->chunk_bytes_aligned };
    void *    out_buf[2] = { dst_spad, dst_spad + pctx->chunk_bytes_aligned };

    struct pad_pipe_state s;
    pad_state_init(&s, pctx, row_start, row_end);

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf, pad_const_next_task, pad_const_push_in, pad_const_compute, pad_push_out);

    FARF(HIGH, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         s.src->ne[0], s.src->ne[1], s.src->ne[2], s.src->ne[3],
         s.dst->ne[0], s.dst->ne[1], s.dst->ne[2], s.dst->ne[3],
         row_start, row_end);
}

static void pad_thread_circular(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;

    const uint32_t row_start = pctx->row_start + pctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + pctx->nrows_per_hvx, pctx->row_start + pctx->total_dst_rows);

    if (row_start >= row_end) {
        return;
    }

    uint8_t * src_spad = pctx->vtcm_src0 + ith * pctx->src0_bytes_per_hvx;
    uint8_t * dst_spad = pctx->vtcm_dst  + ith * pctx->dst_bytes_per_hvx;
    void *    in_buf[2]  = { src_spad, src_spad + pctx->chunk_bytes_aligned };
    void *    out_buf[2] = { dst_spad, dst_spad + pctx->chunk_bytes_aligned };

    struct pad_pipe_state s;
    pad_state_init(&s, pctx, row_start, row_end);

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf, pad_circ_next_task, pad_circ_push_in, pad_circ_compute, pad_push_out);

    FARF(HIGH, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         s.src->ne[0], s.src->ne[1], s.src->ne[2], s.src->ne[3],
         s.dst->ne[0], s.dst->ne[1], s.dst->ne[2], s.dst->ne[3],
         row_start, row_end);
}

int op_pad(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_pad_kernel_params * kparams =
        (const struct htp_pad_kernel_params *) octx->kernel_params;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) || dst->type != src0->type) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t lp0 = octx->op_params[0];
    const int32_t rp0 = octx->op_params[1];
    const int32_t lp1 = octx->op_params[2];
    const int32_t rp1 = octx->op_params[3];
    const int32_t lp2 = octx->op_params[4];
    const int32_t rp2 = octx->op_params[5];
    const int32_t lp3 = octx->op_params[6];
    const int32_t rp3 = octx->op_params[7];

    const uint32_t total_dst_rows = kparams->total_rows;
    const size_t   type_size      = kparams->elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct htp_pad_vtcm_layout layout = { 0 };
    htp_pad_vtcm_layout_build(&layout, (size_t) kparams->chunk_elems * type_size, n_hvx);

    struct htp_pad_context pctx = {
        .octx                = octx,
        .kparams             = kparams,
        .vtcm_src0           = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_src0),
        .vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_dst),
        .src0_bytes_per_hvx  = layout.src0_bytes_per_hvx,
        .dst_bytes_per_hvx   = layout.dst_bytes_per_hvx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_per_hvx       = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_dst_rows      = nrows,
        .row_start           = row_start,
        .chunk_elems         = kparams->chunk_elems,
        .type_size           = type_size,
        .chunk_bytes_aligned = layout.chunk_bytes_aligned,
    };

    FARF(HIGH, "pad-hvx%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    switch (kparams->kernel_type) {
        case HTP_PAD_KERNEL_HVX_CIRCULAR:
            work_queue_run(octx->ctx->hvx_queue, pad_thread_circular, &pctx, n_hvx);
            break;
        default:
            work_queue_run(octx->ctx->hvx_queue, pad_thread,          &pctx, n_hvx);
            break;
    }

    return HTP_STATUS_OK;
}
