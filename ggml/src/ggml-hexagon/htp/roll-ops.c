#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "dma-queue.h"
#include "dma-pipeline.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "roll-ops.h"

struct htp_roll_context {
    struct htp_ops_context *              octx;
    const struct htp_roll_kernel_params * kparams;
    uint8_t *                             vtcm_src;
    uint8_t *                             vtcm_dst;
    size_t                                row_size_aligned;
    size_t                                src_bytes_per_hvx;
    size_t                                dst_bytes_per_hvx;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t nrows_per_hvx;
    uint32_t use_vtcm;
};

static inline uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    if (i < 0) {
        return (uint32_t) (i + (int32_t) ne);
    }
    if ((uint32_t) i >= ne) {
        return (uint32_t) i - ne;
    }
    return (uint32_t) i;
}

#define htp_roll_preamble                                              \
    const struct htp_tensor * src0 = octx->src[0];                     \
    const struct htp_tensor * dst  = octx->dst;                        \
    const struct htp_roll_kernel_params * kparams =                    \
        (const struct htp_roll_kernel_params *) octx->kernel_params;   \
                                                                       \
    const uint32_t __attribute__((unused)) ne0 = dst->ne[0];           \
    const uint32_t __attribute__((unused)) ne1 = dst->ne[1];           \
    const uint32_t __attribute__((unused)) ne2 = dst->ne[2];           \
    const uint32_t __attribute__((unused)) ne3 = dst->ne[3];           \
                                                                       \
    const uint32_t __attribute__((unused)) nb01 = src0->nb[1];         \
    const uint32_t __attribute__((unused)) nb02 = src0->nb[2];         \
    const uint32_t __attribute__((unused)) nb03 = src0->nb[3];         \
                                                                       \
    const uint32_t __attribute__((unused)) nb1 = dst->nb[1];           \
    const uint32_t __attribute__((unused)) nb2 = dst->nb[2];           \
    const uint32_t __attribute__((unused)) nb3 = dst->nb[3];           \
                                                                       \
    const int32_t __attribute__((unused)) s0 = octx->op_params[0];     \
    const int32_t __attribute__((unused)) s1 = octx->op_params[1];     \
    const int32_t __attribute__((unused)) s2 = octx->op_params[2];     \
    const int32_t __attribute__((unused)) s3 = octx->op_params[3];     \
                                                                       \
    const uint32_t __attribute__((unused)) i0_src0 = kparams->i0_src0; \
    const uint32_t __attribute__((unused)) n0      = kparams->n0;

#define htp_roll_dma_preamble dma_queue * q = octx->ctx->dma[0];

static inline void roll_dma_push(dma_queue * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    bytes,
                                 uint32_t    nrows) {
    if (bytes == 0 || nrows == 0) {
        return;
    }

    if (!dma_queue_push(q, dma_make_data(dst, src), dst_stride, src_stride, bytes, nrows)) {
        dma_queue_flush(q);
        dma_queue_push(q, dma_make_data(dst, src),
                       dst_stride, src_stride, bytes, nrows);
    }
}

static inline void roll_dma_push_rows(dma_queue *               q,
                                      const struct htp_tensor * dst,
                                      const struct htp_tensor * src0,
                                      uint32_t                  dst_row,
                                      uint32_t                  src_row,
                                      uint32_t                  nrows,
                                      uint32_t                  row_size,
                                      uint32_t                  i0_src0,
                                      uint32_t                  n0,
                                      uint32_t                  elem_size) {
    const dma_addr_t dst_base = dst->data + (size_t) dst_row * row_size;
    const dma_addr_t src_base = src0->data + (size_t) src_row * row_size;

    roll_dma_push(q, dst_base, src_base + (size_t) i0_src0 * elem_size,
                  row_size, row_size, n0 * elem_size, nrows);
    roll_dma_push(q, dst_base + (size_t) n0 * elem_size, src_base,
                  row_size, row_size, i0_src0 * elem_size, nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
static inline void roll_dma_push_range(dma_queue * q,
                                       dma_addr_t  dst_row,
                                       dma_addr_t  src_row,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0,
                                       uint32_t    elem_size) {
    roll_dma_push(q, dst_row, src_row + (size_t) i0_src0 * elem_size,
                  dst_stride, src_stride, n0 * elem_size, nrows);
    roll_dma_push(q, dst_row + (size_t) n0 * elem_size, src_row,
                  dst_stride, src_stride, i0_src0 * elem_size, nrows);
}

static int roll_dma_contiguous(struct htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    const uint32_t elem_size = kparams->elem_size;
    const uint32_t row_size  = ne0 * elem_size;

    if (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_push_rows(q, dst, src0, 0, 0, ne1 * ne2 * ne3, row_size, i0_src0, n0, elem_size);
        dma_queue_flush(q);
        return HTP_STATUS_OK;
    }

    if (s1 == 0) {
        const uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        for (uint32_t i3 = 0; i3 < ne3; i3++) {
            const uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            const uint32_t dst_row0 = i3 * ne2 * ne1;
            const uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            const uint32_t n2_first = ne2 - i2_src0;

            roll_dma_push_rows(q, dst, src0, dst_row0, src_row0, n2_first * ne1,
                               row_size, i0_src0, n0, elem_size);
            roll_dma_push_rows(q, dst, src0, dst_row0 + n2_first * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, row_size, i0_src0, n0, elem_size);
        }

        dma_queue_flush(q);
        return HTP_STATUS_OK;
    }

    const uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    const uint32_t n1_first = ne1 - i1_src0;

    for (uint32_t i3 = 0; i3 < ne3; i3++) {
        const uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        for (uint32_t i2 = 0; i2 < ne2; i2++) {
            const uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            const uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            const uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_push_rows(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_first, row_size, i0_src0, n0, elem_size);
            roll_dma_push_rows(q, dst, src0, dst_row0 + n1_first, src_row0,
                               i1_src0, row_size, i0_src0, n0, elem_size);
        }
    }

    dma_queue_flush(q);
    return HTP_STATUS_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
static int roll_dma_strided(struct htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    const uint32_t elem_size = kparams->elem_size;
    const uint32_t i1_src0   = htp_roll_wrap(-s1, ne1);
    const uint32_t n1_first  = ne1 - i1_src0;

    for (uint32_t i3 = 0; i3 < ne3; i3++) {
        const uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        for (uint32_t i2 = 0; i2 < ne2; i2++) {
            const uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            const dma_addr_t dst_row0 = dst->data  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            const dma_addr_t src_row0 = src0->data + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_push_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_first, i0_src0, n0, elem_size);
            roll_dma_push_range(q, dst_row0 + (size_t) n1_first * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0, elem_size);
        }
    }

    dma_queue_flush(q);
    return HTP_STATUS_OK;
}

static inline void roll_row_addrs(
    const struct htp_tensor *             src0,
    const struct htp_tensor *             dst,
    const struct htp_roll_kernel_params * kparams,
    uint32_t                              row,
    int32_t                               s1,
    int32_t                               s2,
    int32_t                               s3,
    dma_addr_t *                          src_addr,
    dma_addr_t *                          dst_addr
) {
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t i3  = fastdiv(row, &kparams->div_ne2_ne1);
    const uint32_t rem = row - i3 * ne2 * ne1;
    const uint32_t i2  = fastdiv(rem, &kparams->div_ne1);
    const uint32_t i1  = rem - i2 * ne1;

    const uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
    const uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
    const uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

    *src_addr = src0->data + (size_t) i01 * src0->nb[1] + (size_t) i02 * src0->nb[2] + (size_t) i03 * src0->nb[3];
    *dst_addr = dst->data  + (size_t) i1  * dst->nb[1]  + (size_t) i2  * dst->nb[2]  + (size_t) i3  * dst->nb[3];
}

static void __attribute__((noinline)) roll_copy_row_vtcm(
    uint8_t * restrict       dst_vtcm,
    const uint8_t * restrict src_vtcm,
    uint32_t                 ne0,
    uint32_t                 i0_src0,
    uint32_t                 n0,
    uint32_t                 elem_size
) {
    if (i0_src0 == 0) {
        hvx_copy_aa(dst_vtcm, src_vtcm, ne0 * elem_size, 1);
    } else {
        const uint32_t n0_bytes      = n0 * elem_size;
        const uint32_t i0_src0_bytes = i0_src0 * elem_size;
        hvx_copy_au(dst_vtcm, src_vtcm + i0_src0_bytes, n0_bytes, 1);
        hvx_copy_ua(dst_vtcm + n0_bytes, src_vtcm, i0_src0_bytes, 1);
    }
}

struct roll_pipe_state {
    const struct htp_tensor *             src0;
    const struct htp_tensor *             dst;
    const struct htp_roll_kernel_params * kparams;
    size_t                                row_size;
    size_t                                row_size_aligned;
    uint32_t                              ne0;
    uint32_t                              i0_src0;
    uint32_t                              n0;
    uint32_t                              elem_size;
    int32_t                               s1, s2, s3;
    uint32_t                              row;
    uint32_t                              row_end;
};

static bool roll_next_task(void * ctx, struct dma_pipe_task * task) {
    struct roll_pipe_state * s = (struct roll_pipe_state *) ctx;
    if (s->row >= s->row_end) {
        return false;
    }
    task->row = s->row++;
    task->off = 0;
    task->len = 0;
    task->aux = 0;
    return true;
}

static void roll_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct roll_pipe_state * s = (struct roll_pipe_state *) ctx;
    dma_addr_t src_addr, dst_addr;
    roll_row_addrs(s->src0, s->dst, s->kparams, task->row, s->s1, s->s2, s->s3, &src_addr, &dst_addr);
    dma_queue_push(dma_q, dma_make_data(in_spad, src_addr),
                   s->row_size_aligned, s->row_size, s->row_size, 1);
}

static void roll_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    (void) task;
    struct roll_pipe_state * s = (struct roll_pipe_state *) ctx;
    roll_copy_row_vtcm((uint8_t *) out_spad, (const uint8_t *) in_spad, s->ne0, s->i0_src0, s->n0, s->elem_size);
}

static void roll_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct roll_pipe_state * s = (struct roll_pipe_state *) ctx;
    dma_addr_t src_addr, dst_addr;
    roll_row_addrs(s->src0, s->dst, s->kparams, task->row, s->s1, s->s2, s->s3, &src_addr, &dst_addr);
    dma_queue_push(dma_q, dma_make_data(dst_addr, out_spad),
                   s->row_size, s->row_size_aligned, s->row_size, 1);
}

static void roll_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_roll_context * rctx = (struct htp_roll_context *) data;
    struct htp_ops_context *  octx = rctx->octx;

    htp_roll_preamble;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->nrows);
    if (row_start >= row_end) {
        return;
    }

    dma_queue *    dma_q     = th->dma_q;
    const uint32_t elem_size = kparams->elem_size;
    const size_t   row_size  = (size_t) ne0 * elem_size;

    if (!rctx->use_vtcm) {
        for (uint32_t row = row_start; row < row_end; row++) {
            dma_addr_t src_addr, dst_addr;
            roll_row_addrs(src0, dst, kparams, row, s1, s2, s3, &src_addr, &dst_addr);
            roll_dma_push_range(dma_q, dst_addr, src_addr, nb1, nb01, 1, i0_src0, n0, elem_size);
        }
        dma_queue_flush(dma_q);
        return;
    }

    const size_t row_size_aligned = rctx->row_size_aligned;
    uint8_t *    src_vtcm_base    = rctx->vtcm_src + ith * rctx->src_bytes_per_hvx;
    uint8_t *    dst_vtcm_base    = rctx->vtcm_dst + ith * rctx->dst_bytes_per_hvx;
    void *       in_buf[2]        = { src_vtcm_base, src_vtcm_base + row_size_aligned };
    void *       out_buf[2]       = { dst_vtcm_base, dst_vtcm_base + row_size_aligned };

    struct roll_pipe_state s = {
        .src0             = src0,
        .dst              = dst,
        .kparams          = kparams,
        .row_size         = row_size,
        .row_size_aligned = row_size_aligned,
        .ne0              = ne0,
        .i0_src0          = i0_src0,
        .n0               = n0,
        .elem_size        = elem_size,
        .s1 = s1, .s2 = s2, .s3 = s3,
        .row              = row_start,
        .row_end          = row_end,
    };

    dma_pipe_io_run(dma_q, th->trace, &s, in_buf, out_buf, roll_next_task, roll_push_in, roll_compute, roll_push_out);

    FARF(HIGH, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         row_start, row_end, s0, s1, s2, s3);
}

static int execute_op_roll(struct htp_ops_context * octx) {
    htp_roll_preamble;

    if (kparams->kernel_type == HTP_ROLL_KERNEL_DMA_CONTIGUOUS) {
        return roll_dma_contiguous(octx);
    }
    if (kparams->kernel_type == HTP_ROLL_KERNEL_DMA_STRIDED) {
        return roll_dma_strided(octx);
    }

    const uint32_t total_rows   = kparams->total_rows;
    const uint32_t elem_size    = kparams->elem_size;
    const size_t   dst_row_size = (size_t) ne0 * elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx    = octx->n_hvx;
    const uint32_t use_vtcm = (kparams->vtcm_size > 0) ? 1 : 0;

    struct htp_roll_vtcm_layout layout = { 0 };
    if (use_vtcm) {
        htp_roll_vtcm_layout_build(&layout, dst_row_size, n_hvx);
        if (octx->ctx->vtcm_size < layout.total_bytes) {
            FARF(ERROR, "roll: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;

    struct htp_roll_context rctx = {
        .octx              = octx,
        .kparams           = kparams,
        .vtcm_src          = use_vtcm ? VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_src) : NULL,
        .vtcm_dst          = use_vtcm ? VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_dst) : NULL,
        .row_size_aligned  = layout.row_size_aligned,
        .src_bytes_per_hvx = layout.src_bytes_per_hvx,
        .dst_bytes_per_hvx = layout.dst_bytes_per_hvx,
        .row_start         = row_start,
        .nrows             = nrows,
        .nrows_per_hvx     = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .use_vtcm          = use_vtcm,
    };

    work_queue_run(octx->ctx->hvx_queue, roll_thread, &rctx, n_hvx);

    return HTP_STATUS_OK;
}

int op_roll(struct htp_ops_context * octx) {
    if (octx->src[0]->type != octx->dst->type) {
        return HTP_STATUS_NO_SUPPORT;
    }
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return execute_op_roll(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
