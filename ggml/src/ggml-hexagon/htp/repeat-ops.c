#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
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
#include "repeat-ops.h"

typedef void (*repeat_fill_vtcm_fn_t)(
    uint8_t * restrict       dst_vtcm,
    const uint8_t * restrict src_vtcm,
    uint32_t                 n_elems,
    uint32_t                 reps,
    uint32_t                 dst_elem_size
);

struct htp_repeat_context {
    struct htp_ops_context *                octx;
    const struct htp_repeat_kernel_params * kparams;
    repeat_fill_vtcm_fn_t                   fill_vtcm;

    dma_addr_t data_src;
    dma_addr_t data_dst;
    uint8_t *  vtcm_src;
    uint8_t *  vtcm_dst;
    size_t     src_buf_bytes;
    size_t     dst_buf_bytes;
    size_t     src_bytes_per_hvx;
    size_t     dst_bytes_per_hvx;

    uint32_t nrows_per_hvx;
    uint32_t total_dst_rows;
    uint32_t row_start;
    uint32_t src_elem_size;
    uint32_t dst_elem_size;
    uint32_t tile_elems;
    uint32_t dst_reps;
};

static inline void repeat_row_addrs(
    const struct htp_repeat_context * rctx,
    const struct htp_tensor *         src,
    const struct htp_tensor *         dst,
    uint32_t                          dst_row,
    dma_addr_t *                      src_addr,
    dma_addr_t *                      dst_addr
) {
    const struct htp_repeat_kernel_params * kparams = rctx->kparams;
    const uint32_t dst_row_div_ne1 = fastdiv(dst_row, &kparams->div_ne1);
    const uint32_t i1              = dst_row - dst_row_div_ne1 * dst->ne[1];
    const uint32_t i3              = fastdiv(dst_row, &kparams->div_ne12);
    const uint32_t i2              = dst_row_div_ne1 - i3 * dst->ne[2];

    const uint32_t k1 = fastmodulo(i1, src->ne[1], &kparams->div_ne01);
    const uint32_t k2 = fastmodulo(i2, src->ne[2], &kparams->div_ne02);
    const uint32_t k3 = fastmodulo(i3, src->ne[3], &kparams->div_ne03);

    *src_addr = rctx->data_src + (size_t) k1 * src->nb[1] + (size_t) k2 * src->nb[2] + (size_t) k3 * src->nb[3];
    *dst_addr = rctx->data_dst + (size_t) i1 * dst->nb[1] + (size_t) i2 * dst->nb[2] + (size_t) i3 * dst->nb[3];
}

static inline HVX_Vector repeat_clamp_f16_finite(HVX_Vector v) {
    const HVX_Vector     max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector     min_f16 = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector     inf     = hvx_vec_splat_f32(INFINITY);
    const HVX_VectorPred finite  = Q6_Q_vcmp_gt_VsfVsf(inf, hvx_vec_abs_f32(v));
    const HVX_VectorPred high    = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(v, max_f16));
    const HVX_VectorPred low     = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(min_f16, v));
    v = Q6_V_vmux_QVV(high, max_f16, v);
    return Q6_V_vmux_QVV(low, min_f16, v);
}

static inline void repeat_row_f32_to_f16_a(uint8_t * restrict dst_ptr, const uint8_t * restrict src_ptr, uint32_t n) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src_ptr;
    HVX_Vector       * restrict vdst = (HVX_Vector *) dst_ptr;

    const uint32_t n_vec = n / 64;
    const uint32_t n_rem = n % 64;

    for (uint32_t i = 0; i < n_vec; i++) {
        const HVX_Vector v0 = repeat_clamp_f16_finite(vsrc[2 * i + 0]);
        const HVX_Vector v1 = repeat_clamp_f16_finite(vsrc[2 * i + 1]);
        vdst[i] = hvx_vec_f32_to_f16(v0, v1);
    }

    if (n_rem > 0) {
        const HVX_Vector v0 = repeat_clamp_f16_finite(vsrc[2 * n_vec + 0]);
        const HVX_Vector v1 = (n_rem > 32) ? repeat_clamp_f16_finite(vsrc[2 * n_vec + 1]) : Q6_V_vzero();
        const HVX_Vector v  = hvx_vec_f32_to_f16(v0, v1);
        hvx_vec_store_a(&vdst[n_vec], n_rem * sizeof(__fp16), v);
    }
}

static void __attribute__((noinline)) repeat_fill_vtcm_same(
    uint8_t * restrict       dst_vtcm,
    const uint8_t * restrict src_vtcm,
    uint32_t                 n_elems,
    uint32_t                 reps,
    uint32_t                 dst_elem_size
) {
    const uint32_t unit_bytes = n_elems * dst_elem_size;
    if (reps == 1) {
        hvx_copy_aa(dst_vtcm, src_vtcm, unit_bytes, 1);
        return;
    }

    if (n_elems == 1) {
        const int32_t    bits = Q6_R_vextract_VR(hvx_vmem(src_vtcm), 0);
        const HVX_Vector sv   = (dst_elem_size == 2) ? Q6_Vh_vsplat_R(bits) : Q6_V_vsplat_R(bits);
        hvx_splat_a(dst_vtcm, sv, reps * dst_elem_size, 1);
        return;
    }

    hvx_copy_aa(dst_vtcm, src_vtcm, unit_bytes, 1);
    for (uint32_t r = 1; r < reps; r++) {
        hvx_copy_ua(dst_vtcm + (size_t) r * unit_bytes, src_vtcm, unit_bytes, 1);
    }
}

static void __attribute__((noinline)) repeat_fill_vtcm_f32_to_f16(
    uint8_t * restrict       dst_vtcm,
    const uint8_t * restrict src_vtcm,
    uint32_t                 n_elems,
    uint32_t                 reps,
    uint32_t                 dst_elem_size
) {
    (void) dst_elem_size;
    repeat_row_f32_to_f16_a(dst_vtcm, src_vtcm, n_elems);
    if (reps == 1) {
        return;
    }

    if (n_elems == 1) {
        const int32_t bits = Q6_R_vextract_VR(hvx_vmem(dst_vtcm), 0);
        hvx_splat_a(dst_vtcm, Q6_Vh_vsplat_R(bits), reps, sizeof(__fp16));
        return;
    }

    const size_t unit_bytes = (size_t) n_elems * sizeof(__fp16);
    for (uint32_t r = 1; r < reps; r++) {
        hvx_copy_f16_ua(dst_vtcm + (size_t) r * unit_bytes, dst_vtcm, n_elems);
    }
}

struct repeat_pipe_state {
    const struct htp_repeat_context * rctx;
    const struct htp_tensor *         src;
    const struct htp_tensor *         dst;
    repeat_fill_vtcm_fn_t             fill_vtcm;
    size_t                            src_row_bytes;
    size_t                            dst_row_bytes;
    size_t                            src_buf_bytes;
    size_t                            dst_buf_bytes;
    uint32_t                          ne00;
    uint32_t                          nr0;
    uint32_t                          dst_elem_size;
    uint32_t                          row;
    uint32_t                          row_end;
};

static bool repeat_next_task(void * ctx, struct dma_pipe_task * task) {
    struct repeat_pipe_state * s = (struct repeat_pipe_state *) ctx;
    if (s->row >= s->row_end) {
        return false;
    }
    task->row = s->row++;
    task->off = 0;
    task->len = 0;
    task->aux = 0;
    return true;
}

static void repeat_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct repeat_pipe_state * s = (struct repeat_pipe_state *) ctx;
    dma_addr_t src_addr, dst_addr;
    repeat_row_addrs(s->rctx, s->src, s->dst, task->row, &src_addr, &dst_addr);
    dma_queue_push(dma_q, dma_make_data(in_spad, src_addr),
                   s->src_buf_bytes, s->src_row_bytes, s->src_row_bytes, 1);
}

static void repeat_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    (void) task;
    struct repeat_pipe_state * s = (struct repeat_pipe_state *) ctx;
    s->fill_vtcm((uint8_t *) out_spad, (const uint8_t *) in_spad, s->ne00, s->nr0, s->dst_elem_size);
}

static void repeat_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct repeat_pipe_state * s = (struct repeat_pipe_state *) ctx;
    dma_addr_t src_addr, dst_addr;
    repeat_row_addrs(s->rctx, s->src, s->dst, task->row, &src_addr, &dst_addr);
    dma_queue_push(dma_q, dma_make_data(dst_addr, out_spad),
                   s->dst_row_bytes, s->dst_buf_bytes, s->dst_row_bytes, 1);
}

static void repeat_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context *          octx = rctx->octx;
    const struct htp_tensor *         src  = octx->src[0];
    const struct htp_tensor *         dst  = octx->dst;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->total_dst_rows);
    if (row_start >= row_end) {
        return;
    }

    uint8_t * src_vtcm_base = rctx->vtcm_src + ith * rctx->src_bytes_per_hvx;
    uint8_t * dst_vtcm_base = rctx->vtcm_dst + ith * rctx->dst_bytes_per_hvx;
    void *    in_buf[2]     = { src_vtcm_base, src_vtcm_base + rctx->src_buf_bytes };
    void *    out_buf[2]    = { dst_vtcm_base, dst_vtcm_base + rctx->dst_buf_bytes };

    struct repeat_pipe_state s = {
        .rctx          = rctx,
        .src           = src,
        .dst           = dst,
        .fill_vtcm     = rctx->fill_vtcm,
        .src_row_bytes = (size_t) src->ne[0] * rctx->src_elem_size,
        .dst_row_bytes = (size_t) dst->ne[0] * rctx->dst_elem_size,
        .src_buf_bytes = rctx->src_buf_bytes,
        .dst_buf_bytes = rctx->dst_buf_bytes,
        .ne00          = src->ne[0],
        .nr0           = rctx->kparams->nr0,
        .dst_elem_size = rctx->dst_elem_size,
        .row           = row_start,
        .row_end       = row_end,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf, repeat_next_task, repeat_push_in, repeat_compute, repeat_push_out);

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

static void repeat_thread_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context *          octx = rctx->octx;
    const struct htp_tensor *         src  = octx->src[0];
    const struct htp_tensor *         dst  = octx->dst;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->total_dst_rows);
    if (row_start >= row_end) {
        return;
    }

    const uint32_t ne00          = src->ne[0];
    const uint32_t nr0           = rctx->kparams->nr0;
    const uint32_t src_elem_size = rctx->src_elem_size;
    const uint32_t dst_elem_size = rctx->dst_elem_size;
    const uint32_t tile_elems    = rctx->tile_elems;
    const uint32_t dst_reps      = rctx->dst_reps;

    uint8_t * src_vtcm = rctx->vtcm_src + ith * rctx->src_bytes_per_hvx;
    uint8_t * dst_vtcm = rctx->vtcm_dst + ith * rctx->dst_bytes_per_hvx;

    dma_queue *               dma_q     = th->dma_q;
    struct htp_thread_trace * tr        = th->trace;
    const repeat_fill_vtcm_fn_t fill_vtcm = rctx->fill_vtcm;

    if (tile_elems == ne00) {
        const size_t src_row_bytes  = (size_t) ne00 * src_elem_size;
        const size_t dst_unit_bytes = (size_t) ne00 * dst_elem_size;

        for (uint32_t ir = row_start; ir < row_end; ir++) {
            dma_addr_t src_addr, dst_addr;
            repeat_row_addrs(rctx, src, dst, ir, &src_addr, &dst_addr);

            dma_sync_read(dma_q, src_vtcm, src_addr, src_row_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
            fill_vtcm(dst_vtcm, src_vtcm, ne00, dst_reps, dst_elem_size);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

            for (uint32_t r = 0; r < nr0; r += dst_reps) {
                const uint32_t cnt = MIN(dst_reps, nr0 - r);
                dma_sync_write(dma_q, dst_addr + (size_t) r * dst_unit_bytes, dst_vtcm, (size_t) cnt * dst_unit_bytes);
            }
        }
    } else {
        for (uint32_t ir = row_start; ir < row_end; ir++) {
            dma_addr_t src_addr, dst_addr;
            repeat_row_addrs(rctx, src, dst, ir, &src_addr, &dst_addr);

            for (uint32_t col0 = 0; col0 < ne00; col0 += tile_elems) {
                const uint32_t cnt = MIN(tile_elems, ne00 - col0);
                dma_sync_read(dma_q, src_vtcm, src_addr + (size_t) col0 * src_elem_size, (size_t) cnt * src_elem_size);

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
                fill_vtcm(dst_vtcm, src_vtcm, cnt, 1, dst_elem_size);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

                for (uint32_t i0 = 0; i0 < nr0; i0++) {
                    const dma_addr_t d = dst_addr + ((size_t) i0 * ne00 + col0) * dst_elem_size;
                    dma_sync_write(dma_q, d, dst_vtcm, (size_t) cnt * dst_elem_size);
                }
            }
        }
    }

    FARF(HIGH, "repeat-tiled %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

int op_repeat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (dst->type != src0->type && !(src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16))) {
        FARF(ERROR, "repeat: unsupported types %u -> %u\n", src0->type, dst->type);
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t src_elem_size = (src0->type == HTP_TYPE_F16) ? 2 : 4;
    const uint32_t dst_elem_size = (dst->type  == HTP_TYPE_F16) ? 2 : 4;

    const struct htp_repeat_kernel_params * kparams =
        (const struct htp_repeat_kernel_params *) octx->kernel_params;

    const uint32_t total_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_dst_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx  = octx->n_hvx;
    const uint32_t n_bufs = kparams->pipelined ? 2 : 1;

    struct htp_repeat_vtcm_layout layout;
    htp_repeat_vtcm_layout_build(&layout,
        (size_t) kparams->tile_elems * src_elem_size,
        (size_t) kparams->tile_elems * kparams->dst_reps * dst_elem_size,
        n_bufs, n_hvx);

    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "repeat: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;

    struct htp_repeat_context rctx = {
        .octx              = octx,
        .kparams           = kparams,
        .fill_vtcm         = (src0->type == dst->type) ? repeat_fill_vtcm_same : repeat_fill_vtcm_f32_to_f16,
        .data_src          = (dma_addr_t) src0->data,
        .data_dst          = (dma_addr_t) dst->data,
        .vtcm_src          = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_src),
        .vtcm_dst          = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_dst),
        .src_buf_bytes     = layout.src_buf_bytes,
        .dst_buf_bytes     = layout.dst_buf_bytes,
        .src_bytes_per_hvx = layout.src_bytes_per_hvx,
        .dst_bytes_per_hvx = layout.dst_bytes_per_hvx,
        .nrows_per_hvx     = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_dst_rows    = nrows,
        .row_start         = row_start,
        .src_elem_size     = src_elem_size,
        .dst_elem_size     = dst_elem_size,
        .tile_elems        = kparams->tile_elems,
        .dst_reps          = kparams->dst_reps,
    };

    FARF(HIGH, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u) pipe=%u\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->nr0, kparams->nr1, kparams->nr2, kparams->nr3, kparams->pipelined);

    work_queue_func_t worker = kparams->pipelined ? repeat_thread : repeat_thread_tiled;
    work_queue_run(octx->ctx->hvx_queue, worker, &rctx, n_hvx);

    return HTP_STATUS_OK;
}
