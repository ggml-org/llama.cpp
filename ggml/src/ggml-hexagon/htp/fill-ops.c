#include <HAP_farf.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-common.h"
#include "htp-trace.h"

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "fill-ops.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble                                           \
    const struct htp_tensor * dst = octx->dst;                  \
                                                                \
    const uint32_t ne0 = dst->ne[0];                            \
    const uint32_t __attribute__((unused)) ne1 = dst->ne[1];    \
    const uint32_t __attribute__((unused)) ne2 = dst->ne[2];    \
                                                                \
    const uint32_t __attribute__((unused)) nb1 = dst->nb[1];    \
    const uint32_t __attribute__((unused)) nb2 = dst->nb[2];    \
    const uint32_t __attribute__((unused)) nb3 = dst->nb[3];

struct htp_fill_context {
    struct htp_ops_context *              octx;
    const struct htp_fill_kernel_params * kparams;
    dma_addr_t                            data_dst;
    uint8_t *                             vtcm_tile;
    size_t                                tile_bytes;
    uint32_t                              nrows_per_hvx;
    uint32_t                              total_rows;
    uint32_t                              row_start;
    uint32_t                              splat_bits;
    uint32_t                              elem_size;
};

static void __attribute__((noinline)) fill_splat_vtcm(
    uint8_t * vtcm_tile,
    uint32_t  splat_bits,
    uint32_t  n_bytes,
    uint32_t  elem_size
) {
    const HVX_Vector sv = (elem_size == 2) ? Q6_Vh_vsplat_R((int32_t) splat_bits) : Q6_V_vsplat_R((int32_t) splat_bits);
    hvx_splat_a(vtcm_tile, sv, n_bytes, 1);
}

static inline void fill_dma_write_capped(
    dma_queue *     dma_q,
    uint32_t *      in_flight,
    uint32_t        max_in_flight,
    dma_addr_t      dst_addr,
    const uint8_t * src_vtcm,
    size_t          nbytes
) {
    if (*in_flight >= max_in_flight) {
        dma_queue_pop(dma_q);
        (*in_flight)--;
    }
    dma_queue_push(dma_q, dma_make_data(dst_addr, src_vtcm), nbytes, nbytes, nbytes, 1);
    (*in_flight)++;
}

static void fill_thread_contiguous(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
    struct htp_ops_context *        octx = fctx->octx;
    fill_preamble;

    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_hvx * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_hvx, fctx->row_start + fctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    const size_t   tile_bytes  = fctx->tile_bytes;
    const uint32_t elem_size   = fctx->elem_size;
    const size_t   total_bytes = (size_t) (ir1 - ir0) * ne0 * elem_size;
    const size_t   splat_bytes = MIN(tile_bytes, hex_round_up(total_bytes, 128));
    uint8_t *      vtcm_tile   = fctx->vtcm_tile + ith * tile_bytes;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    fill_splat_vtcm(vtcm_tile, fctx->splat_bits, (uint32_t) splat_bytes, elem_size);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);

    dma_queue *      dma_q         = th->dma_q;
    const uint32_t   max_in_flight = MIN(8u, dma_queue_capacity(dma_q) - 1);
    uint32_t         in_flight     = 0;
    const dma_addr_t dst_base      = fctx->data_dst + (size_t) ir0 * ne0 * elem_size;

    for (size_t off = 0; off < total_bytes; off += tile_bytes) {
        const size_t chunk = MIN(tile_bytes, total_bytes - off);
        fill_dma_write_capped(dma_q, &in_flight, max_in_flight, dst_base + off, vtcm_tile, chunk);
    }

    while (in_flight--) {
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "fill-contig %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

static void fill_thread_strided(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context *       fctx    = (const struct htp_fill_context *) data;
    struct htp_ops_context *              octx    = fctx->octx;
    const struct htp_fill_kernel_params * kparams = fctx->kparams;
    fill_preamble;

    const uint32_t ir0 = fctx->row_start + fctx->nrows_per_hvx * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_hvx, fctx->row_start + fctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    const size_t   tile_bytes  = fctx->tile_bytes;
    const uint32_t elem_size   = fctx->elem_size;
    const size_t   row_bytes   = (size_t) ne0 * elem_size;
    const size_t   splat_bytes = MIN(tile_bytes, hex_round_up(row_bytes, 128));
    uint8_t *      vtcm_tile   = fctx->vtcm_tile + ith * tile_bytes;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    fill_splat_vtcm(vtcm_tile, fctx->splat_bits, (uint32_t) splat_bytes, elem_size);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);

    dma_queue *    dma_q         = th->dma_q;
    const uint32_t max_in_flight = MIN(8u, dma_queue_capacity(dma_q) - 1);
    uint32_t       in_flight     = 0;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t   i3       = fastdiv(ir, &kparams->div_ne1_ne2);
        const uint32_t   rem      = ir - i3 * (ne1 * ne2);
        const uint32_t   i2       = fastdiv(rem, &kparams->div_ne1);
        const uint32_t   i1       = rem - i2 * ne1;
        const dma_addr_t dst_row  = fctx->data_dst + (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3;

        for (size_t off = 0; off < row_bytes; off += tile_bytes) {
            const size_t chunk = MIN(tile_bytes, row_bytes - off);
            fill_dma_write_capped(dma_q, &in_flight, max_in_flight, dst_row + off, vtcm_tile, chunk);
        }
    }

    while (in_flight--) {
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "fill-strided %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

int op_fill(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;
    const struct htp_fill_kernel_params * kparams =
        (const struct htp_fill_kernel_params *) octx->kernel_params;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
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

    struct htp_fill_vtcm_layout layout;
    htp_fill_vtcm_layout_build(&layout, kparams->tile_bytes, n_hvx);
    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "fill: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u ktype=%u\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, kparams->kernel_type);

    float val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], sizeof(float));

    struct htp_fill_context fctx = {
        .octx          = octx,
        .kparams       = kparams,
        .data_dst      = (dma_addr_t) dst->data,
        .vtcm_tile     = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, layout.off_tile),
        .tile_bytes    = layout.tile_bytes,
        .nrows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_rows    = nrows,
        .row_start     = row_start,
        .elem_size     = kparams->elem_size,
    };

    switch (dst->type) {
    case HTP_TYPE_F32:
        memcpy(&fctx.splat_bits, &val_f32, sizeof(uint32_t));
        break;
    case HTP_TYPE_F16: {
        const _Float16 val_f16 = (_Float16) val_f32;
        uint16_t       bits16  = 0;
        memcpy(&bits16, &val_f16, sizeof(uint16_t));
        fctx.splat_bits = bits16;
        break;
    }
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_func_t worker = (kparams->kernel_type == HTP_FILL_KERNEL_CONTIGUOUS)
        ? fill_thread_contiguous
        : fill_thread_strided;

    work_queue_run(octx->ctx->hvx_queue, worker, &fctx, n_hvx);

    return HTP_STATUS_OK;
}
