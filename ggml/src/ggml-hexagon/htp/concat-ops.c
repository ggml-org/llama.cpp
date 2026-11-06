#include "concat-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "dma-copy.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
#include "htp-ctx.h"
#include "htp-fence.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-utils.h"
#include "hvx_hexagon_protos.h"
#include "gather-rows.h"

#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
    struct fastdiv_values div_ne2;
};

static inline dma_addr_t concat_plane_addr(const struct htp_tensor * t, uint32_t p, const struct fastdiv_values * div_ne2, uint32_t ne2) {
    const uint32_t i3 = fastdiv(p, div_ne2);
    const uint32_t i2 = p - i3 * ne2;
    return t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    // dst rows [row_begin, row_end) of this thread; a dst row is a src0 row followed by a src1 column
    const uint32_t row_begin = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t row_end   = MIN(row_begin + cctx->nrows_per_thread, cctx->row_start + cctx->nrows);
    if (row_begin >= row_end) return;

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];

    // tile: block_rows dst rows x block_cols src1 columns
    const uint32_t block_rows = 128;
    const uint32_t block_cols = 256;
    // +1 element spreads the gather lanes across VTCM, so they do not serialize on one path
    const uint32_t spad1_stride = block_rows * sizeof(float) + sizeof(float);

    const HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    const uint32_t block_cols_padded = MIN(hex_round_up(src1_ne0, 32), block_cols);
    const uint32_t src0_row_bytes    = src0_ne0 * sizeof(float);
    const uint32_t src0_row_padded   = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre          = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_stride      = src0_row_padded + block_cols_padded * sizeof(float);

    // two slots per scratchpad: tile b gathers in slot b&1 while dst[b-1] and src1[b+1] DMA in the other
    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;
    const uint32_t spad0_slot = block_rows * spad0_stride;
    const uint32_t spad1_slot = block_cols_padded * spad1_stride;

    // tiles walk (plane, row, col) with col fastest; src0 goes out with the col == 0 tile only
    uint32_t plane = 0;
    uint32_t row   = row_begin;
    uint32_t col   = 0;
    uint32_t slot  = 0;
    bool dst_pending = false;

    const dma_addr_t src1_addr = concat_plane_addr(src1, plane, div_ne2, ne2) + row * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(row_end - row, block_rows) * sizeof(float), MIN(src1_ne0, block_cols));

    const dma_addr_t src0_addr = concat_plane_addr(src0, plane, div_ne2, ne2) + row * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_stride, src0->nb[1], src0_row_bytes, MIN(row_end - row, block_rows));

    while (plane < cctx->nplanes) {
        const uint32_t nrows = MIN(row_end - row, block_rows);
        const uint32_t ncols = MIN(src1_ne0 - col, block_cols);

        uint8_t * spad0 = spad0_base + slot * spad0_slot;
        uint8_t * spad1 = spad1_base + slot * spad1_slot;

        dma_queue_pop(dma_q); // src1
        if (col == 0) {
            dma_queue_pop(dma_q); // src0
        }

        uint32_t next_plane = plane;
        uint32_t next_row   = row;
        uint32_t next_col   = col + block_cols;
        if (next_col >= src1_ne0) {
            next_col = 0;
            next_row = row + block_rows;
            if (next_row >= row_end) {
                next_row = row_begin;
                next_plane++;
            }
        }
        const bool has_next = next_plane < cctx->nplanes;
        const uint32_t next_nrows = MIN(row_end - next_row, block_rows);
        const uint32_t next_ncols = MIN(src1_ne0 - next_col, block_cols);

        // other spad1 slot is free since the previous gather sync
        if (has_next) {
            const dma_addr_t addr = concat_plane_addr(src1, next_plane, div_ne2, ne2) + next_row * src1->nb[1] + next_col * src1->nb[0];
            dma_queue_push(dma_q, dma_make_data(spad1_base + (slot ^ 1) * spad1_slot, addr), spad1_stride, src1->nb[0], next_nrows * sizeof(float), next_ncols);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        for (uint32_t c = 0; c < ncols; c += 32) {
            hvx_transpose_32x32_w_gather(spad0 + src0_row_padded + c * sizeof(float), spad0_stride, spad1 + c * spad1_stride, spad1_stride, offsets, nrows, MIN(ncols - c, 32));
        }
        hvx_gather_sync(spad0 + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        // other spad0 slot is free once dst[b-1] is written
        if (dst_pending) {
            dma_queue_pop(dma_q); // dst
        }
        if (has_next && next_col == 0) {
            const dma_addr_t addr = concat_plane_addr(src0, next_plane, div_ne2, ne2) + next_row * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + (slot ^ 1) * spad0_slot + src0_pre, addr), spad0_stride, src0->nb[1], src0_row_bytes, next_nrows);
        }

        // the col == 0 tile also carries the src0 columns in front
        const uint32_t lead = (col == 0) ? src0_row_bytes : 0;
        const dma_addr_t dst_addr = concat_plane_addr(dst, plane, div_ne2, ne2) + row * dst->nb[1] + src0_row_bytes + col * sizeof(float) - lead;
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0 + src0_row_padded - lead), dst->nb[1], spad0_stride, lead + ncols * sizeof(float), nrows);
        dst_pending = true;

        plane = next_plane;
        row   = next_row;
        col   = next_col;
        slot ^= 1;
    }
    dma_queue_flush(dma_q);
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    // dst rows [row_begin, row_end) of this thread; a dst row is a src0 row followed by a src1 column
    const uint32_t row_begin = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t row_end   = MIN(row_begin + cctx->nrows_per_thread, cctx->row_start + cctx->nrows);
    if (row_begin >= row_end) return;

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];

    // tile: block_rows dst rows x block_cols src1 columns
    const uint32_t block_rows = 128;
    const uint32_t block_cols = 512;
    // +1 element spreads the gather lanes across VTCM, so they do not serialize on one path
    const uint32_t spad1_stride = block_rows * sizeof(__fp16) + sizeof(__fp16);

    const HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    const uint32_t block_cols_padded = MIN(hex_round_up(src1_ne0, 64), block_cols);
    const uint32_t src0_row_bytes    = src0_ne0 * sizeof(__fp16);
    const uint32_t src0_row_padded   = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre          = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_stride      = src0_row_padded + block_cols_padded * sizeof(__fp16);

    // two slots per scratchpad: tile b gathers in slot b&1 while dst[b-1] and src1[b+1] DMA in the other
    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;
    const uint32_t spad0_slot = block_rows * spad0_stride;
    const uint32_t spad1_slot = block_cols_padded * spad1_stride;

    // tiles walk (plane, row, col) with col fastest; src0 goes out with the col == 0 tile only
    uint32_t plane = 0;
    uint32_t row   = row_begin;
    uint32_t col   = 0;
    uint32_t slot  = 0;
    bool dst_pending = false;

    const dma_addr_t src1_addr = concat_plane_addr(src1, plane, div_ne2, ne2) + row * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(row_end - row, block_rows) * sizeof(__fp16), MIN(src1_ne0, block_cols));

    const dma_addr_t src0_addr = concat_plane_addr(src0, plane, div_ne2, ne2) + row * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_stride, src0->nb[1], src0_row_bytes, MIN(row_end - row, block_rows));

    while (plane < cctx->nplanes) {
        const uint32_t nrows = MIN(row_end - row, block_rows);
        const uint32_t ncols = MIN(src1_ne0 - col, block_cols);

        uint8_t * spad0 = spad0_base + slot * spad0_slot;
        uint8_t * spad1 = spad1_base + slot * spad1_slot;

        dma_queue_pop(dma_q); // src1
        if (col == 0) {
            dma_queue_pop(dma_q); // src0
        }

        uint32_t next_plane = plane;
        uint32_t next_row   = row;
        uint32_t next_col   = col + block_cols;
        if (next_col >= src1_ne0) {
            next_col = 0;
            next_row = row + block_rows;
            if (next_row >= row_end) {
                next_row = row_begin;
                next_plane++;
            }
        }
        const bool has_next = next_plane < cctx->nplanes;
        const uint32_t next_nrows = MIN(row_end - next_row, block_rows);
        const uint32_t next_ncols = MIN(src1_ne0 - next_col, block_cols);

        // other spad1 slot is free since the previous gather sync
        if (has_next) {
            const dma_addr_t addr = concat_plane_addr(src1, next_plane, div_ne2, ne2) + next_row * src1->nb[1] + next_col * src1->nb[0];
            dma_queue_push(dma_q, dma_make_data(spad1_base + (slot ^ 1) * spad1_slot, addr), spad1_stride, src1->nb[0], next_nrows * sizeof(__fp16), next_ncols);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        for (uint32_t c = 0; c < ncols; c += 64) {
            hvx_transpose_64x64_h_gather(spad0 + src0_row_padded + c * sizeof(__fp16), spad0_stride, spad1 + c * spad1_stride, spad1_stride, offsets, nrows, MIN(ncols - c, 64));
        }
        hvx_gather_sync(spad0 + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        // other spad0 slot is free once dst[b-1] is written
        if (dst_pending) {
            dma_queue_pop(dma_q); // dst
        }
        if (has_next && next_col == 0) {
            const dma_addr_t addr = concat_plane_addr(src0, next_plane, div_ne2, ne2) + next_row * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + (slot ^ 1) * spad0_slot + src0_pre, addr), spad0_stride, src0->nb[1], src0_row_bytes, next_nrows);
        }

        // the col == 0 tile also carries the src0 columns in front
        const uint32_t lead = (col == 0) ? src0_row_bytes : 0;
        const dma_addr_t dst_addr = concat_plane_addr(dst, plane, div_ne2, ne2) + row * dst->nb[1] + src0_row_bytes + col * sizeof(__fp16) - lead;
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0 + src0_row_padded - lead), dst->nb[1], spad0_stride, lead + ncols * sizeof(__fp16), nrows);
        dst_pending = true;

        plane = next_plane;
        row   = next_row;
        col   = next_col;
        slot ^= 1;
    }
    dma_queue_flush(dma_q);
}

static int concat_regular(struct htp_ops_context * octx, int dim, uint32_t type_size) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    struct htp_tensor view0 = *dst;
    struct htp_tensor view1 = *dst;
    for (int d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    const uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        rstart0 = range0.start;
        nrows0  = range0.count;

        const struct htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        rstart1 = range1.start;
        nrows1  = range1.count;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    return HTP_STATUS_OK;
}

static int concat_transposed(struct htp_ops_context * octx, const struct htp_concat_kernel_params * kparams, uint32_t type_size) {
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * dst = octx->dst;

    const uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_concat_transposed_vtcm_layout layout;
    htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_base            = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad0);
    cctx.spad1_base            = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad1);
    cctx.spad0_size_per_thread = layout.src0_spad_size_per_thread;
    cctx.spad1_size_per_thread = layout.src1_spad_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = kparams->u.transposed.div_ne2;
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    return HTP_STATUS_OK;
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_concat_kernel_params * kparams = (const struct htp_concat_kernel_params *) octx->kernel_params;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int status = HTP_STATUS_OK;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    switch (kparams->kernel_type) {
        case HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            break;

        case HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            break;

        case HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            break;

        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    return octx->status;
}
