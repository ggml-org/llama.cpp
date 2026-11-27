#include "concat-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "dma-copy.h"
#include "hex-fastdiv.h"
#include "htp-trace.h"
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

static inline uint32_t concat_type_size(uint32_t type) {
    return (type == HTP_TYPE_F16) ? 2 : 4;
}

static inline HVX_Vector hvx_vec_clamp_f16_f32(HVX_Vector x) {
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, f16_min), f16_max);
}

static __attribute__((noinline)) void concat_cvt_f16_to_f32_aa(
    uint8_t * restrict dst,
    const uint8_t * restrict src,
    uint32_t n) {

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    uint32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const HVX_VectorPair vp = hvx_vec_f16_to_f32(*vsrc++);
        *vdst++ = Q6_V_lo_W(vp);
        *vdst++ = Q6_V_hi_W(vp);
    }
    if (i < n) {
        const HVX_VectorPair vp = hvx_vec_f16_to_f32(*vsrc);
        const uint32_t rem = n - i;
        if (rem >= 32) {
            *vdst++ = Q6_V_lo_W(vp);
            if (rem > 32) {
                hvx_vec_store_a(vdst, (rem - 32) * sizeof(float), Q6_V_hi_W(vp));
            }
        } else {
            hvx_vec_store_a(vdst, rem * sizeof(float), Q6_V_lo_W(vp));
        }
    }
}

static __attribute__((noinline)) void concat_cvt_f32_to_f16_aa(
    uint8_t * restrict dst,
    const uint8_t * restrict src,
    uint32_t n) {

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;

    uint32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const HVX_Vector v0 = hvx_vec_clamp_f16_f32(vsrc[0]);
        const HVX_Vector v1 = hvx_vec_clamp_f16_f32(vsrc[1]);
        vsrc += 2;
        *vdst++ = hvx_vec_f32_to_f16(v0, v1);
    }
    if (i < n) {
        const uint32_t rem = n - i;
        const HVX_Vector v0 = hvx_vec_clamp_f16_f32(vsrc[0]);
        const HVX_Vector v1 = (rem > 32) ? hvx_vec_clamp_f16_f32(vsrc[1]) : Q6_V_vzero();
        const HVX_Vector out = hvx_vec_f32_to_f16(v0, v1);
        hvx_vec_store_a(vdst, rem * sizeof(__fp16), out);
    }
}

static __attribute__((noinline)) void concat_cvt_src0_rows_f16_to_f32(
    uint8_t * restrict spad0,
    uint8_t * restrict spad2,
    uint32_t spad0_stride,
    uint32_t dst_lead_pre,
    uint32_t lead_ne0,
    uint32_t nrows) {

    if (lead_ne0 == 0) {
        return;
    }
    for (uint32_t r = 0; r < nrows; r++) {
        uint8_t * row_ptr = spad0 + r * spad0_stride;
        concat_cvt_f16_to_f32_aa(spad2, row_ptr, lead_ne0);
        hvx_copy_f32_ua(row_ptr + dst_lead_pre, spad2, lead_ne0);
    }
}

static __attribute__((noinline)) void concat_cvt_src0_rows_f32_to_f16(
    uint8_t * restrict spad0,
    uint8_t * restrict spad2,
    uint32_t spad0_stride,
    uint32_t dst_lead_pre,
    uint32_t lead_ne0,
    uint32_t nrows) {

    if (lead_ne0 == 0) {
        return;
    }
    for (uint32_t r = 0; r < nrows; r++) {
        uint8_t * row_ptr = spad0 + r * spad0_stride;
        concat_cvt_f32_to_f16_aa(spad2, row_ptr, lead_ne0);
        hvx_copy_f16_ua(row_ptr + dst_lead_pre, spad2, lead_ne0);
    }
}

static __attribute__((noinline)) void concat_transpose_f32(
    uint8_t * restrict dst_base,
    uint32_t dst_stride,
    const uint8_t * restrict src_base,
    uint32_t src_stride,
    uint32_t nrows,
    uint32_t ncols) {

    const HVX_Vector offsets = hvx_vec_gather_offsets_w(src_stride);
    for (uint32_t c = 0; c < ncols; c += 32) {
        hvx_transpose_32x32_w_gather(dst_base + c * sizeof(float), dst_stride,
                                     src_base + c * src_stride, src_stride,
                                     offsets, nrows, MIN(ncols - c, 32));
    }
    hvx_gather_sync(dst_base);
}

static __attribute__((noinline)) void concat_transpose_f16(
    uint8_t * restrict dst_base,
    uint32_t dst_stride,
    const uint8_t * restrict src_base,
    uint32_t src_stride,
    uint32_t nrows,
    uint32_t ncols) {

    const HVX_Vector offsets = hvx_vec_gather_offsets_h(src_stride);
    for (uint32_t c = 0; c < ncols; c += 64) {
        hvx_transpose_64x64_h_gather(dst_base + c * sizeof(__fp16), dst_stride,
                                     src_base + c * src_stride, src_stride,
                                     offsets, nrows, MIN(ncols - c, 64));
    }
    hvx_gather_sync(dst_base);
}

static __attribute__((noinline)) void concat_transpose_cvt_f16_to_f32(
    uint8_t * restrict dst_base,
    uint32_t dst_stride,
    const uint8_t * restrict src_base,
    uint32_t src_stride,
    uint8_t * restrict spad2,
    uint32_t nrows,
    uint32_t ncols) {

    const HVX_Vector offsets = hvx_vec_gather_offsets_h(src_stride);
    for (uint32_t c = 0; c < ncols; c += 64) {
        hvx_transpose_64x64_h_gather(spad2, 128,
                                     src_base + c * src_stride, src_stride,
                                     offsets, nrows, MIN(ncols - c, 64));
        for (uint32_t r = 0; r < nrows; r++) {
            const HVX_Vector v = *(const HVX_Vector *) (spad2 + r * 128);
            const HVX_VectorPair vp = hvx_vec_f16_to_f32(v);
            HVX_Vector * restrict vdst = (HVX_Vector *) (dst_base + r * dst_stride + c * sizeof(float));
            vdst[0] = Q6_V_lo_W(vp);
            vdst[1] = Q6_V_hi_W(vp);
        }
    }
}

static __attribute__((noinline)) void concat_transpose_cvt_f32_to_f16(
    uint8_t * restrict dst_base,
    uint32_t dst_stride,
    const uint8_t * restrict src_base,
    uint32_t src_stride,
    uint8_t * restrict spad2,
    uint32_t nrows,
    uint32_t ncols) {

    const HVX_Vector offsets = hvx_vec_gather_offsets_w(src_stride);
    for (uint32_t c = 0; c < ncols; c += 64) {
        const uint32_t chunk_cols = MIN(ncols - c, 64);
        hvx_transpose_32x32_w_gather(spad2, 256,
                                     src_base + c * src_stride, src_stride,
                                     offsets, nrows, MIN(chunk_cols, 32));
        if (chunk_cols > 32) {
            hvx_transpose_32x32_w_gather(spad2 + 128, 256,
                                         src_base + (c + 32) * src_stride, src_stride,
                                         offsets, nrows, chunk_cols - 32);
        }
        if (chunk_cols > 32) {
            for (uint32_t r = 0; r < nrows; r++) {
                const HVX_Vector * restrict vsrc = (const HVX_Vector *) (spad2 + r * 256);
                const HVX_Vector v0 = hvx_vec_clamp_f16_f32(vsrc[0]);
                const HVX_Vector v1 = hvx_vec_clamp_f16_f32(vsrc[1]);
                *(HVX_Vector *) (dst_base + r * dst_stride + c * sizeof(__fp16)) = hvx_vec_f32_to_f16(v0, v1);
            }
        } else {
            const HVX_Vector zero = Q6_V_vzero();
            for (uint32_t r = 0; r < nrows; r++) {
                const HVX_Vector * restrict vsrc = (const HVX_Vector *) (spad2 + r * 256);
                const HVX_Vector v0 = hvx_vec_clamp_f16_f32(vsrc[0]);
                *(HVX_Vector *) (dst_base + r * dst_stride + c * sizeof(__fp16)) = hvx_vec_f32_to_f16(v0, zero);
            }
        }
    }
}

struct htp_concat_context {
    struct htp_ops_context *  octx;
    const struct htp_tensor * lead_src;
    const struct htp_tensor * trans_src;
    const struct htp_tensor * dst_view;
    uint8_t *                 spad0_base;
    uint8_t *                 spad1_base;
    uint8_t *                 spad2_base;
    uint32_t                  spad0_size_per_hvx;
    uint32_t                  spad1_size_per_hvx;
    uint32_t                  spad2_size_per_hvx;
    uint32_t                  row_start;
    uint32_t                  nrows;
    uint32_t                  nrows_per_hvx;
    uint32_t                  nplanes;
    struct fastdiv_values     div_ne2;
};

static inline dma_addr_t concat_plane_addr(const struct htp_tensor * t, uint32_t p, const struct fastdiv_values * div_ne2, uint32_t ne2) {
    const uint32_t i3 = fastdiv(p, div_ne2);
    const uint32_t i2 = p - i3 * ne2;
    return t->data + (i2 * t->nb[2] + i3 * t->nb[3]);
}

static void concat_2d_transposed_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;

    const struct htp_tensor * src0 = cctx->lead_src;
    const struct htp_tensor * src1 = cctx->trans_src;
    const struct htp_tensor * dst  = cctx->dst_view;

    const uint32_t src0_ne0 = src0 ? src0->ne[0] : 0;
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t dst_es  = concat_type_size(dst->type);
    const uint32_t src0_es = src0 ? concat_type_size(src0->type) : dst_es;
    const uint32_t src1_es = concat_type_size(src1->type);

    const uint32_t row_begin = cctx->row_start + ith * cctx->nrows_per_hvx;
    const uint32_t row_end   = MIN(row_begin + cctx->nrows_per_hvx, cctx->row_start + cctx->nrows);
    if (row_begin >= row_end) {
        return;
    }

    dma_queue * dma_q = th->dma_q;
    struct htp_thread_trace * tr = th->trace;

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];

    const uint32_t block_rows = 128;
    const uint32_t block_cols = (src1_es == 2 && dst_es == 2) ? 512 : 256;
    const uint32_t col_align  = (src1_es == 2 || dst_es == 2) ? 64 : 32;
    const uint32_t spad1_stride = block_rows * src1_es + src1_es;

    const uint32_t block_cols_padded   = MIN(hex_round_up(src1_ne0, col_align), block_cols);
    const uint32_t dst_src0_row_bytes  = src0_ne0 * dst_es;
    const uint32_t dst_src0_row_padded = hex_round_up(dst_src0_row_bytes, VLEN);
    const uint32_t dst_src0_pre        = dst_src0_row_padded - dst_src0_row_bytes;
    const uint32_t raw_src0_row_bytes  = src0_ne0 * src0_es;
    const uint32_t raw_src0_row_padded = (src0_es != dst_es) ? hex_round_up(raw_src0_row_bytes, VLEN) : 0;
    const uint32_t spad0_stride        = MAX(dst_src0_row_padded + block_cols_padded * dst_es, raw_src0_row_padded);
    const uint32_t src0_dma_off        = (src0_es == dst_es) ? dst_src0_pre : 0;

    uint8_t * spad0_base = cctx->spad0_base + (size_t) ith * cctx->spad0_size_per_hvx;
    uint8_t * spad1_base = cctx->spad1_base + (size_t) ith * cctx->spad1_size_per_hvx;
    uint8_t * spad2      = cctx->spad2_base ? (cctx->spad2_base + (size_t) ith * cctx->spad2_size_per_hvx) : NULL;
    const uint32_t spad0_slot = block_rows * spad0_stride;
    const uint32_t spad1_slot = block_cols_padded * spad1_stride;

    uint32_t plane = 0;
    uint32_t row   = row_begin;
    uint32_t col   = 0;
    uint32_t slot  = 0;

    const dma_addr_t src1_addr = concat_plane_addr(src1, plane, div_ne2, ne2) + row * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(row_end - row, block_rows) * src1_es, MIN(src1_ne0, block_cols));

    if (src0) {
        const dma_addr_t src0_addr = concat_plane_addr(src0, plane, div_ne2, ne2) + row * src0->nb[1];
        dma_queue_push(dma_q, dma_make_data(spad0_base + src0_dma_off, src0_addr), spad0_stride, src0->nb[1], raw_src0_row_bytes, MIN(row_end - row, block_rows));
    }
    dma_queue_push(dma_q, dma_make_data(dst->data, spad0_base), 0, 0, 0, 0);

    while (plane < cctx->nplanes) {
        const uint32_t nrows = MIN(row_end - row, block_rows);
        const uint32_t ncols = MIN(src1_ne0 - col, block_cols);

        uint8_t * spad0 = spad0_base + slot * spad0_slot;
        uint8_t * spad1 = spad1_base + slot * spad1_slot;

        dma_queue_pop(dma_q); // src1
        if (src0 && col == 0) {
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

        if (has_next) {
            const dma_addr_t addr = concat_plane_addr(src1, next_plane, div_ne2, ne2) + (next_row * src1->nb[1] + next_col * src1->nb[0]);
            dma_queue_push(dma_q, dma_make_data(spad1_base + (slot ^ 1) * spad1_slot, addr), spad1_stride, src1->nb[0], next_nrows * src1_es, next_ncols);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        if (src0 && col == 0 && src0_es != dst_es) {
            if (dst_es == 4) {
                concat_cvt_src0_rows_f16_to_f32(spad0, spad2, spad0_stride, dst_src0_pre, src0_ne0, nrows);
            } else {
                concat_cvt_src0_rows_f32_to_f16(spad0, spad2, spad0_stride, dst_src0_pre, src0_ne0, nrows);
            }
        }

        uint8_t * dst_trans = spad0 + dst_src0_row_padded;
        if (src1_es == dst_es) {
            if (dst_es == 4) {
                concat_transpose_f32(dst_trans, spad0_stride, spad1, spad1_stride, nrows, ncols);
            } else {
                concat_transpose_f16(dst_trans, spad0_stride, spad1, spad1_stride, nrows, ncols);
            }
        } else if (dst_es == 4) {
            concat_transpose_cvt_f16_to_f32(dst_trans, spad0_stride, spad1, spad1_stride, spad2, nrows, ncols);
        } else {
            concat_transpose_cvt_f32_to_f16(dst_trans, spad0_stride, spad1, spad1_stride, spad2, nrows, ncols);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        dma_queue_pop(dma_q); // dst
        if (src0 && has_next && next_col == 0) {
            const dma_addr_t addr = concat_plane_addr(src0, next_plane, div_ne2, ne2) + next_row * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + (slot ^ 1) * spad0_slot + src0_dma_off, addr), spad0_stride, src0->nb[1], raw_src0_row_bytes, next_nrows);
        }

        const uint32_t lead = (src0 && col == 0) ? dst_src0_row_bytes : 0;
        const dma_addr_t dst_addr = concat_plane_addr(dst, plane, div_ne2, ne2) + (row * dst->nb[1] + dst_src0_row_bytes + col * dst_es - lead);
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0 + dst_src0_row_padded - lead), dst->nb[1], spad0_stride, lead + ncols * dst_es, nrows);

        plane = next_plane;
        row   = next_row;
        col   = next_col;
        slot ^= 1;
    }
    dma_queue_flush(dma_q);
}

struct concat_regular_cvt_task {
    struct htp_ops_context *  octx;
    const struct htp_tensor * src;
    const struct htp_tensor * dst;
    uint32_t                  r_start;
    uint32_t                  nrows;
    uint32_t                  rows_per_hvx;
    uint32_t                  src_es;
    uint32_t                  dst_es;
    struct fastdiv_values     div_ne1;
    struct fastdiv_values     div_ne2_ne1;
};

static inline void concat_regular_row_addrs(
    const struct concat_regular_cvt_task * task,
    uint32_t r,
    uint32_t * i1_out,
    dma_addr_t * src_addr_out,
    dma_addr_t * dst_addr_out) {

    const struct htp_tensor * src = task->src;
    const struct htp_tensor * dst = task->dst;
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2_ne1 = src->ne[2] * ne1;

    const uint32_t i3  = fastdiv(r, &task->div_ne2_ne1);
    const uint32_t rem = r - i3 * ne2_ne1;
    const uint32_t i2  = fastdiv(rem, &task->div_ne1);
    const uint32_t i1  = rem - i2 * ne1;

    *i1_out       = i1;
    *src_addr_out = src->data + (i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]);
    *dst_addr_out = dst->data + (i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
}

static void concat_regular_cvt_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct concat_regular_cvt_task * task = (const struct concat_regular_cvt_task *) data;
    const struct htp_tensor * src = task->src;
    const struct htp_tensor * dst = task->dst;

    const uint32_t r_offset = MIN(ith * task->rows_per_hvx, task->nrows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_hvx, task->nrows);
    if (r_beg >= r_end) {
        return;
    }

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    if (ne0 == 0) {
        return;
    }

    const uint32_t src_es = task->src_es;
    const uint32_t dst_es = task->dst_es;
    const uint32_t max_es = MAX(src_es, dst_es);

    uint8_t * base = (uint8_t *) task->octx->ctx->vtcm_base + (size_t) ith * (4 * HTP_CONCAT_REG_BLK_BYTES);
    uint8_t * sbuf[2] = {
        base + 0 * HTP_CONCAT_REG_BLK_BYTES,
        base + 1 * HTP_CONCAT_REG_BLK_BYTES,
    };
    uint8_t * dbuf[2] = {
        base + 2 * HTP_CONCAT_REG_BLK_BYTES,
        base + 3 * HTP_CONCAT_REG_BLK_BYTES,
    };

    dma_queue * q = th->dma_q;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t row_max_bytes = ne0 * max_es;
    if (row_max_bytes <= HTP_CONCAT_REG_BLK_BYTES) {
        uint32_t max_rows_per_blk = HTP_CONCAT_REG_BLK_BYTES / row_max_bytes;
        if (src->nb[1] > 65535 || dst->nb[1] > 65535) {
            max_rows_per_blk = 1;
        }
        if (max_rows_per_blk > DMA_MAX_NROWS) {
            max_rows_per_blk = DMA_MAX_NROWS;
        }

        const uint32_t src_row_bytes = ne0 * src_es;
        const uint32_t dst_row_bytes = ne0 * dst_es;

        uint32_t cur_r = r_beg;
        uint32_t cur_i1 = 0;
        dma_addr_t cur_src_addr = 0;
        dma_addr_t cur_dst_addr = 0;
        concat_regular_row_addrs(task, cur_r, &cur_i1, &cur_src_addr, &cur_dst_addr);
        uint32_t cur_nr = MIN(MIN(r_end - cur_r, max_rows_per_blk), ne1 - cur_i1);

        const size_t s_stride0 = (cur_nr > 1) ? src->nb[1] : src_row_bytes;
        dma_queue_push(q, dma_make_data(sbuf[0], cur_src_addr), src_row_bytes, s_stride0, src_row_bytes, cur_nr);
        dma_queue_push(q, dma_make_data(cur_dst_addr, dbuf[1]), 0, 0, 0, 0);

        uint32_t slot = 0;

        while (cur_r < r_end) {
            const uint32_t next_r = cur_r + cur_nr;
            uint32_t next_i1 = 0;
            dma_addr_t next_src_addr = 0;
            dma_addr_t next_dst_addr = 0;
            uint32_t next_nr = 0;

            dma_queue_pop(q); // sbuf[slot]

            if (next_r < r_end) {
                concat_regular_row_addrs(task, next_r, &next_i1, &next_src_addr, &next_dst_addr);
                next_nr = MIN(MIN(r_end - next_r, max_rows_per_blk), ne1 - next_i1);
                const size_t s_stride_next = (next_nr > 1) ? src->nb[1] : src_row_bytes;
                dma_queue_push(q, dma_make_data(sbuf[slot ^ 1], next_src_addr), src_row_bytes, s_stride_next, src_row_bytes, next_nr);
            }

            dma_queue_pop(q); // dbuf[slot ^ 1]

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur_r);
            if (dst_es == 4) {
                concat_cvt_f16_to_f32_aa(dbuf[slot], sbuf[slot], cur_nr * ne0);
            } else {
                concat_cvt_f32_to_f16_aa(dbuf[slot], sbuf[slot], cur_nr * ne0);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur_r);

            const size_t d_stride = (cur_nr > 1) ? dst->nb[1] : dst_row_bytes;
            dma_queue_push(q, dma_make_data(cur_dst_addr, dbuf[slot]), d_stride, dst_row_bytes, dst_row_bytes, cur_nr);

            cur_r        = next_r;
            cur_i1       = next_i1;
            cur_src_addr = next_src_addr;
            cur_dst_addr = next_dst_addr;
            cur_nr       = next_nr;
            slot        ^= 1;
        }
        dma_queue_flush(q);
    } else {
        const uint32_t max_cols = HTP_CONCAT_REG_BLK_BYTES / max_es;
        uint32_t r = r_beg;
        uint32_t col = 0;
        uint32_t i1 = 0;
        dma_addr_t row_src_addr = 0;
        dma_addr_t row_dst_addr = 0;
        concat_regular_row_addrs(task, r, &i1, &row_src_addr, &row_dst_addr);

        uint32_t cur_cols = MIN(ne0 - col, max_cols);
        const uint32_t s_bytes0 = cur_cols * src_es;
        dma_queue_push(q, dma_make_data(sbuf[0], row_src_addr), s_bytes0, s_bytes0, s_bytes0, 1);
        dma_queue_push(q, dma_make_data(row_dst_addr, dbuf[1]), 0, 0, 0, 0);

        uint32_t slot = 0;

        while (r < r_end) {
            uint32_t next_r = r;
            uint32_t next_col = col + cur_cols;
            uint32_t next_i1 = i1;
            dma_addr_t next_row_src_addr = row_src_addr;
            dma_addr_t next_row_dst_addr = row_dst_addr;
            if (next_col >= ne0) {
                next_col = 0;
                next_r   = r + 1;
                if (next_r < r_end) {
                    concat_regular_row_addrs(task, next_r, &next_i1, &next_row_src_addr, &next_row_dst_addr);
                }
            }

            dma_queue_pop(q); // sbuf[slot]

            if (next_r < r_end) {
                const uint32_t next_cols = MIN(ne0 - next_col, max_cols);
                const uint32_t s_bytes_next = next_cols * src_es;
                dma_queue_push(q, dma_make_data(sbuf[slot ^ 1], next_row_src_addr + next_col * src_es),
                               s_bytes_next, s_bytes_next, s_bytes_next, 1);
            }

            dma_queue_pop(q); // dbuf[slot ^ 1]

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            if (dst_es == 4) {
                concat_cvt_f16_to_f32_aa(dbuf[slot], sbuf[slot], cur_cols);
            } else {
                concat_cvt_f32_to_f16_aa(dbuf[slot], sbuf[slot], cur_cols);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            const uint32_t d_bytes = cur_cols * dst_es;
            dma_queue_push(q, dma_make_data(row_dst_addr + col * dst_es, dbuf[slot]),
                           d_bytes, d_bytes, d_bytes, 1);

            r            = next_r;
            col          = next_col;
            i1           = next_i1;
            row_src_addr = next_row_src_addr;
            row_dst_addr = next_row_dst_addr;
            cur_cols     = MIN(ne0 - col, max_cols);
            slot        ^= 1;
        }
        dma_queue_flush(q);
    }
}

static void concat_regular_side(
    struct htp_ops_context * octx,
    const struct htp_tensor * dst_view,
    const struct htp_tensor * src,
    uint32_t rstart,
    uint32_t nrows,
    const struct fastdiv_values * div_ne1,
    const struct fastdiv_values * div_ne2_ne1) {

    if (nrows == 0 || src->ne[0] == 0) {
        return;
    }

    const uint32_t dst_es = (dst_view->type == HTP_TYPE_F32 || dst_view->type == HTP_TYPE_I32) ? 4 : 2;
    if (src->type == dst_view->type) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst_view, src, dst_es, rstart, nrows);
        dma_queue_flush(q);
        return;
    }

    const uint32_t n_hvx = octx->n_hvx;
    struct concat_regular_cvt_task task;
    task.octx         = octx;
    task.src          = src;
    task.dst          = dst_view;
    task.r_start      = rstart;
    task.nrows        = nrows;
    task.rows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);
    task.src_es       = concat_type_size(src->type);
    task.dst_es       = dst_es;
    task.div_ne1      = *div_ne1;
    task.div_ne2_ne1  = *div_ne2_ne1;

    work_queue_run(octx->ctx->hvx_queue, concat_regular_cvt_worker, &task, n_hvx);
}

static int concat_regular(struct htp_ops_context * octx, const struct htp_concat_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    const int dim = kparams->dim;

    if (src0->type != dst->type || src1->type != dst->type) {
        if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        if (kparams->vtcm_size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
    }

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

    concat_regular_side(octx, &view0, src0, rstart0, nrows0,
                        &kparams->u.regular.div_ne01, &kparams->u.regular.div_ne02_ne01);
    concat_regular_side(octx, &view1, src1, rstart1, nrows1,
                        &kparams->u.regular.div_ne11, &kparams->u.regular.div_ne12_ne11);

    return HTP_STATUS_OK;
}

static int concat_transposed(struct htp_ops_context * octx, const struct htp_concat_kernel_params * kparams) {
    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_hvx   = octx->n_hvx;
    const uint32_t src0_es = concat_type_size(src0->type);
    const uint32_t src1_es = concat_type_size(src1->type);
    const uint32_t dst_es  = concat_type_size(dst->type);

    const bool is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    const bool is_src1_transposed = (src1->nb[0] > src1->nb[1]);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    if (!is_src0_transposed && is_src1_transposed) {
        struct htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, src0->ne[0], src1->ne[0], src0_es, src1_es, dst_es, n_hvx);

        struct htp_concat_context cctx;
        cctx.octx               = octx;
        cctx.lead_src           = src0;
        cctx.trans_src          = src1;
        cctx.dst_view           = dst;
        cctx.spad0_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad0);
        cctx.spad1_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad1);
        cctx.spad2_base         = layout.cvt_spad_size_per_hvx > 0 ? VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad2) : NULL;
        cctx.spad0_size_per_hvx = layout.src0_spad_size_per_hvx;
        cctx.spad1_size_per_hvx = layout.src1_spad_size_per_hvx;
        cctx.spad2_size_per_hvx = layout.cvt_spad_size_per_hvx;
        cctx.row_start          = row_start;
        cctx.nrows              = nrows;
        cctx.nplanes            = dst->ne[2] * dst->ne[3];
        cctx.div_ne2            = kparams->u.transposed.div_ne2;
        cctx.nrows_per_hvx      = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

        work_queue_run(octx->ctx->hvx_queue, concat_2d_transposed_worker, &cctx, n_hvx);
        return HTP_STATUS_OK;
    }

    struct htp_tensor view0 = *dst;
    struct htp_tensor view1 = *dst;
    view0.ne[0] = src0->ne[0];
    view1.ne[0] = src1->ne[0];
    view1.data += src0->ne[0] * dst_es;

    if (src0->ne[0] > 0) {
        struct htp_concat_transposed_vtcm_layout layout0;
        htp_concat_transposed_vtcm_layout_build(&layout0, 0, src0->ne[0], dst_es, src0_es, dst_es, n_hvx);

        struct htp_concat_context cctx0;
        cctx0.octx               = octx;
        cctx0.lead_src           = NULL;
        cctx0.trans_src          = src0;
        cctx0.dst_view           = &view0;
        cctx0.spad0_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout0.off_spad0);
        cctx0.spad1_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout0.off_spad1);
        cctx0.spad2_base         = layout0.cvt_spad_size_per_hvx > 0 ? VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout0.off_spad2) : NULL;
        cctx0.spad0_size_per_hvx = layout0.src0_spad_size_per_hvx;
        cctx0.spad1_size_per_hvx = layout0.src1_spad_size_per_hvx;
        cctx0.spad2_size_per_hvx = layout0.cvt_spad_size_per_hvx;
        cctx0.row_start          = row_start;
        cctx0.nrows              = nrows;
        cctx0.nplanes            = dst->ne[2] * dst->ne[3];
        cctx0.div_ne2            = kparams->u.transposed.div_ne2;
        cctx0.nrows_per_hvx      = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

        work_queue_run(octx->ctx->hvx_queue, concat_2d_transposed_worker, &cctx0, n_hvx);
    }

    if (is_src1_transposed) {
        if (src1->ne[0] > 0) {
            struct htp_concat_transposed_vtcm_layout layout1;
            htp_concat_transposed_vtcm_layout_build(&layout1, 0, src1->ne[0], dst_es, src1_es, dst_es, n_hvx);

            struct htp_concat_context cctx1;
            cctx1.octx               = octx;
            cctx1.lead_src           = NULL;
            cctx1.trans_src          = src1;
            cctx1.dst_view           = &view1;
            cctx1.spad0_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout1.off_spad0);
            cctx1.spad1_base         = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout1.off_spad1);
            cctx1.spad2_base         = layout1.cvt_spad_size_per_hvx > 0 ? VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout1.off_spad2) : NULL;
            cctx1.spad0_size_per_hvx = layout1.src0_spad_size_per_hvx;
            cctx1.spad1_size_per_hvx = layout1.src1_spad_size_per_hvx;
            cctx1.spad2_size_per_hvx = layout1.cvt_spad_size_per_hvx;
            cctx1.row_start          = row_start;
            cctx1.nrows              = nrows;
            cctx1.nplanes            = dst->ne[2] * dst->ne[3];
            cctx1.div_ne2            = kparams->u.transposed.div_ne2;
            cctx1.nrows_per_hvx      = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

            work_queue_run(octx->ctx->hvx_queue, concat_2d_transposed_worker, &cctx1, n_hvx);
        }
    } else if (src1->ne[0] > 0) {
        const uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];
        uint32_t rstart1 = 0, nrows1 = total_rows_1;
        if (octx->ctx->mdev.count > 1) {
            const struct htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
                total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            rstart1 = range1.start;
            nrows1  = range1.count;
        }
        const struct fastdiv_values div_ne11      = init_fastdiv_values(src1->ne[1]);
        const struct fastdiv_values div_ne12_ne11 = init_fastdiv_values(src1->ne[1] * src1->ne[2]);
        concat_regular_side(octx, &view1, src1, rstart1, nrows1, &div_ne11, &div_ne12_ne11);
    }

    return HTP_STATUS_OK;
}

typedef void (*concat_gather_compute_fn_t)(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur);

struct concat_gather_rows_mixed_task {
    struct htp_ops_context *              octx;
    const struct htp_tensor *             a;
    const struct htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_start;
    uint32_t                              rows;
    uint32_t                              rows_per_hvx;
    uint32_t                              src0_es;
    uint32_t                              src1_es;
    uint32_t                              dst_es;
    const struct htp_gather_rows_params * kparams;
    concat_gather_compute_fn_t            compute;
};

static __attribute__((noinline)) void concat_gather_rows_init_tables_f32(
    HVX_Vector * tab, HVX_Vector * inc,
    const struct htp_gather_rows_params * kparams,
    uint32_t a_reg_nb1, uint32_t b_reg_nb1) {

    const int32_t s_a = (int32_t) (kparams->rg * a_reg_nb1);
    const int32_t s_b = (int32_t) (kparams->rg * b_reg_nb1);
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < kparams->vg; k++) {
        for (uint32_t lane = 0; lane < 32; lane++) {
            if (col < kparams->na) {
                t_buf[lane] = (int32_t) (row * a_reg_nb1 + col * 4);
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = (int32_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_reg_nb1);
                s_buf[lane] = s_b;
            }
            if (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
}

static __attribute__((noinline)) void concat_gather_rows_init_tables_f16(
    HVX_Vector * tab, HVX_Vector * inc,
    const struct htp_gather_rows_params * kparams,
    uint32_t a_reg_nb1, uint32_t b_reg_nb1) {

    const int16_t s_a = (int16_t) (kparams->rg * a_reg_nb1);
    const int16_t s_b = (int16_t) (kparams->rg * b_reg_nb1);
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < kparams->vg; k++) {
        for (uint32_t lane = 0; lane < 64; lane++) {
            if (col < kparams->na) {
                t_buf[lane] = (int16_t) (row * a_reg_nb1 + col * 2);
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = (int16_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_reg_nb1);
                s_buf[lane] = s_b;
            }
            if (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
}

static __attribute__((noinline)) void concat_gather_compute_f32_vg1(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (void) cur;
    const uint32_t region_size = kparams->region_size;
    HVX_Vector vcur = tab[0];
    const HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_size, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

static __attribute__((noinline)) void concat_gather_compute_f32_vgn(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    const uint32_t vg = kparams->vg;
    const uint32_t region_size = kparams->region_size;
    for (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_size, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        if (++k == vg) {
            k = 0;
        }
    }
}

static __attribute__((noinline)) void concat_gather_compute_f16_vg1(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (void) cur;
    const uint32_t region_size = kparams->region_size;
    HVX_Vector vcur = tab[0];
    const HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_size, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

static __attribute__((noinline)) void concat_gather_compute_f16_vgn(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    const uint32_t vg = kparams->vg;
    const uint32_t region_size = kparams->region_size;
    for (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_size, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        if (++k == vg) {
            k = 0;
        }
    }
}

static inline void concat_gather_rows_dma_in(
    dma_queue * q,
    const struct concat_gather_rows_mixed_task * task,
    uint8_t * region, uint8_t * raw_a, uint8_t * raw_b,
    uint32_t r, uint32_t nr) {

    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;
    const struct htp_gather_rows_params * kparams = task->kparams;
    const uint32_t src0_es = task->src0_es;
    const uint32_t src1_es = task->src1_es;
    const uint32_t dst_es  = task->dst_es;

    if (kparams->na > 0) {
        uint8_t * dst_a = (src0_es == dst_es) ? region : raw_a;
        const size_t a_size = (size_t) (nr - 1) * a->nb[1] + (size_t) kparams->na * src0_es;
        dma_queue_push(q, dma_make_data(dst_a, a->data + r * a->nb[1]), a_size, a_size, a_size, 1);
    }

    if (b && kparams->nb > 0) {
        uint8_t * dst_b = (src1_es == dst_es) ? (region + kparams->v1) : raw_b;
        if (kparams->b_dense) {
            const size_t b_size = (size_t) (nr - 1) * b->nb[1] + (size_t) kparams->nb * src1_es;
            dma_queue_push(q, dma_make_data(dst_b, b->data + r * b->nb[1]), b_size, b_size, b_size, 1);
        } else {
            const size_t b_size = (size_t) (nr - 1) * b->nb[1] + src1_es;
            const size_t dst_span1 = (src1_es == dst_es)
                ? kparams->span1
                : (((size_t) kparams->span1 / dst_es) * src1_es);
            for (uint32_t k = 0; k < kparams->nb; k++) {
                dma_queue_push(q, dma_make_data(dst_b + k * dst_span1,
                                                b->data + (k * b->nb[0] + r * b->nb[1])),
                               b_size, b_size, b_size, 1);
            }
        }
    }
}

static inline void concat_gather_rows_dma_out(
    dma_queue * q,
    const struct concat_gather_rows_mixed_task * task,
    uint8_t * out, uint32_t r, uint32_t nr) {

    const struct htp_gather_rows_params * kparams = task->kparams;
    const size_t out_size = (size_t) nr * kparams->ne * task->dst_es;
    dma_queue_push(q, dma_make_data(task->dst + r * kparams->ne * task->dst_es, out), out_size, out_size, out_size, 1);
}

static void concat_gather_rows_mixed_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct concat_gather_rows_mixed_task * task = (const struct concat_gather_rows_mixed_task *) data;
    const struct htp_gather_rows_params * kparams = task->kparams;
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;

    const uint32_t r_offset = MIN(ith * task->rows_per_hvx, task->rows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_hvx, task->rows);
    if (r_beg >= r_end) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;

    const uint32_t src0_es = task->src0_es;
    const uint32_t src1_es = task->src1_es;
    const uint32_t dst_es  = task->dst_es;

    struct htp_concat_gather_rows_vtcm_layout layout;
    htp_concat_gather_rows_vtcm_layout_build(&layout, kparams->vg, kparams->region_size, kparams->nr_max,
                                             kparams->na, a->nb[1],
                                             kparams->nb, b ? b->nb[1] : 0, kparams->b_dense != 0, kparams->span1,
                                             kparams->ne, src0_es, src1_es, dst_es);

    uint8_t * base   = VTCM_LAYOUT_PTR(uint8_t, task->octx->ctx->vtcm_base, (size_t) ith * kparams->slice);
    HVX_Vector * tab = VTCM_LAYOUT_PTR(HVX_Vector, base, layout.off_tab);
    HVX_Vector * inc = VTCM_LAYOUT_PTR(HVX_Vector, base, layout.off_inc);
    HVX_Vector * cur = VTCM_LAYOUT_PTR(HVX_Vector, base, layout.off_cur);

    uint8_t * region[2] = {
        VTCM_LAYOUT_PTR(uint8_t, base, layout.off_region0),
        VTCM_LAYOUT_PTR(uint8_t, base, layout.off_region1),
    };
    uint8_t * out[2] = {
        VTCM_LAYOUT_PTR(uint8_t, base, layout.off_out0),
        VTCM_LAYOUT_PTR(uint8_t, base, layout.off_out1),
    };
    uint8_t * raw_a[2] = {
        layout.off_raw_a0 ? VTCM_LAYOUT_PTR(uint8_t, base, layout.off_raw_a0) : NULL,
        layout.off_raw_a1 ? VTCM_LAYOUT_PTR(uint8_t, base, layout.off_raw_a1) : NULL,
    };
    uint8_t * raw_b[2] = {
        layout.off_raw_b0 ? VTCM_LAYOUT_PTR(uint8_t, base, layout.off_raw_b0) : NULL,
        layout.off_raw_b1 ? VTCM_LAYOUT_PTR(uint8_t, base, layout.off_raw_b1) : NULL,
    };

    const uint32_t a_elem_stride = a->nb[1] / src0_es;
    const uint32_t b_elem_stride = b ? (b->nb[1] / src1_es) : 0;
    const uint32_t a_reg_nb1     = a_elem_stride * dst_es;
    const uint32_t b_reg_nb1     = b_elem_stride * dst_es;

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    if (dst_es == 4) {
        concat_gather_rows_init_tables_f32(tab, inc, kparams, a_reg_nb1, b_reg_nb1);
    } else {
        concat_gather_rows_init_tables_f16(tab, inc, kparams, a_reg_nb1, b_reg_nb1);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = th->dma_q;
    const uint32_t n_in = (kparams->na > 0 ? 1 : 0) + ((b && kparams->nb > 0) ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    const uint32_t nr_max = kparams->nr_max;
    const uint32_t nr0    = MIN(nr_max, r_end - r_beg);

    dma_queue_push(q, dma_make_data(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    concat_gather_rows_dma_in(q, task, region[0], raw_a[0], raw_b[0], r_beg, nr0);

    const uint32_t r1 = r_beg + nr_max;
    if (r1 < r_end) {
        const uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_push(q, dma_make_data(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        concat_gather_rows_dma_in(q, task, region[1], raw_a[1], raw_b[1], r1, nr1);
    }

    uint32_t buf = 0;
    for (uint32_t r = r_beg; r < r_end; r += nr_max) {
        const uint32_t nr = MIN(nr_max, r_end - r);
        const uint32_t next_buf = buf ^ 1;

        dma_queue_pop(q); // complete out
        for (uint32_t i = 0; i < n_in; i++) {
            dma_queue_pop(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        if (src0_es != dst_es && kparams->na > 0) {
            const uint32_t a_elems = (nr - 1) * a_elem_stride + kparams->na;
            if (dst_es == 4) {
                concat_cvt_f16_to_f32_aa(region[buf], raw_a[buf], a_elems);
            } else {
                concat_cvt_f32_to_f16_aa(region[buf], raw_a[buf], a_elems);
            }
        }
        if (src1_es != dst_es && b && kparams->nb > 0) {
            const uint32_t b_elems = kparams->b_dense
                ? ((nr - 1) * b_elem_stride + kparams->nb)
                : (kparams->nb * (kparams->span1 / dst_es));
            if (dst_es == 4) {
                concat_cvt_f16_to_f32_aa(region[buf] + kparams->v1, raw_b[buf], b_elems);
            } else {
                concat_cvt_f32_to_f16_aa(region[buf] + kparams->v1, raw_b[buf], b_elems);
            }
        }

        const uint32_t nvec = (dst_es == 4)
            ? ((nr * kparams->ne + 31) >> 5)
            : ((nr * kparams->ne + 63) >> 6);
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        concat_gather_rows_dma_out(q, task, out[buf], r, nr);

        const uint32_t r_prefetch = r + 2 * nr_max;
        if (r_prefetch < r_end) {
            const uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            concat_gather_rows_dma_in(q, task, region[buf], raw_a[buf], raw_b[buf], r_prefetch, nr_prefetch);
        }

        buf = next_buf;
    }
    dma_queue_flush(q);
}

static int concat_gather_rows(
    struct htp_ops_context * octx,
    const struct htp_tensor * src0,
    const struct htp_tensor * src1,
    const struct htp_gather_rows_params * kparams,
    uint32_t n_hvx) {

    const struct htp_tensor * dst = octx->dst;
    if (src0->type == dst->type && src1->type == dst->type) {
        return htp_gather_rows(octx, src0, src1, kparams, n_hvx);
    }

    if (!htp_ops_context_set_n_hvx(octx, n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->slice * octx->n_hvx > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t total_rows = kparams->total_rows;
    uint32_t r_start = 0;
    uint32_t r_count = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        r_start = range.start;
        r_count = range.count;
    }

    if (r_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t dst_es = concat_type_size(dst->type);

    struct concat_gather_rows_mixed_task task;
    task.octx         = octx;
    task.a            = src0;
    task.b            = src1;
    task.dst          = dst->data;
    task.r_start      = r_start;
    task.rows         = r_count;
    task.rows_per_hvx = kparams->rows_per_hvx;
    task.src0_es      = concat_type_size(src0->type);
    task.src1_es      = concat_type_size(src1->type);
    task.dst_es       = dst_es;
    task.kparams      = kparams;

    if (dst_es == 2) {
        task.compute = (kparams->vg == 1) ? concat_gather_compute_f16_vg1 : concat_gather_compute_f16_vgn;
    } else {
        task.compute = (kparams->vg == 1) ? concat_gather_compute_f32_vg1 : concat_gather_compute_f32_vgn;
    }

    work_queue_run(octx->ctx->hvx_queue, concat_gather_rows_mixed_worker, &task, octx->n_hvx);
    return HTP_STATUS_OK;
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_concat_kernel_params * kparams = (const struct htp_concat_kernel_params *) octx->kernel_params;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];

    int status = HTP_STATUS_OK;

    switch (kparams->kernel_type) {
        case HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams);
            break;

        case HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams);
            break;

        case HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = concat_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_hvx);
            break;

        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    return octx->status;
}
