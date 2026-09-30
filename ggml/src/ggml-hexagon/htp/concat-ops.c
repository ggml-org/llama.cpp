#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "dma-queue.h"
#include "htp-vtcm.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"
#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
    uint32_t dim;
    uint32_t nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t elem_start;
    uint32_t nelems;
    uint32_t nplanes;
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1;
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

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 32;
    const uint32_t spad1_stride = block_i * sizeof(float);

    const HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    const uint32_t src0_row_bytes  = src0_ne0 * sizeof(float);
    const uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * sizeof(float);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    const dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * sizeof(float), src1_ne0);

    const dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_pop(dma_q); // src1

    while (p < cctx->nplanes) {
        const uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        for (uint32_t j = 0; j < src1_ne0; j += 32) {
            const uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * sizeof(float);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        if (ni >= end_i) {
            ni = start_i;
            np++;
        }
        const bool has_next = np < cctx->nplanes;
        const uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        if (has_next) {
            const dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * sizeof(float), src1_ne0);
        }

        dma_queue_pop(dma_q); // src0

        const dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);

        if (has_next) {
            dma_queue_pop(dma_q); // next src1
        }
        dma_queue_pop(dma_q); // dst

        // spad0 is free after the dst write
        if (has_next) {
            const dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 64;
    const uint32_t spad1_stride = block_i * sizeof(__fp16);

    const HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    const uint32_t src0_row_bytes  = src0_ne0 * sizeof(__fp16);
    const uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * sizeof(__fp16);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = start_i;

    const dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * sizeof(__fp16), src1_ne0);

    const dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_pop(dma_q); // src1

    while (p < cctx->nplanes) {
        const uint32_t current_block_i = MIN(end_i - i, block_i);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        for (uint32_t j = 0; j < src1_ne0; j += 64) {
            const uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * sizeof(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        if (ni >= end_i) {
            ni = start_i;
            np++;
        }
        const bool has_next = np < cctx->nplanes;
        const uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        if (has_next) {
            const dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * sizeof(__fp16), src1_ne0);
        }

        dma_queue_pop(dma_q); // src0

        const dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

        if (has_next) {
            dma_queue_pop(dma_q); // next src1
        }
        dma_queue_pop(dma_q); // dst

        // spad0 is free after the dst write
        if (has_next) {
            const dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
}

static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const int dim = cctx->dim;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};

    // Per-device element range aligned to prevent false sharing
    const uint32_t elem_start = cctx->elem_start;
    const uint32_t nelems     = cctx->nelems;
    const uint32_t chunk_size = fastdiv(nelems + nth - 1, &octx->n_threads_div);

    const uint32_t start_idx = MIN(elem_start + ith * chunk_size, elem_start + nelems);
    const uint32_t end_idx   = MIN(start_idx + chunk_size, elem_start + nelems);

    // Naive scalar element-wise copy
    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
        uint32_t i0 = idx - idx_div_ne0 * ne[0];

        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];

        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
        uint32_t i3 = idx_div_ne012;

        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];

        uint32_t idx_dim = 0;
        if (dim == 0) idx_dim = i0;
        else if (dim == 1) idx_dim = i1;
        else if (dim == 2) idx_dim = i2;
        else if (dim == 3) idx_dim = i3;

        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;

        uint32_t s0 = i0;
        uint32_t s1 = i1;
        uint32_t s2 = i2;
        uint32_t s3 = i3;

        if (dim == 0 && src == src1) s0 -= src0->ne[0];
        if (dim == 1 && src == src1) s1 -= src0->ne[1];
        if (dim == 2 && src == src1) s2 -= src0->ne[2];
        if (dim == 3 && src == src1) s3 -= src0->ne[3];

        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];

        if (type_size == 4) {
            *(float*)dst_ptr = *(float*)src_ptr;
        } else {
            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
        }
    }
}

static bool concat_dim1_contiguous_dma(struct htp_ops_context * octx, int dim, uint32_t type_size) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (dim != 1 || octx->ctx->mdev.count > 1 ||
        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_I32) ||
        src0->type != dst->type || src1->type != dst->type ||
        src0->ne[0] != dst->ne[0] || src1->ne[0] != dst->ne[0] ||
        src0->ne[2] != dst->ne[2] || src1->ne[2] != dst->ne[2] ||
        src0->ne[3] != dst->ne[3] || src1->ne[3] != dst->ne[3] ||
        dst->ne[1] != src0->ne[1] + src1->ne[1] ||
        !htp_tensor_is_contiguous(src0, type_size) ||
        !htp_tensor_is_contiguous(src1, type_size) ||
        !htp_tensor_is_contiguous(dst, type_size)) {
        return false;
    }

    const uint32_t src0_row_size = src0->ne[0] * type_size;
    const uint32_t src1_row_size = src1->ne[0] * type_size;

    // v75+ dma_queue_push() writes a 2D descriptor directly and does not split overflow.
#if __HVX_ARCH__ >= 75
    if (src0_row_size > 0xffffffu || src1_row_size > 0xffffffu ||
        src0->nb[1] > 0xffffffu || src1->nb[1] > 0xffffffu || dst->nb[1] > 0xffffffu ||
        src0->ne[1] > UINT16_MAX || src1->ne[1] > UINT16_MAX) {
        return false;
    }
#endif

    dma_queue * q = octx->ctx->dma[0];

    for (uint32_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (uint32_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            dma_addr_t dst_addr  = dst->data  + i3 * dst->nb[3]  + i2 * dst->nb[2];
            dma_addr_t src0_addr = src0->data + i3 * src0->nb[3] + i2 * src0->nb[2];
            dma_addr_t src1_addr = src1->data + i3 * src1->nb[3] + i2 * src1->nb[2];

            if (!dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1]);
            }

            dst_addr += src0->ne[1] * dst->nb[1];
            if (!dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1]);
            }
        }
    }

    dma_queue_flush(q);
    return true;
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int dim = octx->op_params[0];

    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
    bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
    bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);

    if (concat_dim1_contiguous_dma(octx, dim, type_size)) {
        return HTP_STATUS_OK;
    }

    uint32_t n_threads = octx->n_threads;
    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);

    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;

    const bool rows_ok = src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size;

    if (dim == 0 && is_src1_transposed && !is_src0_transposed && rows_ok) {
        const uint32_t total_rows = dst->ne[1];
        const size_t dst_data_row_size = dst->ne[0] * type_size;
        uint32_t row_start = 0;
        uint32_t nrows     = total_rows;
        if (octx->ctx->mdev.count > 1) {
            uint32_t rows_per_chunk = 0;
            htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            row_start = range.start;
            nrows     = range.count;
        }

        if (nrows == 0) {
            return HTP_STATUS_OK;
        }

        cctx.row_start = row_start;
        cctx.nrows     = nrows;
        cctx.nplanes   = dst->ne[2] * dst->ne[3];

        uint32_t block_i = (type_size == 4) ? 32 : 64;

        cctx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

        // Allocate VTCM
        uint32_t spad1_stride = block_i * type_size;

        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
        // src0 row is right-aligned to VLEN so the gathered src1 part starts aligned
        uint32_t spad0_row_bytes = hex_round_up(src0->ne[0] * type_size, VLEN) + src1_ne0_padded * type_size;

        octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
        octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride;

        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;

        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        if (type_size == 4) {
            worker_func = concat_2d_f32_transposed;
        } else {
            worker_func = concat_2d_f16_transposed;
        }
    } else {
        if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
            return HTP_STATUS_NO_SUPPORT;
        }

        const uint32_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
        uint32_t elem_start = 0;
        uint32_t nelems     = total_elements;
        if (octx->ctx->mdev.count > 1) {
            const uint32_t elems_per_chunk = HEX_L2_LINE_SIZE / type_size;
            const bool can_split = htp_tensor_mdev_data_aligned(dst) && htp_tensor_is_contiguous(dst, type_size) && !htp_tensor_is_permuted(dst);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_elements, can_split ? elems_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            elem_start = range.start;
            nelems     = range.count;
        }

        if (nelems == 0) {
            return HTP_STATUS_OK;
        }

        cctx.elem_start = elem_start;
        cctx.nelems     = nelems;
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    return HTP_STATUS_OK;
}
