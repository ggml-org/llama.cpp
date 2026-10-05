#ifndef HEX_CPY_DMA_H
#define HEX_CPY_DMA_H

// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
// Used by CPY for the copy itself and by CONCAT, which is two such copies into
// two views of its destination.  Every helper only pushes descriptors; the
// caller flushes the queue when it needs the data.

#include "dma-queue.h"
#include "hex-common.h"
#include "htp-tensor.h"

#include <stddef.h>
#include <stdint.h>

// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
static inline void cpy_dma_sametype_reshape_contig(dma_queue * dma_q,
                                                   dma_addr_t  dst,
                                                   dma_addr_t  src0,
                                                   uint32_t    total_bytes) {
    if (total_bytes == 0) {
        return;
    }

    const uint32_t max_chunk = DMA_SAFE_CHUNK_SIZE;
    while (total_bytes > 0) {
        const uint32_t chunk = MIN(total_bytes, max_chunk);
        if (!dma_queue_push(dma_q, dma_make_data(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
            dma_queue_flush(dma_q);
            dma_queue_push(dma_q, dma_make_data(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
        }
        dst += chunk;
        src0 += chunk;
        total_bytes -= chunk;
    }
}

// One 2d transfer, split at the 16-bit nrows field.
static inline void cpy_dma_push_2d_chunked(dma_queue * dma_q,
                                           dma_addr_t  dst,
                                           dma_addr_t  src,
                                           size_t      dst_stride,
                                           size_t      src_stride,
                                           size_t      row_size,
                                           uint32_t    nrows) {
    if (row_size == 0 || nrows == 0) {
        return;
    }

    while (nrows > 0) {
        const uint32_t cur_rows = MIN(nrows, DMA_MAX_NROWS);
        if (!dma_queue_push(dma_q, dma_make_data(dst, src), dst_stride, src_stride, row_size, cur_rows)) {
            dma_queue_flush(dma_q);
            dma_queue_push(dma_q, dma_make_data(dst, src), dst_stride, src_stride, row_size, cur_rows);
        }
        dst += cur_rows * dst_stride;
        src += cur_rows * src_stride;
        nrows -= cur_rows;
    }
}

// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
// both sides (nb[0] == elem_size).
static inline void cpy_dma_sametype_sameshape(dma_queue *               dma_q,
                                              const struct htp_tensor * dst,
                                              const struct htp_tensor * src0,
                                              uint32_t                  elem_size) {
    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    if (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
        return;
    }

    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];
    const uint32_t nb03 = src0->nb[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const bool contiguous = htp_tensor_is_contiguous(src0, elem_size) && htp_tensor_is_contiguous(dst, elem_size);

    if (contiguous) {
        cpy_dma_sametype_reshape_contig(dma_q, dst->data, src0->data, ne00 * elem_size * ne01 * ne02 * ne03);
        return;
    }

    // The single-descriptor path flattens (i01,i02,i03) into one row index, so every
    // row must sit at a constant stride: nb01 on the source, nb1 on the destination.
    // Walk the outer dims and require each to continue that progression.  A dim of
    // extent 1 spans no rows, so it is skipped -- but its own stride must NOT then be
    // used to justify the next dim's stride, which is what comparing nb03 against
    // ne02*nb02 did: ggml leaves the stride of an extent-1 dim meaningless, so a view
    // could pass the check while its rows were nowhere near that stride.
    uint32_t exp_src          = ne01 * nb01;
    uint32_t exp_dst          = ne01 * nb1;
    bool     contiguous_outer = true;
    if (ne02 != 1) {
        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
    }
    exp_src *= ne02;
    exp_dst *= ne02;
    if (ne03 != 1) {
        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
    }

    if (contiguous_outer) {
        uint32_t total_rows = ne01 * ne02 * ne03;
        cpy_dma_push_2d_chunked(dma_q, dst->data, src0->data, nb1, nb01, ne00 * elem_size, total_rows);
        return;
    }

    for (uint32_t i03 = 0; i03 < ne03; i03++) {
        for (uint32_t i02 = 0; i02 < ne02; i02++) {
            dma_addr_t dst_data  = dst->data + i02 * nb2 + i03 * nb3;
            dma_addr_t src0_data = src0->data + i02 * nb02 + i03 * nb03;
            cpy_dma_push_2d_chunked(dma_q, dst_data, src0_data, nb1, nb01, ne00 * elem_size, ne01);
        }
    }
}

#endif /* HEX_CPY_DMA_H */
