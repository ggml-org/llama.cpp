#include "htp-tensor.h"

#include <string.h>

#include "dma-copy.h"
#include "hex-utils.h"

#define HTP_TENSOR_READ_SMALL_BYTES 256

static inline void htp_tensor_read_l2flush(const htp_thread_t * th, uintptr_t start, size_t len) {
    const uintptr_t s = hex_align_down(start, HEX_L2_LINE_SIZE);
    const uintptr_t e = hex_align_up(start + len, HEX_L2_LINE_SIZE);
    if (e <= s) {
        return;
    }
    htp_trace_event_start(th->trace, HTP_TRACE_EVT_L2FLUSH, 0);
    hex_l2flush((void *) s, e - s);
    htp_trace_event_stop(th->trace, HTP_TRACE_EVT_L2FLUSH, 0);
}

size_t htp_tensor_read(const htp_thread_t * th, void * dst, size_t dst_size, const struct htp_tensor * src) {
    if (!th || !dst || !src) {
        return 0;
    }

    const uint32_t ne0 = src->ne[0];
    const uint32_t ne1 = src->ne[1];
    const uint32_t ne2 = src->ne[2];
    const uint32_t ne3 = src->ne[3];
    if (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) {
        return 0;
    }

    const uint32_t row_bytes = htp_tensor_get_row_size(src->type, ne0);
    if (row_bytes == 0 || (row_bytes % ne0) != 0) {
        return 0;
    }

    const uint32_t elem_size   = row_bytes / ne0;
    const size_t   nrows       = (size_t) ne1 * ne2 * ne3;
    const size_t   total_bytes = (size_t) row_bytes * nrows;
    if (total_bytes > dst_size) {
        return 0;
    }

    const uintptr_t dst_addr  = (uintptr_t) dst;
    const bool      is_contig = htp_tensor_is_contiguous(src, elem_size);

    if (!htp_tensor_is_extended(src) && is_contig && total_bytes <= HTP_TENSOR_READ_SMALL_BYTES) {
        if (!(src->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            htp_tensor_read_l2flush(th, (uintptr_t) src->data, total_bytes);
        }
        memcpy(dst, (const void *) (uintptr_t) src->data, total_bytes);
        return total_bytes;
    }

    htp_tensor_read_l2flush(th, dst_addr, total_bytes);

    dma_queue * dma_q = th->dma_q;
    if (ne0 > 1 && src->nb[0] != elem_size) {
        dma_addr_t cur_dst = (dma_addr_t) dst_addr;
        for (uint32_t i3 = 0; i3 < ne3; ++i3) {
            for (uint32_t i2 = 0; i2 < ne2; ++i2) {
                for (uint32_t i1 = 0; i1 < ne1; ++i1) {
                    const dma_addr_t cur_src = src->data +
                        (dma_addr_t) i1 * src->nb[1] +
                        (dma_addr_t) i2 * src->nb[2] +
                        (dma_addr_t) i3 * src->nb[3];
                    dma_cpy_push_2d_chunked(dma_q, cur_dst, cur_src, elem_size, src->nb[0], elem_size, ne0);
                    cur_dst += row_bytes;
                }
            }
        }
    } else {
        struct htp_tensor dst_t = *src;
        dst_t.data  = (dma_addr_t) dst_addr;
        dst_t.nb[0] = elem_size;
        dst_t.nb[1] = row_bytes;
        dst_t.nb[2] = row_bytes * ne1;
        dst_t.nb[3] = dst_t.nb[2] * ne2;
        dma_cpy_sametype_sameshape(dma_q, &dst_t, src, elem_size);
    }
    dma_queue_flush(dma_q);

    return total_bytes;
}
