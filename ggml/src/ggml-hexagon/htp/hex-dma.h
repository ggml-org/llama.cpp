#ifndef HTP_DMA_H
#define HTP_DMA_H

#include <HAP_farf.h>
#include <hexagon_types.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *dst;
    const void *src;
} dma_ptr;

typedef struct {
    hexagon_udma_descriptor_type1_t * desc;  // descriptor pointers
    hexagon_udma_descriptor_type1_t * tail;  // tail pointer
    dma_ptr                         * dptr;  // dst/src pointers
    uint32_t                          push_idx;
    uint32_t                          pop_idx;
    uint32_t                          capacity;
    uint32_t                          idx_mask;
} dma_queue;

dma_queue * dma_queue_create(size_t capacity);
void        dma_queue_delete(dma_queue * q);
void        dma_queue_flush(dma_queue * q);

// TODO: technically we don't need these and could use Q6_dmstart/wait/etc instead
// but those do not seem to always compiler properly.
static inline void dmstart(void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmstart(%0)" : : "r"(next));
}

static inline void dmlink(void * cur, void * next) {
    asm volatile(" release(%0):at" : : "r"(next));
    asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

static inline unsigned int dmpoll(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmpoll" : "=r"(ret) : : "memory");
    return ret;
}

static inline unsigned int dmwait(void) {
    unsigned int ret = 0;
    asm volatile(" %0 = dmwait" : "=r"(ret) : : "memory");
    return ret;
}

static inline dma_ptr dma_make_ptr(void *dst, const void *src)
{
    dma_ptr p = { dst, src };
    return p;
}

static inline bool dma_queue_push(dma_queue * q,
                                  dma_ptr     dptr,
                                  size_t      dst_row_size,
                                  size_t      src_row_size,
                                  size_t      width, // width in bytes. number of bytes to transfer per row
                                  size_t      nrows) {
    if (((q->push_idx + 1) & q->idx_mask) == q->pop_idx) {
        FARF(ERROR, "dma-push: queue full\n");
        return false;
    }

    hexagon_udma_descriptor_type1_t * desc = &q->desc[q->push_idx];

    desc->next           = NULL;
    desc->length         = 0;
    desc->desctype       = HEXAGON_UDMA_DESC_DESCTYPE_TYPE1;
    desc->dstbypass      = 1;
    desc->srcbypass      = 1;
#if __HVX_ARCH__ >= 73
    desc->dstbypass      = 1;
    desc->srcbypass      = 1;
#else
    desc->dstbypass      = 0;
    desc->srcbypass      = 1;
#endif
    desc->order          = 0;
    desc->dstate         = HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE;
    desc->src            = (void *) dptr.src;
    desc->dst            = (void *) dptr.dst;
    desc->allocation     = 0;
    desc->padding        = 0;
    desc->roiwidth       = width;
    desc->roiheight      = nrows;
    desc->srcstride      = src_row_size;
    desc->dststride      = dst_row_size;
    desc->srcwidthoffset = 0;
    desc->dstwidthoffset = 0;

    q->dptr[q->push_idx] = dptr;

    dmlink(q->tail, desc);
    q->tail = desc;

    // FARF(ERROR, "dma-push: i %u width %u nrows %d dst %p src %p\n", q->push_idx, width, nrows, dptr.dst, dptr.src);
    q->push_idx = (q->push_idx + 1) & q->idx_mask;
    return true;
}

static inline bool dma_queue_push_ddr_to_vtcm(dma_queue * q,
                                              dma_ptr     dptr,
                                              size_t      dst_row_size,
                                              size_t      src_row_size,
                                              size_t      nrows) {
    return dma_queue_push(q, dptr, dst_row_size, src_row_size, src_row_size, nrows);
}


static inline bool dma_queue_push_vtcm_to_ddr(dma_queue * q,
                                              dma_ptr     dptr,
                                              size_t      dst_row_size,
                                              size_t      src_row_size,
                                              size_t      nrows) {
    return dma_queue_push(q, dptr, dst_row_size, src_row_size, dst_row_size, nrows);
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr dptr  = { NULL };

    if (q->push_idx == q->pop_idx) {
        return dptr;
    }

    hexagon_udma_descriptor_type1_t * desc = &q->desc[q->pop_idx];

    // Wait for desc to complete
    while (1) {
        dmpoll();
        if (desc->dstate == HEXAGON_UDMA_DESC_DSTATE_COMPLETE) {
            break;
        }
        // FARF(ERROR, "dma-pop: waiting for DMA : %u\n", q->pop_idx);
    }

    dptr = q->dptr[q->pop_idx];

    // FARF(ERROR, "dma-pop: i %u dst %p src %p\n", q->pop_idx, dptr.dst, dptr.src);
    q->pop_idx = (q->pop_idx + 1) & q->idx_mask;
    return dptr;
}

static inline dma_ptr dma_queue_pop_nowait(dma_queue * q) {
    dma_ptr dptr  = { NULL };

    if (q->push_idx == q->pop_idx) {
        return dptr;
    }

    dptr = q->dptr[q->pop_idx];

    // FARF(ERROR, "dma-pop-nowait: i %u dst %p src %p\n", q->pop_idx, dptr.dst, dptr.src);
    q->pop_idx = (q->pop_idx + 1) & q->idx_mask;
    return dptr;
}

static inline bool dma_queue_empty(dma_queue * q) {
    return q->push_idx == q->pop_idx;
}

static inline uint32_t dma_queue_depth(dma_queue * q) {
    return (q->push_idx - q->pop_idx) & q->idx_mask;
}

static inline uint32_t dma_queue_capacity(dma_queue * q) {
    return q->capacity;
}

// ---------------------------------------------------------------------------
// Overflow-safe DMA push: all UDMA type1 descriptor fields (roiwidth,
// roiheight, srcstride, dststride) are 16-bit, max 65535.  This helper
// transparently handles values that exceed the 16-bit limit and submits
// chained DMA transtions.
//
// Case 1 (fast path): all params fit in 16 bits -> direct dma_queue_push.
// Case 2 (contiguous block): width == srcstride == dststride.  Reshape the
//   flat transfer into a 2D descriptor with sub_width <= 65535.  Produces a
//   single descriptor, preserving async DMA behavior.
// Case 3 (stride overflow): srcstride or dststride > 65535.  Issue rows
//   one at a time.  The first N-1 rows are pushed+popped synchronously;
//   the last row is left async so the caller can pop it.
// ---------------------------------------------------------------------------
#define UDMA_MAX_FIELD_VAL 65535u

static inline bool dma_queue_push_chained(dma_queue *q, dma_ptr dptr, size_t dst_stride, size_t src_stride, size_t width, size_t nrows) {
    // Fast path: everything fits in 16 bits.
    if (__builtin_expect(
            width      <= UDMA_MAX_FIELD_VAL &&
            nrows      <= UDMA_MAX_FIELD_VAL &&
            src_stride <= UDMA_MAX_FIELD_VAL &&
            dst_stride <= UDMA_MAX_FIELD_VAL, 1)) {
        return dma_queue_push(q, dptr, dst_stride, src_stride, width, nrows);
    }

    // Case 2: contiguous block (width == src_stride == dst_stride).
    // Reshape total bytes into sub_width * sub_nrows where sub_width <= 65535.
    if (width == src_stride && width == dst_stride) {
        size_t total = width * nrows;

        // Pick the largest 128-byte-aligned sub_width that divides total evenly.
        size_t sub_width = UDMA_MAX_FIELD_VAL & ~(size_t)127;  // 65408
        while (sub_width > 0 && total % sub_width != 0) {
            sub_width -= 128;
        }
        if (sub_width == 0) {
            // Fallback: use original width (must fit) with adjusted nrows.
            // This shouldn't happen for 128-aligned DMA sizes.
            sub_width = width;
        }
        size_t sub_nrows = total / sub_width;

        // Handle sub_nrows > 65535 by issuing chunked descriptors.
        const uint8_t *src = (const uint8_t *)dptr.src;
        uint8_t       *dst = (uint8_t *)dptr.dst;
        size_t rows_done = 0;
        while (rows_done < sub_nrows) {
            size_t chunk = sub_nrows - rows_done;
            if (chunk > UDMA_MAX_FIELD_VAL) chunk = UDMA_MAX_FIELD_VAL;

            dma_ptr p = dma_make_ptr(dst + rows_done * sub_width, src + rows_done * sub_width);
            if (!dma_queue_push(q, p, sub_width, sub_width, sub_width, chunk))
                return false;

            rows_done += chunk;
            // Complete all chunks without waiting except the last one, so the
            // caller's single dma_queue_pop drains the final descriptor.
            if (rows_done < sub_nrows)
                dma_queue_pop_nowait(q);
        }
        return true;
    }

    // Case 3: stride overflow — fall back to row-by-row.
    {
        const uint8_t *src = (const uint8_t *)dptr.src;
        uint8_t       *dst = (uint8_t *)dptr.dst;
        for (size_t r = 0; r < nrows; ++r) {
          dma_ptr p = dma_make_ptr(dst + r * dst_stride,
                                   src + r * src_stride);
          if (!dma_queue_push(q, p, 0, 0, width, 1))
            return false;
          if (r + 1 < nrows)
            dma_queue_pop_nowait(q);
        }
        return true;
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
