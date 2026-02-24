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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
