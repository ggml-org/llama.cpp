#ifndef HTP_DMA_H
#define HTP_DMA_H

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hexagon_udma_descriptor_type1_t * desc;  // descriptor pointers
    hexagon_udma_descriptor_type1_t * tail;  // tail pointer
    void **                           dst;   // dst pointers
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

static inline bool dma_queue_push(dma_queue *  q,
                                  void *       dst,
                                  const void * src,
                                  size_t       dst_row_size,
                                  size_t       src_row_size,
                                  size_t       nrows) {
    if (((q->push_idx + 1) & q->idx_mask) == q->pop_idx) {
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
    desc->src            = (void *) src;
    desc->dst            = (void *) dst;
    desc->allocation     = 0;
    desc->padding        = 0;
    desc->roiwidth       = src_row_size;
    desc->roiheight      = nrows;
    desc->srcstride      = src_row_size;
    desc->dststride      = dst_row_size;
    desc->srcwidthoffset = 0;
    desc->dstwidthoffset = 0;

    q->dst[q->push_idx] = dst;

    dmlink(q->tail, desc);
    q->tail = desc;

    // FARF(ERROR, "dma-push: i %u len %u dst %p src %p\n", q->push_idx, len, dst, src);
    q->push_idx = (q->push_idx + 1) & q->idx_mask;
    return true;
}

static inline uint8_t * dma_queue_pop(dma_queue * q) {
    if (q->push_idx == q->pop_idx) {
        return NULL;
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

    uint8_t * dst = (uint8_t *) q->dst[q->pop_idx];

    // FARF(ERROR, "dma-pop: i %u dst %p\n", q->pop_idx, dst);
    q->pop_idx = (q->pop_idx + 1) & q->idx_mask;
    return dst;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_H */
