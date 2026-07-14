#include "dma-queue.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

static inline uint32_t pow2_ceil(uint32_t x) {
    if (x <= 1) {
        return 1;
    }
    int p = 2;
    x--;
    while (x >>= 1) {
        p <<= 1;
    }
    return p;
}

static inline uintptr_t align_up(uintptr_t addr, size_t align) {
    return (addr + align - 1) & ~(align - 1);
}

size_t dma_queue_sizeof(size_t capacity) {
    capacity = pow2_ceil(capacity);

    size_t size_q      = sizeof(dma_queue);
    size_t offset_r    = align_up(size_q, HEX_L2_LINE_SIZE);
    size_t size_r      = sizeof(dma_ring);
    size_t offset_desc = align_up(offset_r + size_r, HEX_L2_LINE_SIZE);
    size_t size_desc   = capacity * sizeof(dma_descriptor_2d);
    size_t offset_dptr = align_up(offset_desc + size_desc, HEX_L2_LINE_SIZE);
    size_t size_dptr   = capacity * sizeof(dma_ptr);

    return offset_dptr + size_dptr;
}

size_t dma_queue_alignof(void) {
    return HEX_L2_LINE_SIZE;
}

dma_queue_t dma_queue_init(void * ptr, size_t capacity, uintptr_t vtcm_base, size_t vtcm_size, struct htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t size_q      = sizeof(dma_queue);
    size_t offset_r    = align_up(size_q, HEX_L2_LINE_SIZE);
    size_t size_r      = sizeof(dma_ring);
    size_t offset_desc = align_up(offset_r + size_r, HEX_L2_LINE_SIZE);
    size_t size_desc   = capacity * sizeof(dma_descriptor_2d);
    size_t offset_dptr = align_up(offset_desc + size_desc, HEX_L2_LINE_SIZE);
    size_t size_dptr   = capacity * sizeof(dma_ptr);

    size_t total_size = offset_dptr + size_dptr;
    memset(ptr, 0, total_size);

    dma_queue * q = (dma_queue *) ptr;
    dma_ring * r  = (dma_ring *) ((uintptr_t) ptr + offset_r);

    q->ring    = r;
    q->nocache = 0;
    q->alias   = false;

    r->trace     = trace;
    r->vtcm_base = vtcm_base;
    r->vtcm_end  = vtcm_base + vtcm_size;
    r->capacity  = capacity;
    r->idx_mask  = capacity - 1;
    r->push_idx  = 0;
    r->pop_idx   = 0;

    r->desc = (dma_descriptor_2d *) ((uintptr_t) ptr + offset_desc);
    r->dptr = (dma_ptr *) ((uintptr_t) ptr + offset_dptr);
    r->tail = &r->desc[capacity - 1];

    FARF(HIGH, "dma-queue: capacity %u, unified memory size %zu\n", capacity, total_size);

    return q;
}

void dma_queue_free(dma_queue_t q) {
    (void) q;
}

size_t dma_queue_alias_sizeof(void) {
    return sizeof(dma_queue);
}

dma_queue_t dma_queue_alias_init(void * ptr, dma_queue_t main_q, uint8_t nocache) {
    dma_queue * q = (dma_queue *) ptr;
    memset(q, 0, sizeof(dma_queue));

    q->ring    = main_q->ring;
    q->nocache = nocache;
    q->alias   = true;

    return q;
}

void dma_queue_alias_free(dma_queue_t q) {
    (void) q;
}

void dma_queue_flush(dma_queue_t q) {
    while (dma_queue_pop(q).dst != NULL) ;
}
