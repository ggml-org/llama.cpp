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

dma_queue * dma_queue_create(size_t capacity, uintptr_t vtcm_base, size_t vtcm_size, struct htp_thread_trace * trace) {
    capacity = pow2_ceil(capacity);

    size_t size_q      = sizeof(dma_queue);
    size_t offset_r    = align_up(size_q, 32);
    size_t size_r      = sizeof(dma_ring);
    size_t offset_desc = align_up(offset_r + size_r, 64);
    size_t size_desc   = capacity * sizeof(dma_descriptor_2d);
    size_t offset_dptr = align_up(offset_desc + size_desc, 4);
    size_t size_dptr   = capacity * sizeof(dma_ptr);

    size_t total_size = offset_dptr + size_dptr;

    void * block = memalign(64, total_size);
    if (block == NULL) {
        FARF(ERROR, "%s: failed to allocate unified DMA memory block of size %zu\n", __FUNCTION__, total_size);
        return NULL;
    }
    memset(block, 0, total_size);

    dma_queue * q = (dma_queue *) block;
    dma_ring * r  = (dma_ring *) ((uintptr_t) block + offset_r);

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

    r->desc = (dma_descriptor_2d *) ((uintptr_t) block + offset_desc);
    r->dptr = (dma_ptr *) ((uintptr_t) block + offset_dptr);
    r->tail = &r->desc[capacity - 1];

    FARF(HIGH, "dma-queue: capacity %u, unified memory size %zu\n", capacity, total_size);

    return q;
}

dma_queue * dma_queue_create_alias(dma_queue * main_q, uint8_t nocache) {
    dma_queue * q = (dma_queue *) memalign(32, sizeof(dma_queue));
    if (q == NULL) {
        FARF(ERROR, "%s: failed to allocate DMA queue alias\n", __FUNCTION__);
        return NULL;
    }

    q->ring    = main_q->ring;
    q->nocache = nocache;
    q->alias   = true;

    return q;
}

void dma_queue_delete(dma_queue * q) {
    if (!q) {
        return;
    }
    free(q);
}

void dma_queue_flush(dma_queue * q) {
    while (dma_queue_pop(q).dst != NULL) ;
}
