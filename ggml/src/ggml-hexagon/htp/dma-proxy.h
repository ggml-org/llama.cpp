#ifndef HTP_DMA_PROXY_H
#define HTP_DMA_PROXY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>

#include "dma-queue.h"
#include "hex-utils.h"
#include "htp-thread.h"
#include "htp-trace.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __HVX_ARCH__ > 79
#define DMA_PROXY_POLL_COUNT 2000
#else
#define DMA_PROXY_POLL_COUNT 1
#endif

struct dma_proxy_desc {
    dma_data ddata;
    uint32_t dst_stride;
    uint32_t src_stride;
    uint32_t row_size;
    uint32_t nrows;
};

struct dma_proxy_s {
    struct dma_proxy_desc * desc;
    atomic_uint             idx_write; // monotonic, updated by producer (push)
    uint32_t                idx_pop;   // monotonic, updated by producer (pop)
    uint32_t                idx_mask;
    uint32_t                capacity;
    atomic_uint             seqn;      // incremented on push/kill, used with futex
    atomic_bool             killed;

    atomic_uint             idx_done __attribute__((aligned(HEX_L2_LINE_SIZE))); // monotonic, updated by consumer
    uint32_t                idx_submit; // monotonic, private to consumer

    qurt_thread_t           thread;
    htp_thread_t            th;
    void *                  stack;
};

typedef struct dma_proxy_s * dma_proxy_t;

size_t      dma_proxy_sizeof(size_t capacity, uint32_t stack_size);
size_t      dma_proxy_alignof(void);
dma_proxy_t dma_proxy_init(void * ptr, size_t capacity, uint32_t stack_size, dma_queue_t * dma, struct htp_thread_trace * trace);
void        dma_proxy_free(dma_proxy_t p);

static inline bool dma_proxy_empty(dma_proxy_t p) {
    return p->idx_pop == atomic_load_explicit(&p->idx_write, memory_order_relaxed);
}

static inline bool dma_proxy_push(dma_proxy_t p, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
    uint32_t iw = atomic_load_explicit(&p->idx_write, memory_order_relaxed);
    if ((iw - p->idx_pop) >= p->capacity) {
        FARF(HIGH, "dma-proxy-push: queue is full\n");
        return false;
    }

    struct dma_proxy_desc * d = &p->desc[iw & p->idx_mask];
    d->ddata      = ddata;
    d->dst_stride = (uint32_t) dst_stride;
    d->src_stride = (uint32_t) src_stride;
    d->row_size   = (uint32_t) row_size;
    d->nrows      = (uint32_t) nrows;

    atomic_store_explicit(&p->idx_write, iw + 1, memory_order_release);

    // wake up proxy thread
    unsigned int seqn = atomic_load_explicit(&p->seqn, memory_order_relaxed) + 1;
    atomic_store_explicit(&p->seqn, seqn, memory_order_release);
    qurt_futex_wake(&p->seqn, 1);

    return true;
}

static inline dma_data dma_proxy_pop(dma_proxy_t p) {
    uint32_t ip = p->idx_pop;
    uint32_t iw = atomic_load_explicit(&p->idx_write, memory_order_relaxed);

    dma_data ddata = { 0 };
    if (ip == iw) {
        return ddata;
    }

    while ((int32_t) (atomic_load_explicit(&p->idx_done, memory_order_relaxed) - ip) <= 0) {
        hex_pause();
    }
    atomic_thread_fence(memory_order_acquire);

    ddata = p->desc[ip & p->idx_mask].ddata;
    p->idx_pop = ip + 1;
    return ddata;
}

static inline void dma_proxy_flush(dma_proxy_t p) {
    while (!dma_proxy_empty(p)) {
        dma_proxy_pop(p);
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HTP_DMA_PROXY_H */
