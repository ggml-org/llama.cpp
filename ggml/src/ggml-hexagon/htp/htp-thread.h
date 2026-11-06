#ifndef HTP_THREAD_H
#define HTP_THREAD_H

#include <stdatomic.h>
#include <stdint.h>

#include "dma-queue.h"
#include "hex-profile.h"

typedef struct {
    unsigned int              tid;
    dma_queue *               dma_q;
    struct htp_thread_trace * trace;
} htp_thread_t;

extern atomic_uint htp_thread_count;

static inline void htp_thread_reset(void) {
    atomic_store_explicit(&htp_thread_count, 0, memory_order_relaxed);
}

static inline unsigned int htp_thread_alloc_id(void) {
    return atomic_fetch_add_explicit(&htp_thread_count, 1, memory_order_relaxed);
}

static inline void htp_thread_init(htp_thread_t *            th,
                                   dma_queue_t *             dma,
                                   struct htp_thread_trace * trace) {
    unsigned int tid = htp_thread_alloc_id();
    th->tid   = tid;
    th->dma_q = dma   ? dma[tid]    : NULL;
    th->trace = trace ? &trace[tid] : NULL;
}

#endif /* HTP_THREAD_H */
