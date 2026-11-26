#include "dma-proxy.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <qurt_thread.h>
#include <qurt_futex.h>

#define QURT_LOWEST_PRIO (254)

static bool __attribute__((noinline)) dma_proxy_reap_completed(dma_proxy_t p, bool wait_one) {
    dma_ring * r  = p->th.dma_q->ring0;
    uint32_t   id = atomic_load_explicit(&p->idx_done, memory_order_relaxed);
    if (id == p->idx_submit) {
        return false;
    }

    bool reaped = false;
    dmpoll();
    #pragma clang loop vectorize(disable) interleave(disable)
    while (id != p->idx_submit) {
        volatile dma_descriptor_2d * hw_desc = &r->desc[r->pop_idx];
        if (!hw_desc->done) {
            if (!wait_one) {
                break;
            }
            while (!hw_desc->done) {
                dmpoll();
            }
        }
        dma_ring_pop_nowait(r);
        id++;
        atomic_store_explicit(&p->idx_done, id, memory_order_release);
        wait_one = false;
        reaped   = true;
    }
    return reaped;
}

static void dma_proxy_thread(void * arg) {
    dma_proxy_t p = (dma_proxy_t) arg;

    FARF(HIGH, "dma-proxy-thread (tid %u): started", p->th.tid);

    const uint32_t max_inflight = dma_queue_capacity(p->th.dma_q) - 1;
    unsigned int   poll_cnt     = DMA_PROXY_POLL_COUNT;
    unsigned int   prev_seqn    = 0;

    while (1) {
        bool did_work = false;

        uint32_t iw = atomic_load_explicit(&p->idx_write, memory_order_acquire);
        #pragma clang loop vectorize(disable) interleave(disable)
        while (p->idx_submit != iw) {
            uint32_t id = atomic_load_explicit(&p->idx_done, memory_order_relaxed);
            if ((p->idx_submit - id) >= max_inflight) {
                dma_proxy_reap_completed(p, true);
            }
            const struct dma_proxy_desc * d = &p->desc[p->idx_submit & p->idx_mask];
            const size_t nrows = (d->nrows != 0 && d->row_size != 0) ? d->nrows : 0;
            dma_queue_push(p->th.dma_q, d->ddata, d->dst_stride, d->src_stride, d->row_size, nrows);
            p->idx_submit++;
            did_work = true;
        }

        if (atomic_load_explicit(&p->idx_done, memory_order_relaxed) != p->idx_submit) {
            if (!dma_proxy_reap_completed(p, false) && !did_work) {
                hex_pause();
            }
            poll_cnt = DMA_PROXY_POLL_COUNT;
            continue;
        }

        if (did_work) {
            poll_cnt = DMA_PROXY_POLL_COUNT;
            continue;
        }

        if (atomic_load_explicit(&p->killed, memory_order_acquire)) {
            break;
        }

        unsigned int seqn = atomic_load_explicit(&p->seqn, memory_order_acquire);
        if (seqn != prev_seqn) {
            prev_seqn = seqn;
            poll_cnt  = DMA_PROXY_POLL_COUNT;
            continue;
        }

        if (--poll_cnt) {
            hex_pause();
            continue;
        }

        FARF(HIGH, "dma-proxy-thread: sleeping");
        qurt_futex_wait(&p->seqn, prev_seqn);
        poll_cnt = DMA_PROXY_POLL_COUNT;
    }

    FARF(HIGH, "dma-proxy-thread (tid %u): stopped", p->th.tid);
}

size_t dma_proxy_sizeof(size_t capacity, uint32_t stack_size) {
    capacity = hex_ceil_pow2(capacity > 0 ? capacity : 1);
    size_t size_p     = hex_align_up(sizeof(struct dma_proxy_s), HEX_L2_LINE_SIZE);
    size_t size_desc  = hex_align_up(capacity * sizeof(struct dma_proxy_desc), HEX_L2_LINE_SIZE);
    size_t size_stack = stack_size;
    return size_p + size_desc + size_stack;
}

size_t dma_proxy_alignof(void) {
    return HEX_L2_LINE_SIZE;
}

dma_proxy_t dma_proxy_init(void * ptr, size_t capacity, uint32_t stack_size, dma_queue_t * dma, struct htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity > 0 ? capacity : 1);
    size_t size_p    = hex_align_up(sizeof(struct dma_proxy_s), HEX_L2_LINE_SIZE);
    size_t size_desc = hex_align_up(capacity * sizeof(struct dma_proxy_desc), HEX_L2_LINE_SIZE);

    uint8_t * block = (uint8_t *) ptr;

    dma_proxy_t p = (dma_proxy_t) block;
    block += size_p;
    memset(p, 0, sizeof(struct dma_proxy_s));

    p->capacity   = (uint32_t) capacity;
    p->idx_mask   = (uint32_t) (capacity - 1);
    p->idx_pop    = 0;
    p->idx_submit = 0;
    atomic_init(&p->idx_write, 0);
    atomic_init(&p->idx_done,  0);
    atomic_init(&p->seqn,      0);
    atomic_init(&p->killed,    false);

    p->desc = (struct dma_proxy_desc *) block;
    block += size_desc;
    memset(p->desc, 0, capacity * sizeof(struct dma_proxy_desc));

    p->stack = block;
    memset(p->stack, 0, stack_size);

    htp_thread_init(&p->th, dma, trace);

    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio < 1) {
        prio = 1;
    }
    if (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, p->stack);
    qurt_thread_attr_set_stack_size(&attr, stack_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "dma-proxy");

    int err = qurt_thread_create(&p->thread, &attr, dma_proxy_thread, p);
    if (err) {
        FARF(ERROR, "dma-proxy: thread create failed (%d)", err);
        return NULL;
    }

    FARF(HIGH, "dma-proxy: capacity %u\n", (unsigned) capacity);

    return p;
}

void dma_proxy_free(dma_proxy_t p) {
    if (!p) {
        return;
    }

    dma_proxy_flush(p);
    atomic_store_explicit(&p->killed, true, memory_order_release);

    unsigned int seqn = atomic_load_explicit(&p->seqn, memory_order_relaxed) + 1;
    atomic_store_explicit(&p->seqn, seqn, memory_order_release);
    qurt_futex_wake(&p->seqn, 1);

    if (p->thread) {
        int status = 0;
        (void) qurt_thread_join(p->thread, &status);
        p->thread = 0;
    }
}
