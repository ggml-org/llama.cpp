#pragma clang diagnostic ignored "-Wunused-function"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <qurt_thread.h>
#include <qurt_futex.h>

#include <HAP_compute_res.h>

#include "hmx-queue.h"

#define QURT_LOWEST_PRIO (254)

static inline void hmx_lock(struct hmx_queue *q)
{
    if (!q->hmx_locked) {
        HAP_compute_res_hmx_lock(q->hap_rctx);
        q->hmx_locked = true;
    }
}

static inline void hmx_unlock(struct hmx_queue *q)
{
    if (q->hmx_locked) {
        HAP_compute_res_hmx_unlock(q->hap_rctx);
        q->hmx_locked = false;
    }
}

static inline void hmx_queue_process(struct hmx_queue *q, bool* killed) {
    unsigned int ir = atomic_load(&q->idx_read);

    while (ir != atomic_load(&q->idx_write)) {
        struct hmx_queue_desc *d = &q->desc[ir];
        if (!d->done) {
            FARF(HIGH, "hmx-queue-process: ir %u func %p data %p", ir, d->func, d->data);

            enum hmx_queue_signal sig = (enum hmx_queue_signal) (unsigned int) d->func;
            switch (sig) {
                case HMX_QUEUE_NOOP:    /* noop */;     break;
                case HMX_QUEUE_KILL:    *killed = true; break;
                case HMX_QUEUE_SUSPEND: hmx_unlock(q);  break;
                default:
                    hmx_lock(q);
                    d->func(d->data);
                    break;
            }

            atomic_fetch_add(&d->done, 1);
        }

        ir = (ir + 1) & q->idx_mask;
        atomic_store(&q->idx_read, ir);
    }
}

static void hmx_queue_thread(void * arg) {
    struct hmx_queue * q = (struct hmx_queue *) arg;

    FARF(HIGH, "hmx-queue-thread: started");

    bool killed = false;

    unsigned int poll_cnt  = HMX_QUEUE_POLL_COUNT;
    unsigned int prev_seqn = 0;
    while (!killed) {
        unsigned int seqn = atomic_load(&q->seqn);
        if (seqn == prev_seqn) {
            if (--poll_cnt) { hex_pause(); continue; }
            FARF(HIGH, "hmx-queue-thread: sleeping");
            qurt_futex_wait(&q->seqn, prev_seqn);
            continue;
        }
        prev_seqn = seqn;
        poll_cnt  = HMX_QUEUE_POLL_COUNT;

        FARF(HIGH, "hmx-queue-thread: new work");

        hmx_queue_process(q, &killed);
    }

    FARF(HIGH, "hmx-queue-thread: stopped");
}

struct hmx_queue * hmx_queue_create(size_t capacity, uint32_t hap_rctx) {
    capacity = hex_ceil_pow2(capacity);

    struct hmx_queue * q = (struct hmx_queue *) memalign(32, sizeof(struct hmx_queue));
    if (q == NULL) {
        FARF(ERROR, "%s: failed to allocate DMA queue\n", __FUNCTION__);
        return NULL;
    }
    memset(q, 0, sizeof(struct hmx_queue));
    q->capacity = capacity;
    q->idx_mask = capacity - 1;
    q->hap_rctx = hap_rctx;

    q->desc = (struct hmx_queue_desc *) memalign(64, capacity * sizeof(struct hmx_queue_desc));
    if (!q->desc) {
        FARF(ERROR, "hmx-queue: failed to allocate HMX queue descriptors\n");
        return NULL;
    }
    memset(q->desc, 0, capacity * sizeof(struct hmx_queue_desc));

    const size_t stack_size = HMX_QUEUE_THREAD_STACK_SIZE;
    q->stack = (unsigned char *) memalign(64, stack_size);
    if (!q->stack) {
        FARF(ERROR, "hmx-queue: thread stack allocation failed (%zu bytes)", stack_size);
        return NULL;
    }
    memset(q->stack, 0, stack_size);

    // Match caller thread priority (same pattern as worker-pool.c).
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio < 1) {
        prio = 1;
    }
    if (prio > QURT_LOWEST_PRIO) {
        prio = QURT_LOWEST_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, q->stack);
    qurt_thread_attr_set_stack_size(&attr, stack_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx-queue");

    int err = qurt_thread_create(&q->thread, &attr, hmx_queue_thread, q);
    if (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        return NULL;
    }

    FARF(HIGH, "hmx-queue: capacity %u\n", capacity);

    return q;
}

void hmx_queue_delete(struct hmx_queue * q) {
    if (!q) {
        return;
    }

    // Tell the worker to exit.
    hmx_queue_flush(q);
    hmx_queue_signal(q, HMX_QUEUE_KILL);
    hmx_queue_flush(q);

    int status;
    qurt_thread_join(q->thread, &status);

    free(q->desc);
    free(q->stack);
    free(q);
}
