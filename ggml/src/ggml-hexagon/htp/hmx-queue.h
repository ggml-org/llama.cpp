#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#include <hexagon_types.h>
#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>

#include "hex-utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HMX_QUEUE_THREAD_STACK_SIZE (16 * 1024)
#define HMX_QUEUE_POLL_COUNT        2000

typedef void (*hmx_queue_func)(void *);

// Dummy funcs used as signals
enum hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
};

struct hmx_queue {
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned int     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
};

struct hmx_queue * hmx_queue_create(size_t capacity, uint32_t hap_rctx);
void hmx_queue_delete(struct hmx_queue * q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(struct hmx_queue * q, struct hmx_queue_desc d) {
    unsigned int ir = atomic_load(&q->idx_read);
    unsigned int iw = q->idx_write;

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        return false;
    }

    atomic_store(&d.done, 0);

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    return true;
}

static inline bool hmx_queue_signal(struct hmx_queue *q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));
}

static inline bool hmx_queue_empty(struct hmx_queue * q) {
    return q->idx_pop == q->idx_write;
}

static inline uint32_t hmx_queue_depth(struct hmx_queue * q) {
    return (q->idx_read - q->idx_read) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(struct hmx_queue * q) {
    return q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop(struct hmx_queue * q) {
    unsigned int ip = q->idx_pop;
    unsigned int iw = q->idx_write;

    struct hmx_queue_desc rd = { NULL, NULL };
    if (ip == iw) {
        return rd;
    }

    // Wait for desc to complete
    struct hmx_queue_desc * d = &q->desc[ip];
    while (!atomic_load(&d->done)) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    return rd;
}

static inline void hmx_queue_flush(struct hmx_queue * q) {
    while (hmx_queue_pop(q).func != NULL) ;
}

static inline void hmx_queue_suspend(struct hmx_queue *q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
    hmx_queue_flush(q);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
