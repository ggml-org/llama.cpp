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
#include "htp-trace.h"
#include "htp-thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_COUNT 2000
#else
#define HMX_QUEUE_POLL_COUNT 1
#endif

typedef void (*hmx_queue_func)(const htp_thread_t * th, void *);

// Dummy funcs used as signals (non-zero so NULL remains the empty sentinel)
enum hmx_queue_signal {
    HMX_QUEUE_WAKEUP  = 1,
    HMX_QUEUE_SUSPEND = 2,
    HMX_QUEUE_KILL    = 3,
};

struct hmx_queue_desc {
    hmx_queue_func func;
    void *         data;
};

struct hmx_queue_s {
    struct hmx_queue_desc * desc;
    atomic_uint             idx_write; // monotonic, updated by producer (push)
    uint32_t                idx_pop;   // monotonic, updated by producer (pop)
    uint32_t                idx_mask;
    uint32_t                capacity;
    atomic_uint             seqn;      // incremented for all pushes, used with futex

    atomic_uint             idx_read __attribute__((aligned(HEX_L2_LINE_SIZE))); // monotonic, updated by consumer
    bool                    hmx_locked;

    qurt_thread_t           thread;
    htp_thread_t            th;
    void *                  stack;
    uint32_t                hap_rctx;
};

typedef struct hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t stack_size);
size_t      hmx_queue_alignof(void);
hmx_queue_t hmx_queue_init(void * ptr, size_t capacity, uint32_t stack_size, uint32_t hap_rctx, dma_queue_t * dma, struct htp_thread_trace * trace);
void        hmx_queue_free(hmx_queue_t q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    uint32_t iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    if ((iw - q->idx_pop) >= q->capacity) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        return false;
    }

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw & q->idx_mask] = d;
    atomic_store_explicit(&q->idx_write, iw + 1, memory_order_release);

    // wake up our thread
    unsigned int seqn = atomic_load_explicit(&q->seqn, memory_order_relaxed) + 1;
    atomic_store_explicit(&q->seqn, seqn, memory_order_release);
    qurt_futex_wake(&q->seqn, 1);

    return true;
}

static inline bool hmx_queue_signal(hmx_queue_t q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) (uintptr_t) sig, NULL));
}

static inline bool hmx_queue_empty(hmx_queue_t q) {
    return q->idx_pop == atomic_load_explicit(&q->idx_write, memory_order_relaxed);
}

static inline uint32_t hmx_queue_depth(hmx_queue_t q) {
    return atomic_load_explicit(&q->idx_write, memory_order_relaxed) -
           atomic_load_explicit(&q->idx_read, memory_order_relaxed);
}

static inline uint32_t hmx_queue_capacity(hmx_queue_t q) {
    return q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop_one(hmx_queue_t q) {
    uint32_t ip = q->idx_pop;
    uint32_t iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    struct hmx_queue_desc rd = { NULL, NULL };
    if (ip == iw) {
        return rd;
    }

    // Wait for consumer to finish slot ip
    while ((int32_t) (atomic_load_explicit(&q->idx_read, memory_order_relaxed) - ip) <= 0) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }
    atomic_thread_fence(memory_order_acquire);

    rd = q->desc[ip & q->idx_mask];
    q->idx_pop = ip + 1;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    return rd;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    while (1) {
        struct hmx_queue_desc d = hmx_queue_pop_one(q);

        uintptr_t sig = (uintptr_t) d.func;
        if (sig && sig <= HMX_QUEUE_KILL) {
            continue;
        }

        return d;
    }
}

static inline void hmx_queue_flush(hmx_queue_t q) {
    while (hmx_queue_pop_one(q).func != NULL) ;
}

static inline void hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

static inline void hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
