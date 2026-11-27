#include "work-queue.h"
#include "hex-utils.h"

#include <qurt.h>
#include <qurt_hvx.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

typedef struct {
    htp_thread_t th;
    work_queue_t queue;
} worker_context_t;

struct work_queue_s {
    // Hot dispatch state: written by main thread, read by workers
    // seqn encodes (counter << 8) | (n_threads & 0xff)
    work_queue_func_t func;
    void *            data;
    atomic_uint       seqn;
    atomic_uint       active;
    atomic_bool       killed;

    // Completion barrier: decremented by all workers on every seqn
    atomic_uint       barrier __attribute__((aligned(HEX_L2_LINE_SIZE)));

    // Cold metadata
    uint32_t          last_seqn;
    qurt_thread_t     thread[HTP_MAX_NTHREADS];
    worker_context_t  context[HTP_MAX_NTHREADS];
    void *            stack[HTP_MAX_NTHREADS];
    unsigned int      n_threads;
    unsigned int      n_workers;
};

static void work_queue_thread(void * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;
    unsigned int       id = (unsigned int) (me - q->context);

    FARF(HIGH, "work-queue: thread %u (tid %u) started", id, me->th.tid);

    unsigned int prev_seqn = 0;

    while (!atomic_load_explicit(&q->killed, memory_order_acquire)) {
        unsigned int seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        if (seqn == prev_seqn) {
            if (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } else {
                qurt_hvx_unlock();
                qurt_futex_wait(&q->active, 0);
            }
            continue;
        }

        prev_seqn = seqn;

        unsigned int n = seqn & 0xffu;
        if (id < n) {
            q->func(&me->th, n, id, q->data);
        }

        atomic_fetch_sub_explicit(&q->barrier, 1, memory_order_release);
    }

    FARF(HIGH, "work-queue: thread %u (tid %u) stopped", id, me->th.tid);
}

bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n == 0) {
        return true;
    }

    if (n == 1) {
        func(&q->context[0].th, 1, 0, data);
        return true;
    }

    if (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        return false;
    }

    if (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        work_queue_wakeup(q);
    }

    q->func = func;
    q->data = data;
    atomic_store_explicit(&q->barrier, q->n_workers, memory_order_relaxed);

    // publish job to all workers (every worker decrements barrier before returning)
    atomic_store_explicit(&q->seqn, (++q->last_seqn << 8) | (n & 0xffu), memory_order_release);

    // main thread runs job #0
    func(&q->context[0].th, n, 0, data);

    while (atomic_load_explicit(&q->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    return true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t stack_size) {
    uint32_t n_workers   = n_threads > 1 ? n_threads - 1 : 0;
    size_t   size_stacks = hex_align_up((size_t) stack_size * n_workers, HEX_L2_LINE_SIZE);
    size_t   size_q      = hex_align_up(sizeof(struct work_queue_s), HEX_L2_LINE_SIZE);
    return size_stacks + size_q;
}

size_t work_queue_alignof(void) {
    return 4096;
}

work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t stack_size, dma_queue_t * dma, struct htp_thread_trace * trace) {
    if (n_threads == 0 || n_threads > HTP_MAX_NTHREADS) {
        FARF(ERROR, "work-queue: invalid n_threads %u (max %u)", n_threads, HTP_MAX_NTHREADS);
        return NULL;
    }

    uint32_t        n_workers   = n_threads > 1 ? n_threads - 1 : 0;
    size_t          size_stacks = hex_align_up((size_t) stack_size * n_workers, HEX_L2_LINE_SIZE);
    unsigned char * mem_blob    = (unsigned char *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + size_stacks);
    memset(q, 0, sizeof(struct work_queue_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->last_seqn = 0;

    for (unsigned int i = 0; i < n_threads; i++) {
        htp_thread_init(&q->context[i].th, dma, trace);
        q->context[i].queue = q;
    }

    for (unsigned int i = 0; i < n_workers; i++) {
        q->stack[i]  = mem_blob;
        mem_blob    += stack_size;
        q->thread[i] = 0;
    }

    atomic_init(&q->seqn,    0);
    atomic_init(&q->active,  0);
    atomic_init(&q->killed,  false);
    atomic_init(&q->barrier, 0);

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (unsigned int i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_stack_addr(&attr, q->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);

        char thread_name[32];
        snprintf(thread_name, sizeof(thread_name), "work-queue:%u", i + 1);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        int prio = qurt_thread_get_priority(qurt_thread_get_id());
        if (prio < 1) {
            prio = 1;
        }
        if (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        int err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (void *) &q->context[i + 1]);
        if (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            return NULL;
        }
    }

    return q;
}

void work_queue_free(work_queue_t q) {
    if (!q) {
        return;
    }

    atomic_store_explicit(&q->killed, true, memory_order_release);
    atomic_store_explicit(&q->active, 1,    memory_order_release);
    if (q->n_workers > 0) {
        qurt_futex_wake(&q->active, q->n_workers);
    }

    for (unsigned int i = 0; i < q->n_workers; i++) {
        if (q->thread[i]) {
            int status;
            (void) qurt_thread_join(q->thread[i], &status);
        }
    }
}

void work_queue_wakeup(work_queue_t q) {
    if (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        if (q->n_workers > 0) {
            qurt_futex_wake(&q->active, q->n_workers);
        }
    }
}

void work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}
