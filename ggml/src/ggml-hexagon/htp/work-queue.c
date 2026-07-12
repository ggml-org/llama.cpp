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

// internal structure kept in thread-local storage per instance of work queue
typedef struct {
    work_queue_t  queue;
    unsigned int  id;
} worker_context_t;

struct work_queue_task_s {
    work_queue_func_t func;
    void *            data;
    unsigned int      n_threads;
    atomic_uint       barrier;
};

// internal structure kept in thread-local storage per instance of work queue
struct work_queue_s {
    struct work_queue_task_s queue[WORK_QUEUE_SIZE];

    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    unsigned int       idx_write; // Updated by producer (push)
    uint32_t           idx_mask;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    void *             stack[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned int       n_threads;                          // total threads (workers + main)
    unsigned int       n_workers;                          // number of active threads (just workers)

    atomic_bool        killed;                             // threads need to exit
};

static void work_queue_thread(void * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(HIGH, "work-queue: thread %u started", me->id);

    unsigned int prev_seqn = 0;
    unsigned int poll_cnt  = WORK_QUEUE_POLL_COUNT;

    while (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned int seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        if (seqn == prev_seqn) {
            if (--poll_cnt) {
                hex_pause();
                continue;
            }
            qurt_futex_wait(&q->seqn, prev_seqn);
            poll_cnt = WORK_QUEUE_POLL_COUNT;
            continue;
        }

        prev_seqn = seqn;
        poll_cnt  = WORK_QUEUE_POLL_COUNT;

        // Process all active tasks in the queue
        unsigned int ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
        unsigned int iw = q->idx_write;

        while (ir != iw) {
            struct work_queue_task_s * task = &q->queue[ir];

            unsigned int n = task->n_threads;
            unsigned int i = me->id;
            if (i < n) {
                task->func(n, i, task->data);

                atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
            } else {
                while (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
                    hex_pause();
                }
            }

            ir = (ir + 1) & q->idx_mask;
        }
    }

    FARF(HIGH, "work-queue: thread %u stopped", me->id);
}

bool work_queue_run_async(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        return false;
    }

    unsigned int ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
    unsigned int iw = q->idx_write;

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        return false;
    }

    struct work_queue_task_s * task = &q->queue[iw];
    task->func      = func;
    task->data      = data;
    task->n_threads = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    q->idx_write = (iw + 1) & q->idx_mask;

    // wake up workers
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->seqn, q->n_workers);

    // main thread runs job #0
    func(n, 0, data);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    while (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    return true;
}

work_queue_t work_queue_create(uint32_t n_threads) {
    uint32_t stack_size = WORK_QUEUE_THREAD_STACK_SIZE;
    uint32_t n_workers  = n_threads > 1 ? n_threads - 1 : 0;

    // Allocations
    size_t size = (stack_size * n_workers) + (sizeof(struct work_queue_s));

    unsigned char * mem_blob = (unsigned char *) memalign(4096, size);
    if (!mem_blob) {
        FARF(ERROR, "Could not allocate memory for work queue!!");
        return NULL;
    }

    work_queue_t q = (work_queue_t) (mem_blob + stack_size * n_workers);

    q->n_threads = n_threads;
    q->n_workers = n_workers;

    for (unsigned int i = 0; i < n_workers; i++) {
        q->stack[i]  = mem_blob; mem_blob += stack_size;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].queue = q;
    }

    atomic_init(&q->idx_read, 0);
    atomic_init(&q->seqn,     0);
    q->idx_write = 0;
    q->idx_mask  = WORK_QUEUE_SIZE - 1;
    q->killed    = 0;
    for (int i = 0; i < WORK_QUEUE_SIZE; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        q->queue[i].func      = NULL;
        q->queue[i].data      = NULL;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (unsigned int i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_stack_addr(&attr, q->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);

        char thread_name[32];
        snprintf(thread_name, sizeof(thread_name), "work-queue:%u", i);
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

        int err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (void *) &q->context[i]);
        if (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_delete(q);
            return NULL;
        }
    }

    return q;
}

void work_queue_delete(work_queue_t q) {
    if (!q) { return; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->seqn, q->n_workers);

    for (unsigned int i = 0; i < q->n_workers; i++) {
        if (q->thread[i]) {
            int status;
            (void) qurt_thread_join(q->thread[i], &status);
        }
    }

    // free allocated memory (were allocated as a single buffer starting at stack[0])
    if (q->stack[0]) {
        free(q->stack[0]);
    }
}
