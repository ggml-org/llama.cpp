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
    _Atomic(work_queue_func_t) func;
    _Atomic(void *)           data;
    atomic_uint               n_threads;
    atomic_uint               barrier;
};

// internal structure kept in thread-local storage per instance of work queue
struct work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        n_pub;     // task publication count, separate from wakeups
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    unsigned int       idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[WORK_QUEUE_MAX_N_THREADS];   // thread ID's of the workers
    worker_context_t   context[WORK_QUEUE_MAX_N_THREADS];  // worker contexts
    void *             stack[WORK_QUEUE_MAX_N_THREADS];    // thread stack pointers
    unsigned int       n_threads;                          // total threads (workers + main)
    unsigned int       n_workers;                          // number of active threads (just workers)

    atomic_bool        active;                             // workers are polling/active
    atomic_bool        killed;                             // threads need to exit
    bool               external_mem;                       // memory owned externally

    struct work_queue_task_s queue[] __attribute__((aligned(HEX_L2_LINE_SIZE)));
};

static void work_queue_thread(void * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;

    FARF(HIGH, "work-queue: thread %u started", me->id);

    unsigned int prev_seqn = 0;
    unsigned int last_pub  = 0;  // number of the last task this worker has handled

    while (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned int seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        if (seqn == prev_seqn) {
            if (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } else {
                qurt_futex_wait(&q->seqn, prev_seqn);
            }
            continue;
        }

        prev_seqn = seqn;

        // Tasks finish before the next is published, so only the latest task can need this worker.
        // Track publications separately from wakeups to avoid executing a task twice.
        unsigned int pub = atomic_load_explicit(&q->n_pub, memory_order_acquire);
        if (pub == last_pub) {
            continue;
        }

        struct work_queue_task_s * task = &q->queue[(pub - 1) & q->idx_mask];

        work_queue_func_t func = atomic_load_explicit(&task->func, memory_order_relaxed);
        void *            data = atomic_load_explicit(&task->data, memory_order_relaxed);
        unsigned int      n    = atomic_load_explicit(&task->n_threads, memory_order_relaxed);

        // A nonparticipating worker may race with slot reuse. Retry if this snapshot became stale.
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&q->n_pub, memory_order_relaxed) != pub) {
            continue;
        }

        last_pub = pub;

        unsigned int i = me->id;
        if (i < n) {
            func(n, i, data);

            atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
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

    // task number pub lives in queue[(pub - 1) & idx_mask] == queue[iw] (both counters advance once per task)
    unsigned int pub = atomic_load_explicit(&q->n_pub, memory_order_relaxed) + 1;

    struct work_queue_task_s * task = &q->queue[(pub - 1) & q->idx_mask];

    // Order the previous publication before reusing a slot so a reader of rewritten fields detects it when rechecking n_pub.
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&task->func,      func, memory_order_relaxed);
    atomic_store_explicit(&task->data,      data, memory_order_relaxed);
    atomic_store_explicit(&task->n_threads, n,    memory_order_relaxed);
    atomic_store_explicit(&task->barrier,   n,    memory_order_relaxed);

    q->idx_write = (iw + 1) & q->idx_mask;

    // publish the task (release: its fields and barrier are visible to a worker that reads this n_pub)
    atomic_store_explicit(&q->n_pub, pub, memory_order_release);

    // wake up polling workers
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);

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

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = stack_size * n_workers;
    size_t size_q = hex_align_up(sizeof(struct work_queue_s) + capacity * sizeof(struct work_queue_task_s), HEX_L2_LINE_SIZE);
    return size_stacks + size_q;
}

size_t work_queue_alignof(void) {
    return 4096;
}

work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned char * mem_blob = (unsigned char *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + stack_size * n_workers);
    memset(q, 0, sizeof(struct work_queue_s) + capacity * sizeof(struct work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    for (unsigned int i = 0; i < n_workers; i++) {
        q->stack[i]  = mem_blob; mem_blob += stack_size;
        q->thread[i] = 0;
        q->context[i].id    = i + 1;
        q->context[i].queue = q;
    }

    atomic_init(&q->idx_read, 0);
    atomic_init(&q->seqn,     0);
    atomic_init(&q->active,   false);
    atomic_init(&q->n_pub,    0);
    q->idx_write = 0;
    q->idx_mask  = capacity - 1;
    q->killed    = 0;
    for (int i = 0; i < (int) capacity; i++) {
        atomic_init(&q->queue[i].barrier,   0);
        atomic_init(&q->queue[i].func,      NULL);
        atomic_init(&q->queue[i].data,      NULL);
        atomic_init(&q->queue[i].n_threads, 0);
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
            work_queue_free(q);
            return NULL;
        }
    }

    return q;
}

void work_queue_free(work_queue_t q) {
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
}

void work_queue_wakeup(work_queue_t q) {
    if (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, true, memory_order_release);
        // Increment seqn and wake workers to transition them out of sleep
        atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
        qurt_futex_wake(&q->seqn, q->n_workers);
    }
}

void work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, false, memory_order_release);
}
