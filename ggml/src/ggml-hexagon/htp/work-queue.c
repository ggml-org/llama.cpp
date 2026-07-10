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
    work_queue_callback_t func;
    void *                data;
    unsigned int          n_threads;
    atomic_uint           barrier;
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

    atomic_bool        killed;                             // threads need to exit
};

static void work_queue_thread(void * context) {
    worker_context_t * me    = (worker_context_t *) context;
    work_queue_t       queue = me->queue;

    FARF(HIGH, "work-queue: thread %u started", me->id);

    unsigned int prev_seqn = 0;
    unsigned int poll_cnt  = WORK_QUEUE_POLL_COUNT;
    while (!atomic_load_explicit(&queue->killed, memory_order_relaxed)) {
        unsigned int seqn = atomic_load_explicit(&queue->seqn, memory_order_acquire);
        if (seqn == prev_seqn) {
            // drop HVX context while spinning
            if (poll_cnt > 1 && poll_cnt == WORK_QUEUE_POLL_COUNT) {
                qurt_hvx_unlock();
            }
            if (--poll_cnt) {
                hex_pause();
                continue;
            }
            qurt_futex_wait(&queue->seqn, prev_seqn);
            poll_cnt = WORK_QUEUE_POLL_COUNT;
            continue;
        }

        prev_seqn = seqn;
        poll_cnt  = WORK_QUEUE_POLL_COUNT;

        // Process all active tasks in the queue
        unsigned int ir = atomic_load_explicit(&queue->idx_read, memory_order_relaxed);
        unsigned int iw = queue->idx_write;

        while (ir != iw) {
            struct work_queue_task_s * task = &queue->queue[ir];

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

            ir = (ir + 1) & queue->idx_mask;
        }
    }

    FARF(HIGH, "work-queue: thread %u stopped", me->id);
}

bool work_queue_init(work_queue_t * context, uint32_t n_threads) {
    int err = 0;

    if (NULL == context) {
        FARF(ERROR, "NULL context passed to work_queue_init().");
        return false;
    }

    uint32_t stack_size = WORK_QUEUE_THREAD_STACK_SIZE;
    uint32_t n_workers  = n_threads > 1 ? n_threads - 1 : 0;

    // Allocations
    size_t size = (stack_size * n_workers) + (sizeof(struct work_queue_s));

    unsigned char * mem_blob = (unsigned char *) memalign(4096, size);
    if (!mem_blob) {
        FARF(ERROR, "Could not allocate memory for work queue!!");
        return false;
    }

    work_queue_t queue = (work_queue_t) (mem_blob + stack_size * n_workers);

    queue->n_threads = n_threads;

    // initializations
    for (unsigned int i = 0; i < n_workers; i++) {
        queue->stack[i]  = NULL;
        queue->thread[i] = 0;

        queue->context[i].id    = i + 1; // Thread IDs start at 1
        queue->context[i].queue = queue;
    }

    // initialize task queue indices and descriptors
    queue->idx_write = 0;
    atomic_init(&queue->idx_read, 0);
    queue->idx_mask  = WORK_QUEUE_SIZE - 1;
    queue->seqn      = 0;
    queue->killed    = 0;
    for (int i = 0; i < WORK_QUEUE_SIZE; i++) {
        queue->queue[i].func = NULL;
        queue->queue[i].data = NULL;
        queue->queue[i].n_threads = 0;
        atomic_init(&queue->queue[i].barrier, 0);
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (unsigned int i = 0; i < n_workers; i++) {
        // set up stack
        queue->stack[i] = mem_blob; mem_blob += stack_size;
        qurt_thread_attr_set_stack_addr(&attr, queue->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);

        char thread_name[32];
        snprintf(thread_name, sizeof(thread_name), "0x%8x:worker%u", (unsigned int)(uintptr_t)queue, i);
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

        // launch
        err = qurt_thread_create(&queue->thread[i], &attr, work_queue_thread, (void *) &queue->context[i]);
        if (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_release(&queue);
            return false;
        }
    }
    *context = queue;
    return true;
}

// clean up work queue
void work_queue_release(work_queue_t * context) {
    work_queue_t queue = *context;

    // if no work queue exists, return.
    if (NULL == queue) {
        return;
    }

    uint32_t n_workers = queue->n_threads > 1 ? queue->n_threads - 1 : 0;

    atomic_store_explicit(&queue->killed, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&queue->seqn, 1, memory_order_release);
    qurt_futex_wake(&queue->seqn, n_workers);

    // de-initializations
    for (unsigned int i = 0; i < n_workers; i++) {
        if (queue->thread[i]) {
            int status;
            (void) qurt_thread_join(queue->thread[i], &status);
        }
    }

    // free allocated memory (were allocated as a single buffer starting at stack[0])
    if (queue->stack[0]) {
        free(queue->stack[0]);
    }

    *context = NULL;
}

// async run helper
bool work_queue_run_async(work_queue_t context, work_queue_callback_t func, void * data, unsigned int n) {
    work_queue_t queue = context;
    if (NULL == queue) {
        FARF(ERROR, "work-queue: invalid context");
        return false;
    }

    if (n > queue->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, queue->n_threads);
        return false;
    }

    unsigned int ir = atomic_load_explicit(&queue->idx_read, memory_order_relaxed);
    unsigned int iw = queue->idx_write;

    if (((iw + 1) & queue->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        return false;
    }

    struct work_queue_task_s * task = &queue->queue[iw];
    task->func      = func;
    task->data      = data;
    task->n_threads = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);

    queue->idx_write = (iw + 1) & queue->idx_mask;

    uint32_t n_workers = queue->n_threads > 1 ? queue->n_threads - 1 : 0;

    // wake up workers (using memory_order_release to publish idx_write and queue state changes)
    atomic_fetch_add_explicit(&queue->seqn, 1, memory_order_release);
    qurt_futex_wake(&queue->seqn, n_workers);

    // main thread runs job #0
    func(n, 0, data);

    // main thread decrements barrier (using memory_order_release to publish slice 0 writes to spinning idle threads)
    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    while (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    // pop task
    atomic_store_explicit(&queue->idx_read, (ir + 1) & queue->idx_mask, memory_order_relaxed);

    return true;
}
