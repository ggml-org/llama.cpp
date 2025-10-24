#include "worker-pool.h"

#include <qurt.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif

#include "HAP_farf.h"

#define WORKER_THREAD_STACK_SZ  (2 * 16384)
#define LOWEST_USABLE_QURT_PRIO (254)

struct worker_pool_s;

// internal structure kept in thread-local storage per instance of worker pool
typedef struct {
    struct worker_pool_s * pool;
    unsigned int           id;
} worker_context_t;

// internal structure kept in thread-local storage per instance of worker pool
typedef struct worker_pool_s {
    worker_pool_job_t job[MAX_NUM_WORKERS];      // list of job descriptors
    qurt_thread_t     thread[MAX_NUM_WORKERS];   // thread ID's of the workers
    worker_context_t  context[MAX_NUM_WORKERS];  // worker contexts
    void *            stack[MAX_NUM_WORKERS];    // thread stack pointers
    unsigned int      n_threads;                 // number of workers in this pool

    atomic_uint seqn;                            // seqno used to detect new jobs
    atomic_uint next_job;                        // next job index
    atomic_uint n_pending;                       // number of pending jobs
    atomic_uint n_jobs;                          // number of current jobs
    atomic_bool killed;                          // threads need to exit
} worker_pool_t;

static void worker_pool_main(void * context) {
    worker_context_t * me   = (worker_context_t *) context;
    worker_pool_t *    pool = me->pool;

    FARF(HIGH, "worker-pool: thread %u started", me->id);

    unsigned int prev_seqn = 0;
    while (!atomic_load(&pool->killed)) {
        unsigned int seqn = atomic_load(&pool->seqn);
        if (seqn == prev_seqn) {
            // Nothing to do
            qurt_futex_wait(&pool->seqn, prev_seqn);
            continue;
        }

        // New job
        prev_seqn = seqn;

        unsigned int n = atomic_load(&pool->n_jobs);
        unsigned int i = atomic_fetch_add(&pool->next_job, 1);
        if (i >= n) {
            // Spurios wakeup
            continue;
        }

        pool->job[i].func(n, i, pool->job[i].data);

        atomic_fetch_sub(&pool->n_pending, 1);
    }

    FARF(HIGH, "worker-pool: thread %u stopped", me->id);
}

AEEResult worker_pool_init_with_stack_size(worker_pool_context_t * context, uint32_t n_threads, uint32_t stack_size) {
    int err = 0;

    if (NULL == context) {
        FARF(ERROR, "NULL context passed to worker_pool_init().");
        return AEE_EBADPARM;
    }

    // Allocations
    int size = (stack_size * n_threads) + (sizeof(worker_pool_t));

    unsigned char * mem_blob = (unsigned char *) malloc(size);
    if (!mem_blob) {
        FARF(ERROR, "Could not allocate memory for worker pool!!");
        return AEE_ENOMEMORY;
    }

    worker_pool_t * me = (worker_pool_t *) (mem_blob + stack_size * n_threads);

    // name for the first worker, useful in debugging threads
    char name[19];
    snprintf(name, 12, "0x%8x:", (int) me);
    strcat(name, "worker0");
    me->n_threads = n_threads;

    // initializations
    for (unsigned int i = 0; i < me->n_threads; i++) {
        me->stack[i]  = NULL;
        me->thread[i] = 0;

        me->context[i].id   = i;
        me->context[i].pool = me;
    }

    // initialize job queue
    me->n_pending = 0;
    me->n_jobs    = 0;
    me->next_job  = 0;
    me->seqn      = 0;
    me->killed    = 0;

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (unsigned int i = 0; i < me->n_threads; i++) {
        // set up stack
        me->stack[i] = mem_blob;
        mem_blob += stack_size;
        qurt_thread_attr_set_stack_addr(&attr, me->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);

        // set up name
        qurt_thread_attr_set_name(&attr, name);
        name[17] = (name[17] + 1);
        // name threads context:worker0, context:worker1, .. (recycle at 9, but num threads should be less than that anyway)
        if (name[17] > '9') {
            name[17] = '0';
        }

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
        err = qurt_thread_create(&me->thread[i], &attr, worker_pool_main, (void *) &me->context[i]);
        if (err) {
            FARF(ERROR, "Could not launch worker threads!");
            worker_pool_release((worker_pool_context_t *) &me);
            return AEE_EQURTTHREADCREATE;
        }
    }
    *context = (worker_pool_context_t *) me;
    return AEE_SUCCESS;
}

AEEResult worker_pool_init(worker_pool_context_t * context, uint32_t n_threads) {
    return worker_pool_init_with_stack_size(context, n_threads, WORKER_THREAD_STACK_SZ);
}

// clean up worker pool
void worker_pool_release(worker_pool_context_t * context) {
    worker_pool_t * me = (worker_pool_t *) *context;

    // if no worker pool exists, return error.
    if (NULL == me) {
        return;
    }

    atomic_store(&me->killed, 1);
    atomic_fetch_add(&me->seqn, 1);
    qurt_futex_wake(&me->seqn, me->n_threads);

    // de-initializations
    for (unsigned int i = 0; i < me->n_threads; i++) {
        if (me->thread[i]) {
            int status;
            (void) qurt_thread_join(me->thread[i], &status);
        }
    }

    // free allocated memory (were allocated as a single buffer starting at stack[0])
    if (me->stack[0]) {
        free(me->stack[0]);
    }

    *context = NULL;
}

// run jobs
AEEResult worker_pool_run_jobs(worker_pool_context_t context, worker_pool_job_t * job, unsigned int n) {
    worker_pool_t * me = (worker_pool_t *) context;
    if (NULL == me) {
        FARF(ERROR, "worker-pool: invalid context");
        return AEE_EBADPARM;
    }

    if (n > me->n_threads) {
        FARF(ERROR, "worker-pool: invalid number of jobs %u for n-threads %u", n, me->n_threads);
        return AEE_EBADPARM;
    }

    memcpy(me->job, job, sizeof(worker_pool_job_t) * n);

    if (n > 1) {
        atomic_store(&me->next_job, 1);
        atomic_store(&me->n_jobs, n);
        atomic_store(&me->n_pending, n - 1);

        // wake up workers
        atomic_fetch_add(&me->seqn, 1);
        qurt_futex_wake(&me->seqn, n - 1);
    }

    // main thread runs job #0
    me->job[0].func(n, 0, me->job[0].data);

    if (n > 1) {
        while (atomic_load(&me->n_pending))
            ;
    }

    return 0;
}

// run func
AEEResult worker_pool_run_func(worker_pool_context_t context, worker_callback_t func, void * data, unsigned int n) {
    worker_pool_job_t job[n];

    for (unsigned int i = 0; i < n; i++) {
        job[i].func = func;
        job[i].data = data;
    }

    return worker_pool_run_jobs(context, job, n);
}

AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio) {
    worker_pool_t * me = (worker_pool_t *) context;

    // if no worker pool exists, return error.
    if (!me) {
        return AEE_ENOMORE;
    }

    int result = AEE_SUCCESS;
    if (prio < 1) {
        prio = 1;
    }
    if (prio > LOWEST_USABLE_QURT_PRIO) {
        prio = LOWEST_USABLE_QURT_PRIO;
    }

    for (unsigned int i = 0; i < me->n_threads; i++) {
        int res = qurt_thread_set_priority(me->thread[i], (unsigned short) prio);
        if (0 != res) {
            result = AEE_EBADPARM;
            FARF(ERROR, "QURT failed to set priority of thread %d, ERROR = %d", me->thread[i], res);
        }
    }

    return result;
}

AEEResult worker_pool_retrieve_thread_id(worker_pool_context_t context, unsigned int * tids) {
    worker_pool_t * me = (worker_pool_t *) context;
    if (!me) {
        FARF(ERROR, "worker-pool: invalid context");
        return AEE_EBADPARM;
        ;
    }

    for (int i = 0; i < me->n_threads; i++) {
        tids[i] = me->thread[i];
    }

    return AEE_SUCCESS;
}

AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int * prio) {
    worker_pool_t * me = (worker_pool_t *) context;
    if (!me) {
        FARF(ERROR, "worker-pool: invalid context");
        return AEE_EBADPARM;
    }

    int priority = qurt_thread_get_priority(me->thread[0]);
    if (priority > 0) {
        *prio = priority;
        return 0;
    } else {
        *prio = 0;
        return AEE_EBADSTATE;
    }
}
