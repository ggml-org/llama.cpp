// HMX worker pool implementation.
// Ported from htp-ops-lib/src/dsp/worker_pool.c.
//
// Changes from the original:
//   - All symbols prefixed with hmx_ to coexist with hexagon worker-pool.c
//   - constructor / destructor removed — pool is created explicitly by
//     hmx_manager_setup() and destroyed by hmx_manager_reset()
//   - Static default context removed (not needed)
//   - worker_pool_init() (no-arg) removed
//   - Unused worker_pool_get_thread_priority / worker_pool_available removed

#include "hmx-worker-pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _DEBUG
#  define _DEBUG
#endif
#include "HAP_farf.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "hexagon_protos.h"
#include "qurt.h"
#ifdef __cplusplus
}
#endif

// ---------------------------------------------------------------------------
// Internal definitions
// ---------------------------------------------------------------------------

#define HMX_WORKER_KILL_SIGNAL 31
#define HMX_NUM_JOB_SLOTS     (HMX_MAX_NUM_WORKERS + 1)
#define HMX_LOWEST_PRIO       254

typedef struct {
    qurt_anysignal_t      empty_jobs;
    qurt_anysignal_t      queued_jobs;
    qurt_mutex_t          empty_jobs_mutex;
    qurt_mutex_t          queued_jobs_mutex;
    unsigned int          job_queue_mask;
    unsigned int          num_workers;
    hmx_worker_pool_job_t job[HMX_NUM_JOB_SLOTS];
    qurt_thread_t         thread[HMX_MAX_NUM_WORKERS];
    void                 *stack[HMX_MAX_NUM_WORKERS];
} hmx_pool_t;

typedef union {
    hmx_worker_synctoken_t raw;
    struct {
        unsigned int atomic_countdown;
        unsigned int reserved;
        qurt_sem_t   sem;
    } sync;
} hmx_internal_synctoken_t;

typedef struct {
    hmx_pool_t *pool;
    int          worker_index;
    int          allow_hmx;
} hmx_worker_info_t;

// ---------------------------------------------------------------------------
// Worker main loop
// ---------------------------------------------------------------------------

static void hmx_worker_pool_main(void *context) {
    hmx_worker_info_t *info = (hmx_worker_info_t *)context;
    hmx_pool_t        *me   = info->pool;

    qurt_anysignal_t *signal = &me->queued_jobs;
    unsigned int       mask  = me->job_queue_mask;
    qurt_mutex_t      *mutex = &me->queued_jobs_mutex;

    while (1) {
        qurt_mutex_lock(mutex);
        (void)qurt_anysignal_wait(signal, mask);
        unsigned int sig_rx = Q6_R_ct0_R(mask & qurt_anysignal_get(signal));

        if (sig_rx < HMX_NUM_JOB_SLOTS) {
            hmx_worker_pool_job_t job = me->job[sig_rx];
            (void)qurt_anysignal_clear(signal, (1 << sig_rx));
            (void)qurt_anysignal_set(&me->empty_jobs, (1 << sig_rx));
            qurt_mutex_unlock(mutex);
            job.fptr(job.dptr, info->worker_index);
        } else if (HMX_WORKER_KILL_SIGNAL == sig_rx) {
            qurt_mutex_unlock(mutex);
            break;
        } else {
            FARF(HIGH, "HMX worker pool: invalid job %d", sig_rx);
            qurt_mutex_unlock(mutex);
        }
    }

    qurt_thread_exit(0);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

AEEResult hmx_worker_pool_init_ex(hmx_worker_pool_context_t *context,
                                  int stack_size,
                                  int n_workers,
                                  int allow_hmx) {
    if (stack_size <= 0 || !context) {
        return AEE_EBADPARM;
    }

    size_t size = (stack_size * n_workers)
                + sizeof(hmx_pool_t)
                + sizeof(hmx_worker_info_t) * n_workers;

    unsigned char *mem_blob = (unsigned char *)malloc(size);
    if (!mem_blob) {
        FARF(ERROR, "hmx_worker_pool: alloc failed");
        return AEE_ENOMEMORY;
    }

    hmx_pool_t        *me   = (hmx_pool_t *)(mem_blob + stack_size * n_workers);
    hmx_worker_info_t *info = (hmx_worker_info_t *)(mem_blob + stack_size * n_workers + sizeof(hmx_pool_t));

    char name[19];
    snprintf(name, 12, "%08x:", (int)(uintptr_t)me);
    strcat(name, "hm0");
    me->num_workers = n_workers;

    for (int i = 0; i < n_workers; i++) {
        me->stack[i]  = NULL;
        me->thread[i] = 0;
    }

    qurt_anysignal_init(&me->queued_jobs);
    qurt_anysignal_init(&me->empty_jobs);
    qurt_mutex_init(&me->empty_jobs_mutex);
    qurt_mutex_init(&me->queued_jobs_mutex);
    me->job_queue_mask = (1 << HMX_NUM_JOB_SLOTS) - 1;
    (void)qurt_anysignal_set(&me->empty_jobs, me->job_queue_mask);
    me->job_queue_mask |= (1 << HMX_WORKER_KILL_SIGNAL);

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (int i = 0; i < n_workers; i++) {
        me->stack[i] = mem_blob;
        mem_blob += stack_size;
        qurt_thread_attr_set_stack_addr(&attr, me->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);
        qurt_thread_attr_set_name(&attr, name);
        name[11] = (char)(name[11] + 1);
        if (name[11] > '9') name[11] = '0';

        int prio = qurt_thread_get_priority(qurt_thread_get_id());
        if (prio < 1)               prio = 1;
        if (prio > HMX_LOWEST_PRIO) prio = HMX_LOWEST_PRIO;
        qurt_thread_attr_set_priority(&attr, prio);

        info[i].pool         = me;
        info[i].worker_index = i;
        info[i].allow_hmx    = allow_hmx;

        int err = qurt_thread_create(&me->thread[i], &attr, hmx_worker_pool_main, (void *)&info[i]);
        if (err) {
            FARF(ERROR, "hmx_worker_pool: thread create failed");
            hmx_worker_pool_deinit((hmx_worker_pool_context_t *)&me);
            return AEE_EQURTTHREADCREATE;
        }
    }

    *context = (hmx_worker_pool_context_t)me;
    return AEE_SUCCESS;
}

void hmx_worker_pool_deinit(hmx_worker_pool_context_t *context) {
    hmx_pool_t *me = (hmx_pool_t *)*context;
    if (!me) return;

    (void)qurt_anysignal_set(&me->empty_jobs,  (1 << HMX_WORKER_KILL_SIGNAL));
    (void)qurt_anysignal_set(&me->queued_jobs, (1 << HMX_WORKER_KILL_SIGNAL));

    for (unsigned int i = 0; i < me->num_workers; i++) {
        if (me->thread[i]) {
            int status;
            (void)qurt_thread_join(me->thread[i], &status);
        }
    }

    qurt_mutex_destroy(&me->empty_jobs_mutex);
    qurt_mutex_destroy(&me->queued_jobs_mutex);
    qurt_anysignal_destroy(&me->queued_jobs);
    qurt_anysignal_destroy(&me->empty_jobs);

    if (me->stack[0]) {
        free(me->stack[0]);
    }
    *context = NULL;
}

AEEResult hmx_worker_pool_submit(hmx_worker_pool_context_t context,
                                 hmx_worker_pool_job_t     job) {
    hmx_pool_t *me = (hmx_pool_t *)context;
    if (!me) return AEE_EBADPARM;

    // Recursive submit guard: if a worker thread calls submit, run inline.
    qurt_thread_t id = qurt_thread_get_id();
    for (unsigned int i = 0; i < me->num_workers; i++) {
        if (id == me->thread[i]) {
            job.fptr(job.dptr, i);
            return AEE_SUCCESS;
        }
    }

    qurt_mutex_t     *mutex  = &me->empty_jobs_mutex;
    qurt_anysignal_t *signal = &me->empty_jobs;
    unsigned int       mask  = me->job_queue_mask;

    qurt_mutex_lock(mutex);
    (void)qurt_anysignal_wait(signal, mask);
    unsigned int bitfield = qurt_anysignal_get(signal);

    if (bitfield & (1 << HMX_WORKER_KILL_SIGNAL)) {
        qurt_mutex_unlock(mutex);
        return AEE_ENOMORE;
    }

    unsigned int sig_rx = Q6_R_ct0_R(mask & bitfield);
    me->job[sig_rx]     = job;
    (void)qurt_anysignal_clear(signal, (1 << sig_rx));
    (void)qurt_anysignal_set(&me->queued_jobs, (1 << sig_rx));
    qurt_mutex_unlock(mutex);

    return AEE_SUCCESS;
}

// ---------------------------------------------------------------------------
// Sync token
// ---------------------------------------------------------------------------

void hmx_worker_pool_synctoken_init(hmx_worker_synctoken_t *token, unsigned int njobs) {
    hmx_internal_synctoken_t *internal = (hmx_internal_synctoken_t *)token;
    internal->sync.atomic_countdown = njobs;
    qurt_sem_init_val(&internal->sync.sem, 0);
}

void hmx_worker_pool_synctoken_jobdone(hmx_worker_synctoken_t *token) {
    hmx_internal_synctoken_t *internal = (hmx_internal_synctoken_t *)token;
    if (0 == hmx_worker_pool_atomic_dec_return(&internal->sync.atomic_countdown)) {
        (void)qurt_sem_up(&internal->sync.sem);
    }
}

void hmx_worker_pool_synctoken_wait(hmx_worker_synctoken_t *token) {
    hmx_internal_synctoken_t *internal = (hmx_internal_synctoken_t *)token;
    (void)qurt_sem_down(&internal->sync.sem);
    (void)qurt_sem_destroy(&internal->sync.sem);
}

AEEResult hmx_worker_pool_set_thread_priority(hmx_worker_pool_context_t context, unsigned int prio) {
    hmx_pool_t *me = (hmx_pool_t *)context;
    if (!me) return AEE_ENOMORE;

    if (prio < 1)               prio = 1;
    if (prio > HMX_LOWEST_PRIO) prio = HMX_LOWEST_PRIO;

    int result = AEE_SUCCESS;
    for (unsigned int i = 0; i < me->num_workers; i++) {
        int res = qurt_thread_set_priority(me->thread[i], (unsigned short)prio);
        if (res != 0) {
            result = AEE_EBADPARM;
            FARF(ERROR, "hmx_worker_pool: set priority failed for thread %d", me->thread[i]);
        }
    }
    return result;
}
