// HMX worker pool header.
// Ported from htp-ops-lib/include/dsp/worker_pool.h.
//
// This is a separate worker pool used exclusively by HMX operations.
// It coexists with the existing hexagon worker-pool.h which serves HVX ops.
// Key differences from the hexagon worker pool:
//   - Callback signature: void (*)(void *data, int worker_index)
//   - Submit/synctoken API for fan-out parallel dispatch
//   - No auto-init (constructor/destructor removed)

#ifndef HMX_WORKER_POOL_H
#define HMX_WORKER_POOL_H

#include <AEEStdDef.h>
#include <AEEStdErr.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Types ----------------------------------------------------------------

// Callback signature: receives data pointer and worker index.
typedef void (*hmx_worker_callback_t)(void *data, int worker_index);

// Opaque pool context.
typedef void *hmx_worker_pool_context_t;

// Job descriptor.
typedef struct {
    hmx_worker_callback_t fptr;
    void                 *dptr;
} hmx_worker_pool_job_t;

// Opaque synchronisation token (allocated on caller stack).
typedef struct {
    unsigned int dummy[8]; // large enough for counter + semaphore
} hmx_worker_synctoken_t __attribute__((aligned(8)));

// --- Constants ------------------------------------------------------------

#define HMX_MAX_NUM_WORKERS 6

// --- API ------------------------------------------------------------------

// Initialise worker pool with explicit parameters.
//   stack_size  – per-worker stack size in bytes
//   n_workers   – number of worker threads
//   allow_hmx   – reserved (unused, kept for API compatibility)
AEEResult hmx_worker_pool_init_ex(hmx_worker_pool_context_t *context,
                                  int stack_size,
                                  int n_workers,
                                  int allow_hmx);

// Tear down worker pool and release resources.
void hmx_worker_pool_deinit(hmx_worker_pool_context_t *context);

// Submit a single job to the pool.
AEEResult hmx_worker_pool_submit(hmx_worker_pool_context_t context,
                                 hmx_worker_pool_job_t     job);

// Synctoken: initialise for n jobs.
void hmx_worker_pool_synctoken_init(hmx_worker_synctoken_t *token, unsigned int njobs);
// Worker calls this when done.
void hmx_worker_pool_synctoken_jobdone(hmx_worker_synctoken_t *token);
// Submitter waits for all jobs.
void hmx_worker_pool_synctoken_wait(hmx_worker_synctoken_t *token);

// Thread priority control.
AEEResult hmx_worker_pool_set_thread_priority(hmx_worker_pool_context_t context, unsigned int prio);

// --- Inline atomics -------------------------------------------------------

static inline unsigned int hmx_worker_pool_atomic_inc_return(unsigned int *target) {
    unsigned int result;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%2)\n"
        "       %0 = add(%0, #1)\n"
        "       memw_locked(%2, p0) = %0\n"
        "       if !p0 jump 1b\n"
        : "=&r"(result), "+m"(*target)
        : "r"(target)
        : "p0");
    return result;
}

static inline unsigned int hmx_worker_pool_atomic_dec_return(unsigned int *target) {
    unsigned int result;
    __asm__ __volatile__(
        "1:     %0 = memw_locked(%2)\n"
        "       %0 = add(%0, #-1)\n"
        "       memw_locked(%2, p0) = %0\n"
        "       if !p0 jump 1b\n"
        : "=&r"(result), "+m"(*target)
        : "r"(target)
        : "p0");
    return result;
}

#ifdef __cplusplus
}
#endif

// Convenience macros for task-state boilerplate (same as htp-ops-lib).
#define HMX_EXPAND_COMMON_TASK_STATE_MEMBERS \
    hmx_worker_synctoken_t sync_ctx;         \
    unsigned int           task_id;          \
    int                    n_tasks;          \
    int                    n_tot_chunks;     \
    int                    n_chunks_per_task;

#define HMX_INIT_COMMON_TASK_STATE_MEMBERS(state, tot_chunks, chunks_per_task)               \
    do {                                                                                     \
        state.task_id           = 0;                                                         \
        state.n_tasks           = (tot_chunks + chunks_per_task - 1) / chunks_per_task;      \
        state.n_tot_chunks      = tot_chunks;                                                \
        state.n_chunks_per_task = chunks_per_task;                                           \
    } while (0)

#endif // HMX_WORKER_POOL_H
