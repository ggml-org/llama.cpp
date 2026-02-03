#ifndef HTP_WORKER_POOL_H
#define HTP_WORKER_POOL_H

// MACRO enables function to be visible in shared-library case.
#define WORKERPOOL_API __attribute__((visibility("default")))

#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// signature of callbacks to be invoked by worker threads
typedef void (*worker_callback_t)(unsigned int n, unsigned int i, void *);

/// Typedef of worker_pool context
typedef void * worker_pool_context_t;

/// descriptor for requested callback
typedef struct {
    worker_callback_t func;
    void *            data;
} worker_pool_job_t;

/// Maximum supported number of worker threads.
#define MAX_NUM_WORKERS 10

// Initialize worker pool.
WORKERPOOL_API AEEResult worker_pool_init(worker_pool_context_t * context, uint32_t n_threads);

// Initialize worker pool with custom stack size
WORKERPOOL_API AEEResult worker_pool_init_with_stack_size(worker_pool_context_t * context,
                                                          uint32_t                n_threads,
                                                          uint32_t                stack_size);

// Kill worker threads and release worker pool resources
WORKERPOOL_API void worker_pool_release(worker_pool_context_t * context);

// Run jobs with the worker pool.
WORKERPOOL_API AEEResult worker_pool_run_jobs(worker_pool_context_t context, worker_pool_job_t * job, unsigned int n);

WORKERPOOL_API AEEResult worker_pool_run_func(worker_pool_context_t context,
                                              worker_callback_t     func,
                                              void *                data,
                                              unsigned int          n);

WORKERPOOL_API AEEResult worker_pool_set_thread_priority(worker_pool_context_t context, unsigned int prio);
WORKERPOOL_API AEEResult worker_pool_get_thread_priority(worker_pool_context_t context, unsigned int * prio);
WORKERPOOL_API AEEResult worker_pool_retrieve_thread_id(worker_pool_context_t context, unsigned int * tids);

#ifdef __cplusplus
}
#endif

#endif  // #ifndef HTP_WORKER_POOL_H
