#ifndef HMX_WORKER_H
#define HMX_WORKER_H

// Async HMX worker: single dedicated thread for HMX compute,
// allowing the main thread to run HVX/DMA work in parallel.
//
// Lifecycle per matmul op:
//   hmx_worker_begin  — worker thread acquires HMX lock
//   hmx_worker_submit — fire a job (non-blocking)
//   hmx_worker_wait   — block until current job completes
//   ...               — repeat submit/wait as needed
//   hmx_worker_end    — worker thread releases HMX lock
//
// Design: single-producer single-consumer, 1 in-flight job max.

#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*hmx_worker_fn_t)(void * data);

typedef struct hmx_worker_context * hmx_worker_context_t;

// Create worker thread.  Thread starts idle (no HMX lock held).
AEEResult hmx_worker_init(hmx_worker_context_t * ctx, uint32_t stack_size, uint32_t vtcm_rctx);

// Destroy worker thread.  Must not be called while a job is in-flight.
void hmx_worker_release(hmx_worker_context_t ctx);

// Worker thread acquires HMX lock.  Blocks until lock is held.
AEEResult hmx_worker_begin(hmx_worker_context_t ctx);

// Submit a job (non-blocking).  Caller must have called wait() for any
// previous job before submitting a new one.
// |data| must remain valid until the corresponding wait() returns.
AEEResult hmx_worker_submit(hmx_worker_context_t ctx, hmx_worker_fn_t fn, void * data);

// Block until the current in-flight job completes.
// Returns immediately if no job is in-flight.
AEEResult hmx_worker_wait(hmx_worker_context_t ctx);

// Ensure no in-flight job, then worker thread releases HMX lock.
// Blocks until unlock is complete.
AEEResult hmx_worker_end(hmx_worker_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif /* HMX_WORKER_H */
