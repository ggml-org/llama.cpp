#include "hmx-worker.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <qurt.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

enum hmx_worker_cmd {
    HMX_WORKER_CMD_BEGIN,  // acquire HMX lock
    HMX_WORKER_CMD_JOB,    // execute fn(data)
    HMX_WORKER_CMD_END,    // release HMX lock
    HMX_WORKER_CMD_KILL,   // exit thread
};

struct hmx_worker_context {
    // Command channel: main thread → worker
    atomic_uint         cmd_seqn;  // bumped by main thread for each command
    enum hmx_worker_cmd cmd_type;
    hmx_worker_fn_t     fn;
    void *              data;

    // Completion channel: worker → main thread
    atomic_uint done_seqn;  // set to cmd_seqn when command completes

    // Configuration
    uint32_t vtcm_rctx;

    // Thread resources
    qurt_thread_t thread;
    void *        stack;  // single allocation: stack + context
};

// ---------------------------------------------------------------------------
// Worker thread entry point
// ---------------------------------------------------------------------------

static void hmx_worker_main(void * arg) {
    struct hmx_worker_context * ctx = (struct hmx_worker_context *) arg;

    FARF(HIGH, "hmx-worker: thread started");

    unsigned int prev_seqn = 0;
    for (;;) {
        unsigned int seqn = atomic_load_explicit(&ctx->cmd_seqn, memory_order_acquire);
        if (seqn == prev_seqn) {
            qurt_futex_wait(&ctx->cmd_seqn, prev_seqn);
            continue;
        }
        prev_seqn = seqn;

        switch (ctx->cmd_type) {
            case HMX_WORKER_CMD_BEGIN:
                HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
                break;

            case HMX_WORKER_CMD_JOB:
                ctx->fn(ctx->data);
                break;

            case HMX_WORKER_CMD_END:
                HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
                break;

            case HMX_WORKER_CMD_KILL:
                atomic_store_explicit(&ctx->done_seqn, seqn, memory_order_release);
                qurt_futex_wake(&ctx->done_seqn, 1);
                FARF(HIGH, "hmx-worker: thread stopped");
                return;
        }

        atomic_store_explicit(&ctx->done_seqn, seqn, memory_order_release);
        qurt_futex_wake(&ctx->done_seqn, 1);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Issue a command to the worker (non-blocking).
static void hmx_worker_issue(struct hmx_worker_context * ctx,
                             enum hmx_worker_cmd         type,
                             hmx_worker_fn_t             fn,
                             void *                      data) {
    ctx->cmd_type = type;
    ctx->fn       = fn;
    ctx->data     = data;
    atomic_fetch_add_explicit(&ctx->cmd_seqn, 1, memory_order_release);
    qurt_futex_wake(&ctx->cmd_seqn, 1);
}

// Block until the worker has completed the most recently issued command.
static void hmx_worker_drain(struct hmx_worker_context * ctx) {
    unsigned int expected = atomic_load_explicit(&ctx->cmd_seqn, memory_order_acquire);
    while (atomic_load_explicit(&ctx->done_seqn, memory_order_acquire) != expected) {
        qurt_futex_wait(&ctx->done_seqn, atomic_load_explicit(&ctx->done_seqn, memory_order_relaxed));
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#define LOWEST_USABLE_QURT_PRIO (254)

AEEResult hmx_worker_init(hmx_worker_context_t * out, uint32_t stack_size, uint32_t vtcm_rctx) {
    if (!out) {
        return AEE_EBADPARM;
    }

    // Single allocation: stack followed by context struct.
    size_t          total = stack_size + sizeof(struct hmx_worker_context);
    unsigned char * blob  = (unsigned char *) malloc(total);
    if (!blob) {
        FARF(ERROR, "hmx-worker: allocation failed (%zu bytes)", total);
        return AEE_ENOMEMORY;
    }
    memset(blob, 0, total);

    struct hmx_worker_context * ctx = (struct hmx_worker_context *) (blob + stack_size);
    ctx->stack                      = blob;
    ctx->vtcm_rctx                  = vtcm_rctx;
    atomic_init(&ctx->cmd_seqn, 0);
    atomic_init(&ctx->done_seqn, 0);

    // Match caller thread priority (same pattern as worker-pool.c).
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio < 1) {
        prio = 1;
    }
    if (prio > LOWEST_USABLE_QURT_PRIO) {
        prio = LOWEST_USABLE_QURT_PRIO;
    }

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, blob);
    qurt_thread_attr_set_stack_size(&attr, stack_size);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_attr_set_name(&attr, "hmx_worker");

    int err = qurt_thread_create(&ctx->thread, &attr, hmx_worker_main, ctx);
    if (err) {
        FARF(ERROR, "hmx-worker: thread create failed (%d)", err);
        free(blob);
        return AEE_EQURTTHREADCREATE;
    }

    *out = ctx;
    return AEE_SUCCESS;
}

void hmx_worker_release(hmx_worker_context_t ctx) {
    if (!ctx) {
        return;
    }

    // Tell the worker to exit.
    hmx_worker_issue(ctx, HMX_WORKER_CMD_KILL, NULL, NULL);
    hmx_worker_drain(ctx);

    int status;
    qurt_thread_join(ctx->thread, &status);

    free(ctx->stack);
}

AEEResult hmx_worker_begin(hmx_worker_context_t ctx) {
    hmx_worker_issue(ctx, HMX_WORKER_CMD_BEGIN, NULL, NULL);
    hmx_worker_drain(ctx);  // wait until HMX lock is acquired
    return AEE_SUCCESS;
}

AEEResult hmx_worker_submit(hmx_worker_context_t ctx, hmx_worker_fn_t fn, void * data) {
    // Caller is expected to have called wait() for any previous job.
    // Safety: drain any residual (should be instant in normal flow).
    hmx_worker_drain(ctx);
    hmx_worker_issue(ctx, HMX_WORKER_CMD_JOB, fn, data);
    return AEE_SUCCESS;
}

AEEResult hmx_worker_wait(hmx_worker_context_t ctx) {
    hmx_worker_drain(ctx);
    return AEE_SUCCESS;
}

AEEResult hmx_worker_end(hmx_worker_context_t ctx) {
    hmx_worker_drain(ctx);  // ensure no in-flight job
    hmx_worker_issue(ctx, HMX_WORKER_CMD_END, NULL, NULL);
    hmx_worker_drain(ctx);  // wait until HMX lock is released
    return AEE_SUCCESS;
}
