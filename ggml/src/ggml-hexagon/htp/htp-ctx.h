#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "htp-dma.h"
#include "worker-pool.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>

#define HTP_MAX_NTHREADS 10

// FIXME: move these into matmul-ops
#define HTP_SPAD_SRC0_NROWS 16
#define HTP_SPAD_SRC1_NROWS 16
#define HTP_SPAD_DST_NROWS  2

// Main context for htp DSP backend
struct htp_context {
    dspqueue_t            queue;
    dma_queue *           dma[HTP_MAX_NTHREADS];
    worker_pool_context_t worker_pool;
    uint32_t              n_threads;

    int thread_id;
    int thread_prio;

    uint8_t * vtcm_base;
    size_t    vtcm_size;
    uint32_t  vtcm_rctx;

    atomic_bool vtcm_valid;
    atomic_bool vtcm_inuse;
    atomic_bool vtcm_needs_release;

    uint32_t opmask;
};

#endif /* HTP_CTX_H */
