#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "hex-dma.h"
#include "worker-pool.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>

#define HTP_MAX_NTHREADS 10

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

    // HMX acceleration fields (v73+, enabled by compile-time HTP_HAS_HMX)
#ifdef HTP_HAS_HMX
    int        hmx_enabled;       // Runtime flag: HMX initialisation succeeded
    size_t     vtcm_scratch_size; // Usable dynamic scratch (vtcm_size minus tail reservation)
    uint8_t *  exp2_table;        // Precomputed exp2 table address at VTCM tail
#endif
};

#endif /* HTP_CTX_H */
