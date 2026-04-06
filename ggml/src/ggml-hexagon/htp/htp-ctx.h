#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "hex-dma.h"
#include "htp-msg.h"
#include "worker-pool.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>

#define HTP_MAX_NTHREADS 10

#define HTP_MAX_MMAPS    16
#define HTP_MAX_VMEM     3865473024UL // ~ 3.6GB

struct htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t pinned;
};

// Main context for htp DSP backend
struct htp_context {
    dspqueue_t             queue;
    dma_queue *            dma[HTP_MAX_NTHREADS];
    struct htp_mmap        mmap[HTP_MAX_MMAPS];
    worker_pool_context_t  worker_pool;
    uint32_t               n_threads;

    int                    thread_id;
    int                    thread_prio;

    int                    hmx_enabled;       // Runtime flag: HMX initialisation succeeded

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    size_t                 vtcm_scratch_size; // Usable dynamic scratch (vtcm_size minus tail reservation)
};

#endif /* HTP_CTX_H */
