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
    uint32_t age;
};

// Main context for htp DSP backend
struct htp_context {
    dspqueue_t             queue;
    dma_queue *            dma[HTP_MAX_NTHREADS];
    worker_pool_context_t  worker_pool;
    uint32_t               n_threads;

    int         thread_id;
    int         thread_prio;

    uint8_t *   vtcm_base;
    size_t      vtcm_size;
    uint32_t    vtcm_rctx;
    atomic_bool vtcm_valid;

    // HMX acceleration fields (v73+, enabled by compile-time HTP_HAS_HMX)
#ifdef HTP_HAS_HMX
    int         hmx_enabled;       // Runtime flag: HMX initialisation succeeded
    size_t      vtcm_scratch_size; // Usable dynamic scratch (vtcm_size minus tail reservation)
#endif

    // Cached src1 spad position from the last quantize pass.
    // When SKIP_QUANTIZE is set the Q8 activation data is already in VTCM
    // at this address; the matmul must read from here instead of recomputing
    // the offset (which depends on the current op's src0 size).
    uint8_t *   prev_src1_spad;

    uint32_t    opmask;

    struct htp_mmap mmap[HTP_MAX_MMAPS];

    struct {
        struct htp_general_req req;
        struct htp_op_buf      bufs[HTP_OP_MAX_BUFS];
        struct htp_tensor      tens[HTP_OP_MAX_TENSORS];
        struct htp_op_req       ops[HTP_OP_MAX_REQS];
    } op_stage;
};

#endif /* HTP_CTX_H */
