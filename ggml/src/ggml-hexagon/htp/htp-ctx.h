#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "hex-dma.h"
#include "htp-ops.h"
#include "hmx-worker.h"
#include "worker-pool.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>

#define HTP_MAX_NTHREADS 10
#define HTP_MAX_MMAPS    16

// Memory mapping
struct htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t pinned;
};

// Scratchpad state
struct htp_spad {
    const struct htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

// Context while processing an Op
// TODO: fold this into the main context
struct htp_ops_context {
    struct htp_context * ctx;

    enum htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];

    const struct htp_tensor * src[HTP_OP_MAX_INPUTS];
    const struct htp_tensor * dst;

    // TODO convert these to an array
    struct htp_spad src0_spad;
    struct htp_spad src1_spad;
    struct htp_spad src2_spad;
    struct htp_spad src3_spad;
    struct htp_spad dst_spad;

    uint32_t n_threads;
    uint32_t flags;
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

    int                    hmx_enabled;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    struct htp_ops_context octx;

#ifdef HTP_HAS_HMX
    hmx_worker_context_t   hmx_worker; // Async HMX worker for pipeline overlap
#endif
};

int op_matmul(struct htp_ops_context * octx);
int op_matmul_id(struct htp_ops_context * octx);
int op_binary(struct htp_ops_context * octx);
int op_unary(struct htp_ops_context * octx);
int op_sum_rows(struct htp_ops_context * octx);
int op_activations(struct htp_ops_context * octx);
int op_softmax(struct htp_ops_context * octx);
int op_add_id(struct htp_ops_context * octx);
int op_rope(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
int op_set_rows(struct htp_ops_context * octx);
int op_get_rows(struct htp_ops_context * octx);
int op_cpy(struct htp_ops_context * octx);
int op_repeat(struct htp_ops_context * octx);
int op_argsort(struct htp_ops_context * octx);
int op_ssm_conv(struct htp_ops_context * octx);
int op_cumsum(struct htp_ops_context * octx);

#endif /* HTP_CTX_H */
