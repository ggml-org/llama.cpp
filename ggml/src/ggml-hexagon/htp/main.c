#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <AEEStdErr.h>
#include <dspqueue.h>
#include <HAP_compute_res.h>
#include <HAP_etm_config.h>
#include <HAP_mem.h>
#include <HAP_power.h>
#include <HAP_ps.h>
#include <qurt.h>
#include <qurt_thread.h>
#include <qurt_memory.h>
#include <remote.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "worker-pool.h"

AEEResult htp_iface_open(const char * uri, remote_handle64 * handle) {
    struct htp_context * ctx;
    int                  err = 0;

    ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return AEE_ENOMEMORY;
    }

    // Use the context structure as the handle
    *handle = (remote_handle64) ctx;

    // Enable FARF logs
    HAP_setFARFRuntimeLoggingParams(0xffff, NULL, 0);

    // Set client class
    {
        HAP_power_request_t request;
        memset(&request, 0, sizeof(HAP_power_request_t));
        request.type    = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            return err;
        }
    }

    {
        HAP_power_request_t request;
        memset(&request, 0, sizeof(request));

        request.type                              = HAP_power_set_DCVS_v3;
        request.dcvs_v3.set_dcvs_enable           = TRUE;
        request.dcvs_v3.dcvs_enable               = TRUE;
        request.dcvs_v3.dcvs_option               = HAP_DCVS_V2_PERFORMANCE_MODE;
        request.dcvs_v3.set_bus_params            = TRUE;
        request.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.bus_params.target_corner  = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.set_core_params           = TRUE;
        request.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.set_sleep_disable         = TRUE;
        request.dcvs_v3.sleep_disable             = TRUE;
        if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            return err;
        }

        memset(&request, 0, sizeof(request));
        request.type         = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;
        if ((err = HAP_power_set((void *) ctx, &request)) != 0) {
            return err;
        }
    }

    {
        // Power on HMX
        HAP_power_request_t request;
        memset(&request, 0, sizeof(HAP_power_request_t));
        request.type         = HAP_power_set_HMX;
        request.hmx.power_up = TRUE;
        FARF(ALWAYS, "Powering HMX on\n");
        err = HAP_power_set((void *) &ctx, &request);
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "Error powering on HMX.");
            return err;
        }
    }

    return AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    struct htp_context * ctx = (struct htp_context *) handle;

    if (!ctx) {
        return AEE_EBADPARM;
    }

    if (ctx->queue) {
        FARF(ERROR, "Closing handle with queue still open");
        return AEE_EITEMBUSY;
    }

    // release the mmaps (if any)
    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        if (ctx->mmap[i].size) {
            HAP_munmap2((void *) ctx->mmap[i].base, ctx->mmap[i].size);
        }
    }

    free(ctx);
    return AEE_SUCCESS;
}

AEEResult htp_iface_enable_etm(remote_handle64 handle) {
    int err = HAP_user_etm_enable();
    if (err) {
        if (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable is not supported\n");
        } else {
            FARF(ERROR, "Error executing HAP_user_etm_enable with error code : 0x%x\n", err);
        }
    }
    return err;
}

AEEResult htp_iface_disable_etm(remote_handle64 handle) {
    int err = HAP_user_etm_disable();
    if (err) {
        if (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_disable is not supported\n");
        } else {
            FARF(ERROR, "Error executing HAP_user_etm_disable with error code : 0x%x\n", err);
        }
    }
    return err;
}

AEEResult htp_iface_unmap_buffers(remote_handle64 handle) {
    struct htp_context * ctx = (struct htp_context *) handle;

    if (!ctx) {
        return AEE_EBADPARM;
    }

    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (m->size) {
            FARF(HIGH, "unmmap : base %p fd %u size %u", (void*) m->base, m->fd, (uint32_t) m->size);
            HAP_munmap2((void *) m->base, m->size);
            m->size = 0;
            m->base = NULL;
            m->fd   = -1;
        }
    }

    return AEE_SUCCESS;
}

static void vtcm_acquire(struct htp_context * ctx) {
    if (ctx->vtcm_valid) return;

    int err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 1000000);
    if (err != 0) {
        FARF(ERROR, "Failed to acquire VTCM: 0x%08x", (unsigned)err);
        abort();
    }

    ctx->vtcm_valid = true;
}

static void vtcm_release(struct htp_context * ctx) {
    if (!ctx->vtcm_valid) return;

    HAP_compute_res_release_cached(ctx->vtcm_rctx);
    ctx->vtcm_valid = false;
}

static int vtcm_alloc(struct htp_context * ctx) {
    unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB default
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    // Allocate VTCM for scratch pads
    uint32_t rctx = HAP_compute_res_acquire(&attr, 1000000 /* timeout */);
    if (!rctx) {
        FARF(ERROR, "failed to allocate %zu bytes VTCM\n", ctx->vtcm_size);
        return AEE_ENOMEMORY;
    }

    void * vtcm_ptr;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_size) != 0) {
        HAP_compute_res_release(rctx);
        FARF(ERROR, "failed to allocate %zu bytes VTCM (new)\n", ctx->vtcm_size);
        return AEE_ENOMEMORY;
    }

    ctx->vtcm_base          = (uint8_t *) vtcm_ptr;
    ctx->vtcm_size          = vtcm_size;
    ctx->vtcm_rctx          = rctx;
    ctx->vtcm_valid         = false;

    return 0;
}

static void vtcm_free(struct htp_context * ctx) {
    if (ctx->vtcm_rctx) {
        HAP_compute_res_release(ctx->vtcm_rctx);
        ctx->vtcm_base = 0;
        ctx->vtcm_rctx = 0;
    }
}

static void htp_packet_callback(dspqueue_t queue, int error, void * context);
static void htp_error_callback(dspqueue_t queue, int error, void * context);

AEEResult htp_iface_start(remote_handle64 handle, uint32 sess_id, uint64 dsp_queue_id, uint32 n_hvx, uint32 use_hmx) {
    struct htp_context * ctx = (struct htp_context *) handle;

    if (!ctx) {
        return AEE_EBADPARM;
    }

    if (ctx->queue) {
        FARF(ERROR, "Queue already open");
        return AEE_EITEMBUSY;
    }

    // Import queue created on the CPU
    int err = dspqueue_import(dsp_queue_id,         // Queue ID from dspqueue_export
                              htp_packet_callback,  // Packet callback
                              htp_error_callback,   // Error callback; no errors expected on the DSP
                              (void *) ctx,         // Callback context
                              &ctx->queue);

    if (err) {
        FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
        return err;
    }

    ctx->thread_id   = qurt_thread_get_id();
    ctx->thread_prio = qurt_thread_get_priority(ctx->thread_id);

    // allocate VTCM
    err = vtcm_alloc(ctx);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to allocate VTCM");
        return AEE_ENOMEMORY;
    }

#ifdef HTP_HAS_HMX
    if (use_hmx) {
        ctx->hmx_enabled       = 1;
        ctx->vtcm_scratch_size = ctx->vtcm_size;
        FARF(HIGH, "HMX enabled: vtcm-scratch %zu", ctx->vtcm_scratch_size);
    } else {
        ctx->hmx_enabled       = 0;
        ctx->vtcm_scratch_size = ctx->vtcm_size;
        FARF(HIGH, "HMX disabled (use_hmx=0)");
    }
#endif

    qurt_sysenv_max_hthreads_t hw_threads;
    qurt_sysenv_get_max_hw_threads(&hw_threads);
    uint32_t hw_nhvx = (qurt_hvx_get_units() >> 8) & 0xFF;

    if (n_hvx == 0) {
        n_hvx = hw_nhvx;
    }
    if (n_hvx > hw_threads.max_hthreads) {
        n_hvx = hw_threads.max_hthreads;
    }
    if (n_hvx > HTP_MAX_NTHREADS) {
        n_hvx = HTP_MAX_NTHREADS;
    }

    ctx->n_threads = n_hvx;
    for (int i = 0; i < ctx->n_threads; i++) {
        // see discussion https://github.com/ggml-org/llama.cpp/pull/18151#discussion_r2632388541
        ctx->dma[i] = dma_queue_create(128);
    }

    // init worker pool
    err = worker_pool_init(&ctx->worker_pool, n_hvx);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "Unable to create worker pool");
        return err;
    }

    FARF(HIGH, "session %u started: n-hvx %u vtcm-size %zu vtcm-rctx %u n-threads %u thread-id %d thread-prio %d \n",
         sess_id, hw_nhvx, ctx->vtcm_size, ctx->vtcm_rctx, ctx->n_threads, ctx->thread_id, ctx->thread_prio);

    return AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    struct htp_context * ctx = (struct htp_context *) handle;
    if (!ctx) {
        return AEE_EBADPARM;
    }

    if (!ctx->queue) {
        FARF(ERROR, "Queue not open");
        return AEE_EBADSTATE;
    }

    // Close queue. dspqueue_close() will also wait for callbacks to finish.
    int err    = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err != 0) {
        FARF(ERROR, "Queue close failed with 0x%08x", (unsigned) err);
        return err;
    }

    if (ctx->worker_pool) {
        // Release worker pool
        worker_pool_release(&ctx->worker_pool);
    }

    for (int i = 0; i < ctx->n_threads; i++) {
        dma_queue_delete(ctx->dma[i]);
    }
#ifdef HTP_HAS_HMX
    if (ctx->hmx_enabled) {
        ctx->hmx_enabled = 0;
    }
#endif


    vtcm_free(ctx);

    return AEE_SUCCESS;
}

static void htp_error_callback(dspqueue_t queue, int error, void * context) {
    // No errors expected on the DSP.
    FARF(ERROR, "Error callback: 0x%08x", (unsigned) error);
}

struct profile_data {
    uint64_t usecs;
    uint64_t cycles;
    uint64_t pkts;
};

static inline void profile_start(struct profile_data * d) {
    d->usecs  = HAP_perf_get_qtimer_count();
    d->cycles = hex_get_cycles();
    d->pkts   = hex_get_pktcnt();
}

static inline void profile_stop(struct profile_data * d) {
    d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
    d->cycles = hex_get_cycles() - d->cycles;
    d->pkts   = hex_get_pktcnt() - d->pkts;
}

static void execute_op(struct htp_ops_context * octx) {
    switch (octx->op) {
        case HTP_OP_MUL_MAT:
            op_matmul(octx);
            break;

        case HTP_OP_MUL_MAT_ID:
            op_matmul_id(octx);
            break;

        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
        case HTP_OP_ADD_ID:
            op_binary(octx);
            break;

        case HTP_OP_RMS_NORM:
        case HTP_OP_SCALE:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
            op_unary(octx);
            break;

        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_SWIGLU_OAI:
        case HTP_OP_GLU_GEGLU:
            op_activations(octx);
            break;

        case HTP_OP_SOFTMAX:
            op_softmax(octx);
            break;

        case HTP_OP_ROPE:
            op_rope(octx);
            break;

        case HTP_OP_FLASH_ATTN_EXT:
            op_flash_attn_ext(octx);
            break;

        case HTP_OP_SET_ROWS:
            op_set_rows(octx);
            break;

        case HTP_OP_GET_ROWS:
            op_get_rows(octx);
            break;

        case HTP_OP_SUM_ROWS:
            op_sum_rows(octx);
            break;

        case HTP_OP_CPY:
            op_cpy(octx);
            break;

        case HTP_OP_REPEAT:
            op_repeat(octx);
            break;

        case HTP_OP_ARGSORT:
            op_argsort(octx);
            break;

        case HTP_OP_SSM_CONV:
            op_ssm_conv(octx);
            break;

        default:
            FARF(ERROR, "Unknown Op %u", octx->op);
            break;
    }
}

static void prep_op_buf(struct htp_context *ctx, uint32_t idx, struct htp_op_buf *b) {
    FARF(HIGH, "prep-buf #%u : fd %u size %u flags 0x%x", idx, b->fd, (uint32_t) b->size, b->flags);

    b->base = NULL;

    // See if the buffer is already mapped
    // Age mapings and find the oldest as we go
    struct htp_mmap *o_mm = ctx->mmap;
    uint32_t        o_age = 0;

    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (m->fd == b->fd) {
            b->base = m->base;
            m->age  = 0;
        } else {
            if (++m->age > o_age) {
                o_age = m->age;
                o_mm  = m;
            }
        }
    }

    if (!b->base) {
        // New buffer, add to mappings
        struct htp_mmap *m = o_mm;
        if (m->size) {
            // Replacing an older entry, unmap first
            FARF(HIGH, "unmmap : base %p fd %u size %u", (void*) m->base, m->fd, (uint32_t) m->size);
            HAP_munmap2((void *) m->base, m->size);
        }

        void *va = HAP_mmap2(NULL, b->size, HAP_PROT_READ | HAP_PROT_WRITE, 0, b->fd, 0);
        if (va == (void*)-1) {
            FARF(ERROR, "mmap failed : va %p fd %u size %u", va, b->fd, (uint32_t) b->size);
            abort(); // can't do much else at this point
        }

        m->base = b->base = (uint64_t) va;
        m->fd   = b->fd;
        m->size = b->size;
        m->age  = 0;
    }
}

static void prep_tensor(struct htp_context *ctx, struct htp_op_buf *bufs, uint32_t idx, struct htp_tensor *t) {
    uint32_t offset = t->data;
    uint32_t size   = t->size;
    uint32_t bi     = t->bi;

    t->data = bufs[bi].base + offset; // update data to the actual pointer

    FARF(HIGH, "prep-tensor #%u: bi %u offset %u size %u data %p : %u:%u:%u:%u", idx, t->bi, offset, t->size, (void*) t->data,
        t->ne[0], t->ne[1], t->ne[3], t->ne[3]);
}

static void proc_op_req(struct htp_ops_context * octx, struct htp_tensor *tens, uint32_t idx, struct htp_op_req * op) {
    memcpy(octx->op_params, op->params, sizeof(octx->op_params));
    octx->flags = op->flags;
    octx->op    = op->opcode;

    FARF(HIGH, "proc-op #%u: opcode %u flags 0x%x", idx, octx->op, octx->flags);

    // Prep input tensors
    for (uint32_t i=0; i<HTP_OP_MAX_INPUTS; i++) {
        struct htp_tensor *src = op->src[i] == 0xffff ? NULL : tens + op->src[i];

        octx->src[i] = src;
        if (!src) continue;

        if (!(src->flags & HTP_TENSOR_FLUSHED) && (src->flags & HTP_TENSOR_COMPUTE)) {
            // invalidate compute buffers on input
            hex_l2clear((void *) src->data, src->size);
        }

        FARF(HIGH, "prep-src #%u: data %p size %u : %u:%u:%u:%u", op->src[i], (void*) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[3], src->ne[3]);
    }

    // Prep output tensor
    struct htp_tensor *dst = tens + op->dst;

    octx->dst = dst;

    FARF(HIGH, "prep-dst #%u: data %p size %u : %u:%u:%u:%u", op->dst, (void*) dst->data, dst->size,
        dst->ne[0], dst->ne[1], dst->ne[3], dst->ne[3]);

    execute_op(octx);

    // flush buffers on output
    hex_l2flush((void *) dst->data, dst->size);
    dst->flags |= HTP_TENSOR_FLUSHED;

    FARF(HIGH, "post-dst #%u: data %p size %u : %u:%u:%u:%u", op->dst, (void*) dst->data, dst->size,
        dst->ne[0], dst->ne[1], dst->ne[3], dst->ne[3]);
}

static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
    struct htp_context * ctx = (struct htp_context *) context;

    int err;
    while (1) {
        uint8_t  *m_ptr  = (uint8_t *) &ctx->op_stage;
        uint32_t  m_size = sizeof(ctx->op_stage);

        struct dspqueue_buffer dbufs[2];
        uint32_t        n_dbufs = 2;
        uint32_t        flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, dbufs, m_size, &m_size, (uint8_t *) m_ptr);
        if (err == AEE_EWOULDBLOCK) {
            return;
        }

        if (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            return;
        }

        const uint32_t h_size = sizeof(struct htp_general_req);

        struct htp_general_req* h = (struct htp_general_req*) m_ptr; m_ptr += h_size;
        const uint32_t n_bufs = h->n_bufs;
        const uint32_t n_tens = h->n_tensors;
        const uint32_t n_ops  = h->n_ops;

        const uint32_t b_size = sizeof(struct htp_op_buf) * n_bufs;
        const uint32_t t_size = sizeof(struct htp_tensor) * n_tens;
        const uint32_t o_size = sizeof(struct htp_op_req) * n_ops;

        if (m_size < h_size + b_size + t_size + o_size) {
            FARF(ERROR, "invalid opreq batch size %u", m_size);
            continue;
        }

        struct htp_op_buf* bufs = (struct htp_op_buf*) m_ptr; m_ptr += b_size;
        struct htp_tensor* tens = (struct htp_tensor*) m_ptr; m_ptr += t_size;
        struct htp_op_req*  ops = (struct htp_op_req*) m_ptr;

        FARF(HIGH, "processing opreq batch: n-bufs %u n-tensors %u n-ops %u : m-size %u h-size %u b-size %u t-size %u o-size %u",
                n_bufs, n_tens, n_ops, m_size, h_size, b_size, t_size, o_size);

        for (uint32_t i=0; i < n_bufs; i++) {
            prep_op_buf(ctx, i, &bufs[i]);
        }

        for (uint32_t i=0; i < n_tens; i++) {
            prep_tensor(ctx, bufs, i, &tens[i]);
        }

        vtcm_acquire(ctx);

        struct htp_ops_context octx;
        memset(&octx, 0, sizeof(octx));
        octx.n_threads = ctx->n_threads;
        octx.ctx       = ctx;

        for (uint32_t i=0; i < n_ops; i++) {
            struct profile_data prof;
            profile_start(&prof);

            proc_op_req(&octx, tens, i, &ops[i]);

            profile_stop(&prof);
            ops[i].prof_usecs  = prof.usecs;
            ops[i].prof_cycles = prof.cycles;
            ops[i].prof_pkts   = prof.pkts;
        }

        // dspqueue_write_early_wakeup_noblock(ctx->queue, 10, 0);

        vtcm_release(ctx);

        // Prep response struct
        struct htp_general_rsp rsp;
        rsp.status      = HTP_STATUS_OK; // FIXME
        err = dspqueue_write(queue, 0, 0, NULL /* n_bufs, bufs */, sizeof(rsp), (const uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
        if (err != 0) {
            FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
        }
    }
}
