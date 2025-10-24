#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"

#define FARF_ERROR  1
#define FARF_HIGH   1
#define FARF_MEDIUM 0
#define FARF_LOW    0
#include <AEEStdErr.h>
#include <dspqueue.h>
#include <HAP_compute_res.h>
#include <HAP_etm_config.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <HAP_ps.h>
#include <qurt.h>
#include <qurt_thread.h>
#include <remote.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "ops-utils.h"
#include "worker-pool.h"

AEEResult htp_iface_open(const char * uri, remote_handle64 * handle) {
    struct htp_context * ctx;
    int                  err = 0;

    ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return AEE_ENOMEMORY;
    }

    // Use the context structure as a handle
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

static int vtcm_acquire(struct htp_context * ctx) {
    if (!ctx->vtcm_valid) {
        // Temporarily bump thread priority to make sure it's higher than other sessions.
        // This way the resource manager will notify the other thread to release VTCM.
        // Note that we need to reaquire VTCM at normal priority for this to work next time.
        qurt_thread_set_priority(qurt_thread_get_id(), ctx->thread_prio - 10);
        HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 1000000);
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
        qurt_thread_set_priority(qurt_thread_get_id(), ctx->thread_prio);

        HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 1000000);
        ctx->vtcm_valid = true;
    }

    ctx->vtcm_inuse = true;
    return 0;
}

static int vtcm_release(struct htp_context * ctx) {
    ctx->vtcm_inuse = false;

    if (ctx->vtcm_valid && ctx->vtcm_needs_release) {
        ctx->vtcm_valid         = false;
        ctx->vtcm_needs_release = false;
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }

    return 0;
}

static int vtcm_release_callback(unsigned int rctx, void * state) {
    struct htp_context * ctx = (struct htp_context *) state;

    if (!ctx || ctx->vtcm_rctx != rctx) {
        return AEE_EBADPARM;
    }

    // If VTCM is not inuse (not processing Ops) release it right here
    // otherwise we'll release it once we're done with the current Op.

    if (ctx->vtcm_inuse) {
        ctx->vtcm_needs_release = false;
        return 0;
    }

    ctx->vtcm_valid = false;
    HAP_compute_res_release_cached(ctx->vtcm_rctx);

    return 0;
}

static int vtcm_alloc(struct htp_context * ctx) {
    unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB default
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size);
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, (void *) ctx);
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
    ctx->vtcm_inuse         = false;
    ctx->vtcm_needs_release = false;

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

AEEResult htp_iface_start(remote_handle64 handle, uint32 sess_id, uint64 dsp_queue_id, uint32 n_hvx) {
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
        ctx->dma[i] = dma_queue_create(HTP_SPAD_SRC0_NROWS * 2);
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
    d->cycles = htp_get_cycles();
    d->pkts   = htp_get_pktcnt();
}

static inline void profile_stop(struct profile_data * d) {
    d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
    d->cycles = htp_get_cycles() - d->cycles;
    d->pkts   = htp_get_pktcnt() - d->pkts;
}

static int send_htp_rsp(struct htp_context *     c,
                        uint32_t                 op,
                        uint32_t                 status,
                        struct dspqueue_buffer * bufs,
                        size_t                   n_bufs,
                        struct profile_data *    prof) {
    // Prep response struct
    struct htp_general_rsp rsp;
    rsp.op          = op;
    rsp.status      = status;
    rsp.prof_usecs  = prof->usecs;
    rsp.prof_cycles = prof->cycles;
    rsp.prof_pkts   = prof->pkts;

    int err = dspqueue_write(c->queue,
                             0,                       // Flags
                             n_bufs,
                             bufs,                    // Buffer references
                             sizeof(rsp),
                             (const uint8_t *) &rsp,  // Message
                             DSPQUEUE_TIMEOUT_NONE);

    if (err != 0) {
        FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
    }

    return err;
}

static void proc_matmul_req(struct htp_context *     ctx,
                            struct htp_general_req * req,
                            struct dspqueue_buffer * bufs,
                            size_t                   n_bufs) {
    // Prep response buffer structs (needed for error responses, etc)
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));
    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[2].fd     = bufs[2].fd;
    rsp_bufs[2].ptr    = bufs[2].ptr;
    rsp_bufs[2].size   = bufs[2].size;
    rsp_bufs[2].offset = bufs[2].offset;
    rsp_bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                         DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.src1                   = req->src1;
    octx.dst                    = req->dst;
    octx.flags                  = req->flags;
    octx.op                     = req->op;

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.dst.data  = (uint32_t) bufs[2].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_matmul(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 3, &prof);
}

static void proc_matmul_id_req(struct htp_context *     ctx,
                               struct htp_general_req * req,
                               struct dspqueue_buffer * bufs,
                               size_t                   n_bufs) {
    // Prep response buffer structs (needed for error responses, etc)
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));
    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[2].fd     = bufs[2].fd;
    rsp_bufs[2].ptr    = bufs[2].ptr;
    rsp_bufs[2].size   = bufs[2].size;
    rsp_bufs[2].offset = bufs[2].offset;
    rsp_bufs[2].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[3].fd     = bufs[3].fd;
    rsp_bufs[3].ptr    = bufs[3].ptr;
    rsp_bufs[3].size   = bufs[3].size;
    rsp_bufs[3].offset = bufs[3].offset;
    rsp_bufs[3].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                         DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.src1                   = req->src1;
    octx.src2                   = req->src2;
    octx.dst                    = req->dst;
    octx.flags                  = req->flags;
    octx.op                     = req->op;

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.src2.data = (uint32_t) bufs[2].ptr;
    octx.dst.data  = (uint32_t) bufs[3].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_matmul_id(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 4, &prof);
}

static void proc_binary_req(struct htp_context * ctx, struct htp_general_req * req, struct dspqueue_buffer * bufs) {
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));

    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[2].fd     = bufs[2].fd;
    rsp_bufs[2].ptr    = bufs[2].ptr;
    rsp_bufs[2].offset = bufs[2].offset;
    rsp_bufs[2].size   = bufs[2].size;
    rsp_bufs[2].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                         DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.src1                   = req->src1;
    octx.dst                    = req->dst;
    octx.flags                  = req->flags;
    octx.op                     = req->op;

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.dst.data  = (uint32_t) bufs[2].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_binary(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 3, &prof);
}

static void proc_add_id_req(struct htp_context * ctx, struct htp_general_req * req, struct dspqueue_buffer * bufs) {
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));

    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[2].fd     = bufs[2].fd;
    rsp_bufs[2].ptr    = bufs[2].ptr;
    rsp_bufs[2].offset = bufs[2].offset;
    rsp_bufs[2].size   = bufs[2].size;
    rsp_bufs[2].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[3].fd     = bufs[3].fd;
    rsp_bufs[3].ptr    = bufs[3].ptr;
    rsp_bufs[3].offset = bufs[3].offset;
    rsp_bufs[3].size   = bufs[3].size;
    rsp_bufs[3].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                         DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.src1                   = req->src1;
    octx.src2                   = req->src2;
    octx.dst                    = req->dst;
    octx.flags                  = req->flags;
    octx.op                     = req->op;

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    octx.src2.data = (uint32_t) bufs[2].ptr;
    octx.dst.data  = (uint32_t) bufs[3].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_binary(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 4, &prof);
}

static void proc_unary_req(struct htp_context * ctx, struct htp_general_req * req, struct dspqueue_buffer * bufs) {
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));

    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                         DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.dst                    = req->dst;
    octx.flags                  = req->flags;
    octx.op                     = req->op;

    memcpy(octx.op_params, req->op_params, sizeof(octx.op_params));

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.dst.data  = (uint32_t) bufs[1].ptr;
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_unary(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, 2, &prof);
}

static void proc_activations_req(struct htp_context *     ctx,
                                 struct htp_general_req * req,
                                 struct dspqueue_buffer * bufs,
                                 uint32_t                 n_bufs) {
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));

    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    int write_idx = 1;
    if (3 == n_bufs) {
        rsp_bufs[1].fd     = bufs[1].fd;
        rsp_bufs[1].ptr    = bufs[1].ptr;
        rsp_bufs[1].offset = bufs[1].offset;
        rsp_bufs[1].size   = bufs[1].size;
        rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

        write_idx = 2;
    }

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[write_idx].fd     = bufs[write_idx].fd;
    rsp_bufs[write_idx].ptr    = bufs[write_idx].ptr;
    rsp_bufs[write_idx].offset = bufs[write_idx].offset;
    rsp_bufs[write_idx].size   = bufs[write_idx].size;
    rsp_bufs[write_idx].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                                 DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                                 DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    if (3 == n_bufs) {
        octx.src1 = req->src1;
    }
    octx.dst   = req->dst;
    octx.flags = req->flags;
    octx.op    = req->op;

    memcpy(octx.op_params, req->op_params, sizeof(octx.op_params));

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    if (3 == n_bufs) {
        octx.src1.data = (uint32_t) bufs[1].ptr;
        octx.dst.data  = (uint32_t) bufs[2].ptr;
    } else {
        octx.dst.data = (uint32_t) bufs[1].ptr;
    }
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        if (octx.op == HTP_OP_SOFTMAX) {
            rsp_status = op_softmax(&octx);
        } else {
            rsp_status = op_activations(&octx);
        }
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, n_bufs, &prof);
}

static void proc_rope_req(struct htp_context *     ctx,
                          struct htp_general_req * req,
                          struct dspqueue_buffer * bufs,
                          uint32_t                 n_bufs) {
    struct dspqueue_buffer rsp_bufs[HTP_MAX_PACKET_BUFFERS];
    memset(rsp_bufs, 0, sizeof(rsp_bufs));

    rsp_bufs[0].fd     = bufs[0].fd;
    rsp_bufs[0].ptr    = bufs[0].ptr;
    rsp_bufs[0].offset = bufs[0].offset;
    rsp_bufs[0].size   = bufs[0].size;
    rsp_bufs[0].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    rsp_bufs[1].fd     = bufs[1].fd;
    rsp_bufs[1].ptr    = bufs[1].ptr;
    rsp_bufs[1].offset = bufs[1].offset;
    rsp_bufs[1].size   = bufs[1].size;
    rsp_bufs[1].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

    int write_idx = 2;
    if (4 == n_bufs) {
        rsp_bufs[write_idx].fd     = bufs[write_idx].fd;
        rsp_bufs[write_idx].ptr    = bufs[write_idx].ptr;
        rsp_bufs[write_idx].offset = bufs[write_idx].offset;
        rsp_bufs[write_idx].size   = bufs[write_idx].size;
        rsp_bufs[write_idx].flags  = DSPQUEUE_BUFFER_FLAG_DEREF;  // Release reference

        write_idx++;
    }

    // We had written to the output buffer, we'd also need to flush it
    rsp_bufs[write_idx].fd     = bufs[write_idx].fd;
    rsp_bufs[write_idx].ptr    = bufs[write_idx].ptr;
    rsp_bufs[write_idx].offset = bufs[write_idx].offset;
    rsp_bufs[write_idx].size   = bufs[write_idx].size;
    rsp_bufs[write_idx].flags  = (DSPQUEUE_BUFFER_FLAG_DEREF |                 // Release reference
                                 DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |          // Flush NSP
                                 DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT);  // Invalidate CPU

    // Setup Op context
    struct htp_ops_context octx = { 0 };
    octx.ctx                    = ctx;
    octx.src0                   = req->src0;
    octx.src1                   = req->src1;
    if (4 == n_bufs) {
        octx.src2 = req->src2;
    }
    octx.dst   = req->dst;
    octx.flags = req->flags;
    octx.op    = req->op;

    memcpy(octx.op_params, req->op_params, sizeof(octx.op_params));

    // Update data pointers
    octx.src0.data = (uint32_t) bufs[0].ptr;
    octx.src1.data = (uint32_t) bufs[1].ptr;
    if (4 == n_bufs) {
        octx.src2.data = (uint32_t) bufs[2].ptr;
        octx.dst.data  = (uint32_t) bufs[3].ptr;
    } else {
        octx.dst.data = (uint32_t) bufs[2].ptr;
    }
    octx.n_threads = ctx->n_threads;

    struct profile_data prof;
    profile_start(&prof);

    uint32_t rsp_status = HTP_STATUS_INTERNAL_ERR;
    if (vtcm_acquire(ctx) == AEE_SUCCESS) {
        rsp_status = op_rope(&octx);
        vtcm_release(ctx);
    }

    profile_stop(&prof);
    send_htp_rsp(ctx, req->op, rsp_status, rsp_bufs, n_bufs, &prof);
}

static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
    struct htp_context * ctx = (struct htp_context *) context;

    // Repeatedly read packets from the queue until it's empty. We don't
    // necessarily get a separate callback for each packet, and new packets
    // may arrive while we're processing the previous one. This ensures we
    // keep the DSP busy as much as possible and avoid waiting for the CPU.

    while (1) {
        struct htp_general_req req;
        uint32_t               req_size;

        struct dspqueue_buffer bufs[HTP_MAX_PACKET_BUFFERS];
        uint32_t               n_bufs;
        uint32_t               flags;

        // Read packet from queue
        int err = dspqueue_read_noblock(queue, &flags,
                                        HTP_MAX_PACKET_BUFFERS,  // Maximum number of buffer references
                                        &n_bufs,                 // Number of buffer references
                                        bufs,                    // Buffer references
                                        sizeof(req),             // Max message length
                                        &req_size,               // Message length
                                        (uint8_t *) &req);       // Message

        if (err == AEE_EWOULDBLOCK) {
            // Consumed all packets available for now
            return;
        }

        if (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            return;
        }

        if (req_size != sizeof(req)) {
            FARF(ERROR, "Invalid request size");
            continue;
        }

        if (req.flags & HTP_OPFLAGS_EARLY_WAKEUP) {
            // Host wants early notification
            dspqueue_write_early_wakeup_noblock(ctx->queue, 10, 0);
        }

        // Process packet based on its message type
        switch (req.op) {
            case HTP_OP_MUL_MAT:
                if (n_bufs != 3) {
                    FARF(ERROR, "Bad matmul-req buffer list");
                    continue;
                }
                proc_matmul_req(ctx, &req, bufs, n_bufs);
                break;

            case HTP_OP_MUL_MAT_ID:
                if (n_bufs != 4) {
                    FARF(ERROR, "Bad matmul-id-req buffer list");
                    continue;
                }
                proc_matmul_id_req(ctx, &req, bufs, n_bufs);
                break;

            case HTP_OP_MUL:
            case HTP_OP_ADD:
            case HTP_OP_SUB:
                if (n_bufs != 3) {
                    FARF(ERROR, "Bad binary-req buffer list");
                    continue;
                }
                proc_binary_req(ctx, &req, bufs);
                break;

            case HTP_OP_RMS_NORM:
                if (n_bufs != 2) {
                    FARF(ERROR, "Bad unary-req buffer list");
                    continue;
                }

                proc_unary_req(ctx, &req, bufs);
                break;

            case HTP_OP_UNARY_SILU:
                if (n_bufs != 2) {
                    FARF(ERROR, "Bad act-req buffer list");
                    continue;
                }
                proc_activations_req(ctx, &req, bufs, n_bufs);
                break;

            case HTP_OP_GLU_SWIGLU:
            case HTP_OP_SOFTMAX:
                if ((n_bufs != 2) && (n_bufs != 3)) {
                    FARF(ERROR, "Bad act-req buffer list");
                    continue;
                }
                proc_activations_req(ctx, &req, bufs, n_bufs);
                break;

            case HTP_OP_ADD_ID:
                if (n_bufs != 4) {
                    FARF(ERROR, "Bad add-id-req buffer list");
                    continue;
                }
                proc_add_id_req(ctx, &req, bufs);
                break;

            case HTP_OP_ROPE:
                if ((n_bufs != 3) && (n_bufs != 4)) {
                    FARF(ERROR, "Bad rope-req buffer list");
                    continue;
                }
                proc_rope_req(ctx, &req, bufs, n_bufs);
                break;

            default:
                FARF(ERROR, "Unknown Op %u", req.op);
                break;
        }
    }
}
