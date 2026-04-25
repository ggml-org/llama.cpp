#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

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

#include "hex-utils.h"
#include "hex-dma.h"
#include "hmx-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "htp_iface.h"
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

    {
        // Set HMX clock
        HAP_power_request_t request;
        memset(&request, 0, sizeof(HAP_power_request_t));
        request.type = HAP_power_set_HMX_v2;
        request.hmx_v2.set_clock = TRUE;
        request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
        request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
        request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
        request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
        FARF(ALWAYS, "Setting HMX clock\n");
        err = HAP_power_set((void *) &ctx, &request);
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "Error setting HMX clock.");
            return err;
        }
    }

    return AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    int err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
    if (err) {
        if (err == AEE_EVERSIONNOTSUPPORT) {
            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
        } else {
            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
        }
    }
    return err;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, const htp_iface_pmu_conf* pmu_conf) {
    struct htp_context * ctx = (struct htp_context *) handle;
    if (!ctx) {
        return AEE_EBADPARM;
    }

    if (mode == HTP_PROF_PMU) {
        const uint32_t* events = pmu_conf->events;

        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
        for (; i < HEX_NUM_PMU_COUNTERS/2; i++) {
            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
        }

        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
        for (i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
            cfg |= (((events[i] >> 8) & 3) << (i * 2));
        }

        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);

        // Configure PMU registers
        qurt_pmu_set(QURT_PMUCFG,     cfg);
        qurt_pmu_set(QURT_PMUEVTCFG,  evtcfg);
        qurt_pmu_set(QURT_PMUEVTCFG1, evtcfg1);
        qurt_pmu_enable(1);
    }

    ctx->profiler = mode;

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
#if __HVX_ARCH__ > 73
            HAP_munmap2((void *) ctx->mmap[i].base, ctx->mmap[i].size);
#else
            HAP_munmap((void *) ctx->mmap[i].base, ctx->mmap[i].size);
#endif
            ctx->mmap[i].size = 0;
            ctx->mmap[i].base = NULL;
            ctx->mmap[i].fd   = -1;
        }
    }

    if (ctx->profiler) {
        qurt_pmu_enable(1);
    }

    if (ctx->etm) {
        HAP_user_etm_disable();
    }

    free(ctx);
    return AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32 pinned) {
    struct htp_context * ctx = (struct htp_context *) handle;
    if (!ctx) {
        return AEE_EBADPARM;
    }

    // See if we already have this mapping
    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (m->fd == fd) {
            m->pinned = pinned;
            return AEE_SUCCESS;
        }
    }

    // Add new mapping
    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (!m->size) {
            FARF(HIGH, "mmap : fd %u size %u pinned %u", fd, size, pinned);
#if __HVX_ARCH__ > 73
            void *va = HAP_mmap2(NULL, size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#else
            if (size > HTP_MMAP_MAX_VMEM) { // HAP_mmap has a size limit of 2GB
                FARF(ERROR, "mmap failed : size %u exceeds 2GB limit for HAP_mmap", (uint32_t) size);
                abort(); // can't do much else at this point
            }

            void *va = HAP_mmap(NULL, size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
            if (va == (void*)-1) {
                FARF(ERROR, "mmap failed : va %p fd %u size %u", va, fd, (uint32_t) size);
                return AEE_EFAILED;
            }

            m->base   = (uint64_t) va;
            m->fd     = fd;
            m->size   = size;
            m->pinned = pinned;

            return AEE_SUCCESS;
        }
    }

    return AEE_ENOMEMORY;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
    struct htp_context * ctx = (struct htp_context *) handle;
    if (!ctx) {
        return AEE_EBADPARM;
    }

    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (fd < 0 || m->fd == fd) {
            FARF(HIGH, "unmmap : base %p fd %u size %u", (void*) m->base, m->fd, (uint32_t) m->size);
#if __HVX_ARCH__ > 73
            HAP_munmap2((void *) m->base, m->size);
#else
            HAP_munmap((void *) m->base, m->size);
#endif
            m->size   = 0;
            m->base   = NULL;
            m->fd     = -1;
            m->pinned = 0;
        }
    }

    return AEE_SUCCESS;
}

static void vtcm_acquire(struct htp_context * ctx) {
    if (!ctx->vtcm_valid) {
        int err = HAP_compute_res_acquire_cached(ctx->vtcm_rctx, 1000000u);
        if (err != 0) {
            FARF(ERROR, "ggml-hex: failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }

        ctx->vtcm_needs_release = false;
        ctx->vtcm_valid = true;

        // Drop the priority to make sure we get the release callback from other GGML-HTP and QNN-HTP sessions
        HAP_compute_res_update_priority(ctx->vtcm_rctx, ctx->thread_prio + 10);
    }
}

static void vtcm_release(struct htp_context * ctx) {
    if (ctx->vtcm_valid) {
        ctx->vtcm_valid         = false;
        ctx->vtcm_needs_release = false;
        HAP_compute_res_release_cached(ctx->vtcm_rctx);
    }
}

static int vtcm_release_callback(unsigned int rctx, void * state) {
    struct htp_context * ctx = (struct htp_context *) state;
    ctx->vtcm_needs_release = true;
    return 0;
}

static int vtcm_alloc(struct htp_context * ctx) {
    unsigned int vtcm_size = 8 * 1024 * 1024;  // 8MB default
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size, vtcm_size, vtcm_size); // single page
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
    ctx->hmx_enabled = use_hmx;
    ctx->hmx_queue   = NULL;
    if (use_hmx) {
        ctx->hmx_queue = hmx_queue_create(16, ctx->vtcm_rctx);
        if (!ctx->hmx_queue) {
            FARF(ERROR, "hmx-queue-create failed");
            ctx->hmx_enabled = false;
        }
    }
    FARF(HIGH, "HMX %s (use_hmx=%d)", ctx->hmx_enabled ? "enabled" : "disabled", use_hmx);
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
    if (ctx->hmx_queue) {
        hmx_queue_delete(ctx->hmx_queue);
        ctx->hmx_queue = NULL;
    }
    ctx->hmx_enabled = false;
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
    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
};

static inline void profile_start(uint32_t mode, struct profile_data * d) {
    switch (mode) {
        case HTP_PROF_PMU:
            hex_get_pmu(d->pmu_counters);
            // fallthrough
        case HTP_PROF_BASIC:
            d->usecs  = HAP_perf_get_qtimer_count();
            d->cycles = hex_get_cycles();
            break;
        default:
            break;
    }
}

static inline void profile_stop(uint32_t mode, struct profile_data * d) {
    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
    switch (mode) {
        case HTP_PROF_PMU:
            hex_get_pmu(pmu_counters);
            for (int i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
            }
            // fallthrough
        case HTP_PROF_BASIC:
            d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
            d->cycles = hex_get_cycles() - d->cycles;
            break;
        default:
            break;
    }
}

static int execute_op(struct htp_ops_context * octx) {
    switch (octx->op) {
        case HTP_OP_MUL_MAT:
            return op_matmul(octx);

        case HTP_OP_MUL_MAT_ID:
            return op_matmul_id(octx);

        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
        case HTP_OP_ADD_ID:
            return op_binary(octx);

        case HTP_OP_RMS_NORM:
        case HTP_OP_SCALE:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
            return op_unary(octx);

        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_SWIGLU_OAI:
        case HTP_OP_GLU_GEGLU:
            return op_activations(octx);

        case HTP_OP_SOFTMAX:
            return op_softmax(octx);

        case HTP_OP_ROPE:
            return op_rope(octx);

        case HTP_OP_FLASH_ATTN_EXT:
            return op_flash_attn_ext(octx);

        case HTP_OP_SET_ROWS:
            return op_set_rows(octx);

        case HTP_OP_GET_ROWS:
            return op_get_rows(octx);

        case HTP_OP_SUM_ROWS:
            return op_sum_rows(octx);

        case HTP_OP_CPY:
            return op_cpy(octx);

        case HTP_OP_REPEAT:
            return op_repeat(octx);

        case HTP_OP_ARGSORT:
            return op_argsort(octx);

        case HTP_OP_SSM_CONV:
            return op_ssm_conv(octx);

        case HTP_OP_CUMSUM:
            return op_cumsum(octx);

        case HTP_OP_FILL:
            return op_fill(octx);

        case HTP_OP_DIAG:
            return op_diag(octx);

        case HTP_OP_SOLVE_TRI:
            return op_solve_tri(octx);

        case HTP_OP_INVALID:
            break;

        // No default to catch missing cases
    }

    FARF(ERROR, "Unknown Op %u", octx->op);
    return -1;
}

static inline bool reuse_buf(struct htp_context *ctx, uint32_t *m_reuse, struct htp_buf_desc *b) {
    b->base = NULL;

    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = ctx->mmap + i;
        if (m->size && m->fd == b->fd) {
            b->base   = m->base;
            *m_reuse |= (1 << i);
            return true;
        }
    }

    return false;
}

static inline void drop_mmap(struct htp_context *ctx, struct htp_mmap *m) {
    if (m->size && !m->pinned) {
        FARF(HIGH, "unmap : fd %u base %p size %u pinned %u", m->fd, (void*) m->base, (uint32_t) m->size, m->pinned);
#if __HVX_ARCH__ > 73
        HAP_munmap2((void *) m->base, m->size);
#else
        HAP_munmap((void *) m->base, m->size);
#endif
        m->size = 0;
        m->base = 0;
        m->fd   = -1;
    }
}

static inline void mmap_buf(struct htp_context *ctx, struct htp_buf_desc *b) {
    if (b->base) return; // already mapped

    // find unused mapping
    for (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
        struct htp_mmap *m = &ctx->mmap[i];
        if (!m->size) {
#if __HVX_ARCH__ > 73
            void *va = HAP_mmap2(NULL, b->size, HAP_PROT_READ | HAP_PROT_WRITE, 0, b->fd, 0);
#else
            if (b->size > HTP_MMAP_MAX_VMEM) { // HAP_mmap has a size limit of 2GB
                FARF(ERROR, "mmap failed : size %u exceeds 2GB limit for HAP_mmap", (uint32_t) b->size);
                abort(); // can't do much else at this point
            }

            void *va = HAP_mmap(NULL, b->size, HAP_PROT_READ | HAP_PROT_WRITE, 0, b->fd, 0);
#endif
            if (va == (void*)-1) {
                FARF(ERROR, "mmap failed : va %p fd %u size %u", va, b->fd, (uint32_t) b->size);
                abort(); // can't do much else at this point
            }

            m->base   = b->base = (uint64_t) va;
            m->fd     = b->fd;
            m->size   = b->size;
            m->pinned = 0;

            FARF(HIGH, "mmap : fd %u base %p size %u pinned %u", m->fd, (void*) m->base, (uint32_t) m->size, m->pinned);
            return;
        }
    }
}

static void prep_op_bufs(struct htp_context *ctx, struct htp_buf_desc *bufs, uint32_t n_bufs) {
    uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
    uint32_t b_reuse = 0; // buf reuse count

    size_t   m_vmem  = 0; // mapped vmem
    size_t   e_vmem  = 0; // extra  vmem

    // See what we can reuse
    for (uint32_t i=0; i < n_bufs; i++) {
        struct htp_buf_desc *b = bufs + i;
        if (reuse_buf(ctx, &m_reuse, b)) { b_reuse++; } else { e_vmem += b->size; }
        FARF(HIGH, "prep-buf #%u : pass0 fd %u base %p size %u flags 0x%x", i, b->fd, (void*) b->base, (uint32_t) b->size, b->flags);
    }

    if (b_reuse == n_bufs) return; // all bufs reuse existing mappings

    // See how much vmem we have mmaped right now
    for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) { m_vmem += ctx->mmap[i].size; }

    FARF(HIGH, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu n-bufs %u b-reuse %u", m_vmem, e_vmem, n_bufs, b_reuse);

    if ((m_vmem + e_vmem) > HTP_OP_MAX_VMEM) {
        // Drop unused mappings
        for (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
            bool used = m_reuse & (1<<i);
            if (!used) { drop_mmap(ctx, ctx->mmap + i); }
        }
    }

    // Create missing mappings
    for (uint32_t i=0; i < n_bufs; i++) {
        struct htp_buf_desc *b = bufs + i;
        mmap_buf(ctx, b);
        FARF(HIGH, "prep-buf #%u : pass1 fd %u base %p size %u flags 0x%x", i, b->fd, (void*) b->base, (uint32_t) b->size, b->flags);
    }
}

static void prep_tensor(struct htp_context *ctx, struct htp_buf_desc *bufs, uint32_t idx, struct htp_tensor *t) {
    uint32_t offset = t->data;
    uint32_t size   = t->size;
    uint32_t bi     = t->bi;

    t->data = bufs[bi].base + offset; // update data to the actual pointer

    FARF(HIGH, "prep-tensor #%u: bi %u offset %u size %u data %p : %u:%u:%u:%u", idx, t->bi, offset, t->size, (void*) t->data,
        t->ne[0], t->ne[1], t->ne[3], t->ne[3]);
}

static void prep_tensors(struct htp_context *ctx, struct htp_buf_desc *bufs, struct htp_tensor *tens, uint32_t n_tens) {
    for (uint32_t i=0; i < n_tens; i++) {
        prep_tensor(ctx, bufs, i, tens + i);
    }
}

static void proc_op_req(struct htp_ops_context * octx, struct htp_tensor *tens, uint32_t idx, struct htp_op_desc * op) {
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
            // flush compute buffers on input
            hex_l2flush((void *) src->data, src->size);
        }

        FARF(HIGH, "prep-src #%u: data %p size %u : %u:%u:%u:%u", op->src[i], (void*) src->data, src->size,
            src->ne[0], src->ne[1], src->ne[3], src->ne[3]);
    }

    // Prep output tensor
    struct htp_tensor *dst = tens + op->dst;

    octx->dst = dst;

    FARF(HIGH, "prep-dst #%u: data %p size %u : %u:%u:%u:%u", op->dst, (void*) dst->data, dst->size,
        dst->ne[0], dst->ne[1], dst->ne[3], dst->ne[3]);

    (void) execute_op(octx);

    // flush buffers on output
    hex_l2flush((void *) dst->data, dst->size);
    dst->flags |= HTP_TENSOR_FLUSHED;

    FARF(HIGH, "post-dst #%u: data %p size %u : %u:%u:%u:%u", op->dst, (void*) dst->data, dst->size,
        dst->ne[0], dst->ne[1], dst->ne[3], dst->ne[3]);
}

#define DSPQUEUE_POLL_TIMEOUT_USEC 100
#define DSPQUEUE_POLL_COUNT        100

static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
    struct htp_context * ctx = (struct htp_context *) context;

    int err;

    uint32_t poll_count = DSPQUEUE_POLL_COUNT;

    vtcm_acquire(ctx);

    while (!ctx->vtcm_needs_release) {
        struct htp_opbatch_req req;
        uint32_t r_size = sizeof(req);

        struct dspqueue_buffer dbuf;
        uint32_t n_dbufs = 1;
        uint32_t flags   = 0;

        err = dspqueue_read_noblock(queue, &flags, n_dbufs, &n_dbufs, &dbuf, r_size, &r_size, (uint8_t *) &req);
        if (err == AEE_EWOULDBLOCK) {
            if (--poll_count) {
                qurt_sleep(DSPQUEUE_POLL_TIMEOUT_USEC);
                continue;
            }
            break;
        }

        if (err != 0) {
            FARF(ERROR, "dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            break;
        }

        if (r_size < sizeof(req) || n_dbufs != 1) {
            FARF(ERROR, "invalid request : size %u n-dbufs %u", r_size, n_dbufs);
            continue;
        }

        // Reset poll count for valid requests
        poll_count = DSPQUEUE_POLL_COUNT;

        const uint32_t n_bufs = req.n_bufs;
        const uint32_t n_tens = req.n_tensors;
        const uint32_t n_ops  = req.n_ops;

        const uint32_t b_size = sizeof(struct htp_buf_desc)  * n_bufs;
        const uint32_t t_size = sizeof(struct htp_tensor)    * n_tens;
        const uint32_t o_size = sizeof(struct htp_op_desc)   * n_ops;
        const uint32_t p_size = sizeof(struct htp_prof_desc) * n_ops;

        if (dbuf.size < b_size + t_size + o_size + p_size) {
            FARF(ERROR, "invalid opbatch memory block size %u", dbuf.size);
            break;
        }

        FARF(HIGH, "processing opbatch #%u: n-bufs %u n-tensors %u n-ops %u : m-size %u b-size %u t-size %u o-size %u", req.id,
                n_bufs, n_tens, n_ops, dbuf.size, b_size, t_size, o_size);

        // Setup descriptor pointers
        uint8_t * m_ptr = dbuf.ptr;
        struct htp_buf_desc* bufs = (struct htp_buf_desc*)  m_ptr; m_ptr += b_size;
        struct htp_tensor*   tens = (struct htp_tensor*)    m_ptr; m_ptr += t_size;
        struct htp_op_desc*   ops = (struct htp_op_desc*)   m_ptr; m_ptr += o_size;
        struct htp_prof_desc* pds = (struct htp_prof_desc*) m_ptr;

        prep_op_bufs(ctx, bufs, n_bufs);
        prep_tensors(ctx, bufs, tens, n_tens);

        struct htp_ops_context *octx = &ctx->octx;
        memset(octx, 0, sizeof(*octx));
        octx->n_threads = ctx->n_threads;
        octx->ctx       = ctx;

        for (uint32_t i=0; i < n_ops; i++) {
            struct profile_data prof;

            profile_start(ctx->profiler, &prof);

            proc_op_req(octx, tens, i, &ops[i]);

            profile_stop(ctx->profiler, &prof);

            if (ctx->profiler) {
                pds[i].opcode = ops[i].opcode;
                pds[i].usecs  = prof.usecs;
                pds[i].cycles = prof.cycles;
                for (int j = 0; j < HEX_NUM_PMU_COUNTERS; j++) {
                    pds[i].pmu[j] = prof.pmu_counters[j];
                }
            }
        }

        // dspqueue_write_early_wakeup_noblock(ctx->queue, 10, 0);

        struct htp_opbatch_rsp rsp;
        rsp.id        = req.id;
        rsp.status    = HTP_STATUS_OK;
        rsp.n_bufs    = n_bufs;
        rsp.n_tensors = n_tens;
        rsp.n_ops     = n_ops;

        dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

        err = dspqueue_write(queue, 0, 1, &dbuf, sizeof(rsp), (const uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
        if (err != 0) {
            FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
            break;
        }
    }

    vtcm_release(ctx);
}
