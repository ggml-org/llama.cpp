// HMX manager implementation.
// Ported from htp-ops-lib/src/dsp/hmx_mgr.c.
//
// Changes from the original:
//   - Uses hmx_worker_pool_init_ex / hmx_worker_pool_deinit (prefixed API)
//   - Adds default HVX worker pool for parallel dequantize/transfer tasks
//   - Adds global VTCM/DMA state accessors
//   - HMX locking simplified: uses HAP_compute_res_hmx_lock/unlock on the
//     VTCM resource context (which already has HMX capability from vtcm_alloc).
//     This replaces the original separate HAP_compute_res_acquire for HMX +
//     SHARED lock + custom spin-lock design.

#include "hmx-mgr.h"
#include "hmx-worker-pool.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "qurt.h"
#ifdef __cplusplus
}
#endif

// VTCM resource context (from vtcm_alloc, already HMX-capable).
// Used by hmx_unit_acquire/release for HAP_compute_res_hmx_lock/unlock.
static uint32_t hmx_rctx;

// Global worker pools
hmx_worker_pool_context_t hmx_worker_pool_ctx;
hmx_worker_pool_context_t hmx_default_pool_ctx;

// Number of HVX 128-byte contexts
unsigned int hmx_num_hvx128_contexts;

// Global VTCM / DMA state
uint8_t *hmx_vtcm_base;
size_t   hmx_vtcm_scratch_size;
uint8_t *hmx_exp2_table;
void    *hmx_dma_ptr;

void hmx_set_vtcm_state(uint8_t *vtcm_base, size_t scratch_size,
                         uint8_t *exp2_table, void *dma, uint32_t rctx) {
    hmx_vtcm_base         = vtcm_base;
    hmx_vtcm_scratch_size = scratch_size;
    hmx_exp2_table        = exp2_table;
    hmx_dma_ptr           = dma;
    hmx_rctx              = rctx;
}

void hmx_manager_setup(void) {
    // Query HVX context count from hardware
    hmx_num_hvx128_contexts = (qurt_hvx_get_units() >> 8) & 0xFF;
    if (hmx_num_hvx128_contexts == 0) {
        hmx_num_hvx128_contexts = 1; // fallback
    }

    // No separate HAP_compute_res_acquire needed — vtcm_rctx already has
    // HMX capability (set_hmx_param in vtcm_alloc).  HMX locking goes
    // through hmx_unit_acquire/release which use vtcm_rctx directly.

    // Create HMX worker pool (1 worker for HMX tile compute)
    int err = hmx_worker_pool_init_ex(&hmx_worker_pool_ctx, 8192, 1, 0);
    if (err) {
        FARF(ALWAYS, "%s: HMX worker pool init failed", __func__);
    }

    // Create default HVX worker pool (N workers for parallel data processing)
    err = hmx_worker_pool_init_ex(&hmx_default_pool_ctx, 8192,
                                   hmx_num_hvx128_contexts, 0);
    if (err) {
        FARF(ALWAYS, "%s: default HVX worker pool init failed", __func__);
    }
}

void hmx_manager_reset(void) {
    if (hmx_default_pool_ctx) {
        hmx_worker_pool_deinit(&hmx_default_pool_ctx);
    }

    if (hmx_worker_pool_ctx) {
        hmx_worker_pool_deinit(&hmx_worker_pool_ctx);
    }
}

void hmx_unit_acquire(void) {
    if (!hmx_rctx) return;

    int err = HAP_compute_res_hmx_lock(hmx_rctx);
    if (err) {
        FARF(ALWAYS, "HAP_compute_res_hmx_lock failed: 0x%x", err);
    }
}

void hmx_unit_release(void) {
    if (!hmx_rctx) return;

    HAP_compute_res_hmx_unlock(hmx_rctx);
}
