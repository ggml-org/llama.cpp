// HMX manager — owns the HMX worker pool singletons and global VTCM/DMA state.
// HMX hardware locking uses HAP_compute_res_hmx_lock/unlock on the VTCM resource
// context (vtcm_rctx), which already has HMX capability from vtcm_alloc().
// Ported from htp-ops-lib/include/dsp/hmx_mgr.h.

#ifndef HMX_MGR_H
#define HMX_MGR_H

#include "hmx-worker-pool.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Global HMX worker pool context (1 worker for HMX tile compute).
extern hmx_worker_pool_context_t hmx_worker_pool_ctx;

// Default HVX worker pool context (N workers for parallel dequantize / transfer).
extern hmx_worker_pool_context_t hmx_default_pool_ctx;

// Number of available HVX 128-byte mode contexts on this device.
extern unsigned int hmx_num_hvx128_contexts;

// Global VTCM state — set by main.c during HMX initialisation via hmx_set_vtcm_state().
extern uint8_t *hmx_vtcm_base;
extern size_t   hmx_vtcm_scratch_size;
extern uint8_t *hmx_exp2_table;

// Global DMA queue for HMX operations (from htp_context.hmx_dma).
extern void *hmx_dma_ptr;  // actually dma_queue *, void * to avoid header dep

void hmx_manager_setup(void);
void hmx_manager_reset(void);

// Set VTCM / DMA / resource-context state (called from main.c after vtcm_alloc).
// rctx is the VTCM compute-resource context that already has HMX capability;
// it is used by hmx_unit_acquire/release for HAP_compute_res_hmx_lock/unlock.
void hmx_set_vtcm_state(uint8_t *vtcm_base, size_t scratch_size,
                         uint8_t *exp2_table, void *dma, uint32_t rctx);

// Exclusive HMX hardware lock/unlock via HAP_compute_res_hmx_lock/unlock.
// All HMX tile instructions must be bracketed by these calls.
void hmx_unit_acquire(void);
void hmx_unit_release(void);

#ifdef __cplusplus
}
#endif

#endif // HMX_MGR_H
