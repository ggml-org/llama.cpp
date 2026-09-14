#ifndef HTP_FENCE_H
#define HTP_FENCE_H

#include <stdatomic.h>
#include <stdint.h>

#include <HAP_farf.h>

#include "hex-utils.h"
#include "htp-ops.h"
#include "htp-ctx.h"

static inline atomic_uint * htp_mdev_fence_slot(const void * fence_base, uint32_t idx) {
    return (atomic_uint *) ((const uint8_t *) fence_base + (size_t) idx * HTP_FENCE_SLOT_SIZE);
}

static inline void htp_fence_write(void * fence_ptr, uint32_t seq, uint32_t status) {
    atomic_uint * fence = (atomic_uint *) fence_ptr;
    atomic_store(&fence[1], status);
    atomic_store(&fence[0], seq);
    asm volatile ("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) fence);
}

static inline void htp_fence_read(const void * fence_ptr, uint32_t * seq, uint32_t * status) {
    const atomic_uint * fence = (const atomic_uint *) fence_ptr;
    Q6_dccleaninva_A((void *) fence);
    asm volatile ("syncht" : : : "memory");
    *seq = atomic_load(&fence[0]);
    *status = atomic_load(&fence[1]);
}

static inline void htp_mdev_group_barrier(struct htp_ops_context * octx) {
    struct htp_context * ctx = octx->ctx;
    if (ctx->mdev.count <= 1) {
        return;
    }

    const uint32_t seq = ++ctx->mdev.fence_seq;

    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);

    const uint32_t mdev_idx   = ctx->mdev.idx;
    const uint32_t mdev_count = ctx->mdev.count;

    uint8_t * fence_base   = ctx->mdev.fence_base;
    atomic_uint * my_fence = htp_mdev_fence_slot(fence_base, mdev_idx);
    htp_fence_write(my_fence, seq, octx->status);

    for (uint32_t d = 0; d < mdev_count; d++) {
        if (d == mdev_idx) continue;
        atomic_uint * peer_fence = htp_mdev_fence_slot(fence_base, d);
        uint64_t spins = 0;
        while (1) {
            uint32_t peer_seq;
            uint32_t peer_status;
            htp_fence_read(peer_fence, &peer_seq, &peer_status);
            if ((int32_t)(peer_seq - seq) >= 0) {
                if (peer_status > HTP_STATUS_OK) {
                    FARF(ERROR, "ggml-hex: mdev %u peer %u failed with status %u : seq 0x%08x\n",
                         mdev_idx, d, peer_status, seq);
                    htp_ops_context_set_status(octx, peer_status);
                }
                break;
            }
            if (++spins == 10000) {
                FARF(ALWAYS, "ggml-hex: mdev %u waiting for mdev %u : seq 0x%08x (b %u op %u) my-fence %p peer-fence %p peer-seq 0x%08x (diff %d)\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, my_fence, peer_fence, peer_seq, (int32_t)(peer_seq - seq));
            }
            if (spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: mdev %u timeout waiting for mdev %u : seq 0x%08x (b %u op %u) peer-fence %p peer-seq 0x%08x\n",
                     mdev_idx, d, seq, seq >> 12, seq & 0xfff, peer_fence, peer_seq);
                htp_ops_context_set_status(octx, HTP_STATUS_INTERNAL_ERR);
                break;
            }
            hex_pause();
        }
    }
    asm volatile ("syncht" : : : "memory");

    if (octx->status > HTP_STATUS_OK) {
        htp_fence_write(my_fence, seq, octx->status);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, (uint16_t) seq);
}

#endif // HTP_FENCE_H
