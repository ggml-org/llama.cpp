#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-types.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hex-dma.h"

#define htp_diag_tensors_preamble                           \
    const struct htp_tensor * restrict src0 = octx->src[0]; \
    const struct htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    const uint32_t ne02 = src0->ne[2];                      \
                                                            \
    const uint32_t ne0 = dst->ne[0];                        \
    const uint32_t ne1 = dst->ne[1];                        \
                                                            \
    const uint32_t nb02 = src0->nb[2];                      \
    const uint32_t nb03 = src0->nb[3];                      \
                                                            \
    const uint32_t nb1 = dst->nb[1];                        \
    const uint32_t nb2 = dst->nb[2];                        \
    const uint32_t nb3 = dst->nb[3];

struct htp_diag_context {
    struct htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    struct htp_diag_context * dctx = (struct htp_diag_context *) data; \
    struct htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

static inline void hvx_diag_row_f32(const float * restrict src, float * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

static void diag_thread_f32_dma(unsigned int nth, unsigned int ith, void * data) {
    htp_diag_preamble;
    dma_queue * dma_queue = octx->ctx->dma[ith];

    const uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    const uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    if (ib0 >= ib1) {
        return;
    }

    const size_t src_batch_size         = dctx->src_batch_size;
    const size_t dst_row_size           = dctx->dst_row_size;
    const size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    const size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t ib = ib0; ib < ib1; ib++) {
        const uint32_t i3 = ib / ne02;
        const uint32_t i2 = ib % ne02;

        const uint8_t * src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_push_ddr_to_vtcm(dma_queue,
                                   dma_make_ptr(src_spad, src_batch),
                                   src_batch_size_aligned, src_batch_size, 1);
        dma_queue_flush(dma_queue);

        const float * src_spad_f32 = (const float *) src_spad;
        float       * dst_spad_f32 = (float *) dst_spad;

        for (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            uint8_t * dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_push_vtcm_to_ddr(dma_queue,
                                       dma_make_ptr(dst_row, dst_spad),
                                       dst_row_size, dst_row_size_aligned, 1);
            dma_queue_flush(dma_queue);
        }
    }

    FARF(HIGH, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

static void diag_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    htp_diag_preamble;

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    const uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    const uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    for (uint32_t ib = ib0; ib < ib1; ib++) {
        const uint32_t i3 = ib / ne02;
        const uint32_t i2 = ib % ne02;

        const float * restrict src_batch = (const float *)(src_data + i3 * nb03 + i2 * nb02);

        for (uint32_t i1 = 0; i1 < ne1; i1++) {
            float * restrict dst_row = (float *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(HIGH, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

int op_diag_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t total_batches = src0->ne[2] * src0->ne[3];
    const size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    if (octx->ctx->mdev.count > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == sizeof(float)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        if (can_split) {
            if (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } else if (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_LINE_SIZE)) : 1;
            } else {
                can_split = false;
            }
        }

        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        batch_start = range.start;
        nbatches    = range.count;
    }

    if (nbatches == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads     = octx->n_threads;

    const size_t src_batch_size         = src0->ne[0] * sizeof(float);
    const size_t dst_row_size           = dst->ne[0] * sizeof(float);
    const size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    const size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    const size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = NULL;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = NULL;

    struct htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    if (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    return HTP_STATUS_OK;
}

int op_diag(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    int err = HTP_STATUS_OK;

    switch (dst->type) {
        case HTP_TYPE_F32:
            err = op_diag_f32(octx);
            break;
        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
