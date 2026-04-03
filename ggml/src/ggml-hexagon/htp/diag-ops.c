#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "hex-dma.h"

#define htp_diag_tensors_preamble                    \
    struct htp_tensor * restrict src0 = &octx->src0; \
    struct htp_tensor * restrict dst  = &octx->dst;  \
                                                     \
    const uint32_t ne00 = src0->ne[0];               \
    const uint32_t ne01 = src0->ne[1];               \
    const uint32_t ne02 = src0->ne[2];               \
    const uint32_t ne03 = src0->ne[3];               \
                                                     \
    const uint32_t ne0 = dst->ne[0];                 \
    const uint32_t ne1 = dst->ne[1];                 \
    const uint32_t ne2 = dst->ne[2];                 \
    const uint32_t ne3 = dst->ne[3];                 \
                                                     \
    const uint32_t nb00 = src0->nb[0];               \
    const uint32_t nb01 = src0->nb[1];               \
    const uint32_t nb02 = src0->nb[2];               \
    const uint32_t nb03 = src0->nb[3];               \
                                                     \
    const uint32_t nb0 = dst->nb[0];                 \
    const uint32_t nb1 = dst->nb[1];                 \
    const uint32_t nb2 = dst->nb[2];                 \
    const uint32_t nb3 = dst->nb[3];

struct htp_diag_context {
    struct htp_ops_context * octx;
    uint32_t rows_per_thread;
    uint32_t total_rows;
};

#define htp_diag_preamble                                            \
    struct htp_diag_context * dctx = (struct htp_diag_context *) data; \
    struct htp_ops_context *  octx = dctx->octx;                     \
    htp_diag_tensors_preamble;

// ---------------------------------------------------------------------------
// Per thread worker: Direct computation (no DMA needed for this simple op)
// ---------------------------------------------------------------------------

static void diag_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    htp_diag_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    // Total rows in the 3D/4D batch (ne2 * ne3)
    const uint32_t ir0 = dctx->rows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + dctx->rows_per_thread, dctx->total_rows);

    // For diag:
    // - Input shape:  [ne00, 1, ne02, ne03]
    // - Output shape: [ne0, ne1, ne2, ne3] where ne0 == ne1 == ne00
    //
    // For each batch (i2, i3):
    //   For each output row i1 in [0, ne1):
    //     dst[i0, i1, i2, i3] = (i0 == i1) ? src[i1, 0, i2, i3] : 0

    for (uint32_t ir = ir0; ir < ir1; ir++) {
        // Decompose ir into i3, i2
        const uint32_t i3 = ir / ne02;
        const uint32_t i2 = ir % ne02;

        // For each output row i1
        for (uint32_t i1 = 0; i1 < ne1; i1++) {
            float * restrict d = (float *)((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            const float * restrict s = (const float *)((char *) src_data + i3 * nb03 + i2 * nb02);

            // Set elements before diagonal to 0
            for (uint32_t i0 = 0; i0 < i1; i0++) {
                d[i0] = 0.0f;
            }

            // Set diagonal element
            d[i1] = s[i1];

            // Set elements after diagonal to 0
            for (uint32_t i0 = i1 + 1; i0 < ne0; i0++) {
                d[i0] = 0.0f;
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_diag_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * dst  = &octx->dst;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // Total batches (iterate over ne2 * ne3)
    const uint32_t total_rows = src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = MIN(octx->n_threads, total_rows);

    struct htp_diag_context dctx = {
        .octx            = octx,
        .rows_per_thread = (total_rows + n_threads - 1) / n_threads,
        .total_rows      = total_rows,
    };

    worker_pool_run_func(octx->ctx->worker_pool, diag_thread_f32, &dctx, n_threads);

    return HTP_STATUS_OK;
}

int op_diag(struct htp_ops_context * octx) {
    int                 err = HTP_STATUS_OK;
    struct htp_tensor * dst = &octx->dst;

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
