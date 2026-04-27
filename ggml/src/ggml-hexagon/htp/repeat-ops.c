#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"

struct htp_repeat_context {
    struct htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3

    size_t   type_size;
};

static void repeat_job_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = rctx->nr0;
    const uint32_t nr1 = rctx->nr1;
    const uint32_t nr2 = rctx->nr2;
    const uint32_t nr3 = rctx->nr3;

    const size_t row_bytes = ne00 * rctx->type_size;

    const uint32_t row_start = rctx->nrows_per_thread * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->total_dst_rows);

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        const uint32_t i1 = dst_row % ne1;
        const uint32_t i2 = (dst_row / ne1) % ne2;
        const uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        const uint32_t k1 = i1 % ne01;
        const uint32_t k2 = i2 % ne02;
        const uint32_t k3 = i3 % ne03;

        const uint8_t * src_row = (const uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        for (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u usec %u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_repeat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    if (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        return HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    switch (src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        case HTP_TYPE_F16: type_size = 2; break;
        default:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const uint32_t n_threads = MIN(octx->n_threads, total_dst_rows);

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_per_thread = (total_dst_rows + n_threads - 1) / n_threads,
        .total_dst_rows   = total_dst_rows,
        .type_size        = type_size,
    };

    FARF(HIGH, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    worker_pool_run_func(octx->ctx->worker_pool, repeat_job_per_thread, &rctx, n_threads);

    return HTP_STATUS_OK;
}
