#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hvx-copy.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#define fill_preamble \
    const struct htp_tensor * dst = octx->dst; \
    \
    const uint32_t ne0 = dst->ne[0]; \
    const uint32_t ne1 = dst->ne[1]; \
    const uint32_t ne2 = dst->ne[2]; \
    const uint32_t ne3 = dst->ne[3]; \
    \
    const uint32_t nb1 = dst->nb[1]; \
    const uint32_t nb2 = dst->nb[2]; \
    const uint32_t nb3 = dst->nb[3]; \
    \
    const uint32_t nr = ne1 * ne2 * ne3;

struct htp_fill_context {
    struct htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t total_rows;  // ne1 * ne2 * ne3
    bool     opt_path;
    HVX_Vector splat_vec;
    uint32_t   elem_size;
};

static void fill_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
    struct htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    const uint32_t ir0 = fctx->nrows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->total_rows);

    uint64_t t1 = HAP_perf_get_qtimer_count();

    if (fctx->opt_path) {
        // Opt path: tensor is fully contiguous, treat as flat array
        const uint32_t elem_start = ir0 * ne0;
        const uint32_t elem_end = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
    } else {
        // Non-contiguous path: must respect strides
        for (uint32_t ir = ir0; ir < ir1; ++ir) {
            const uint32_t i1 = ir % ne1;
            const uint32_t i2 = (ir / ne1) % ne2;
            const uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
        }
    }

    uint64_t t2 = HAP_perf_get_qtimer_count();
    FARF(HIGH, "fill %u/%u: rows %u:%u usec %u\n",
         ith, nth, ir0, ir1, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_fill(struct htp_ops_context * octx) {
    fill_preamble;

    if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    const uint32_t n_threads = MIN(nr, octx->n_threads);

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    const bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (int) opt_path);

    float val_f32 = 0.f;
    memcpy(&val_f32, &octx->op_params[0], sizeof(float));

    struct htp_fill_context fctx = {
        .octx             = octx,
        .nrows_per_thread = (nr + n_threads - 1) / n_threads,
        .total_rows       = nr,
        .opt_path         = opt_path,
    };

    switch (dst->type) {
    case HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_size = sizeof(float);
        break;
    case HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_size = sizeof(_Float16);
        break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_pool_run_func(octx->ctx->worker_pool, fill_thread, &fctx, n_threads);

    return HTP_STATUS_OK;
}
