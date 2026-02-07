#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"


#define sum_rows_preamble                       \
    struct htp_tensor *src0 =  &octx->src0;\
    struct htp_tensor *dst  = &octx->dst;  \
                                           \
    const uint32_t ne00 = src0->ne[0];     \
    const uint32_t ne01 = src0->ne[1];     \
    const uint32_t ne02 = src0->ne[2];     \
    const uint32_t ne03 = src0->ne[3];     \
                                           \
    const uint32_t nb00 = src0->nb[0];     \
    const uint32_t nb01 = src0->nb[1];     \
    const uint32_t nb02 = src0->nb[2];     \
    const uint32_t nb03 = src0->nb[3];     \
                                           \
    const uint32_t  ne0 = dst->ne[0];      \
    const uint32_t  ne1 = dst->ne[1];      \
    const uint32_t  ne2 = dst->ne[2];      \
    const uint32_t  ne3 = dst->ne[3];      \
                                           \
    const uint32_t  nb0 = dst->nb[0];      \
    const uint32_t  nb1 = dst->nb[1];      \
    const uint32_t  nb2 = dst->nb[2];      \
    const uint32_t  nb3 = dst->nb[3];      \

static int sum_rows_thread_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    sum_rows_preamble;

    const uint32_t src0_nrows_per_thread  = octx->src0_nrows_per_thread;
    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return HTP_STATUS_OK;
    }

    int opt_path   = 0;
    if ((0 == hex_is_aligned((void *) src0->data, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_path = 1;
    }

    const uint8_t * restrict data_src = (const uint8_t *) src0->data;
    uint8_t * restrict data_dst       = (uint8_t *) dst->data;

    const float * restrict src_th = (float *) (data_src + (src0_start_row * src0_row_size));
    float * restrict dst_th       = (float *) (data_dst + (src0_start_row * dst_row_size));

    for (uint32_t ir = 0; ir < src0_nrows_per_thread; ir++) {
        const float * restrict src_local = src_th + (ir * ne00);

        if (ir + 1 < src0_nrows_per_thread) {
            hex_l2fetch(src_local + ne00, src0_row_size, src0_row_size, 1);
        }

        if (1 == opt_path) {
            dst_th[ir] = hvx_reduce_sum_f32_a((const uint8_t *) src_local, ne00);
        } else {
            dst_th[ir] = hvx_reduce_sum_f32((const uint8_t *) src_local, ne00);
        }
    }

    return HTP_STATUS_OK;
}

static void sum_rows_work_f32(unsigned int n, unsigned int i, void *data) {
    sum_rows_thread_f32((struct htp_ops_context *) data, n, i);
}

int op_sum_rows(struct htp_ops_context * octx) {
    sum_rows_preamble;

    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int      n_threads  = octx->n_threads;
    const uint32_t src0_nrows = ne01 * ne02 * ne03;

    uint32_t n_jobs = MIN(n_threads, src0_nrows);
    octx->src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

    worker_pool_run_func(octx->ctx->worker_pool, sum_rows_work_f32, octx, n_jobs);

    return HTP_STATUS_OK;
}

