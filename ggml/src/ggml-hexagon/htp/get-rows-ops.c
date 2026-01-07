#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

#define get_rows_preamble \
    const uint32_t ne00 = octx->src0.ne[0]; \
    const uint32_t ne01 = octx->src0.ne[1]; \
    const uint32_t ne02 = octx->src0.ne[2]; \
    const uint32_t ne03 = octx->src0.ne[3]; \
                                            \
    const uint32_t ne10 = octx->src1.ne[0]; \
    const uint32_t ne11 = octx->src1.ne[1]; \
    const uint32_t ne12 = octx->src1.ne[2]; \
                                            \
    const uint32_t nb01 = octx->src0.nb[1]; \
    const uint32_t nb02 = octx->src0.nb[2]; \
    const uint32_t nb03 = octx->src0.nb[3]; \
                                            \
    const uint32_t nb10 = octx->src1.nb[0]; \
    const uint32_t nb11 = octx->src1.nb[1]; \
    const uint32_t nb12 = octx->src1.nb[2]; \
                                            \
    const uint32_t nb1 = octx->dst.nb[1];   \
    const uint32_t nb2 = octx->dst.nb[2];   \
    const uint32_t nb3 = octx->dst.nb[3];   \
                                            \
    const uint32_t nr = ne10 * ne11 * ne12;

static int get_rows_thread_f32_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    get_rows_preamble;

    // parallelize by src1 elements (which correspond to dst rows)
    const uint32_t dr  = octx->src1_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    const bool is_i32 = (octx->src1.type == HTP_TYPE_I32);

    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &octx->get_rows_div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &octx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src1.data + i10*nb10 + i11*nb11 + i12*nb12;

        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            // invalid index, skip for now to avoid crash
            continue;
        }

        const uintptr_t src0_ptr = octx->src0.data + i01*nb01 + i11*nb02 + i12*nb03;
        const uintptr_t dst_ptr  = octx->dst.data  + i10*nb1  + i11*nb2  + i12*nb3;
        hvx_copy_fp32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, ne00);
    }

    return HTP_STATUS_OK;
}

static void get_rows_work_f32_f32(unsigned int n, unsigned int i, void *data) {
    get_rows_thread_f32_f32((struct htp_ops_context *) data, n, i);
}

int op_get_rows(struct htp_ops_context * octx) {
    get_rows_preamble;

    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src1.type != HTP_TYPE_I32 && octx->src1.type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    octx->get_rows_div_ne10      = init_fastdiv_values(octx->src1.ne[0]);
    octx->get_rows_div_ne10_ne11 = init_fastdiv_values(octx->src1.ne[0] * octx->src1.ne[1]);

    const uint32_t n_jobs = MIN(nr, octx->n_threads);
    octx->src1_nrows_per_thread = (nr + n_jobs - 1) / n_jobs;

    worker_pool_run_func(octx->ctx->worker_pool, get_rows_work_f32_f32, octx, n_jobs);
    return HTP_STATUS_OK;
}
