#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#define set_rows_preamble \
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
    const uint32_t ne1 = octx->dst.ne[1];   \
                                            \
    const uint32_t nr  = ne01;

static int set_rows_thread_f32_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    set_rows_preamble;

    // parallelize by rows of src0
    const uint32_t dr  = octx->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    const bool is_i32 = (octx->src1.type == HTP_TYPE_I32);

    for (uint32_t i03 = 0; i03 < ne03; ++i03) {
        for (uint32_t i02 = 0; i02 < ne02; ++i02) {
            for (uint32_t i = ir0; i < ir1; ++i) {
                const uint32_t i12 = fastmodulo(i03, ne12, &octx->set_rows_div_ne12);
                const uint32_t i11 = fastmodulo(i02, ne11, &octx->set_rows_div_ne11);
                const uint32_t i10 = i;

                const uintptr_t src1_addr = octx->src1.data + i10*nb10 + i11*nb11 + i12*nb12;

                uint32_t i1 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;
                if (i1 >= ne1) {
                    // ignore invalid indices
                    continue;
                }

                const uintptr_t src0_ptr = octx->src0.data + i*nb01 + i02*nb02 + i03*nb03;
                const uintptr_t dst_ptr  = octx->dst.data  + i1*nb1 + i02*nb2  + i03*nb3;

                // copy row
                hvx_copy_f32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, ne00);
            }
        }
    }

    return HTP_STATUS_OK;
}

static int set_rows_thread_f16_f32(struct htp_ops_context * octx, const int nth, const int ith) {
    set_rows_preamble;

    // parallelize by rows of src0
    const uint32_t dr  = octx->src0_nrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    const bool is_i32 = (octx->src1.type == HTP_TYPE_I32);

    for (uint32_t i03 = 0; i03 < ne03; ++i03) {
        for (uint32_t i02 = 0; i02 < ne02; ++i02) {
            for (uint32_t i = ir0; i < ir1; ++i) {
                const uint32_t i12 = fastmodulo(i03, ne12, &octx->set_rows_div_ne12);
                const uint32_t i11 = fastmodulo(i02, ne11, &octx->set_rows_div_ne11);
                const uint32_t i10 = i;

                const uintptr_t src1_addr = octx->src1.data + i10*nb10 + i11*nb11 + i12*nb12;

                uint32_t i1 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;
                if (i1 >= ne1) {
                    // ignore invalid indices
                    continue;
                }

                const uint8_t* src0_ptr = (const uint8_t *) octx->src0.data + i*nb01 + i02*nb02 + i03*nb03;
                uint8_t*       dst_ptr  = (uint8_t *)       octx->dst.data  + i1*nb1 + i02*nb2  + i03*nb3;

                hvx_copy_f16_f32_uu(dst_ptr, src0_ptr, ne00);
            }
        }
    }

    return HTP_STATUS_OK;
}

static void set_rows_work_f16_f32(unsigned int n, unsigned int i, void *data) {
    set_rows_thread_f16_f32((struct htp_ops_context *) data, n, i);
}

static void set_rows_work_f32_f32(unsigned int n, unsigned int i, void *data) {
    set_rows_thread_f32_f32((struct htp_ops_context *) data, n, i);
}

int op_set_rows(struct htp_ops_context * octx) {
    set_rows_preamble;

    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst.type != HTP_TYPE_F32 && octx->dst.type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src1.type != HTP_TYPE_I32 && octx->src1.type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    octx->set_rows_div_ne12 = init_fastdiv_values(ne12);
    octx->set_rows_div_ne11 = init_fastdiv_values(ne11);

    const uint32_t n_jobs = MIN(nr, octx->n_threads);
    octx->src0_nrows_per_thread = (nr + n_jobs - 1) / n_jobs;

    switch(octx->dst.type) {
    case HTP_TYPE_F32:
        worker_pool_run_func(octx->ctx->worker_pool, set_rows_work_f32_f32, octx, n_jobs);
        break;
    case HTP_TYPE_F16:
        worker_pool_run_func(octx->ctx->worker_pool, set_rows_work_f16_f32, octx, n_jobs);
        break;
    default:
        return HTP_STATUS_NO_SUPPORT;
    }

    return HTP_STATUS_OK;
}
