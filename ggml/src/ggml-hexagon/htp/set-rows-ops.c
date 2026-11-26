#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dma-queue.h"
#include "dma-pipeline.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hvx-quant.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/set-rows-ops.h"

#define set_rows_preamble                                              \
    const uint32_t __attribute__((unused)) ne00 = octx->src[0]->ne[0]; \
    const uint32_t __attribute__((unused)) ne01 = octx->src[0]->ne[1]; \
    const uint32_t __attribute__((unused)) ne02 = octx->src[0]->ne[2]; \
    const uint32_t __attribute__((unused)) ne03 = octx->src[0]->ne[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) ne10 = octx->src[1]->ne[0]; \
    const uint32_t __attribute__((unused)) ne11 = octx->src[1]->ne[1]; \
    const uint32_t __attribute__((unused)) ne12 = octx->src[1]->ne[2]; \
    const uint32_t __attribute__((unused)) ne13 = octx->src[1]->ne[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) nb01 = octx->src[0]->nb[1]; \
    const uint32_t __attribute__((unused)) nb02 = octx->src[0]->nb[2]; \
    const uint32_t __attribute__((unused)) nb03 = octx->src[0]->nb[3]; \
                                                                       \
    const uint32_t __attribute__((unused)) nb10 = octx->src[1]->nb[0]; \
    const uint32_t __attribute__((unused)) nb11 = octx->src[1]->nb[1]; \
    const uint32_t __attribute__((unused)) nb12 = octx->src[1]->nb[2]; \
                                                                       \
    const uint32_t __attribute__((unused)) nb1 = octx->dst->nb[1];     \
    const uint32_t __attribute__((unused)) nb2 = octx->dst->nb[2];     \
    const uint32_t __attribute__((unused)) nb3 = octx->dst->nb[3];     \
                                                                       \
    const uint32_t __attribute__((unused)) ne0 = octx->dst->ne[0];     \
    const uint32_t __attribute__((unused)) ne1 = octx->dst->ne[1];     \
    const uint32_t __attribute__((unused)) ne2 = octx->dst->ne[2];     \
    const uint32_t __attribute__((unused)) ne3 = octx->dst->ne[3];     \
                                                                       \
    const uint32_t __attribute__((unused)) nr  = ne01;

typedef void (*set_rows_compute_fn_t)(void * restrict dst, const void * restrict src, uint32_t n);

struct set_rows_context {
    struct htp_ops_context * octx;
    const struct htp_set_rows_kernel_params * kparams;
    struct htp_set_rows_vtcm_layout vtcm_layout;
    set_rows_compute_fn_t compute;
    uint8_t * vtcm_base;
    const void * src1_data;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_hvx;
    bool     is_i32;
};

static __attribute__((noinline)) void set_rows_row_f32_f32(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_copy_f32_uu((uint8_t *) dst, (const uint8_t *) src, n);
}

static __attribute__((noinline)) void set_rows_row_f16_f32(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_copy_f16_f32_uu((uint8_t *) dst, (const uint8_t *) src, n);
}

static __attribute__((noinline)) void set_rows_row_q8_0_f32(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_quantize_row_q8_0_f32(dst, (const float *) src, n);
}

static __attribute__((noinline)) void set_rows_row_f32_f16(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_copy_f32_f16_uu((uint8_t *) dst, (const uint8_t *) src, n);
}

static __attribute__((noinline)) void set_rows_row_f16_f16(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_copy_f16_uu((uint8_t *) dst, (const uint8_t *) src, n);
}

static __attribute__((noinline)) void set_rows_row_q8_0_f16(void * restrict dst, const void * restrict src, uint32_t n) {
    hvx_quantize_row_q8_0_f16(dst, (const __fp16 *) src, n);
}

struct set_rows_pipe_state {
    dma_addr_t            src0_data;
    uintptr_t             src1_data;
    dma_addr_t            dst_data;
    set_rows_compute_fn_t compute;
    uint32_t              ne00;
    uint32_t              ne02;
    uint32_t              ne1;
    uint32_t              ne11;
    uint32_t              ne12;
    uint32_t              nb01, nb02, nb03;
    uint32_t              nb10, nb11, nb12;
    uint32_t              nb1,  nb2,  nb3;
    uint32_t              src0_row_size;
    uint32_t              dst_row_size;
    size_t                src0_spad_half;
    size_t                dst_spad_half;
    uint32_t              ir0;
    uint32_t              nrows_per_hvx;
    uint32_t              step;
    uint32_t              total_steps;
    uint32_t              ci_step;
    uint32_t              ci02;
    uint32_t              ci03;
    uint32_t              ci11_base;
    uint32_t              ci12_base;
    bool                  is_i32;
};

static bool set_rows_next_task(void * ctx, struct dma_pipe_task * task) {
    struct set_rows_pipe_state * s = (struct set_rows_pipe_state *) ctx;
    if (s->step >= s->total_steps) {
        return false;
    }

    const uint32_t  i         = s->ir0 + s->ci_step;
    const uintptr_t src1_addr = s->src1_data + (size_t) i * s->nb10 + (size_t) s->ci11_base * s->nb11 + (size_t) s->ci12_base * s->nb12;
    const uint64_t  idx       = s->is_i32 ? (uint64_t) *(const int32_t *) src1_addr : (uint64_t) *(const int64_t *) src1_addr;

    task->row = i;
    task->off = s->ci02;
    task->len = s->ci03;
    task->aux = (idx < (uint64_t) s->ne1) ? (uint32_t) idx : UINT32_MAX;

    s->step++;
    s->ci_step++;
    if (s->ci_step == s->nrows_per_hvx) {
        s->ci_step = 0;
        s->ci02++;
        s->ci11_base++;
        if (s->ci11_base == s->ne11) {
            s->ci11_base = 0;
        }
        if (s->ci02 == s->ne02) {
            s->ci02 = 0;
            s->ci03++;
            s->ci12_base++;
            if (s->ci12_base == s->ne12) {
                s->ci12_base = 0;
            }
        }
    }
    return true;
}

static void set_rows_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct set_rows_pipe_state * s = (struct set_rows_pipe_state *) ctx;
    if (task->aux >= s->ne1) {
        dma_queue_push(dma_q, dma_make_data(in_spad, 0), 0, 0, 0, 0);
        return;
    }
    const dma_addr_t src0_addr = s->src0_data + (size_t) task->row * s->nb01 + (size_t) task->off * s->nb02 + (size_t) task->len * s->nb03;
    dma_queue_push(dma_q, dma_make_data(in_spad, src0_addr),
                   s->src0_spad_half, s->src0_row_size, s->src0_row_size, 1);
}

static void set_rows_pipe_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct set_rows_pipe_state * s = (struct set_rows_pipe_state *) ctx;
    if (task->aux < s->ne1) {
        s->compute(out_spad, in_spad, s->ne00);
    }
}

static void set_rows_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct set_rows_pipe_state * s = (struct set_rows_pipe_state *) ctx;
    if (task->aux >= s->ne1) {
        dma_queue_push(dma_q, dma_make_data(0, out_spad), 0, 0, 0, 0);
        return;
    }
    const dma_addr_t dst_addr = s->dst_data + (size_t) task->aux * s->nb1 + (size_t) task->off * s->nb2 + (size_t) task->len * s->nb3;
    dma_queue_push(dma_q, dma_make_data(dst_addr, out_spad),
                   s->dst_row_size, s->dst_spad_half, s->dst_row_size, 1);
}

static void set_rows_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct set_rows_context * srctx = (struct set_rows_context *) data;
    struct htp_ops_context *  octx  = srctx->octx;
    set_rows_preamble;

    const uint32_t dr  = srctx->tasks_per_hvx;
    const uint32_t ir0 = srctx->task_start + dr * ith;
    if (ir0 >= srctx->task_start + srctx->tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);

    const struct htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_src0) + ith * vtcm_layout->src0_bytes_per_hvx;
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_dst)  + ith * vtcm_layout->dst_bytes_per_hvx;
    void *    in_buf[2]  = { vtcm_src0, vtcm_src0 + vtcm_layout->src0_spad_half_size };
    void *    out_buf[2] = { vtcm_dst,  vtcm_dst  + vtcm_layout->dst_spad_half_size };

    const uint32_t nrows_per_hvx = ir1 - ir0;
    const uint32_t idx_elem_size = srctx->is_i32 ? sizeof(int32_t) : sizeof(int64_t);

    struct set_rows_pipe_state s = {
        .src0_data      = octx->src[0]->data,
        .src1_data      = (uintptr_t) srctx->src1_data,
        .dst_data       = octx->dst->data,
        .compute        = srctx->compute,
        .ne00           = ne00,
        .ne02           = ne02,
        .ne1            = ne1,
        .ne11           = ne11,
        .ne12           = ne12,
        .nb01 = nb01, .nb02 = nb02, .nb03 = nb03,
        .nb10 = idx_elem_size, .nb11 = ne10 * idx_elem_size, .nb12 = ne10 * ne11 * idx_elem_size,
        .nb1  = nb1,  .nb2  = nb2,  .nb3  = nb3,
        .src0_row_size  = htp_tensor_get_row_size(octx->src[0]->type, ne00),
        .dst_row_size   = htp_tensor_get_row_size(octx->dst->type, ne00),
        .src0_spad_half = vtcm_layout->src0_spad_half_size,
        .dst_spad_half  = vtcm_layout->dst_spad_half_size,
        .ir0            = ir0,
        .nrows_per_hvx  = nrows_per_hvx,
        .step           = 0,
        .total_steps    = ne03 * ne02 * nrows_per_hvx,
        .ci_step        = 0,
        .ci02           = 0,
        .ci03           = 0,
        .ci11_base      = 0,
        .ci12_base      = 0,
        .is_i32         = srctx->is_i32,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    set_rows_next_task, set_rows_push_in, set_rows_pipe_compute, set_rows_push_out);
}

int op_set_rows(struct htp_ops_context * octx) {
    const struct htp_set_rows_kernel_params * kparams = (const struct htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    const uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        tasks      = range.count;
    }

    if (tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, (uint32_t) kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_hvx = fastdiv(tasks + n_hvx - 1, &octx->n_hvx_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->src[0]->type, octx->dst->type, ne00, n_hvx);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    srctx.is_i32  = (octx->src[1]->type == HTP_TYPE_I32);
    srctx.compute = NULL;

    if (octx->src[0]->type == HTP_TYPE_F32) {
        switch (octx->dst->type) {
            case HTP_TYPE_F32:  srctx.compute = set_rows_row_f32_f32;  break;
            case HTP_TYPE_F16:  srctx.compute = set_rows_row_f16_f32;  break;
            case HTP_TYPE_Q8_0: srctx.compute = set_rows_row_q8_0_f32; break;
            default:            return HTP_STATUS_NO_SUPPORT;
        }
    } else if (octx->src[0]->type == HTP_TYPE_F16) {
        switch (octx->dst->type) {
            case HTP_TYPE_F32:  srctx.compute = set_rows_row_f32_f16;  break;
            case HTP_TYPE_F16:  srctx.compute = set_rows_row_f16_f16;  break;
            case HTP_TYPE_Q8_0: srctx.compute = set_rows_row_q8_0_f16; break;
            default:            return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t src1_bytes = hex_round_up(htp_tensor_nbytes(octx->src[1]), 128);
    void * src1_buf = (src1_bytes <= octx->ctx->ddr_spad_size) ? octx->ctx->ddr_spad_base : memalign(128, src1_bytes);
    if (!src1_buf) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    if (htp_tensor_read(&octx->ctx->main_th, src1_buf, src1_bytes, octx->src[1]) == 0) {
        if (src1_buf != octx->ctx->ddr_spad_base) {
            free(src1_buf);
        }
        return HTP_STATUS_INTERNAL_ERR;
    }
    srctx.src1_data = src1_buf;

    FARF(HIGH, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-hvx %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_hvx * n_hvx,
         srctx.vtcm_layout.dst_bytes_per_hvx  * n_hvx,
         n_hvx);

    work_queue_run(octx->ctx->hvx_queue, set_rows_thread, &srctx, n_hvx);

    if (src1_buf != octx->ctx->ddr_spad_base) {
        free(src1_buf);
    }

    return HTP_STATUS_OK;
}
