#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hvx-quant.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nr  = ne01;

struct set_rows_context {
    struct htp_ops_context * octx;
    const struct htp_set_rows_kernel_params * kparams;
    struct htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_EXPR)                                                \
static void set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {       \
    struct set_rows_context * srctx = (struct set_rows_context *)data;                                           \
    struct htp_ops_context * octx = srctx->octx;                                                                 \
    const struct htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    const uint32_t dr  = kparams->tasks_per_thread;                                                              \
    const uint32_t ir0 = dr * ith;                                                                               \
    if (ir0 >= kparams->total_tasks) {                                                                           \
        return;                                                                                                  \
    }                                                                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, kparams->total_tasks);                                                    \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                 \
    const struct htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    const uint32_t src0_row_size = ne00 * sizeof(float);                                                         \
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    const uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    const uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    for (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        const uintptr_t src0_ptr = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                          \
        dma_queue_push(dma_queue,                                                                                \
                       dma_make_ptr((void *)octx->dst->data,                                                     \
                                    vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                      \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_push(dma_queue,                                                                                \
                       dma_make_ptr((void *)(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size),           \
                                    (const void *)src0_ptr),                                                     \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        if (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            if (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_step = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_base = 0;                                                                                      \
    uint32_t ci12_base = 0;                                                                                      \
    for (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        void * dst_spad = (void *) dma_queue_pop(dma_queue).src;                                                 \
        void * src_spad = (void *) dma_queue_pop(dma_queue).dst;                                                 \
        uint32_t i = ir0 + ci_step;                                                                              \
        const uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        const IDX_TYPE i1 = *(const IDX_TYPE *)src1_addr;                                                        \
        const bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        const uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        if (valid_i1) {                                                                                          \
            COMPUTE_EXPR;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        if (valid_i1) {                                                                                          \
            const uintptr_t dst_ptr = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                     \
            dma_queue_push(dma_queue,                                                                            \
                           dma_make_ptr((void *)dst_ptr, (const void *)dst_spad),                                \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } else {                                                                                                 \
            dma_queue_push(dma_queue,                                                                            \
                           dma_make_ptr((void *)octx->dst->data, (const void *)dst_spad),                        \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        const uint32_t next_step = step + 2;                                                                     \
        if (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            const uintptr_t psrc0_ptr = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                    \
            dma_queue_push(dma_queue,                                                                            \
                           dma_make_ptr((void *)src_spad, (const void *)psrc0_ptr),                              \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            if (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                if (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        if (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            if (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            if (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                if (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_queue);                                                                                  \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })

int op_set_rows(struct htp_ops_context * octx) {
    const struct htp_set_rows_kernel_params * kparams = (const struct htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((const void *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    struct set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, kparams->n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = NULL;
    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    switch (octx->dst->type) {
        case HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  break;
        case HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  break;
        case HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; break;
        default:            return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, kparams->n_threads);

    return HTP_STATUS_OK;
}
