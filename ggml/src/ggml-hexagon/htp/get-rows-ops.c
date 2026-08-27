#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "hvx-quant.h"
#include "get-rows-ops.h"
#include "work-queue.h"

struct get_rows_context {
    struct htp_ops_context * octx;
    const struct htp_get_rows_kernel_params * kparams;
    struct htp_get_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
};

#define get_rows_preamble \
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
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
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
    const uint32_t nr = ne10 * ne11 * ne12;

#define GET_ROWS_THREAD_ST_FN(IDX_TYPE)                                                                                \
static void get_rows_thread_st_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {                            \
    struct get_rows_context * grctx = (struct get_rows_context *)data;                                                 \
    struct htp_ops_context * octx = grctx->octx;                                                                       \
    const struct htp_get_rows_kernel_params * kparams = grctx->kparams;                                                \
    get_rows_preamble;                                                                                                 \
    const uint32_t dr  = kparams->tasks_per_thread;                                                                    \
    const uint32_t ir0 = dr * ith;                                                                                     \
    if (ir0 >= kparams->total_tasks) {                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    const uint32_t ir1 = MIN(ir0 + dr, kparams->total_tasks);                                                          \
    const uint32_t row_size_bytes = htp_tensor_get_row_size(octx->src[0]->type, ne00);                                 \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                       \
    for (uint32_t i = ir0; i < ir1; ++i) {                                                                             \
        const uint32_t i12 = fastdiv(i, &kparams->div_ne10_ne11);                                                      \
        const uint32_t rem = i - i12 * ne11 * ne10;                                                                    \
        const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);                                                         \
        const uint32_t i10 = rem - i11 * ne10;                                                                         \
        const IDX_TYPE * src1_ptr = (const IDX_TYPE *)(octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12);           \
        const uint32_t i01 = (uint32_t)*src1_ptr;                                                                      \
        assert(i01 < ne01);                                                                                            \
        const uint32_t q02 = fastdiv(i11, &kparams->div_ne02);                                                         \
        const uint32_t i02 = i11 - q02 * ne02;                                                                         \
        const uint32_t q03 = fastdiv(i12, &kparams->div_ne03);                                                         \
        const uint32_t i03 = i12 - q03 * ne03;                                                                         \
        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;                                \
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;                                 \
        while (!dma_queue_push(dma_queue, dma_make_ptr((void *)dst_ptr, (const void *)src0_ptr), nb1, nb01,            \
                               row_size_bytes, 1)) {                                                                   \
            dma_queue_pop(dma_queue);                                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
    dma_queue_flush(dma_queue);                                                                                        \
}

GET_ROWS_THREAD_ST_FN(int32_t)
GET_ROWS_THREAD_ST_FN(int64_t)

#define GET_ROWS_THREAD_DT_FN(TYPE_NAME, SRC0_SIZE_EXPR, IDX_TYPE, COMPUTE_EXPR)                                       \
static void get_rows_thread_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {                 \
    struct get_rows_context * grctx = (struct get_rows_context *)data;                                                 \
    struct htp_ops_context * octx = grctx->octx;                                                                       \
    const struct htp_get_rows_kernel_params * kparams = grctx->kparams;                                                \
    get_rows_preamble;                                                                                                 \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                             \
    const uint32_t dr  = kparams->tasks_per_thread;                                                                    \
    const uint32_t ir0 = dr * ith;                                                                                     \
    if (ir0 >= kparams->total_tasks) {                                                                                 \
        return;                                                                                                        \
    }                                                                                                                  \
    const uint32_t ir1 = MIN(ir0 + dr, kparams->total_tasks);                                                          \
    const uint32_t chunks_per_row = kparams->chunks_per_row;                                                           \
    const uint32_t chunk_size     = kparams->chunk_size;                                                               \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                       \
    const struct htp_get_rows_vtcm_layout * vtcm_layout = &grctx->vtcm_layout;                                         \
    uint8_t * vtcm_src0 = grctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;         \
    uint8_t * vtcm_dst  = grctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;          \
    for (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, spad_idx++) {                      \
        const uint32_t i = ir0 + step;                                                                                 \
        const uint32_t row_idx   = fastdiv(i, &kparams->div_chunks_per_row);                                           \
        const uint32_t chunk_idx = i - row_idx * chunks_per_row;                                                       \
        const uint32_t i12 = fastdiv(row_idx, &kparams->div_ne10_ne11);                                                \
        const uint32_t rem = row_idx - i12 * ne11 * ne10;                                                              \
        const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);                                                         \
        const uint32_t i10 = rem - i11 * ne10;                                                                         \
        const IDX_TYPE * src1_ptr = (const IDX_TYPE *)(octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12);           \
        const uint32_t i01 = (uint32_t)*src1_ptr;                                                                      \
        assert(i01 < ne01);                                                                                            \
        const uint32_t q02 = fastdiv(i11, &kparams->div_ne02);                                                         \
        const uint32_t i02 = i11 - q02 * ne02;                                                                         \
        const uint32_t q03 = fastdiv(i12, &kparams->div_ne03);                                                         \
        const uint32_t i03 = i12 - q03 * ne03;                                                                         \
        const uint32_t offset = chunk_idx * chunk_size;                                                                \
        const uint32_t cur_elems = (offset < ne00) ? MIN(chunk_size, ne00 - offset) : 0;                               \
        const uint32_t cur_src0_bytes = SRC0_SIZE_EXPR(cur_elems);                                                     \
        const uint32_t cur_dst_bytes  = cur_elems * sizeof(float);                                                     \
        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03 + SRC0_SIZE_EXPR(offset);       \
        dma_queue_push(dma_queue,                                                                                      \
                       dma_make_ptr((void *)(uintptr_t)octx->dst->data,                                                \
                                    vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                            \
                       cur_dst_bytes, vtcm_layout->dst_spad_half_size, cur_dst_bytes, 0);                              \
        dma_queue_push(dma_queue,                                                                                      \
                       dma_make_ptr((void *)(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size),                 \
                                    (const void *)src0_ptr),                                                           \
                       vtcm_layout->src0_spad_half_size, cur_src0_bytes, cur_src0_bytes, 1);                           \
    }                                                                                                                  \
    for (uint32_t step = 0; step < ir1 - ir0; ++step) {                                                                \
        const uint32_t i = ir0 + step;                                                                                 \
        void * dst_spad = (void *) dma_queue_pop(dma_queue).src;                                                       \
        void * src_spad = (void *) dma_queue_pop(dma_queue).dst;                                                       \
        const uint32_t row_idx   = fastdiv(i, &kparams->div_chunks_per_row);                                           \
        const uint32_t chunk_idx = i - row_idx * chunks_per_row;                                                       \
        const uint32_t i12 = fastdiv(row_idx, &kparams->div_ne10_ne11);                                                \
        const uint32_t rem = row_idx - i12 * ne11 * ne10;                                                              \
        const uint32_t i11 = fastdiv(rem, &kparams->div_ne10);                                                         \
        const uint32_t i10 = rem - i11 * ne10;                                                                         \
        const uint32_t offset = chunk_idx * chunk_size;                                                                \
        const uint32_t cur_elems = (offset < ne00) ? MIN(chunk_size, ne00 - offset) : 0;                               \
        const uint32_t cur_dst_bytes  = cur_elems * sizeof(float);                                                     \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, i);                                                          \
        COMPUTE_EXPR;                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, i);                                                           \
        const uintptr_t dst_ptr  = octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3 + offset * sizeof(float);             \
        dma_queue_push(dma_queue,                                                                                      \
                       dma_make_ptr((void *)dst_ptr, (const void *)dst_spad),                                          \
                       cur_dst_bytes, vtcm_layout->dst_spad_half_size, cur_dst_bytes, 1);                              \
        const uint32_t next_step = step + 2;                                                                           \
        if (next_step < ir1 - ir0) {                                                                                   \
            const uint32_t pi = ir0 + next_step;                                                                       \
            const uint32_t prow_idx   = fastdiv(pi, &kparams->div_chunks_per_row);                                     \
            const uint32_t pchunk_idx = pi - prow_idx * chunks_per_row;                                                \
            const uint32_t pi12 = fastdiv(prow_idx, &kparams->div_ne10_ne11);                                          \
            const uint32_t prem = prow_idx - pi12 * ne11 * ne10;                                                       \
            const uint32_t pi11 = fastdiv(prem, &kparams->div_ne10);                                                   \
            const uint32_t pi10 = prem - pi11 * ne10;                                                                  \
            const IDX_TYPE * psrc1_ptr = (const IDX_TYPE *)(octx->src[1]->data + pi10*nb10 + pi11*nb11 + pi12*nb12);   \
            const uint32_t pi01 = (uint32_t)*psrc1_ptr;                                                                \
            assert(pi01 < ne01);                                                                                       \
            const uint32_t pq02 = fastdiv(pi11, &kparams->div_ne02);                                                   \
            const uint32_t pi02 = pi11 - pq02 * ne02;                                                                  \
            const uint32_t pq03 = fastdiv(pi12, &kparams->div_ne03);                                                   \
            const uint32_t pi03 = pi12 - pq03 * ne03;                                                                  \
            const uint32_t poffset = pchunk_idx * chunk_size;                                                          \
            const uint32_t pcur_elems = (poffset < ne00) ? MIN(chunk_size, ne00 - poffset) : 0;                        \
            const uint32_t pcur_src0_bytes = SRC0_SIZE_EXPR(pcur_elems);                                               \
            const uintptr_t psrc0_ptr =                                                                                \
                octx->src[0]->data + pi01*nb01 + pi02*nb02 + pi03*nb03 + SRC0_SIZE_EXPR(poffset);                      \
            dma_queue_push(dma_queue,                                                                                  \
                           dma_make_ptr((void *)src_spad, (const void *)psrc0_ptr),                                    \
                           vtcm_layout->src0_spad_half_size, pcur_src0_bytes, pcur_src0_bytes, 1);                     \
        }                                                                                                              \
    }                                                                                                                  \
    dma_queue_flush(dma_queue);                                                                                        \
}

#define F32_BYTES(n)  ((n) * sizeof(float))
#define F16_BYTES(n)  ((n) * sizeof(__fp16))
#define Q8_0_BYTES(n) (((n) / 32) * sizeof(block_q8_0))

GET_ROWS_THREAD_DT_FN(f32,  F32_BYTES,  int32_t, { if (cur_elems > 0) hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, cur_elems); })
GET_ROWS_THREAD_DT_FN(f32,  F32_BYTES,  int64_t, { if (cur_elems > 0) hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, cur_elems); })

GET_ROWS_THREAD_DT_FN(f16,  F16_BYTES,  int32_t, { hvx_dequantize_row_f16_f32((float *)dst_spad, src_spad, ne00); })
GET_ROWS_THREAD_DT_FN(f16,  F16_BYTES,  int64_t, { hvx_dequantize_row_f16_f32((float *)dst_spad, src_spad, ne00); })

GET_ROWS_THREAD_DT_FN(q8_0, Q8_0_BYTES, int32_t, { hvx_dequantize_row_q8_0_f32((float *)dst_spad, src_spad, ne00); })
GET_ROWS_THREAD_DT_FN(q8_0, Q8_0_BYTES, int64_t, { hvx_dequantize_row_q8_0_f32((float *)dst_spad, src_spad, ne00); })

int op_get_rows(struct htp_ops_context * octx) {
    const struct htp_get_rows_kernel_params * kparams = (const struct htp_get_rows_kernel_params *) octx->kernel_params;

    if (octx->src[0]->type != HTP_TYPE_F32 &&
        octx->src[0]->type != HTP_TYPE_F16 &&
        octx->src[0]->type != HTP_TYPE_Q8_0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct get_rows_context grctx;
    grctx.octx = octx;
    grctx.kparams = kparams;
    grctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    const uint32_t ne00 = octx->src[0]->ne[0];
    htp_get_rows_vtcm_layout_build(&grctx.vtcm_layout, octx->src[0]->type, ne00, kparams->n_threads);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    work_queue_func_t q_func = NULL;
    if (kparams->use_dma) {
        q_func = (work_queue_func_t)(is_i32 ? get_rows_thread_st_int32_t : get_rows_thread_st_int64_t);
    } else {
        switch (octx->src[0]->type) {
            case HTP_TYPE_F32:  q_func = (work_queue_func_t)(is_i32 ? get_rows_thread_f32_int32_t  : get_rows_thread_f32_int64_t);  break;
            case HTP_TYPE_F16:  q_func = (work_queue_func_t)(is_i32 ? get_rows_thread_f16_int32_t  : get_rows_thread_f16_int64_t);  break;
            case HTP_TYPE_Q8_0: q_func = (work_queue_func_t)(is_i32 ? get_rows_thread_q8_0_int32_t : get_rows_thread_q8_0_int64_t); break;
            default:            return HTP_STATUS_NO_SUPPORT;
        }
    }

    work_queue_run(octx->ctx->work_queue, q_func, &grctx, kparams->n_threads);
    return HTP_STATUS_OK;
}
