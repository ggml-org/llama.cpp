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
#include "hvx-utils.h"
#include "hvx-quant.h"

struct get_rows_context {
    struct htp_ops_context * octx;
    uint32_t tasks_per_thread;
    uint32_t total_tasks;
    uint32_t chunks_per_row;
    uint32_t chunk_size;
    struct fastdiv_values get_rows_div_ne10;
    struct fastdiv_values get_rows_div_ne10_ne11;
    struct fastdiv_values get_rows_div_chunks_per_row;
    struct fastdiv_values get_rows_div_ne02;
    struct fastdiv_values get_rows_div_ne03;
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

static inline uint32_t get_row_size_bytes(int type, uint32_t ne00) {
    switch (type) {
        case HTP_TYPE_F32:  return ne00 * 4;
        case HTP_TYPE_F16:  return ne00 * 2;
        case HTP_TYPE_Q8_0: return (ne00 / 32) * 34;
        default:            return 0;
    }
}

#define GET_ROWS_THREAD_DMA_FN(IDX_TYPE)                                                                               \
static void get_rows_thread_dma_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {                           \
    struct get_rows_context * grctx = (struct get_rows_context *)data;                                                 \
    struct htp_ops_context * octx = grctx->octx;                                                                       \
    get_rows_preamble;                                                                                                 \
    const uint32_t dr  = grctx->tasks_per_thread;                                                                      \
    const uint32_t ir0 = dr * ith;                                                                                     \
    if (ir0 >= grctx->total_tasks) {                                                                                   \
        return;                                                                                                        \
    }                                                                                                                  \
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);                                                            \
    const uint32_t row_size_bytes = get_row_size_bytes(octx->src[0]->type, ne00);                                       \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                       \
    for (uint32_t i = ir0; i < ir1; ++i) {                                                                             \
        const uint32_t i12 = fastdiv(i, &grctx->get_rows_div_ne10_ne11);                                               \
        const uint32_t rem = i - i12 * ne11 * ne10;                                                                    \
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);                                                  \
        const uint32_t i10 = rem - i11 * ne10;                                                                         \
        const IDX_TYPE * src1_ptr = (const IDX_TYPE *)(octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12);           \
        const uint32_t i01 = (uint32_t)*src1_ptr;                                                                      \
        assert(i01 < ne01);                                                                                            \
        const uint32_t q02 = fastdiv(i11, &grctx->get_rows_div_ne02);                                                  \
        const uint32_t i02 = i11 - q02 * ne02;                                                                         \
        const uint32_t q03 = fastdiv(i12, &grctx->get_rows_div_ne03);                                                  \
        const uint32_t i03 = i12 - q03 * ne03;                                                                         \
        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03;                                 \
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;                                 \
        while (!dma_queue_push(dma_queue, dma_make_ptr((void *)dst_ptr, (const void *)src0_ptr), nb1, nb01,             \
                               row_size_bytes, 1)) {                                                                   \
            dma_queue_pop(dma_queue);                                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
    dma_queue_flush(dma_queue);                                                                                        \
}

GET_ROWS_THREAD_DMA_FN(int32_t)
GET_ROWS_THREAD_DMA_FN(int64_t)

#define GET_ROWS_THREAD_HVX_FN(TYPE_NAME, IDX_TYPE, COMPUTE_EXPR)                                                      \
static void get_rows_thread_hvx_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {             \
    struct get_rows_context * grctx = (struct get_rows_context *)data;                                                 \
    struct htp_ops_context * octx = grctx->octx;                                                                       \
    get_rows_preamble;                                                                                                 \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                             \
    const uint32_t dr  = grctx->tasks_per_thread;                                                                      \
    const uint32_t ir0 = dr * ith;                                                                                     \
    if (ir0 >= grctx->total_tasks) {                                                                                   \
        return;                                                                                                        \
    }                                                                                                                  \
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);                                                            \
    const uint32_t chunks_per_row = grctx->chunks_per_row;                                                             \
    const uint32_t chunk_size     = grctx->chunk_size;                                                                 \
    for (uint32_t i = ir0; i < ir1; ++i) {                                                                             \
        const uint32_t row_idx   = fastdiv(i, &grctx->get_rows_div_chunks_per_row);                                    \
        const uint32_t chunk_idx = i - row_idx * chunks_per_row;                                                       \
        const uint32_t i12 = fastdiv(row_idx, &grctx->get_rows_div_ne10_ne11);                                         \
        const uint32_t rem = row_idx - i12 * ne11 * ne10;                                                              \
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);                                                  \
        const uint32_t i10 = rem - i11 * ne10;                                                                         \
        const IDX_TYPE * src1_ptr = (const IDX_TYPE *)(octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12);           \
        const uint32_t i01 = (uint32_t)*src1_ptr;                                                                      \
        assert(i01 < ne01);                                                                                            \
        const uint32_t q02 = fastdiv(i11, &grctx->get_rows_div_ne02);                                                  \
        const uint32_t i02 = i11 - q02 * ne02;                                                                         \
        const uint32_t q03 = fastdiv(i12, &grctx->get_rows_div_ne03);                                                  \
        const uint32_t i03 = i12 - q03 * ne03;                                                                         \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, i);                                                          \
        COMPUTE_EXPR;                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, i);                                                           \
    }                                                                                                                  \
}

GET_ROWS_THREAD_HVX_FN(f32, int32_t, {
    const uint32_t offset = chunk_idx * chunk_size;
    if (offset < ne00) {
        const uint32_t copy_size = MIN(chunk_size, ne00 - offset);
        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03 + offset * sizeof(float);
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3  + offset * sizeof(float);
        hvx_copy_f32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, copy_size);
    }
})

GET_ROWS_THREAD_HVX_FN(f32, int64_t, {
    const uint32_t offset = chunk_idx * chunk_size;
    if (offset < ne00) {
        const uint32_t copy_size = MIN(chunk_size, ne00 - offset);
        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03 + offset * sizeof(float);
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3  + offset * sizeof(float);
        hvx_copy_f32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, copy_size);
    }
})

GET_ROWS_THREAD_HVX_FN(f16, int32_t, {
    const void * src0_ptr = (const void *)((const uint8_t *) octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03);
    float *      dst_ptr  = (float *)      ((uint8_t *)       octx->dst->data  + i10*nb1  + i11*nb2  + i12*nb3);
    hvx_dequantize_row_f16_f32(dst_ptr, src0_ptr, ne00);
})

GET_ROWS_THREAD_HVX_FN(f16, int64_t, {
    const void * src0_ptr = (const void *)((const uint8_t *) octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03);
    float *      dst_ptr  = (float *)      ((uint8_t *)       octx->dst->data  + i10*nb1  + i11*nb2  + i12*nb3);
    hvx_dequantize_row_f16_f32(dst_ptr, src0_ptr, ne00);
})

GET_ROWS_THREAD_HVX_FN(q8_0, int32_t, {
    const void * src0_ptr = (const void *)((const uint8_t *) octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03);
    float *      dst_ptr  = (float *)      ((uint8_t *)       octx->dst->data  + i10*nb1  + i11*nb2  + i12*nb3);
    hvx_dequantize_row_q8_0_f32(dst_ptr, src0_ptr, ne00);
})

GET_ROWS_THREAD_HVX_FN(q8_0, int64_t, {
    const void * src0_ptr = (const void *)((const uint8_t *) octx->src[0]->data + i01*nb01 + i02*nb02 + i03*nb03);
    float *      dst_ptr  = (float *)      ((uint8_t *)       octx->dst->data  + i10*nb1  + i11*nb2  + i12*nb3);
    hvx_dequantize_row_q8_0_f32(dst_ptr, src0_ptr, ne00);
})

int op_get_rows(struct htp_ops_context * octx) {
    get_rows_preamble;

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
    grctx.get_rows_div_ne10      = init_fastdiv_values(octx->src[1]->ne[0]);
    grctx.get_rows_div_ne10_ne11 = init_fastdiv_values(octx->src[1]->ne[0] * octx->src[1]->ne[1]);
    grctx.get_rows_div_ne02      = init_fastdiv_values(octx->src[0]->ne[2]);
    grctx.get_rows_div_ne03      = init_fastdiv_values(octx->src[0]->ne[3]);

    const uint32_t nb00 = octx->src[0]->nb[0];
    const uint32_t nb0  = octx->dst->nb[0];

    const bool can_use_dma = (octx->src[0]->type == octx->dst->type) && (nb01 == nb1);
    const bool use_dma = can_use_dma && (ne00 >= 2048);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    if (use_dma) {
        grctx.chunks_per_row = 1;
        grctx.chunk_size = ne00;
        grctx.total_tasks = nr;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(1);

        const uint32_t n_threads = MIN(nr, octx->n_threads);
        grctx.tasks_per_thread = (nr + n_threads - 1) / n_threads;

        worker_callback_t dma_worker = is_i32 ? get_rows_thread_dma_int32_t : get_rows_thread_dma_int64_t;
        worker_pool_run_func(octx->ctx->worker_pool, dma_worker, &grctx, n_threads);
    } else {
        uint32_t chunks_per_row = 1;
        uint32_t chunk_size = ne00;
        uint32_t total_tasks = nr;

        if (octx->src[0]->type == HTP_TYPE_F32 && nr < octx->n_threads) {
            const uint32_t min_chunk_size = 1024;
            uint32_t max_chunks = ne00 / min_chunk_size;
            if (max_chunks == 0) {
                max_chunks = 1;
            }
            chunks_per_row = MIN((octx->n_threads + nr - 1) / nr, max_chunks);
            chunk_size = (ne00 + chunks_per_row - 1) / chunks_per_row;
            total_tasks = nr * chunks_per_row;
        }

        grctx.chunks_per_row = chunks_per_row;
        grctx.chunk_size = chunk_size;
        grctx.total_tasks = total_tasks;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(chunks_per_row);

        const uint32_t n_threads = MIN(total_tasks, octx->n_threads);
        grctx.tasks_per_thread = (total_tasks + n_threads - 1) / n_threads;

        worker_callback_t hvx_worker = NULL;
        switch (octx->src[0]->type) {
            case HTP_TYPE_F32:
                hvx_worker = is_i32 ? get_rows_thread_hvx_f32_int32_t : get_rows_thread_hvx_f32_int64_t;
                break;
            case HTP_TYPE_F16:
                hvx_worker = is_i32 ? get_rows_thread_hvx_f16_int32_t : get_rows_thread_hvx_f16_int64_t;
                break;
            case HTP_TYPE_Q8_0:
                hvx_worker = is_i32 ? get_rows_thread_hvx_q8_0_int32_t : get_rows_thread_hvx_q8_0_int64_t;
                break;
            default:
                return HTP_STATUS_NO_SUPPORT;
        }

        worker_pool_run_func(octx->ctx->worker_pool, hvx_worker, &grctx, n_threads);
    }
    return HTP_STATUS_OK;
}
