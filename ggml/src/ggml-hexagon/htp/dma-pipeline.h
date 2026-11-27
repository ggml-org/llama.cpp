#ifndef HTP_DMA_PIPELINE_H
#define HTP_DMA_PIPELINE_H

#include <stdbool.h>
#include <stdint.h>

#include "dma-queue.h"
#include "htp-trace.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dma_pipe_task {
    uint32_t row;
    uint32_t off;
    uint32_t len;
    uint32_t aux;
};

typedef bool (*dma_pipe_next_task_fn)(void * ctx, struct dma_pipe_task * task);
typedef void (*dma_pipe_push_in_fn)(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad);
typedef void (*dma_pipe_push_in2_fn)(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in0_spad, void * in1_spad);
typedef void (*dma_pipe_compute_io_fn)(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad);
typedef void (*dma_pipe_compute_io2_fn)(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in0_spad, void * in1_spad);
typedef void (*dma_pipe_compute_1_fn)(void * ctx, const struct dma_pipe_task * task, void * spad);
typedef void (*dma_pipe_push_out_fn)(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad);

// Double-buffered DDR -> VTCM -> HVX -> VTCM -> DDR pipeline.
// Uses dummy output pushes in the prologue and dma_queue_pop().src/.dst to
// recycle VTCM slot pointers directly through the DMA ring.
// push_in/push_out must always push exactly one descriptor per task (use a
// zero-nrows dummy push carrying the VTCM pointer when skipping a transfer).
static inline void dma_pipe_io_run(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    void *                    ctx,
    void *                    in_buf[2],
    void *                    out_buf[2],
    dma_pipe_next_task_fn     next_task,
    dma_pipe_push_in_fn       push_in,
    dma_pipe_compute_io_fn    compute,
    dma_pipe_push_out_fn      push_out
) {
    struct dma_pipe_task t[2];
    uint32_t head = 0, tail = 0;

    for (uint32_t b = 0; b < 2; ++b) {
        if (!next_task(ctx, &t[tail & 1])) {
            break;
        }
        dma_queue_push(dma_q, dma_make_data(0, out_buf[b]), 0, 0, 0, 0);
        push_in(dma_q, ctx, &t[tail & 1], in_buf[b]);
        tail++;
    }

    while (head != tail) {
        const struct dma_pipe_task cur = t[head & 1];
        head++;

        void * out_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * in_spad  = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);
        compute(ctx, &cur, out_spad, in_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);

        push_out(dma_q, ctx, &cur, out_spad);

        if (next_task(ctx, &t[tail & 1])) {
            push_in(dma_q, ctx, &t[tail & 1], in_spad);
            tail++;
        }
    }

    dma_queue_flush(dma_q);
}

// Double-buffered 2-input DDR -> VTCM -> HVX -> VTCM -> DDR pipeline.
static inline void dma_pipe_io2_run(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    void *                    ctx,
    void *                    in0_buf[2],
    void *                    in1_buf[2],
    void *                    out_buf[2],
    dma_pipe_next_task_fn     next_task,
    dma_pipe_push_in2_fn      push_in,
    dma_pipe_compute_io2_fn   compute,
    dma_pipe_push_out_fn      push_out
) {
    struct dma_pipe_task t[2];
    uint32_t head = 0, tail = 0;

    for (uint32_t b = 0; b < 2; ++b) {
        if (!next_task(ctx, &t[tail & 1])) {
            break;
        }
        dma_queue_push(dma_q, dma_make_data(0, out_buf[b]), 0, 0, 0, 0);
        push_in(dma_q, ctx, &t[tail & 1], in0_buf[b], in1_buf[b]);
        tail++;
    }

    while (head != tail) {
        const struct dma_pipe_task cur = t[head & 1];
        head++;

        void * out_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;
        void * in0_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;
        void * in1_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);
        compute(ctx, &cur, out_spad, in0_spad, in1_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);

        push_out(dma_q, ctx, &cur, out_spad);

        if (next_task(ctx, &t[tail & 1])) {
            push_in(dma_q, ctx, &t[tail & 1], in0_spad, in1_spad);
            tail++;
        }
    }

    dma_queue_flush(dma_q);
}

// Double-buffered DDR -> VTCM -> HVX pipeline for input-only reductions.
static inline void dma_pipe_in_run(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    void *                    ctx,
    void *                    in_buf[2],
    dma_pipe_next_task_fn     next_task,
    dma_pipe_push_in_fn       push_in,
    dma_pipe_compute_1_fn     compute
) {
    struct dma_pipe_task t[2];
    uint32_t head = 0, tail = 0;

    for (uint32_t b = 0; b < 2; ++b) {
        if (!next_task(ctx, &t[tail & 1])) {
            break;
        }
        push_in(dma_q, ctx, &t[tail & 1], in_buf[b]);
        tail++;
    }

    while (head != tail) {
        const struct dma_pipe_task cur = t[head & 1];
        head++;

        void * in_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);
        compute(ctx, &cur, in_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);

        if (next_task(ctx, &t[tail & 1])) {
            push_in(dma_q, ctx, &t[tail & 1], in_spad);
            tail++;
        }
    }

    dma_queue_flush(dma_q);
}

// Bounded in-flight HVX -> VTCM -> DDR pipeline for output-only generators.
static inline void dma_pipe_out_run(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    void *                    ctx,
    void * const *            out_bufs,
    uint32_t                  n_bufs,
    dma_pipe_next_task_fn     next_task,
    dma_pipe_compute_1_fn     compute,
    dma_pipe_push_out_fn      push_out
) {
    for (uint32_t b = 0; b < n_bufs; ++b) {
        dma_queue_push(dma_q, dma_make_data(0, out_bufs[b]), 0, 0, 0, 0);
    }

    struct dma_pipe_task cur;
    while (next_task(ctx, &cur)) {
        void * out_spad = (void *) (uintptr_t) dma_queue_pop(dma_q).src;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);
        compute(ctx, &cur, out_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) cur.row);

        push_out(dma_q, ctx, &cur, out_spad);
    }

    dma_queue_flush(dma_q);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // HTP_DMA_PIPELINE_H
