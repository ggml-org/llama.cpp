#include "htp-tensor.h"

#include <qurt.h>
#include <qurt_memory.h>

#include "hex-common.h"
#include "hex-utils.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "work-queue.h"

struct l2flush_task {
    struct htp_thread_trace * trace;
    uint32_t start;
    uint32_t end;
    uint32_t chunk_size;
    uint32_t ti;
};

static void l2flush_thread_worker(unsigned int n, unsigned int i, void * data) {
    struct l2flush_task * task = (struct l2flush_task *) data;
    const uint32_t start = task->start;
    const uint32_t end   = task->end;
    const uint32_t ti    = task->ti;
    const uint32_t chunk_size = task->chunk_size;

    const uint32_t thread_s = start + i * chunk_size;
    if (thread_s >= end) {
        return;
    }
    uint32_t thread_e = thread_s + chunk_size;
    if (thread_e > end) {
        thread_e = end;
    }

    struct htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, ti);
    hex_l2flush((void *) (uintptr_t) thread_s, thread_e - thread_s);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, ti);
}

static void flush_all_dcache(struct htp_context * ctx) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    bitmap_reset(ctx->dirty_map, HTP_OP_MAX_TENSORS);
}

static void flush_tensor_range(struct htp_context * ctx, const struct htp_tensor * t) {
    struct htp_thread_trace * tr = &ctx->trace[0];

    if (t->size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        struct l2flush_task task;
        task.start = hex_align_down((size_t) t->data, HEX_L2_LINE_SIZE);
        task.end   = hex_align_up((size_t) t->data + t->size, HEX_L2_LINE_SIZE);
        task.ti    = t->ti;
        task.trace = ctx->trace;

        const uint32_t total_size = task.end - task.start;
        const uint32_t n_blocks   = (total_size + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
        const uint32_t blocks_per_thread = fastdiv(n_blocks + ctx->n_threads - 1, &ctx->n_threads_div);
        task.chunk_size = blocks_per_thread * HEX_L2_BLOCK_SIZE;

        work_queue_run(ctx->work_queue, l2flush_thread_worker, &task, ctx->n_threads);
    } else {
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        hex_l2flush((void *) t->data, t->size);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
    }

    htp_tensor_make_clean(t, ctx->dirty_map);
}

void htp_tensor_flush(struct htp_context * ctx, const struct htp_tensor * t) {
    if (!bitmap_test(ctx->dirty_map, t->ti)) {
        return;
    }

    if (t->size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        return;
    }

    flush_tensor_range(ctx, t);
}

// One dirty tensor's line-aligned range, placed in the flattened global block space.
struct l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HEX_L2_BLOCK_SIZE chunks (last may be partial)
};

struct l2flush_multi_task {
    struct htp_thread_trace * trace;
    struct l2flush_range      ranges[HTP_OP_MAX_INPUTS];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_thread;
};

static void l2flush_multi_worker(unsigned int n, unsigned int i, void * data) {
    (void) n;
    struct l2flush_multi_task * task = (struct l2flush_multi_task *) data;

    const uint32_t gb_first = i * task->blocks_per_thread;
    uint32_t       gb_last  = gb_first + task->blocks_per_thread;
    if (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    if (gb_first >= gb_last) {
        return;
    }

    struct htp_thread_trace * tr = &task->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    for (uint32_t r = 0; r < task->n_ranges; r++) {
        const struct l2flush_range * rg = &task->ranges[r];
        const uint32_t rb_first = rg->block_first;
        const uint32_t rb_last  = rg->block_first + rg->n_blocks;

        const uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        const uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        if (lo >= hi) {
            continue;
        }

        const uint32_t s = rg->start + (lo - rb_first) * HEX_L2_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HEX_L2_BLOCK_SIZE;
        if (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((void *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    uint64_t total_dirty = 0;
    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (t && bitmap_test(ctx->dirty_map, t->ti)) {
            total_dirty += t->size;
        }
    }

    if (total_dirty == 0) {
        return;
    }

    if (total_dirty > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        return;
    }

    // Aggregate is small enough to walk. Thread it across all dirty ranges at once
    // when it is worth the dispatch, otherwise flush sequentially.
    if (total_dirty > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        struct l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = 0;

        uint32_t block_acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            const struct htp_tensor * t = tensors[i];
            if (!t || !bitmap_test(ctx->dirty_map, t->ti)) {
                continue;
            }
            // Clear as we go: dedups a tensor passed as multiple srcs (e.g. mul(x,x)).
            htp_tensor_make_clean(t, ctx->dirty_map);

            struct l2flush_range * rg = &task.ranges[task.n_ranges++];
            rg->start = hex_align_down((size_t) t->data, HEX_L2_LINE_SIZE);
            rg->end   = hex_align_up((size_t) t->data + t->size, HEX_L2_LINE_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
        return;
    }

    struct htp_thread_trace * tr = &ctx->trace[0];
    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (!t || !bitmap_test(ctx->dirty_map, t->ti)) {
            continue;
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        hex_l2flush((void *) t->data, t->size);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        htp_tensor_make_clean(t, ctx->dirty_map);
    }
}
