#include "htp-tensor.h"

#include <qurt.h>
#include <qurt_memory.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "hex-utils.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "work-queue.h"

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

static void flush_all_dcache(struct htp_context * ctx) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, sizeof(ctx->dirty_ranges));
}

static void l2flush_multi_worker(unsigned int n, unsigned int i, void * data) {
    struct l2flush_multi_task * task = (struct l2flush_multi_task *) data;
    (void) n;

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

void htp_tensor_dirty_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    const struct htp_tensor * pending[HTP_OP_MAX_OUTPUTS];
    uint32_t n_pending = 0;

    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (!t) continue;

        uint32_t t_start = t->data;
        uint32_t t_end   = t_start + t->size;

        bool merged = false;
        for (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
            struct htp_dirty_range * r = &ctx->dirty_ranges[j];
            if (!r->start) continue;

            if (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end)     ? t_end   : r->end;
                r->start = new_start;
                r->end   = new_end;
                merged   = true;
            }
        }

        if (!merged) {
            pending[n_pending++] = t;
        }
    }

    if (n_pending == 0) {
        return;
    }

    uint32_t empty_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t active_indices[HTP_MAX_DIRTY_RANGES];
    uint32_t n_active = 0;
    uint32_t n_empty  = 0;
    for (uint32_t j = 0; j < HTP_MAX_DIRTY_RANGES; j++) {
        if (ctx->dirty_ranges[j].start) {
            active_indices[n_active++] = j;
        } else {
            empty_indices[n_empty++]   = j;
        }
    }

    if (n_pending <= n_empty) {
        for (uint32_t i = 0; i < n_pending; i++) {
            uint32_t idx = empty_indices[i];
            struct htp_dirty_range * r = &ctx->dirty_ranges[idx];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
            r->bi    = pending[i]->bi;
        }
        return;
    }

    uint32_t n_evict = n_pending - n_empty;
    uint32_t total_evict_size = 0;
    for (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        struct htp_dirty_range * r = &ctx->dirty_ranges[idx];
        total_evict_size += r->end - r->start;
    }

    if (total_evict_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        for (uint32_t i = 0; i < n_pending; i++) {
            struct htp_dirty_range * r = &ctx->dirty_ranges[i];
            r->start = pending[i]->data;
            r->end   = pending[i]->data + pending[i]->size;
            r->bi    = pending[i]->bi;
        }
        return;
    }

    if (total_evict_size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1 && n_evict <= HTP_OP_MAX_INPUTS) {
        struct l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = n_evict;

        uint32_t block_acc = 0;
        for (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            struct htp_dirty_range * r = &ctx->dirty_ranges[idx];

            struct l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_LINE_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_LINE_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_thread = fastdiv(block_acc + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } else {
        struct htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        for (uint32_t i = 0; i < n_evict; i++) {
            uint32_t idx = active_indices[i];
            struct htp_dirty_range * r = &ctx->dirty_ranges[idx];
            uint32_t size = r->end - r->start;
            hex_l2flush((void *) (uintptr_t) r->start, size);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }

    for (uint32_t i = 0; i < n_evict; i++) {
        uint32_t idx = active_indices[i];
        struct htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[i]->data;
        r->end   = pending[i]->data + pending[i]->size;
        r->bi    = pending[i]->bi;
    }

    for (uint32_t i = 0; i < n_empty; i++) {
        uint32_t idx = empty_indices[i];
        struct htp_dirty_range * r = &ctx->dirty_ranges[idx];
        r->start = pending[n_evict + i]->data;
        r->end   = pending[n_evict + i]->data + pending[n_evict + i]->size;
        r->bi    = pending[n_evict + i]->bi;
    }
}

static void make_tensor_clean(struct htp_context * ctx, const struct htp_tensor * t) {
    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    for (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (!r->start) continue;

        if (r->start < t_end && t_start < r->end) {
            if (t_start <= r->start && r->end <= t_end) {
                r->start = 0;
            } else if (t_start <= r->start) {
                r->start = t_end;
            } else if (r->end <= t_end) {
                r->end = t_start;
            }
        }
    }
}

static inline bool is_tensor_dirty(struct htp_context * ctx, const struct htp_tensor * t) {
    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    for (uint32_t i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (!r->start) continue;

        if (r->start < t_end && t_start < r->end) {
            return true;
        }
    }
    return false;
}

void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    const struct htp_tensor * dirty_tensors[HTP_OP_MAX_INPUTS];
    uint32_t n_dirty = 0;
    uint64_t total_dirty = 0;

    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (t && (t->flags & HTP_TENSOR_COMPUTE) && is_tensor_dirty(ctx, t)) {
            dirty_tensors[n_dirty++] = t;
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

    if (total_dirty >= HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        struct l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = 0;

        uint32_t block_acc = 0;
        for (uint32_t i = 0; i < n_dirty; i++) {
            const struct htp_tensor * t = dirty_tensors[i];
            make_tensor_clean(ctx, t);

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
    for (uint32_t i = 0; i < n_dirty; i++) {
        const struct htp_tensor * t = dirty_tensors[i];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        hex_l2flush((void *) (uintptr_t) t->data, t->size);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        make_tensor_clean(ctx, t);
    }
}
