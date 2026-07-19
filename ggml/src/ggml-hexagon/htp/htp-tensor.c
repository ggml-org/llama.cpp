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

#define DEBUG_CACHE 0

#if DEBUG_CACHE
#define LOG_CACHE(fmt, ...) FARF(ALWAYS, fmt, ##__VA_ARGS__)
#else
#define LOG_CACHE(fmt, ...)
#endif

static void flush_all_dcache(struct htp_context * ctx);
static void flush_address_range(struct htp_context * ctx, uint32_t start, uint32_t size, uint32_t ti);

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

static void flush_address_range(struct htp_context * ctx, uint32_t start, uint32_t size, uint32_t ti) {
    struct htp_thread_trace * tr = &ctx->trace[0];

    if (size > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        struct l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = 1;

        struct l2flush_range * rg = &task.ranges[0];
        rg->start = hex_align_down((size_t) start, HEX_L2_LINE_SIZE);
        rg->end   = hex_align_up((size_t) start + size, HEX_L2_LINE_SIZE);
        rg->block_first = 0;
        rg->n_blocks = (rg->end - rg->start + HEX_L2_BLOCK_SIZE - 1) / HEX_L2_BLOCK_SIZE;

        task.total_blocks      = rg->n_blocks;
        task.blocks_per_thread = fastdiv(rg->n_blocks + ctx->n_threads - 1, &ctx->n_threads_div);

        work_queue_run(ctx->work_queue, l2flush_multi_worker, &task, ctx->n_threads);
    } else {
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, ti);
        hex_l2flush((void *) (uintptr_t) start, size);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, ti);
    }
}

void htp_tensor_make_dirty(struct htp_context * ctx, const struct htp_tensor * t) {
    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;
    uint32_t bi      = t->bi;

    LOG_CACHE("make_dirty: t #%u bi %u range [%p, %p)", t->ti, t->bi, (void*)(uintptr_t)t_start, (void*)(uintptr_t)t_end);

    bool merged = false;

    for (int i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (r->start != 0) {
            if (r->start <= t_end && t_start <= r->end) {
                uint32_t new_start = (t_start < r->start) ? t_start : r->start;
                uint32_t new_end   = (t_end > r->end) ? t_end : r->end;
                LOG_CACHE("  merge into r[%d] [%p, %p) -> [%p, %p)", i,
                    (void*)(uintptr_t)r->start, (void*)(uintptr_t)r->end,
                    (void*)(uintptr_t)new_start, (void*)(uintptr_t)new_end);
                r->start = new_start;
                r->end   = new_end;
                merged = true;
            }
        }
    }

    if (merged) {
        return;
    }

    for (int i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (r->start == 0) {
            r->start = t_start;
            r->end   = t_end;
            r->bi    = bi;
            LOG_CACHE("  added to r[%d]", i);
            return;
        }
    }

    // 3. Cache is full. Find the slot with the smallest size (lowest flush penalty)
    int min_idx = 0;
    uint32_t min_size = ctx->dirty_ranges[0].end - ctx->dirty_ranges[0].start;
    for (int i = 1; i < HTP_MAX_DIRTY_RANGES; i++) {
        uint32_t size = ctx->dirty_ranges[i].end - ctx->dirty_ranges[i].start;
        if (size < min_size) {
            min_size = size;
            min_idx = i;
        }
    }

    LOG_CACHE("  cache full! evicting r[%d] size %u", min_idx, min_size);

    if (min_size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        // Eviction cost is too high, flush all instead
        flush_all_dcache(ctx);
        // Put new range in the first slot
        struct htp_dirty_range * first = &ctx->dirty_ranges[0];
        first->start = t_start;
        first->end   = t_end;
        first->bi    = bi;
    } else {
        struct htp_dirty_range * r = &ctx->dirty_ranges[min_idx];
        flush_address_range(ctx, r->start, min_size, 0);
        r->start = t_start;
        r->end   = t_end;
        r->bi    = bi;
    }
}

void htp_tensor_make_clean(struct htp_context * ctx, const struct htp_tensor * t) {
    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    LOG_CACHE("make_clean: t #%u bi %u range [%p, %p)", t->ti, t->bi, (void*)(uintptr_t)t_start, (void*)(uintptr_t)t_end);

    for (int i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (r->start != 0) {
            if (r->start < t_end && t_start < r->end) {
                LOG_CACHE("  overlap with r[%d] range [%p, %p)", i, (void*)(uintptr_t)r->start, (void*)(uintptr_t)r->end);
                if (t_start <= r->start && r->end <= t_end) {
                    LOG_CACHE("    complete cover -> invalidate");
                    r->start = 0;
                } else if (t_start <= r->start) {
                    LOG_CACHE("    overlap head -> trim start to %p", (void*)(uintptr_t)t_end);
                    r->start = t_end;
                } else if (r->end <= t_end) {
                    LOG_CACHE("    overlap tail -> trim end to %p", (void*)(uintptr_t)t_start);
                    r->end = t_start;
                } else {
                    LOG_CACHE("    overlap middle -> leave unmodified");
                }
            }
        }
    }
}

static inline bool is_tensor_dirty(struct htp_context * ctx, const struct htp_tensor * t) {
    uint32_t t_start = t->data;
    uint32_t t_end   = t_start + t->size;

    for (int i = 0; i < HTP_MAX_DIRTY_RANGES; i++) {
        struct htp_dirty_range * r = &ctx->dirty_ranges[i];
        if (r->start != 0) {
            if (r->start < t_end && t_start < r->end) {
                LOG_CACHE("is_dirty: t #%u bi %u range [%p, %p) -> dirty (matches r[%d] [%p, %p))",
                    t->ti, t->bi, (void*)(uintptr_t)t_start, (void*)(uintptr_t)t_end,
                    i, (void*)(uintptr_t)r->start, (void*)(uintptr_t)r->end);
                return true;
            }
        }
    }
    LOG_CACHE("is_dirty: t #%u bi %u range [%p, %p) -> clean",
        t->ti, t->bi, (void*)(uintptr_t)t_start, (void*)(uintptr_t)t_end);
    return false;
}

static void flush_all_dcache(struct htp_context * ctx) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    LOG_CACHE("flush_all_dcache: clearing all dirty ranges");
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    memset(ctx->dirty_ranges, 0, sizeof(ctx->dirty_ranges));
}

static void flush_tensor_range(struct htp_context * ctx, const struct htp_tensor * t) {
    flush_address_range(ctx, t->data, t->size, t->ti);
    htp_tensor_make_clean(ctx, t);
}

void htp_tensor_flush(struct htp_context * ctx, const struct htp_tensor * t) {
    if (!is_tensor_dirty(ctx, t)) {
        return;
    }

    if (t->size > HEX_L2_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        return;
    }

    flush_tensor_range(ctx, t);
}

void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    uint64_t total_dirty = 0;
    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (t && is_tensor_dirty(ctx, t)) {
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

    if (total_dirty > HEX_L2_FLUSH_WQ_THRESHOLD && ctx->n_threads > 1) {
        struct l2flush_multi_task task;
        task.trace    = ctx->trace;
        task.n_ranges = 0;

        uint32_t block_acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            const struct htp_tensor * t = tensors[i];
            if (!t || !is_tensor_dirty(ctx, t)) {
                continue;
            }
            htp_tensor_make_clean(ctx, t);

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
        if (!t || !is_tensor_dirty(ctx, t)) {
            continue;
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        hex_l2flush((void *) (uintptr_t) t->data, t->size);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, t->ti);
        htp_tensor_make_clean(ctx, t);
    }
}
