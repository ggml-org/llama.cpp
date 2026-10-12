#include "gather-rows.h"

#include <assert.h>

#include "hex-common.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "work-queue.h"

typedef void (*gather_rows_compute_fn_t)(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur);

struct gather_rows_task {
    struct htp_ops_context *              octx;
    const struct htp_tensor *             a;
    const struct htp_tensor *             b;
    dma_addr_t                            dst;
    uint32_t                              r_start;
    uint32_t                              rows;
    uint32_t                              rows_per_thread;
    const struct htp_gather_rows_params * kparams;
    gather_rows_compute_fn_t              compute;
};

static inline __attribute__((always_inline)) void gather_rows_dma_in(
    dma_queue * q, const struct gather_rows_task * task,
    uint8_t * region, uint32_t r, uint32_t nr, uint32_t elem_size) {
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;
    const struct htp_gather_rows_params * kparams = task->kparams;

    const size_t a_size = (nr - 1) * a->nb[1] + kparams->na * elem_size;
    dma_queue_push(q, dma_make_data(region, a->data + r * a->nb[1]), a_size, a_size, a_size, 1);
    if (b) {
        if (kparams->b_dense) {
            const size_t b_size = (nr - 1) * b->nb[1] + kparams->nb * elem_size;
            dma_queue_push(q, dma_make_data(region + kparams->v1, b->data + r * b->nb[1]), b_size, b_size, b_size, 1);
        } else {
            const size_t b_size = (nr - 1) * b->nb[1] + elem_size;
            for (uint32_t k = 0; k < kparams->nb; k++) {
                dma_queue_push(q, dma_make_data(region + kparams->v1 + k * kparams->span1,
                                                b->data + k * b->nb[0] + r * b->nb[1]), b_size, b_size, b_size, 1);
            }
        }
    }
}

static inline void gather_rows_dma_out(dma_queue * q, const struct gather_rows_task * task,
                                       uint8_t * out, uint32_t r, uint32_t nr, uint32_t elem_size) {
    const struct htp_gather_rows_params * kparams = task->kparams;
    const size_t out_size = nr * kparams->ne * elem_size;
    dma_queue_push(q, dma_make_data(task->dst + r * kparams->ne * elem_size, out), out_size, out_size, out_size, 1);
}

static __attribute__((noinline)) void gather_rows_init_tables_f32(
    HVX_Vector * tab, HVX_Vector * inc,
    const struct htp_gather_rows_params * kparams,
    const struct htp_tensor * a, const struct htp_tensor * b) {

    const int32_t s_a = kparams->rg * a->nb[1];
    const int32_t s_b = b ? kparams->rg * b->nb[1] : 0;
    const uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < kparams->vg; k++) {
        for (uint32_t lane = 0; lane < 32; lane++) {
            if (col < kparams->na) {
                t_buf[lane] = row * a->nb[1] + col * 4;
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1;
                s_buf[lane] = s_b;
            }
            if (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
}

static __attribute__((noinline)) void gather_rows_init_tables_f16(
    HVX_Vector * tab, HVX_Vector * inc,
    const struct htp_gather_rows_params * kparams,
    const struct htp_tensor * a, const struct htp_tensor * b) {

    const int16_t s_a = (int16_t) (kparams->rg * a->nb[1]);
    const int16_t s_b = (int16_t) (b ? kparams->rg * b->nb[1] : 0);
    const uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < kparams->vg; k++) {
        for (uint32_t lane = 0; lane < 64; lane++) {
            if (col < kparams->na) {
                t_buf[lane] = (int16_t) (row * a->nb[1] + col * 2);
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = (int16_t) (kparams->v1 + (col - kparams->na) * kparams->span1 + row * b_nb1);
                s_buf[lane] = s_b;
            }
            if (++col == kparams->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
}

static void gather_rows_compute_f32_vg1(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (void) cur;
    const uint32_t region_size = kparams->region_size;
    HVX_Vector vcur = tab[0];
    const HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_size, vcur);
        vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
    }
}

static void gather_rows_compute_f32_vgn(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    const uint32_t vg = kparams->vg;
    const uint32_t region_size = kparams->region_size;
    for (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, region_size, cur[k]);
        cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
        if (++k == vg) {
            k = 0;
        }
    }
}

static void gather_rows_compute_f16_vg1(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    (void) cur;
    const uint32_t region_size = kparams->region_size;
    HVX_Vector vcur = tab[0];
    const HVX_Vector vinc = inc[0];
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_size, vcur);
        vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
    }
}

static void gather_rows_compute_f16_vgn(
    uint8_t * restrict out, const uint8_t * restrict region, uint32_t nvec,
    const struct htp_gather_rows_params * restrict kparams,
    const HVX_Vector * restrict tab, const HVX_Vector * restrict inc, HVX_Vector * restrict cur) {
    const uint32_t vg = kparams->vg;
    const uint32_t region_size = kparams->region_size;
    for (uint32_t k = 0; k < vg; k++) {
        cur[k] = tab[k];
    }
    uint32_t k = 0;
    #pragma unroll(4)
    for (uint32_t v = 0; v < nvec; v++) {
        Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, region_size, cur[k]);
        cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
        if (++k == vg) {
            k = 0;
        }
    }
}

static void gather_rows_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct gather_rows_task * task = (const struct gather_rows_task *) data;
    const struct htp_gather_rows_params * kparams = task->kparams;
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;

    const uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    if (r_beg >= r_end) {
        return;
    }

    struct htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * kparams->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + kparams->vg;
    HVX_Vector * cur = inc + kparams->vg;
    uint8_t * region_base = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_base, region_base + kparams->buf_stride };
    uint8_t * out[2]    = { region[0] + kparams->out_off, region[1] + kparams->out_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_rows_init_tables_f32(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    const uint32_t n_in   = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);
    const uint32_t nr_max = kparams->nr_max;
    const uint32_t nr0    = MIN(nr_max, r_end - r_beg);

    dma_queue_push(q, dma_make_data(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 4);

    const uint32_t r1 = r_beg + nr_max;
    if (r1 < r_end) {
        const uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_push(q, dma_make_data(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 4);
    }

    uint32_t buf = 0;
    for (uint32_t r = r_beg; r < r_end; r += nr_max) {
        const uint32_t nr = MIN(nr_max, r_end - r);
        const uint32_t next_buf = buf ^ 1;

        dma_queue_pop(q); // complete out
        for (uint32_t i = 0; i < n_in; i++) {
            dma_queue_pop(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        const uint32_t nvec = (nr * kparams->ne + 31) >> 5;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 4);

        const uint32_t r_prefetch = r + 2 * nr_max;
        if (r_prefetch < r_end) {
            const uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 4);
        }

        buf = next_buf;
    }
    dma_queue_flush(q);
}

static void gather_rows_thread_f16(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct gather_rows_task * task = (const struct gather_rows_task *) data;
    const struct htp_gather_rows_params * kparams = task->kparams;
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;

    const uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    if (r_beg >= r_end) {
        return;
    }

    struct htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * kparams->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + kparams->vg;
    HVX_Vector * cur = inc + kparams->vg;
    uint8_t * region_base = (uint8_t *) (cur + kparams->vg);

    uint8_t * region[2] = { region_base, region_base + kparams->buf_stride };
    uint8_t * out[2]    = { region[0] + kparams->out_off, region[1] + kparams->out_off };

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    gather_rows_init_tables_f16(tab, inc, kparams, a, b);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];
    const uint32_t n_in = 1 + (task->b ? (kparams->b_dense ? 1 : kparams->nb) : 0);

    const uint32_t nr_max = kparams->nr_max;
    const uint32_t nr0 = MIN(nr_max, r_end - r_beg);
    dma_queue_push(q, dma_make_data(task->dst, out[0]), 0, 0, 0, 0); // dummy out
    gather_rows_dma_in(q, task, region[0], r_beg, nr0, 2);

    const uint32_t r1 = r_beg + nr_max;
    if (r1 < r_end) {
        const uint32_t nr1 = MIN(nr_max, r_end - r1);
        dma_queue_push(q, dma_make_data(task->dst, out[1]), 0, 0, 0, 0); // dummy out
        gather_rows_dma_in(q, task, region[1], r1, nr1, 2);
    }

    uint32_t buf = 0;
    for (uint32_t r = r_beg; r < r_end; r += nr_max) {
        const uint32_t nr = MIN(nr_max, r_end - r);
        const uint32_t next_buf = buf ^ 1;

        dma_queue_pop(q); // complete out
        for (uint32_t i = 0; i < n_in; i++) {
            dma_queue_pop(q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        const uint32_t nvec = (nr * kparams->ne + 63) >> 6;
        task->compute(out[buf], region[buf], nvec, kparams, tab, inc, cur);
        hvx_gather_sync(out[buf]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        gather_rows_dma_out(q, task, out[buf], r, nr, 2);

        const uint32_t r_prefetch = r + 2 * nr_max;
        if (r_prefetch < r_end) {
            const uint32_t nr_prefetch = MIN(nr_max, r_end - r_prefetch);
            gather_rows_dma_in(q, task, region[buf], r_prefetch, nr_prefetch, 2);
        }

        buf = next_buf;
    }
    dma_queue_flush(q);
}

int htp_gather_rows(
    struct htp_ops_context * octx,
    const struct htp_tensor * a,
    const struct htp_tensor * b,
    const struct htp_gather_rows_params * kparams,
    uint32_t n_threads) {

    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    assert(kparams->slice * octx->n_threads <= octx->ctx->vtcm_size);

    const uint32_t total_rows = kparams->total_rows;
    uint32_t r_start = 0;
    uint32_t r_count = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, kparams->rg, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        r_start = range.start;
        r_count = range.count;
    }

    if (r_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t elem_size = (a->type == HTP_TYPE_F16) ? 2 : 4;

    struct gather_rows_task task;
    task.octx            = octx;
    task.a               = a;
    task.b               = b;
    task.dst             = octx->dst->data;
    task.r_start         = r_start;
    task.rows            = r_count;
    task.kparams         = kparams;
    task.rows_per_thread = kparams->rows_per_thread;

    if (elem_size == 2) {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f16_vg1 : gather_rows_compute_f16_vgn;
    } else {
        task.compute = (kparams->vg == 1) ? gather_rows_compute_f32_vg1 : gather_rows_compute_f32_vgn;
    }

    work_queue_func_t worker_func = (elem_size == 2) ? gather_rows_thread_f16 : gather_rows_thread_f32;
    work_queue_run(octx->ctx->work_queue, worker_func, &task, octx->n_threads);
    return HTP_STATUS_OK;
}
