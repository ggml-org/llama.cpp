#pragma once

// Copy many short f32 rows (a few elements each) through VTCM: DMA the source spans in, rearrange with vgather, DMA the result out.
// Per-row DMA or scalar copies of such rows wait on memory for every row.
// dst row i = a[i][0..na) followed by b[i][0..nb) (b column k at b->data + k*b->nb[0]); dst rows are contiguous.

#include "hex-common.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct hvx_gather_rows_job {
    struct htp_ops_context * octx;
    const struct htp_tensor * a;
    const struct htp_tensor * b;
    dma_addr_t dst;
    uint32_t na, nb, ne;           // words per row: from a, from b, total
    uint32_t rows;
    uint32_t rg, vg;               // offsets repeat every rg rows = vg vectors
    uint32_t rows_per_thread;
    uint32_t nr_max;               // rows per VTCM chunk
    uint32_t v1, span1;            // region layout: a rows at 0, b column k at v1 + k*span1
    uint32_t region_size, out_off, slice;
};

static inline void hvx_gather_rows_dma(dma_queue * q, dma_addr_t dst, dma_addr_t src, size_t size) {
    if (!dma_queue_push(q, dma_make_data(dst, src), size, size, size, 1)) {
        dma_queue_flush(q);
        dma_queue_push(q, dma_make_data(dst, src), size, size, size, 1);
    }
}

static void hvx_gather_rows_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct hvx_gather_rows_job * j = (const struct hvx_gather_rows_job *) data;
    const struct htp_tensor * a = j->a;
    const struct htp_tensor * b = j->b;

    const uint32_t r_beg = MIN(ith * j->rows_per_thread, j->rows);
    const uint32_t r_end = MIN(r_beg + j->rows_per_thread, j->rows);
    if (r_beg >= r_end) {
        return;
    }

    uint8_t * base   = j->octx->ctx->vtcm_base + ith * j->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + j->vg;
    HVX_Vector * cur = inc + j->vg;
    uint8_t * sync   = (uint8_t *) (cur + j->vg);
    uint8_t * region = sync + 128;
    uint8_t * out    = region + j->out_off;

    // byte offsets of output word e = k*32 + lane, for one period
    for (uint32_t k = 0; k < j->vg; k++) {
        int32_t * t = (int32_t *) (tab + k);
        int32_t * s = (int32_t *) (inc + k);
        for (uint32_t lane = 0; lane < 32; lane++) {
            const uint32_t e   = k * 32 + lane;
            const uint32_t row = e / j->ne;
            const uint32_t col = e - row * j->ne;
            if (col < j->na) {
                t[lane] = row * a->nb[1] + col * 4;
                s[lane] = j->rg * a->nb[1];
            } else {
                t[lane] = j->v1 + (col - j->na) * j->span1 + row * b->nb[1];
                s[lane] = j->rg * b->nb[1];
            }
        }
    }

    dma_queue * q = j->octx->ctx->dma[ith];

    for (uint32_t r = r_beg; r < r_end; r += j->nr_max) {
        const uint32_t nr = MIN(j->nr_max, r_end - r);

        hvx_gather_rows_dma(q, (dma_addr_t) region, a->data + r * a->nb[1], (nr - 1) * a->nb[1] + j->na * 4);
        for (uint32_t k = 0; k < j->nb; k++) {
            hvx_gather_rows_dma(q, (dma_addr_t) (region + j->v1 + k * j->span1), b->data + k * b->nb[0] + r * b->nb[1], (nr - 1) * b->nb[1] + 4);
        }
        dma_queue_flush(q);

        for (uint32_t k = 0; k < j->vg; k++) {
            cur[k] = tab[k];
        }

        const uint32_t nvec = (nr * j->ne + 31) / 32;
        uint32_t k = 0;
        for (uint32_t v = 0; v < nvec; v++) {
            Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, j->region_size, cur[k]);
            cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
            if (++k == j->vg) {
                k = 0;
            }
        }

        // vector loads wait for the gathers; DMA does not
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t v = 0; v < nvec; v++) {
            acc = Q6_V_vor_VV(acc, *(const HVX_Vector *) (out + v * 128));
        }
        *(HVX_Vector *) sync = acc;

        hvx_gather_rows_dma(q, j->dst + r * j->ne * 4, (dma_addr_t) out, nr * j->ne * 4);
        dma_queue_flush(q);
    }
}

// Returns false when the shape does not fit; the caller keeps its own path then.
static inline bool hvx_gather_rows_run(struct htp_ops_context * octx, const struct htp_tensor * a, uint32_t na,
                                       const struct htp_tensor * b, uint32_t nb, dma_addr_t dst, uint32_t rows) {
    const uint32_t ne = na + nb;
    if (ne == 0 || ne > 32 || rows == 0 || octx->ctx->mdev.count > 1 ||
        (a->nb[1] % 4) != 0 || (b && ((b->nb[0] % 4) != 0 || (b->nb[1] % 4) != 0))) {
        return false;
    }

    struct hvx_gather_rows_job j;
    j.octx = octx;
    j.a    = a;
    j.b    = b;
    j.dst  = dst;
    j.na   = na;
    j.nb   = nb;
    j.ne   = ne;
    j.rows = rows;
    j.vg   = ne / hex_gcd_u32(ne, 32);
    j.rg   = j.vg * 32 / ne;

    const uint32_t n_threads = octx->n_threads;
    j.slice = (uint32_t) (octx->ctx->vtcm_size / n_threads) & ~127u;
    j.rows_per_thread = hex_round_up((rows + n_threads - 1) / n_threads, j.rg);

    const uint32_t fixed   = (3 * j.vg + 1) * 128 + 3 * 128;
    const uint32_t per_row = a->nb[1] + (b ? nb * b->nb[1] : 0) + ne * 4;
    if (j.slice <= fixed + per_row * j.rg) {
        return false;
    }
    j.nr_max = MIN((j.slice - fixed) / per_row, j.rows_per_thread);
    j.nr_max = j.nr_max / j.rg * j.rg;
    if (j.nr_max == 0) {
        return false;
    }

    j.v1          = hex_round_up((j.nr_max - 1) * a->nb[1] + na * 4, 128);
    j.span1       = b ? (j.nr_max - 1) * b->nb[1] + 4 : 0;
    j.region_size = j.v1 + nb * j.span1;
    j.out_off     = hex_round_up(j.region_size, 128);

    work_queue_run(octx->ctx->work_queue, hvx_gather_rows_thread, &j, n_threads);
    return true;
}
