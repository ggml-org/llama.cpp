#ifndef HTP_GATHER_ROWS_H
#define HTP_GATHER_ROWS_H

#include "hex-common.h"
#include "htp-vtcm.h"
#include <stdint.h>
#include <stdbool.h>

#ifndef GGML_MAX_DIMS
#define GGML_MAX_DIMS 4
#endif

struct htp_gather_rows_params {
    uint32_t na;
    uint32_t nb;
    uint32_t ne;
    uint32_t rg;
    uint32_t vg;
    uint32_t nr_max;
    uint32_t v1;
    uint32_t span1;
    uint32_t region_size;
    uint32_t slice;
    uint32_t total_rows;
    uint32_t rows_per_thread;
    uint8_t  b_dense;
    uint8_t  pad[3];
};

struct htp_gather_rows_vtcm_layout {
    size_t off_tab;
    size_t off_inc;
    size_t off_cur;
    size_t off_region0;
    size_t off_out0;
    size_t off_region1;
    size_t off_out1;
    size_t total_per_thread;
};

static inline void htp_gather_rows_vtcm_layout_build(
    struct htp_gather_rows_vtcm_layout * L,
    uint32_t vg,
    uint32_t region_size,
    uint32_t nr_max,
    uint32_t ne,
    uint32_t elem_size) {

    const size_t vec_bytes      = (size_t) vg * 128;
    const size_t region_aligned = hex_round_up(region_size, 128);
    const size_t out_aligned    = hex_round_up(region_aligned + (size_t) nr_max * ne * elem_size, 128) - region_aligned;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_tab,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_inc,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_cur,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_region0, region_aligned);
    VTCM_LAYOUT_ALLOC(off, off_out0,    out_aligned);
    VTCM_LAYOUT_ALLOC(off, off_region1, region_aligned);
    VTCM_LAYOUT_ALLOC(off, off_out1,    out_aligned);
    L->total_per_thread = off;
}

static inline bool htp_gather_rows_solve_layout(
    struct htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t elem_size,
    uint32_t total_rows,
    uint32_t n_threads,
    uint32_t mdev_count,
    uint32_t vtcm_size) {

    const uint32_t max_ne0 = 128 / elem_size;
    const uint32_t ne = na + nb;
    if (ne == 0 || ne > max_ne0) {
        return false;
    }

    if (elem_size != 4 && elem_size != 2) {
        return false;
    }

    const uint32_t v_lanes = 128 / elem_size;
    const uint32_t g       = hex_gcd_u32(ne, v_lanes);
    const uint32_t rg      = v_lanes / g;
    const uint32_t vg      = ne / g;

    const uint32_t slice = (vtcm_size / n_threads) & ~127u;
    const uint32_t total_chunks = total_rows / rg;
    const uint32_t chunks_per_dev = (total_chunks + mdev_count - 1) / mdev_count;
    const uint32_t max_rows_per_dev = (mdev_count > 1 && total_chunks >= mdev_count)
        ? (chunks_per_dev * rg + (total_rows % rg))
        : total_rows;
    const uint32_t rows_per_thread = hex_round_up((max_rows_per_dev + n_threads - 1) / n_threads, rg);

    const uint32_t fixed = 3 * vg * 128;
    const uint32_t per_row = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0) + ne * elem_size;
    if (slice <= fixed + 2 * per_row * rg) {
        return false;
    }

    const uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_thread);
    if (elem_size == 2) {
        const uint32_t per_row_region = a_nb1 + (nb > 0 ? (b_dense ? b_nb1 : nb * b_nb1) : 0);
        if (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_size = 0;

    while (nr_max > 0) {
        v1 = hex_round_up((nr_max - 1) * a_nb1 + na * elem_size, 128);
        if (nb > 0) {
            if (b_dense) {
                span1       = elem_size;
                region_size = v1 + (nr_max - 1) * b_nb1 + nb * elem_size;
            } else {
                span1       = (nr_max - 1) * b_nb1 + elem_size;
                region_size = v1 + nb * span1;
            }
        } else {
            span1       = 0;
            region_size = v1;
        }

        struct htp_gather_rows_vtcm_layout layout;
        htp_gather_rows_vtcm_layout_build(&layout, vg, region_size, nr_max, ne, elem_size);

        if ((elem_size != 2 || region_size <= 32768) && layout.total_per_thread <= slice) {
            break;
        }

        nr_max -= rg;
    }

    if (nr_max == 0) {
        return false;
    }

    kparams->na              = na;
    kparams->nb              = nb;
    kparams->ne              = ne;
    kparams->rg              = rg;
    kparams->vg              = vg;
    kparams->nr_max          = nr_max;
    kparams->v1              = v1;
    kparams->span1           = span1;
    kparams->region_size     = region_size;
    kparams->slice           = slice;
    kparams->total_rows      = total_rows;
    kparams->rows_per_thread = rows_per_thread;
    kparams->b_dense         = b_dense ? 1 : 0;
    kparams->pad[0]          = 0;
    kparams->pad[1]          = 0;
    kparams->pad[2]          = 0;
    return true;
}

struct htp_gather_transpose_params {
    uint32_t elem_size;
    uint32_t ne_a;
    uint32_t ne_b;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t nb_a_src;
    uint32_t nb_b_dst;
    uint32_t nb2_src;
    uint32_t nb3_src;
    uint32_t nb2_dst;
    uint32_t nb3_dst;
    uint32_t tile_a;
    uint32_t tile_b;
};

static inline void htp_gather_transpose_layout_build(const struct htp_gather_transpose_params * tp,
                                                     struct htp_gather_rows_params *            gp,
                                                     struct htp_gather_rows_vtcm_layout *       layout) {
    const uint32_t lane = tp->elem_size % 4 == 0 ? 4 : 2;
    const uint32_t ne   = tp->elem_size / lane;
    const uint32_t g    = hex_gcd_u32(ne, 128 / lane);
    gp->na = gp->ne      = ne;
    gp->nb               = 0;
    gp->rg               = (128 / lane) / g;
    gp->vg               = ne / g;
    gp->region_size      = (tp->tile_a - 1) * tp->tile_b * tp->elem_size + tp->elem_size - 1;
    const uint32_t pitch = hex_round_up(tp->tile_a * tp->elem_size, 128);
    htp_gather_rows_vtcm_layout_build(layout, gp->vg, tp->tile_a * tp->tile_b * tp->elem_size, tp->tile_b, pitch / lane,
                                      lane);
}

#ifdef __cplusplus
struct ggml_tensor;
static inline bool htp_gather_rows_tensor_outer_contiguous(const struct ggml_tensor * t) {
    size_t next_nb = ((const size_t *) t->nb)[1] * ((const int64_t *) t->ne)[1];
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        if (((const int64_t *) t->ne)[i] != 1 && ((const size_t *) t->nb)[i] != next_nb) {
            return false;
        }
        next_nb *= ((const int64_t *) t->ne)[i];
    }
    return true;
}
#else
struct htp_ops_context;
struct htp_tensor;

int htp_gather_transpose(struct htp_ops_context *                   octx,
                         const struct htp_gather_transpose_params * kparams,
                         uint32_t                                   n_threads);

int htp_gather_rows(
    struct htp_ops_context * octx,
    const struct htp_tensor * a,
    const struct htp_tensor * b,
    const struct htp_gather_rows_params * kparams,
    uint32_t n_threads);
#endif

#endif // HTP_GATHER_ROWS_H
