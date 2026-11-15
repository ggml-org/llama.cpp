#ifndef HTP_CONCAT_OPS_H
#define HTP_CONCAT_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include "gather-rows.h"
#include <stdint.h>

enum htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ROWS = 3,
};

#define HTP_CONCAT_REG_BLK_BYTES (32 * 1024)

struct htp_concat_regular_params {
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02_ne01;
    struct fastdiv_values div_ne11;
    struct fastdiv_values div_ne12_ne11;
};

struct htp_concat_transposed_params {
    struct fastdiv_values div_ne2;
};

struct htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_hvx;
    uint8_t  pad;

    uint32_t vtcm_size;

    union {
        struct htp_concat_regular_params    regular;
        struct htp_concat_transposed_params transposed;
        struct htp_gather_rows_params       gather;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#endif

struct htp_concat_gather_rows_vtcm_layout {
    size_t off_tab;
    size_t off_inc;
    size_t off_cur;
    size_t off_region0;
    size_t off_out0;
    size_t off_region1;
    size_t off_out1;
    size_t off_raw_a0;
    size_t off_raw_a1;
    size_t off_raw_b0;
    size_t off_raw_b1;
    size_t total_per_hvx;
};

static inline void htp_concat_gather_rows_vtcm_layout_build(
    struct htp_concat_gather_rows_vtcm_layout * L,
    uint32_t vg,
    uint32_t region_size,
    uint32_t nr_max,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense, uint32_t span1,
    uint32_t ne,
    uint32_t src0_es, uint32_t src1_es, uint32_t dst_es) {

    const size_t vec_bytes      = (size_t) vg * 128;
    const size_t region_aligned = hex_round_up(region_size, 128);
    const size_t out_aligned    = hex_round_up(region_aligned + (size_t) nr_max * ne * dst_es, 128) - region_aligned;

    size_t raw_a_aligned = 0;
    if (src0_es != dst_es && na > 0) {
        raw_a_aligned = hex_round_up((size_t) (nr_max - 1) * a_nb1 + (size_t) na * src0_es, 128);
    }

    size_t raw_b_aligned = 0;
    if (src1_es != dst_es && nb > 0) {
        const size_t raw_span1 = ((size_t) span1 / dst_es) * src1_es;
        const size_t raw_b_bytes = b_dense
            ? ((size_t) (nr_max - 1) * b_nb1 + (size_t) nb * src1_es)
            : ((size_t) nb * raw_span1);
        raw_b_aligned = hex_round_up(raw_b_bytes, 128);
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_tab,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_inc,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_cur,     vec_bytes);
    VTCM_LAYOUT_ALLOC(off, off_region0, region_aligned);
    VTCM_LAYOUT_ALLOC(off, off_out0,    out_aligned);
    VTCM_LAYOUT_ALLOC(off, off_region1, region_aligned);
    VTCM_LAYOUT_ALLOC(off, off_out1,    out_aligned);
    if (raw_a_aligned > 0) {
        VTCM_LAYOUT_ALLOC(off, off_raw_a0, raw_a_aligned);
        VTCM_LAYOUT_ALLOC(off, off_raw_a1, raw_a_aligned);
    } else {
        L->off_raw_a0 = 0;
        L->off_raw_a1 = 0;
    }
    if (raw_b_aligned > 0) {
        VTCM_LAYOUT_ALLOC(off, off_raw_b0, raw_b_aligned);
        VTCM_LAYOUT_ALLOC(off, off_raw_b1, raw_b_aligned);
    } else {
        L->off_raw_b0 = 0;
        L->off_raw_b1 = 0;
    }
    L->total_per_hvx = off;
}

static inline bool htp_concat_gather_rows_solve_layout(
    struct htp_gather_rows_params * kparams,
    uint32_t na, uint32_t a_nb1,
    uint32_t nb, uint32_t b_nb1, bool b_dense,
    uint32_t src0_es, uint32_t src1_es, uint32_t dst_es,
    uint32_t total_rows,
    uint32_t n_hvx,
    uint32_t mdev_count,
    uint32_t vtcm_size) {

    if (src0_es == dst_es && src1_es == dst_es) {
        return htp_gather_rows_solve_layout(kparams, na, a_nb1, nb, b_nb1, b_dense, dst_es, total_rows, n_hvx, mdev_count, vtcm_size);
    }

    const uint32_t max_ne0 = 128 / dst_es;
    const uint32_t ne = na + nb;
    if (ne == 0 || ne > max_ne0) {
        return false;
    }

    if ((dst_es != 4 && dst_es != 2) || (src0_es != 4 && src0_es != 2) || (src1_es != 4 && src1_es != 2)) {
        return false;
    }
    if ((a_nb1 % src0_es) != 0 || (b_nb1 % src1_es) != 0) {
        return false;
    }

    const uint32_t a_reg_nb1 = (a_nb1 / src0_es) * dst_es;
    const uint32_t b_reg_nb1 = (b_nb1 / src1_es) * dst_es;

    const uint32_t v_lanes = 128 / dst_es;
    const uint32_t g       = hex_gcd_u32(ne, v_lanes);
    const uint32_t rg      = v_lanes / g;
    const uint32_t vg      = ne / g;

    const uint32_t slice = (vtcm_size / n_hvx) & ~127u;
    const uint32_t total_chunks = total_rows / rg;
    const uint32_t chunks_per_dev = (total_chunks + mdev_count - 1) / mdev_count;
    const uint32_t max_rows_per_dev = (mdev_count > 1 && total_chunks >= mdev_count)
        ? (chunks_per_dev * rg + (total_rows % rg))
        : total_rows;
    const uint32_t rows_per_hvx = hex_round_up((max_rows_per_dev + n_hvx - 1) / n_hvx, rg);

    const uint32_t fixed = 3 * vg * 128;
    const uint32_t per_row_reg_a = na > 0 ? a_reg_nb1 : 0;
    const uint32_t per_row_reg_b = nb > 0 ? (b_dense ? b_reg_nb1 : nb * b_reg_nb1) : 0;
    const uint32_t per_row_raw_a = (src0_es != dst_es && na > 0) ? a_nb1 : 0;
    const uint32_t per_row_raw_b = (src1_es != dst_es && nb > 0) ? (b_dense ? b_nb1 : nb * b_nb1) : 0;
    const uint32_t per_row = per_row_reg_a + per_row_reg_b + per_row_raw_a + per_row_raw_b + ne * dst_es;
    if (slice <= fixed + 2 * per_row * rg) {
        return false;
    }

    const uint32_t avail_per_buf = (slice - fixed) / 2;
    uint32_t nr_max = MIN(avail_per_buf / per_row, rows_per_hvx);
    if (dst_es == 2) {
        const uint32_t per_row_region = per_row_reg_a + per_row_reg_b;
        if (per_row_region > 0) {
            nr_max = MIN(nr_max, (32768u - 256u) / per_row_region);
        }
    }
    nr_max = (nr_max / rg) * rg;

    uint32_t v1          = 0;
    uint32_t span1       = 0;
    uint32_t region_size = 0;

    while (nr_max > 0) {
        v1 = na > 0 ? hex_round_up((nr_max - 1) * a_reg_nb1 + na * dst_es, 128) : 0;
        if (nb > 0) {
            if (b_dense) {
                span1       = dst_es;
                region_size = v1 + (nr_max - 1) * b_reg_nb1 + nb * dst_es;
            } else {
                span1       = (nr_max - 1) * b_reg_nb1 + dst_es;
                region_size = v1 + nb * span1;
            }
        } else {
            span1       = 0;
            region_size = v1;
        }

        struct htp_concat_gather_rows_vtcm_layout layout;
        htp_concat_gather_rows_vtcm_layout_build(&layout, vg, region_size, nr_max,
                                                 na, a_nb1, nb, b_nb1, b_dense, span1, ne,
                                                 src0_es, src1_es, dst_es);

        if ((dst_es != 2 || region_size <= 32768) && layout.total_per_hvx <= slice) {
            break;
        }

        nr_max -= rg;
    }

    if (nr_max == 0) {
        return false;
    }

    kparams->na           = na;
    kparams->nb           = nb;
    kparams->ne           = ne;
    kparams->rg           = rg;
    kparams->vg           = vg;
    kparams->nr_max       = nr_max;
    kparams->v1           = v1;
    kparams->span1        = span1;
    kparams->region_size  = region_size;
    kparams->slice        = slice;
    kparams->total_rows   = total_rows;
    kparams->rows_per_hvx = rows_per_hvx;
    kparams->b_dense      = b_dense ? 1 : 0;
    kparams->pad[0]       = 0;
    kparams->pad[1]       = 0;
    kparams->pad[2]       = 0;
    return true;
}

struct htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_size_per_hvx;
    uint32_t src1_spad_size_per_hvx;
    uint32_t cvt_spad_size_per_hvx;
    size_t   off_spad0;
    size_t   off_spad1;
    size_t   off_spad2;
    uint32_t total_bytes;
};

static inline void htp_concat_transposed_vtcm_layout_build(
    struct htp_concat_transposed_vtcm_layout * L,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t src0_es,
    uint32_t src1_es,
    uint32_t dst_es,
    uint32_t n_hvx) {

    // src1_ne0 is processed in chunks of block_cols tokens, so VTCM does not grow with the batch
    uint32_t block_rows = 128;
    uint32_t block_cols = (src1_es == 2 && dst_es == 2) ? 512 : 256;
    uint32_t col_align  = (src1_es == 2 || dst_es == 2) ? 64 : 32;
    // +1 element spreads the gather lanes across VTCM, so they do not serialize on one path
    uint32_t spad1_stride = block_rows * src1_es + src1_es;
    uint32_t block_cols_padded = MIN(hex_round_up(src1_ne0, col_align), block_cols);

    uint32_t dst_src0_row_padded = hex_round_up(src0_ne0 * dst_es, 128);
    uint32_t raw_src0_row_padded = (src0_es != dst_es) ? hex_round_up(src0_ne0 * src0_es, 128) : 0;
    uint32_t spad0_row_bytes     = MAX(dst_src0_row_padded + block_cols_padded * dst_es, raw_src0_row_padded);

    uint32_t cvt1_bytes = (src1_es != dst_es) ? (block_rows * 64 * src1_es) : 0;
    uint32_t cvt0_bytes = (src0_es != dst_es) ? dst_src0_row_padded : 0;

    // double-buffered: two slots per thread
    L->src0_spad_size_per_hvx = 2 * block_rows * spad0_row_bytes;
    L->src1_spad_size_per_hvx = 2 * block_cols_padded * spad1_stride;
    L->cvt_spad_size_per_hvx  = hex_round_up(MAX(cvt1_bytes, cvt0_bytes), 128);

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_spad0, (size_t) n_hvx * L->src0_spad_size_per_hvx);
    VTCM_LAYOUT_ALLOC(off, off_spad1, (size_t) n_hvx * L->src1_spad_size_per_hvx);
    if (L->cvt_spad_size_per_hvx > 0) {
        VTCM_LAYOUT_ALLOC(off, off_spad2, (size_t) n_hvx * L->cvt_spad_size_per_hvx);
    } else {
        L->off_spad2 = 0;
    }
    L->total_bytes = (uint32_t) off;
}

#endif // HTP_CONCAT_OPS_H
