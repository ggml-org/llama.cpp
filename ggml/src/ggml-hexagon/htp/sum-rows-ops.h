#ifndef HTP_SUM_ROWS_OPS_H
#define HTP_SUM_ROWS_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"

struct htp_sum_rows_kernel_params {
    uint32_t              n_hvx;
    uint32_t              elems_per_hvx;
    uint32_t              block;
    uint32_t              chunk_elems;
    uint32_t              src0_contig;
    uint32_t              vtcm_size;
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne012;
};

struct htp_sum_vtcm_layout {
    size_t chunk_bytes;
    size_t src0_bytes_per_hvx;
    size_t dst_bytes_per_hvx;
    size_t src0_off;
    size_t dst_off;
    size_t total_bytes;
};

static inline void htp_sum_vtcm_layout_build(
    struct htp_sum_vtcm_layout * L,
    uint32_t chunk_elems,
    size_t   src0_elem_size,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->chunk_bytes        = hex_round_up((size_t) chunk_elems * src0_elem_size, 128);
    L->src0_bytes_per_hvx = 2 * L->chunk_bytes;
    L->dst_bytes_per_hvx  = 128;

    VTCM_LAYOUT_ALLOC(off, src0_off, (size_t) n_hvx * L->src0_bytes_per_hvx);
    VTCM_LAYOUT_ALLOC(off, dst_off,  (size_t) n_hvx * L->dst_bytes_per_hvx);
    L->total_bytes = off;
}

struct htp_sum_rows_vtcm_layout {
    size_t src0_row_size_aligned;
    size_t src0_half_bytes;
    size_t dst_half_bytes;
    size_t src0_bytes_per_hvx;
    size_t dst_bytes_per_hvx;
    size_t src0_off;
    size_t dst_off;
    size_t total_bytes;
};

static inline void htp_sum_rows_vtcm_layout_build(
    struct htp_sum_rows_vtcm_layout * L,
    size_t   src0_row_size,
    size_t   dst_elem_size,
    uint32_t block,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    L->src0_half_bytes       = L->src0_row_size_aligned * block;
    L->dst_half_bytes        = hex_round_up((size_t) block * dst_elem_size, 128);
    L->src0_bytes_per_hvx    = 2 * L->src0_half_bytes;
    L->dst_bytes_per_hvx     = 2 * L->dst_half_bytes;

    VTCM_LAYOUT_ALLOC(off, src0_off, (size_t) n_hvx * L->src0_bytes_per_hvx);
    VTCM_LAYOUT_ALLOC(off, dst_off,  (size_t) n_hvx * L->dst_bytes_per_hvx);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SUM_ROWS_OPS_H
