#ifndef HTP_SOLVE_TRI_OPS_H
#define HTP_SOLVE_TRI_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

struct htp_solve_tri_vtcm_layout {
    size_t a_row_aligned;
    size_t a_bytes_per_hvx;
    size_t bx_bytes_per_hvx;
    size_t off_a;
    size_t off_bx;
    size_t total_bytes;
};

static inline void htp_solve_tri_vtcm_layout_build(
    struct htp_solve_tri_vtcm_layout * L,
    uint32_t n,
    uint32_t a_full,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->a_row_aligned    = hex_round_up((size_t) n * sizeof(float), 128);
    L->a_bytes_per_hvx  = a_full ? ((size_t) n * L->a_row_aligned) : L->a_row_aligned;
    L->bx_bytes_per_hvx = (size_t) n * 128;

    VTCM_LAYOUT_ALLOC(off, off_a,  (size_t) n_hvx * L->a_bytes_per_hvx);
    VTCM_LAYOUT_ALLOC(off, off_bx, (size_t) n_hvx * L->bx_bytes_per_hvx);
    L->total_bytes = off;
}

struct htp_solve_tri_kernel_params {
    uint32_t n_hvx;
    uint32_t total_batches;
    uint32_t total_jobs;
    uint32_t k_chunks;
    uint32_t col_block;
    uint32_t batched;
    uint32_t a_full;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne02;
    struct fastdiv_values div_k_chunks;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_solve_tri_kernel_params) <= 128, "htp_solve_tri_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_solve_tri_kernel_params) <= 128, "htp_solve_tri_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SOLVE_TRI_OPS_H
