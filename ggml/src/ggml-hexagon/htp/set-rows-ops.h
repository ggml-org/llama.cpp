#ifndef HTP_SET_ROWS_OPS_H
#define HTP_SET_ROWS_OPS_H

#include "hex-fastdiv.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"

struct htp_set_rows_kernel_params {
    uint32_t n_threads;
    uint32_t total_tasks;
    uint32_t vtcm_size;
};

struct htp_set_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

static inline void htp_set_rows_vtcm_layout_build(
    struct htp_set_rows_vtcm_layout * L,
    int dst_type,
    uint32_t ne00,
    uint32_t n_threads) {

    size_t src0_row_size = ne00 * sizeof(float);
    size_t dst_row_size  = htp_tensor_get_row_size(dst_type, ne00);

    size_t src0_row_size_aligned = (src0_row_size + 255) & ~255;
    size_t dst_row_size_aligned  = (dst_row_size + 255) & ~255;

    L->src0_spad_half_size = src0_row_size_aligned;
    L->dst_spad_half_size  = dst_row_size_aligned;

    L->src0_bytes_per_thread = src0_row_size_aligned * 2;
    L->dst_bytes_per_thread  = dst_row_size_aligned * 2;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_set_rows_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SET_ROWS_OPS_H
