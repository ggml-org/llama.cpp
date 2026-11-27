#ifndef HTP_ROLL_OPS_H
#define HTP_ROLL_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

enum htp_roll_kernel_type {
    HTP_ROLL_KERNEL_DMA_CONTIGUOUS = 1,
    HTP_ROLL_KERNEL_DMA_STRIDED    = 2,
    HTP_ROLL_KERNEL_HVX            = 3,
};

struct htp_roll_vtcm_layout {
    size_t row_size_aligned;
    size_t src_bytes_per_hvx;
    size_t dst_bytes_per_hvx;
    size_t off_src;
    size_t off_dst;
    size_t total_bytes;
};

static inline void htp_roll_vtcm_layout_build(
    struct htp_roll_vtcm_layout * L,
    size_t   row_size,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->row_size_aligned  = hex_round_up(row_size, 128);
    L->src_bytes_per_hvx = 2 * L->row_size_aligned;
    L->dst_bytes_per_hvx = 2 * L->row_size_aligned;

    VTCM_LAYOUT_ALLOC(off, off_src, (size_t) n_hvx * L->src_bytes_per_hvx);
    VTCM_LAYOUT_ALLOC(off, off_dst, (size_t) n_hvx * L->dst_bytes_per_hvx);
    L->total_bytes = off;
}

struct htp_roll_kernel_params {
    uint8_t  kernel_type;
    uint32_t n_hvx;
    uint32_t total_rows;
    uint32_t i0_src0;
    uint32_t n0;
    uint32_t elem_size;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2_ne1;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_roll_kernel_params) <= 128, "htp_roll_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_roll_kernel_params) <= 128, "htp_roll_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ROLL_OPS_H
