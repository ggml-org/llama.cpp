#ifndef HTP_FILL_OPS_H
#define HTP_FILL_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

enum htp_fill_kernel_type {
    HTP_FILL_KERNEL_CONTIGUOUS = 0,
    HTP_FILL_KERNEL_STRIDED    = 1,
};

struct htp_fill_vtcm_layout {
    size_t tile_bytes;
    size_t off_tile;
    size_t total_bytes;
};

static inline void htp_fill_vtcm_layout_build(
    struct htp_fill_vtcm_layout * L,
    size_t   tile_bytes,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->tile_bytes = hex_round_up(tile_bytes, 128);
    VTCM_LAYOUT_ALLOC(off, off_tile, (size_t) n_hvx * L->tile_bytes);
    L->total_bytes = off;
}

struct htp_fill_kernel_params {
    uint8_t  kernel_type;
    uint32_t n_hvx;
    uint32_t total_rows;
    uint32_t elem_size;
    uint32_t tile_bytes;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne1_ne2;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fill_kernel_params) <= 128, "htp_fill_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_fill_kernel_params) <= 128, "htp_fill_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_FILL_OPS_H
