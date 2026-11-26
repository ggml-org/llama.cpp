#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#include <stdbool.h>
#include <string.h>

#include "hex-fastdiv.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"

enum htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

struct htp_get_rows_kernel_params {
    uint8_t  kernel_type;
    uint32_t n_hvx;
    uint32_t total_tasks;
    uint32_t vtcm_size;

    // Fastdiv helpers
    struct fastdiv_values div_ne10;
    struct fastdiv_values div_ne10_ne11;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne03;
};

struct htp_get_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_hvx;
    size_t dst_bytes_per_hvx;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

static inline void htp_get_rows_vtcm_layout_build(
    struct htp_get_rows_vtcm_layout * L,
    int kernel_type,
    int src0_type,
    int dst_type,
    uint32_t ne00,
    uint32_t n_hvx) {

    memset(L, 0, sizeof(*L));

    if (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        if (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            const size_t tile_stride = htp_tensor_get_tile_padded_size(src0_type);
            const uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(src0_type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } else {
            src0_row_size = htp_tensor_get_row_size(src0_type, ne00);
        }

        const bool direct_f16_spad =
            (kernel_type == HTP_GET_ROWS_KERNEL_TILED && src0_type == HTP_TYPE_F16 && dst_type == HTP_TYPE_F16) ||
            (kernel_type == HTP_GET_ROWS_KERNEL_FLAT  && src0_type == HTP_TYPE_F32 && dst_type == HTP_TYPE_F16);
        const size_t dst_spad_row_size = htp_tensor_get_row_size(direct_f16_spad ? dst_type : HTP_TYPE_F32, ne00);

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (dst_spad_row_size + 255) & ~255;
        L->src0_bytes_per_hvx    = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_hvx     = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_hvx * n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_hvx  * n_hvx);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
