#ifndef HTP_IM2COL_OPS_H
#define HTP_IM2COL_OPS_H

#include "hex-common.h"
#include <stdint.h>

enum htp_im2col_kernel_type {
    HTP_IM2COL_KERNEL_UNSUPPORTED    = 0,
    HTP_IM2COL_KERNEL_PATCHEMBED_DMA = 1,
    HTP_IM2COL_KERNEL_BLOCKED_DMA    = 2,
};

struct htp_im2col_vtcm_layout {
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_bytes_per_thread;
    uint32_t dst_bytes_per_thread;
    uint32_t total_bytes;
};

static inline void htp_im2col_vtcm_layout_build(
    struct htp_im2col_vtcm_layout * layout,
    uint32_t src_row_bytes,
    uint32_t dst_row_bytes,
    uint32_t n_threads) {

    layout->src_bytes_per_thread = 2 * src_row_bytes + 256;
    layout->dst_bytes_per_thread = 2 * dst_row_bytes;

    layout->off_src     = 0;
    layout->off_dst     = layout->off_src + layout->src_bytes_per_thread * n_threads;
    layout->total_bytes = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
}

struct htp_im2col_kernel_params {
    uint8_t  kernel_type;
    uint8_t  is_2d;
    uint8_t  is_3d;
    uint8_t  n_threads;

    uint32_t vtcm_size;
    uint32_t off_src;
    uint32_t off_dst;
    uint32_t src_bytes_per_thread;
    uint32_t dst_bytes_per_thread;
    uint32_t src_row_bytes;
    uint32_t dst_row_bytes;

    uint32_t owb;
    uint32_t wb;
    uint32_t nbpr;

    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;
    uint32_t total_rows;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_im2col_kernel_params) <= 128, "htp_im2col_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_IM2COL_OPS_H
