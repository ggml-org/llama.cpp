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
    uint32_t n_threads;
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
    size_t off_src1;
    size_t src1_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

static inline void htp_get_rows_vtcm_layout_build(
    struct htp_get_rows_vtcm_layout * L,
    int kernel_type,
    int type,
    uint32_t ne00,
    uint32_t total_tasks,
    int idx_type,
    uint32_t n_threads) {

    memset(L, 0, sizeof(*L));

    const size_t elem_size = (idx_type == HTP_TYPE_I64) ? sizeof(int64_t) : sizeof(int32_t);
    const size_t src1_raw_bytes = total_tasks * elem_size;
    const size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    L->src1_bytes = src1_bytes;
    const size_t src1_alloc_bytes = (src1_bytes + 255) & ~255;

    if (kernel_type != HTP_GET_ROWS_KERNEL_SAMETYPE) {
        size_t src0_row_size = 0;
        if (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
            const size_t tile_stride = htp_tensor_get_tile_padded_size(type);
            const uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_row_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } else {
            src0_row_size = htp_tensor_get_row_size(type, ne00);
        }

        L->src0_spad_half_size   = (src0_row_size + 255) & ~255;
        L->dst_spad_half_size    = (ne00 * sizeof(float) + 255) & ~255;
        L->src0_bytes_per_thread = 2 * L->src0_spad_half_size;
        L->dst_bytes_per_thread  = 2 * L->dst_spad_half_size;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread  * n_threads);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
