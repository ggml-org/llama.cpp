#ifndef HTP_POOL_OPS_H
#define HTP_POOL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "hex-common.h"
#include "htp-vtcm.h"

enum htp_pool_kernel_type {
    HTP_POOL_KERNEL_GENERAL = 0,
    HTP_POOL_KERNEL_HVX,
    HTP_POOL_KERNEL_HVX_NARROW,
    HTP_POOL_KERNEL_GLOBAL,
    HTP_POOL_KERNEL_BLOCK,
};

struct htp_pool_2d_kernel_params {
    uint8_t  kernel_type;
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t kernel_x;
    uint32_t kernel_y;
    uint32_t stride_x;
    uint32_t stride_y;
    int32_t  pad_x;
    int32_t  pad_y;
    uint32_t n_threads;
    uint32_t planes;
    uint32_t planes_per_chunk;
    uint32_t pool_op;
    uint32_t avg_divide_count;
    uint32_t ox_lo;
    uint32_t ox_hi;
    float    inv_kernel_area;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

struct htp_pool_vtcm_layout {
    size_t   total_bytes;
    size_t   off_src;
    size_t   off_dst;
    uint32_t src_plane_bytes;
    uint32_t dst_plane_bytes;
    size_t   src_bytes_per_thread;
    size_t   dst_bytes_per_thread;
    size_t   src_spad_half_size;
    size_t   dst_spad_half_size;
};

static inline void htp_pool_vtcm_layout_build(
    struct htp_pool_vtcm_layout * L,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    L->src_plane_bytes      = src_x * src_y * sizeof(float);
    L->dst_plane_bytes      = dst_x * dst_y * sizeof(float);
    L->src_spad_half_size   = hex_round_up(L->src_plane_bytes + 256, 128);
    L->dst_spad_half_size   = hex_round_up(L->dst_plane_bytes, 128);
    L->src_bytes_per_thread = 2 * L->src_spad_half_size;
    L->dst_bytes_per_thread = 2 * L->dst_spad_half_size;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src, L->src_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst, L->dst_bytes_per_thread * n_threads);
    L->total_bytes = off;
}

static inline bool htp_pool_solve_layout(
    struct htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads,
    size_t   vtcm_budget
) {
    htp_pool_vtcm_layout_build(layout, src_x, src_y, dst_x, dst_y, n_threads);
    return layout->total_bytes <= vtcm_budget;
}

#endif // HTP_POOL_OPS_H
