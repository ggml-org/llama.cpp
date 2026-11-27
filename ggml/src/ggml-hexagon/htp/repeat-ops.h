#ifndef HTP_REPEAT_OPS_H
#define HTP_REPEAT_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

struct htp_repeat_vtcm_layout {
    size_t src_buf_bytes;
    size_t dst_buf_bytes;
    size_t src_bytes_per_hvx;
    size_t dst_bytes_per_hvx;
    size_t off_src;
    size_t off_dst;
    size_t total_bytes;
};

static inline void htp_repeat_vtcm_layout_build(
    struct htp_repeat_vtcm_layout * L,
    size_t   src_tile_bytes,
    size_t   dst_tile_bytes,
    uint32_t n_bufs,
    uint32_t n_hvx
) {
    size_t off = 0;

    L->src_buf_bytes     = hex_round_up(src_tile_bytes, 128);
    L->dst_buf_bytes     = hex_round_up(dst_tile_bytes, 128);
    L->src_bytes_per_hvx = (size_t) n_bufs * L->src_buf_bytes;
    L->dst_bytes_per_hvx = (size_t) n_bufs * L->dst_buf_bytes;

    VTCM_LAYOUT_ALLOC(off, off_src, (size_t) n_hvx * L->src_bytes_per_hvx);
    VTCM_LAYOUT_ALLOC(off, off_dst, (size_t) n_hvx * L->dst_bytes_per_hvx);
    L->total_bytes = off;
}

struct htp_repeat_kernel_params {
    uint32_t n_hvx;
    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;
    uint32_t tile_elems;
    uint32_t dst_reps;
    uint32_t pipelined;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne03;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_repeat_kernel_params) <= 128, "htp_repeat_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_repeat_kernel_params) <= 128, "htp_repeat_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_REPEAT_OPS_H
