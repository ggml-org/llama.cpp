#ifndef HTP_CONV2D_OPS_H
#define HTP_CONV2D_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

enum htp_conv2d_kernel_type {
    HTP_CONV2D_KERNEL_UNSUPPORTED = 0,
    HTP_CONV2D_KERNEL_HMX_WINDOW,       // HMX convolution with sliding-window staging
    HTP_CONV2D_KERNEL_HMX_TILED,        // tiled implicit-GEMM convolution on HMX
};

// HMX_WINDOW output staging buffers per thread
#define HTP_CONV2D_WIN_OBUF 2

// HMX scales of one row tile at the start of VTCM: 32 (scale, bias) fp16 pairs and 128 zero bytes
#define HTP_CONV2D_HMX_SCALES_SIZE 256

struct htp_conv2d_tiled_params {
    uint32_t w_kstride;
    uint32_t k_size;
    uint32_t k_group;
    uint32_t k_group_pad;
    uint32_t n_groups;
    uint32_t n_k_tiles;
    uint32_t n_ow_tiles;
    uint32_t seg_tiles;
    uint32_t stage_row;
    uint32_t n_stage;
    uint32_t wk_chunk;
    struct fastdiv_values div_ow_tiles;
    struct fastdiv_values div_oh;
};

struct htp_conv2d_window_params {
    uint32_t fold;
    uint32_t n_cc;
    uint32_t act_rt_last;
    uint32_t sr;
    uint32_t ihs;
    uint32_t max_off;
    uint32_t n_bc;
    uint32_t seg_b;
    uint32_t seg_t;
    uint32_t w_cs;
    struct fastdiv_values div_sr;
    struct fastdiv_values div_ihs;
    struct fastdiv_values div_kw;
    struct fastdiv_values div_tg;
    struct fastdiv_values div_cc;
    struct fastdiv_values div_cs;
};

struct htp_conv2d_kernel_params {
    uint8_t  kernel_type;
    uint8_t  n_threads;
    uint8_t  cwhn;
    uint8_t  w_kmajor;

    uint32_t total_rows;
    uint32_t n_row_tiles;
    uint32_t ct_chunk;
    uint32_t rt_block;
    uint32_t stage_size;
    uint32_t thr_size;
    uint32_t vtcm_size;

    union {
        struct htp_conv2d_tiled_params  tiled;
        struct htp_conv2d_window_params window;
    };
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#endif

enum htp_conv2d_dw_kernel_type {
    HTP_CONV2D_DW_KERNEL_UNSUPPORTED = 0,
    HTP_CONV2D_DW_KERNEL_CWHN,             // CWHN channel-contiguous layout
    HTP_CONV2D_DW_KERNEL_WHCN_S1,          // WHCN general stride-1
    HTP_CONV2D_DW_KERNEL_WHCN_S2,          // WHCN general stride-2
    HTP_CONV2D_DW_KERNEL_WHCN_S1_K3X3,     // WHCN aligned w%32==0, 3x3 s=1 p=1 d=1
    HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W32, // WHCN w=32->ow=16, 3x3 s=2 p=1 d=1 (2 rows per vector)
    HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3_W64, // WHCN w=64->ow=32, 3x3 s=2 p=1 d=1 (1 vector rolling kh)
    HTP_CONV2D_DW_KERNEL_WHCN_S2_K3X3,     // WHCN aligned w%32==0 (w>=96), 3x3 s=2 p=1 d=1
    HTP_CONV2D_DW_KERNEL_COUNT,
};

struct htp_conv2d_dw_kernel_params {
    uint8_t  kernel_type;
    uint8_t  n_threads;
    uint8_t  cwhn;
    uint8_t  is_f16;

    uint32_t total_tasks;
    uint32_t vtcm_size;
    uint32_t w;
    uint32_t h;
    uint32_t c;
    uint32_t n;
    uint32_t kw;
    uint32_t kh;
    uint32_t ow;
    uint32_t oh;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
    uint32_t oh_chunk;
    uint32_t n_chunks;
    uint32_t row_src_bytes;
    uint32_t row_dst_bytes;
    uint32_t pad_top_stride;
    struct fastdiv_values div_c;
    struct fastdiv_values div_n_chunks;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_dw_kernel_params) <= 128, "htp_conv2d_dw_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_dw_kernel_params) <= 128, "htp_conv2d_dw_kernel_params is too large for kernel_params blob");
#endif

struct htp_conv2d_dw_vtcm_layout {
    size_t   total_bytes;
    size_t   off_w;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_spad_half_size;
    size_t   dst_spad_half_size;
    size_t   src_bytes_per_thread;
    size_t   dst_bytes_per_thread;
};

static inline void htp_conv2d_dw_vtcm_layout_build(
    struct htp_conv2d_dw_vtcm_layout * L,
    uint32_t max_src_bytes,
    uint32_t max_dst_bytes,
    uint32_t w_size,
    uint32_t n_threads
) {
    L->src_spad_half_size   = hex_round_up(max_src_bytes + 256, 128);
    L->dst_spad_half_size   = hex_round_up(max_dst_bytes + 128, 128);
    L->src_bytes_per_thread = 2 * L->src_spad_half_size;
    L->dst_bytes_per_thread = 2 * L->dst_spad_half_size;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_w, hex_round_up(w_size, 128));
    VTCM_LAYOUT_ALLOC(off, off_src, L->src_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst, L->dst_bytes_per_thread * n_threads);
    L->total_bytes = off;
}

struct htp_conv2d_vtcm_layout {
    size_t total_bytes;
    size_t off_scales;
    size_t off_w;
    size_t off_act[2];
    size_t off_out[2];
    size_t off_thr;

    size_t act_size;
};

static inline void htp_conv2d_vtcm_layout_build(
    struct htp_conv2d_vtcm_layout * L,
    uint32_t rt_block,
    size_t   w_size,
    size_t   act_size,
    size_t   out_size,
    size_t   thr_size,
    uint32_t n_threads
) {
    const size_t sc = hex_round_up((size_t) rt_block * HTP_CONV2D_HMX_SCALES_SIZE, 2048);
    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_scales, sc);
    VTCM_LAYOUT_ALLOC(off, off_w,      w_size);
    VTCM_LAYOUT_ALLOC(off, off_act[0], act_size);
    VTCM_LAYOUT_ALLOC(off, off_act[1], act_size);
    VTCM_LAYOUT_ALLOC(off, off_out[0], out_size);
    VTCM_LAYOUT_ALLOC(off, off_out[1], out_size);
    VTCM_LAYOUT_ALLOC(off, off_thr,    (size_t) n_threads * thr_size);
    L->act_size    = act_size;
    L->total_bytes = off;
}

#endif // HTP_CONV2D_OPS_H
