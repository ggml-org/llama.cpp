#ifndef HTP_CONV2D_OPS_H
#define HTP_CONV2D_OPS_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"

enum htp_conv2d_kernel_type {
    HTP_CONV2D_KERNEL_UNSUPPORTED = 0,
    HTP_CONV2D_KERNEL_HMX_WINDOW,       // HMX convolution with sliding-window staging
    HTP_CONV2D_KERNEL_HMX_TILED,        // tiled implicit-GEMM convolution on HMX
};

// HMX_WINDOW output staging buffers per thread
#define HTP_CONV2D_WIN_OBUF 2

// HMX scales of one row tile at the start of VTCM: 32 (scale, bias) fp16 pairs and 128 zero bytes
#define HTP_CONV2D_HMX_SCALES_SIZE 256

struct htp_conv2d_kernel_params {
    uint32_t n_threads;
    uint32_t total_rows;
    uint32_t kernel_type;

    uint32_t cwhn;
    uint32_t w_kmajor;
    uint32_t n_row_tiles;
    uint32_t ct_chunk;
    uint32_t rt_block;
    uint32_t stage_size;

    uint32_t off_w;
    uint32_t off_act;
    uint32_t act_size;
    uint32_t off_out;
    uint32_t out_size;
    uint32_t off_thr;
    uint32_t thr_size;
    uint32_t vtcm_size;

    union {
        struct {
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
        } tiled;
        struct {
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
        } window;
    };
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CONV2D_OPS_H
