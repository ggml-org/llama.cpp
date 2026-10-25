#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#include <stdint.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

struct htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_size;
    uint32_t silu;

    struct fastdiv_values div_d_inner_tile;
};

struct htp_ssm_conv_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;
    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

static inline void htp_ssm_conv_vtcm_layout_build(
    struct htp_ssm_conv_vtcm_layout * L,
    uint32_t d_inner_per_thread,
    uint32_t d_conv,
    uint32_t n_t,
    uint32_t d_inner_tile,
    uint32_t n_threads
) {
    const size_t src1_raw_bytes = hex_round_up((size_t) d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
    const size_t src1_T_bytes   = hex_round_up((size_t) d_conv * d_inner_per_thread * sizeof(float), 128);

    L->src1_bytes_per_thread = src1_raw_bytes + src1_T_bytes;

    if (n_t == 1) {
        const size_t src0_tile_raw_bytes = hex_round_up((size_t) d_inner_tile * d_conv * sizeof(float), 128);
        const size_t src0_T_bytes        = hex_round_up((size_t) d_conv * d_inner_tile * sizeof(float), 128);
        L->src0_bytes_per_thread         = 2 * src0_tile_raw_bytes + src0_T_bytes;
        L->dst_bytes_per_thread          = 2 * hex_round_up((size_t) d_inner_tile * sizeof(float), 128);
    } else {
        const uint32_t ncs          = d_conv - 1 + n_t;
        const size_t   src0_block_T = hex_round_up((size_t) ncs * 32 * sizeof(float), 128);
        const size_t   src0_tile_raw = hex_round_up((size_t) d_inner_tile * ncs * sizeof(float), 128);
        L->src0_bytes_per_thread    = 2 * src0_tile_raw + src0_block_T;
        L->dst_bytes_per_thread     = 2 * hex_round_up((size_t) d_inner_tile * n_t * sizeof(float), 128);
    }

    L->src0_bytes = L->src0_bytes_per_thread * n_threads;
    L->src1_bytes = L->src1_bytes_per_thread * n_threads;
    L->dst_bytes  = L->dst_bytes_per_thread * n_threads;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src1, L->src1_bytes);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SSM_CONV_H
