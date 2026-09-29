#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"

struct htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_per_thread;
    uint32_t d_inner_tile;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;

    // CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): n_state_rows
    // is conv_kernel_size - 1, i.e. src0->ne[0] when fused_concat is set. fused_concat==0
    // means the normal unfused dispatch (src0 = full conv_input, computed as before).
    uint32_t fused_concat;
    uint32_t n_state_rows;

    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SSM_CONV_H
