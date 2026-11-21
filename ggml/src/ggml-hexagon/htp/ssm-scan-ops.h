#ifndef HTP_SSM_SCAN_OPS_H
#define HTP_SSM_SCAN_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-fastdiv.h"

#define HTP_SSM_SCAN_ROW_TILE_DEFAULT   32U
#define HTP_SSM_SCAN_TOKEN_TILE_DEFAULT 32U
#define HTP_SSM_SCAN_HMX_TOKEN_TILE     64U
#define HTP_SSM_SCAN_HMX_MAX_DIM        128U

enum htp_ssm_scan_kernel_type {
    HTP_SSM_SCAN_KERNEL_UNSUPPORTED = 0,
    HTP_SSM_SCAN_KERNEL_HVX_DECODE  = 1,
    HTP_SSM_SCAN_KERNEL_HVX_PREFILL = 2,
    HTP_SSM_SCAN_KERNEL_HMX         = 3,
};

struct htp_ssm_scan_kernel_params {
    uint8_t  kernel_type;
    uint8_t  fast_step4;
    uint16_t row_tile;
    uint16_t token_tile;
    uint16_t n_hvx;

    uint32_t n_chunks;
    uint32_t d_state;
    uint32_t n_a;
    uint32_t head_dim;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t heads_per_group;
    uint32_t n_row_blocks;
    uint32_t group_units;
    uint32_t units_per_seq;
    uint32_t total_units;
    uint32_t units_per_hvx;

    uint32_t y_token_stride;
    uint32_t y_seq_stride;
    uint32_t y_bytes;

    uint32_t vtcm_per_hvx;
    uint32_t vtcm_size;

    struct fastdiv_values div_units_per_seq;
    struct fastdiv_values div_group_units;
    struct fastdiv_values div_n_row_blocks;
    struct fastdiv_values div_n_hvx;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_scan_kernel_params) <= 128, "htp_ssm_scan_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_ssm_scan_kernel_params) <= 128, "htp_ssm_scan_kernel_params is too large for kernel_params blob");
#endif

// Decode updates state out of place (off_state -> off_state_out) so writeback(N) and fetch(N+2)
// never share a VTCM buffer. Prefill updates state in place across token chunks using off_state[0].
struct htp_ssm_scan_vtcm_layout {
    size_t off_state[2];
    size_t off_state_out[2];
    size_t off_B[2];
    size_t off_C[2];
    size_t off_x[2];
    size_t off_dt[2];
    size_t off_A[2];
    size_t off_y[2];

    size_t off_hmx_scales;
    size_t off_hmx_c_row;
    size_t off_hmx_b_t_col;
    size_t off_hmx_b_col;
    size_t off_hmx_s_t_col;
    size_t off_hmx_q_m;
    size_t off_hmx_y_tiles;
    size_t off_hmx_x_tilde_col;
    size_t off_hmx_x_hat_t_row;
    size_t off_hmx_L;
    size_t off_hmx_p;

    size_t state_row_stride;
    size_t xy_row_stride;
    size_t bytes_per_hvx;
    size_t total_bytes;
};

static inline void htp_ssm_scan_vtcm_layout_build(
    struct htp_ssm_scan_vtcm_layout * layout,
    uint32_t d_state,
    uint32_t row_tile,
    uint32_t token_tile,
    uint32_t n_hvx,
    bool use_hmx
) {
    const size_t state_row_stride  = hex_align_up((size_t) d_state * sizeof(float), 128);
    const size_t state_slot_bytes  = (size_t) row_tile * state_row_stride;
    const size_t bc_slot_bytes     = (size_t) token_tile * state_row_stride;
    const size_t xy_row_stride     = hex_align_up((size_t) row_tile * sizeof(float), 128);
    const size_t xy_slot_bytes     = (size_t) token_tile * xy_row_stride;
    const size_t scalar_slot_bytes = hex_align_up((size_t) token_tile * sizeof(float), 128);
    const size_t a_slot_bytes      = state_row_stride;

    size_t off = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        layout->off_state[i]     = off; off += state_slot_bytes;
        layout->off_state_out[i] = off; off += state_slot_bytes;
        layout->off_B[i]         = off; off += bc_slot_bytes;
        layout->off_C[i]         = off; off += bc_slot_bytes;
        layout->off_x[i]         = off; off += xy_slot_bytes;
        layout->off_dt[i]        = off; off += scalar_slot_bytes;
        layout->off_A[i]         = off; off += a_slot_bytes;
        layout->off_y[i]         = off; off += xy_slot_bytes;
    }

    if (use_hmx) {
        const size_t n_ds_tiles      = (size_t) (d_state / 32U);
        const size_t n_d_tiles       = (size_t) (row_tile / 32U);
        const size_t tile_64x64_sz   = 2U * 2U * 2048U;
        const size_t tile_64xds_sz   = 2U * n_ds_tiles * 2048U;
        const size_t tile_64xd_sz    = 2U * n_d_tiles * 2048U;
        const size_t tile_dxds_sz    = n_d_tiles * n_ds_tiles * 2048U;
        const size_t l_f32_sz        = 64U * 64U * sizeof(float);
        const size_t p_f32_sz        = hex_align_up(4U * 64U * sizeof(float), 2048U);

        off = hex_align_up(off, 2048U);
        layout->off_hmx_scales      = off; off += 2048U;
        layout->off_hmx_c_row       = off; off += tile_64xds_sz;
        layout->off_hmx_b_t_col     = off; off += tile_64xds_sz;
        layout->off_hmx_b_col       = off; off += tile_64xds_sz;
        layout->off_hmx_s_t_col     = off; off += tile_dxds_sz;
        layout->off_hmx_q_m         = off; off += tile_64x64_sz;
        layout->off_hmx_y_tiles     = off; off += tile_64xd_sz;
        layout->off_hmx_x_tilde_col = off; off += tile_64xd_sz;
        layout->off_hmx_x_hat_t_row = off; off += tile_64xd_sz;
        layout->off_hmx_L           = off; off += l_f32_sz;
        layout->off_hmx_p           = off; off += p_f32_sz;
        off = hex_align_up(off, 2048U);
    } else {
        layout->off_hmx_scales      = 0;
        layout->off_hmx_c_row       = 0;
        layout->off_hmx_b_t_col     = 0;
        layout->off_hmx_b_col       = 0;
        layout->off_hmx_s_t_col     = 0;
        layout->off_hmx_q_m         = 0;
        layout->off_hmx_y_tiles     = 0;
        layout->off_hmx_x_tilde_col = 0;
        layout->off_hmx_x_hat_t_row = 0;
        layout->off_hmx_L           = 0;
        layout->off_hmx_p           = 0;
    }

    layout->state_row_stride = state_row_stride;
    layout->xy_row_stride    = xy_row_stride;
    layout->bytes_per_hvx    = off;
    layout->total_bytes      = off * n_hvx;
}

#endif // HTP_SSM_SCAN_OPS_H
