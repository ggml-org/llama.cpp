#ifndef HTP_SSM_SCAN_OPS_H
#define HTP_SSM_SCAN_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "hex-common.h"
#include "hex-fastdiv.h"

#define HTP_SSM_SCAN_ROW_TILE_DEFAULT   32U
#define HTP_SSM_SCAN_TOKEN_TILE_DEFAULT 32U

enum htp_ssm_scan_kernel_type {
    HTP_SSM_SCAN_KERNEL_FALLBACK    = 0,
    HTP_SSM_SCAN_KERNEL_MAMBA2_VTCM = 1,
};

enum htp_ssm_scan_state_io {
    HTP_SSM_SCAN_STATE_IO_DMA = 0,
    HTP_SSM_SCAN_STATE_IO_HVX = 1,
};

struct htp_ssm_scan_kernel_params {
    uint8_t  kernel_type;
    uint8_t  decode_state_io;
    uint16_t row_tile;
    uint16_t token_tile;
    uint16_t n_threads;

    uint32_t d_state;
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t heads_per_group;
    uint32_t n_row_blocks;
    uint32_t group_units;
    uint32_t units_per_seq;
    uint32_t total_units;

    uint32_t vtcm_per_thread;
    uint32_t vtcm_size;

    struct fastdiv_values div_units_per_seq;
    struct fastdiv_values div_group_units;
    struct fastdiv_values div_n_row_blocks;
    struct fastdiv_values div_heads_per_group;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_scan_kernel_params) <= 128, "htp_ssm_scan_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_ssm_scan_kernel_params) <= 128, "htp_ssm_scan_kernel_params is too large for kernel_params blob");
#endif

struct htp_ssm_scan_vtcm_layout {
    size_t off_state[2];
    size_t off_B[2];
    size_t off_C[2];
    size_t off_x[2];
    size_t off_dt[2];
    size_t off_A[2];
    size_t off_y[2];

    size_t xy_row_stride;
    size_t bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_ssm_scan_vtcm_layout_build(
    struct htp_ssm_scan_vtcm_layout * layout,
    uint32_t d_state,
    uint32_t row_tile,
    uint32_t token_tile,
    uint32_t n_threads
) {
    const size_t state_slot_bytes  = hex_align_up((size_t) row_tile * d_state * sizeof(float), 128);
    const size_t bc_slot_bytes     = hex_align_up((size_t) token_tile * d_state * sizeof(float), 128);
    const size_t xy_row_stride     = hex_align_up((size_t) row_tile * sizeof(float), 128);
    const size_t xy_slot_bytes     = (size_t) token_tile * xy_row_stride;
    const size_t scalar_slot_bytes = hex_align_up((size_t) token_tile * sizeof(float), 128);

    size_t off = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        layout->off_state[i] = off; off += state_slot_bytes;
        layout->off_B[i]     = off; off += bc_slot_bytes;
        layout->off_C[i]     = off; off += bc_slot_bytes;
        layout->off_x[i]     = off; off += xy_slot_bytes;
        layout->off_dt[i]    = off; off += scalar_slot_bytes;
        layout->off_A[i]     = off; off += 128;
        layout->off_y[i]     = off; off += xy_slot_bytes;
    }

    layout->xy_row_stride    = xy_row_stride;
    layout->bytes_per_thread = off;
    layout->total_bytes      = off * n_threads;
}

#endif // HTP_SSM_SCAN_OPS_H
