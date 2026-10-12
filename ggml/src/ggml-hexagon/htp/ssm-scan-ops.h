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
    uint32_t n_group;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t heads_per_group;
    uint32_t n_row_blocks;
    uint32_t group_units;
    uint32_t units_per_seq;
    uint32_t total_units;
    uint32_t units_per_thread;
    uint32_t heads_per_thread;

    uint32_t state_slot_bytes;
    uint32_t xy_row_stride;
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
    size_t off_dA[2];
    size_t off_A[2];
    size_t off_y[2];

    size_t state_slot_bytes;
    size_t xy_row_stride;
    size_t bytes_per_thread;
    size_t total_bytes;
};

static inline bool htp_ssm_scan_layout_size(size_t count, size_t size, size_t * result) {
    size_t bytes;
    if (hex_mul_overflow(count, size, &bytes) || bytes > SIZE_MAX - 127) {
        return false;
    }
    *result = hex_align_up(bytes, 128);
    return true;
}

static inline bool htp_ssm_scan_layout_alloc(size_t * off, size_t size, size_t * result) {
    *result = *off;
    return !hex_add_overflow(*off, size, off);
}

static inline bool htp_ssm_scan_vtcm_layout_build(
    struct htp_ssm_scan_vtcm_layout * layout,
    uint32_t d_state,
    uint32_t row_tile,
    uint32_t token_tile,
    uint32_t n_threads
) {
    size_t state_elements;
    size_t state_slot_bytes;
    size_t bc_elements;
    size_t bc_slot_bytes;
    size_t xy_row_stride;
    size_t xy_slot_bytes;
    size_t scalar_slot_bytes;
    size_t off = 0;

    if (hex_mul_overflow(row_tile, d_state, &state_elements) ||
        hex_mul_overflow(token_tile, d_state, &bc_elements) ||
        !htp_ssm_scan_layout_size(state_elements, sizeof(float), &state_slot_bytes) ||
        !htp_ssm_scan_layout_size(bc_elements, sizeof(float), &bc_slot_bytes) ||
        !htp_ssm_scan_layout_size(row_tile, sizeof(float), &xy_row_stride) ||
        hex_mul_overflow(token_tile, xy_row_stride, &xy_slot_bytes) ||
        !htp_ssm_scan_layout_size(token_tile, sizeof(float), &scalar_slot_bytes)) {
        return false;
    }

    for (uint32_t i = 0; i < 2; ++i) {
        if (!htp_ssm_scan_layout_alloc(&off, state_slot_bytes, &layout->off_state[i]) ||
            !htp_ssm_scan_layout_alloc(&off, bc_slot_bytes, &layout->off_B[i]) ||
            !htp_ssm_scan_layout_alloc(&off, bc_slot_bytes, &layout->off_C[i]) ||
            !htp_ssm_scan_layout_alloc(&off, xy_slot_bytes, &layout->off_x[i]) ||
            !htp_ssm_scan_layout_alloc(&off, scalar_slot_bytes, &layout->off_dt[i]) ||
            !htp_ssm_scan_layout_alloc(&off, scalar_slot_bytes, &layout->off_dA[i]) ||
            !htp_ssm_scan_layout_alloc(&off, 128, &layout->off_A[i]) ||
            !htp_ssm_scan_layout_alloc(&off, xy_slot_bytes, &layout->off_y[i])) {
            return false;
        }
    }

    layout->state_slot_bytes = state_slot_bytes;
    layout->xy_row_stride    = xy_row_stride;
    layout->bytes_per_thread = off;
    return !hex_mul_overflow(off, n_threads, &layout->total_bytes);
}

#endif // HTP_SSM_SCAN_OPS_H
