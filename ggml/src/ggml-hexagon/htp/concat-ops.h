#ifndef HTP_CONCAT_OPS_H
#define HTP_CONCAT_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include "gather-rows.h"
#include <stdint.h>

enum htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ROWS = 3,
};

struct htp_concat_transposed_params {
    struct fastdiv_values div_ne2;
};

struct htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_threads;
    uint8_t  pad;

    uint32_t vtcm_size;

    union {
        struct htp_concat_transposed_params transposed;
        struct htp_gather_rows_params       gather;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#endif

struct htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_size_per_thread;
    uint32_t src1_spad_size_per_thread;
    size_t   off_spad0;
    size_t   off_spad1;
    uint32_t total_bytes;
};

static inline void htp_concat_transposed_vtcm_layout_build(
    struct htp_concat_transposed_vtcm_layout * L,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_size,
    uint32_t n_threads) {

    // src1_ne0 is processed in chunks of block_cols tokens, so VTCM does not grow with the batch
    uint32_t block_rows = 128;
    uint32_t block_cols = (type_size == 4) ? 256 : 512;
    // +1 element spreads the gather lanes across VTCM, so they do not serialize on one path
    uint32_t spad1_stride = block_rows * type_size + type_size;
    uint32_t block_cols_padded = MIN(hex_round_up(src1_ne0, (type_size == 4) ? 32 : 64), block_cols);
    uint32_t spad0_row_bytes = hex_round_up(src0_ne0 * type_size, 128) + block_cols_padded * type_size;

    // double-buffered: two slots per thread
    L->src0_spad_size_per_thread = 2 * block_rows * spad0_row_bytes;
    L->src1_spad_size_per_thread = 2 * block_cols_padded * spad1_stride;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_spad0, (size_t) n_threads * L->src0_spad_size_per_thread);
    VTCM_LAYOUT_ALLOC(off, off_spad1, (size_t) n_threads * L->src1_spad_size_per_thread);
    L->total_bytes = (uint32_t) off;
}

#endif // HTP_CONCAT_OPS_H
