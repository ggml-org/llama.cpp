#ifndef HTP_CPY_OPS_H
#define HTP_CPY_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include "gather-rows.h"
#include <stdint.h>

enum htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_GATHER_ROWS        = 6,
    HTP_COPY_KERNEL_TRANSPOSE          = 7,
};

struct htp_copy_convert_params {
    uint32_t              blk_rows;         // rows moved per DMA descriptor
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02_ne01;
};

struct htp_copy_reshape_params {
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1_ne0;
    struct fastdiv_values div_ne2_ne1_ne0;
    struct fastdiv_values div_ne00;
    struct fastdiv_values div_ne01_ne00;
    struct fastdiv_values div_ne02_ne01_ne00;
};

struct htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_size;
    uint8_t  dst_type_size;
    uint8_t  n_threads;

    uint32_t total_elems;
    uint32_t total_rows;
    uint32_t vtcm_size;

    union {
        struct htp_copy_convert_params convert;
        struct htp_copy_reshape_params reshape;
        struct htp_gather_rows_params  gather;
        struct htp_gather_transpose_params transpose;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#define HTP_COPY_VTCM_BLK_BYTES (32 * 1024)

struct htp_copy_convert_vtcm_layout {
    size_t   total_bytes;
    size_t   off_src0;
    size_t   off_dst;
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_rows;
    uint32_t src0_buf_size;
    uint32_t dst_buf_size;
    uint32_t spad0_size_per_thread;
    uint32_t spad1_size_per_thread;
};

static inline void htp_copy_convert_vtcm_layout_build(
    struct htp_copy_convert_vtcm_layout * L,
    uint32_t ne00,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t blk_rows,
    uint32_t n_threads) {

    const uint32_t src0_row_stride = hex_round_up(ne00 * src_type_size, 256);
    const uint32_t dst_row_stride  = hex_round_up(ne00 * dst_type_size, 256);

    L->src0_row_stride       = src0_row_stride;
    L->dst_row_stride        = dst_row_stride;
    L->blk_rows              = blk_rows;
    L->src0_buf_size         = blk_rows * src0_row_stride;
    L->dst_buf_size          = blk_rows * dst_row_stride;
    L->spad0_size_per_thread = 2 * L->src0_buf_size;
    L->spad1_size_per_thread = 2 * L->dst_buf_size;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, (size_t) n_threads * L->spad0_size_per_thread);
    VTCM_LAYOUT_ALLOC(off, off_dst,  (size_t) n_threads * L->spad1_size_per_thread);
    L->total_bytes = off;
}

// ne01 bounds a block: rows of one block share the dim-1 stride.
static inline void htp_copy_convert_solve_layout(
    struct htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t n_threads) {

    const uint32_t src0_row_stride = hex_round_up(ne00 * src_type_size, 256);
    const uint32_t dst_row_stride  = hex_round_up(ne00 * dst_type_size, 256);

    const uint32_t max_row = src0_row_stride > dst_row_stride ? src0_row_stride : dst_row_stride;
    uint32_t blk_rows = HTP_COPY_VTCM_BLK_BYTES / max_row;
    if (blk_rows > ne01) {
        blk_rows = ne01;
    }
    if (blk_rows == 0) {
        blk_rows = 1;
    }

    htp_copy_convert_vtcm_layout_build(layout, ne00, src_type_size, dst_type_size, blk_rows, n_threads);
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
