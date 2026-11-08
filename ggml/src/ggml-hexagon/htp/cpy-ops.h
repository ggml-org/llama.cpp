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

// Batched 2D transpose of elem_size elements (dim 0 dense in dst, dim 1 dense in src)
struct htp_copy_transpose_params {
    uint32_t ne[3];          // dims 0..2
    uint32_t src_nb[3];      // src strides for dims 0, 2, 3
    uint32_t dst_nb[3];      // dst strides for dims 1, 2, 3
    struct fastdiv_values div_nkb;
    struct fastdiv_values div_n0b;
    struct fastdiv_values div_ne2;
    uint32_t blk0;           // dim-0 elements per tile
    uint32_t blkk;           // dim-1 elements per tile
    uint32_t n0b;            // tiles along dim 0
    uint32_t nkb;            // tiles along dim 1
    uint32_t in_stride;      // VTCM row pitch of staged src tile
    uint32_t out_stride;     // VTCM row pitch of transposed tile
    uint32_t in_buf_size;
    uint32_t out_buf_size;
    uint32_t total_tiles;
    uint8_t  elem_size;
    uint8_t  packed;         // gather straight to dst order
    uint8_t  log2_m;         // log2(gather lanes per element)
    uint8_t  log2_r;         // log2(dim-0 elements per vector)
};

struct htp_copy_kernel_params {
    uint8_t  kernel_type;
    uint8_t  src0_type_size;
    uint8_t  dst_type_size;
    uint8_t  n_hvx;

    uint32_t total_elems;
    uint32_t total_rows;
    uint32_t vtcm_size;

    union {
        struct htp_copy_convert_params convert;
        struct htp_copy_reshape_params reshape;
        struct htp_gather_rows_params  gather;
        struct htp_copy_transpose_params transpose;
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
    uint32_t spad0_size_per_hvx;
    uint32_t spad1_size_per_hvx;
};

static inline void htp_copy_convert_vtcm_layout_build(
    struct htp_copy_convert_vtcm_layout * L,
    uint32_t ne00,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t blk_rows,
    uint32_t n_hvx) {

    const uint32_t src0_row_stride = hex_round_up(ne00 * src_type_size, 256);
    const uint32_t dst_row_stride  = hex_round_up(ne00 * dst_type_size, 256);

    L->src0_row_stride       = src0_row_stride;
    L->dst_row_stride        = dst_row_stride;
    L->blk_rows              = blk_rows;
    L->src0_buf_size         = blk_rows * src0_row_stride;
    L->dst_buf_size          = blk_rows * dst_row_stride;
    L->spad0_size_per_hvx    = 2 * L->src0_buf_size;
    L->spad1_size_per_hvx    = 2 * L->dst_buf_size;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, (size_t) n_hvx * L->spad0_size_per_hvx);
    VTCM_LAYOUT_ALLOC(off, off_dst,  (size_t) n_hvx * L->spad1_size_per_hvx);
    L->total_bytes = off;
}

// ne01 bounds a block: rows of one block share the dim-1 stride.
static inline void htp_copy_convert_solve_layout(
    struct htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t n_hvx) {

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

    htp_copy_convert_vtcm_layout_build(layout, ne00, src_type_size, dst_type_size, blk_rows, n_hvx);
}

#define HTP_COPY_TRANSPOSE_BLK_BYTES (64 * 1024)

static inline bool htp_copy_transpose_layout_build(struct htp_copy_transpose_params * tp, uint32_t blk_bytes) {
    const uint32_t ne0  = tp->ne[0];
    const uint32_t nek  = tp->ne[1];
    const uint32_t es   = tp->elem_size;
    const uint32_t epv  = 128 / es;

    uint32_t blk0;
    uint32_t blkk;
    if (tp->packed) {
        // Packed: vector holds epv / ne0 whole dst rows
        blk0 = ne0;
        blkk = blk_bytes / (ne0 * es);
        if (es == 2 && blkk > 16384 / ne0 - 64) {
            blkk = 16384 / ne0 - 64;
        }
        if (blkk > nek) {
            blkk = nek;
        }
        if (blkk == 0) {
            return false;
        }
    } else if (nek <= epv) {
        // Stage short contiguous dim and expand dim 0
        blkk = nek;
        blk0 = blk_bytes / (nek * es);
        blk0 -= blk0 % epv;
        if (blk0 > hex_round_up(ne0, epv)) {
            blk0 = hex_round_up(ne0, epv);
        }
        if (blk0 < epv) {
            return false;
        }
    } else {
        blk0 = hex_round_up(ne0, epv);
        if (blk0 > 256) {
            blk0 = 256;
        }
        blkk = blk_bytes / (blk0 * es);
        blkk -= blkk % epv;
        // Cap f16 blkk to 256 for 16-bit gather offsets
        const uint32_t max_blkk = (es == 2 ? 512 : 4096) / es;
        if (blkk > max_blkk) {
            blkk = max_blkk;
        }
        if (blkk > hex_round_up(nek, epv)) {
            blkk = hex_round_up(nek, epv);
        }
        if (blkk < epv) {
            return false;
        }
    }

    tp->blk0         = blk0;
    tp->blkk         = blkk;
    tp->n0b          = (ne0 + blk0 - 1) / blk0;
    tp->nkb          = (nek + blkk - 1) / blkk;
    // Pad row pitch by 1 element to avoid VTCM bank conflicts
    const uint32_t pad = (es == 2) ? 2 : 4;
    tp->in_stride    = blkk * es + pad;
    tp->out_stride   = blk0 * es;
    // Extra vector for packed gather overhang
    tp->in_buf_size  = hex_round_up(blk0 * tp->in_stride + (tp->packed ? 128 : 0), 128);
    tp->out_buf_size = hex_round_up(blkk * tp->out_stride, 128);
    tp->log2_m       = es == 2 ? 0 : __builtin_ctz(es) - 2;
    tp->log2_r       = __builtin_ctz(tp->packed ? ne0 : epv);
    return true;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
