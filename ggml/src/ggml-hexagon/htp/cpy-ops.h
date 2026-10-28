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

// Same-extent copy to a contiguous dst where dst dim 0 is strided in src and src dim
// dim_k is contiguous: a 2D transpose batched over dims dim_p and dim_q. ne/nb describe
// a canonical view of src (in dst dim order) with extent-1 dims dropped and dims that
// stay adjacent in src merged.
struct htp_copy_transpose_params {
    uint32_t ne[4];
    uint32_t nb[4];          // src strides
    uint8_t  dim_k;
    uint8_t  dim_p;
    uint8_t  dim_q;
    uint8_t  packed;         // blk0 == ne[0] divides the vector lanes: gather straight to dst order
    uint32_t blk0;           // dim-0 elements per tile
    uint32_t blkk;           // dim-k elements per tile
    uint32_t n0b;            // tiles along dim 0
    uint32_t nkb;            // tiles along dim k
    uint32_t in_stride;      // VTCM row pitch of a staged src tile (blkk elements)
    uint32_t out_stride;     // VTCM row pitch of a transposed tile (blk0 elements)
    uint32_t in_buf_size;
    uint32_t out_buf_size;
    uint32_t total_tiles;
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

// Tiles are built from 32x32 (f32) or 64x64 (f16) vgather transposes. Two staged and
// two transposed buffers per thread keep one DMA in and one DMA out in flight.
#define HTP_COPY_TRANSPOSE_BLK_BYTES (64 * 1024)

static inline bool htp_copy_transpose_layout_build(
    struct htp_copy_transpose_params * tp,
    uint32_t ne0,
    uint32_t nek,
    uint32_t type_size,
    uint32_t blk_bytes,
    bool     k_is_dim1) {

    const uint32_t lanes = 128 / type_size;

    uint32_t blk0;
    uint32_t blkk;
    bool     packed = false;
    if (k_is_dim1 && ne0 < lanes && lanes % ne0 == 0) {
        // One vector holds lanes / ne0 whole dst rows, and with dim k right after dim 0
        // the transposed tile is one contiguous dst run.
        packed = true;
        blk0 = ne0;
        blkk = blk_bytes / (ne0 * type_size);
        // vgather halfword offsets are unsigned 16-bit; cap keeps them under 32 KB since larger tiles measured slower
        if (type_size == 2 && blkk > 16384 / ne0 - 64) {
            blkk = 16384 / ne0 - 64;
        }
        if (blkk > nek) {
            blkk = nek;
        }
        if (blkk == 0) {
            return false;
        }
    } else if (nek <= lanes) {
        // Stage the short contiguous dim exactly and make the tile long along dim 0.
        blkk = nek;
        blk0 = blk_bytes / (nek * type_size);
        blk0 -= blk0 % lanes;
        if (blk0 > hex_round_up(ne0, lanes)) {
            blk0 = hex_round_up(ne0, lanes);
        }
        if (blk0 < lanes) {
            return false;
        }
    } else {
        blk0 = hex_round_up(ne0, lanes);
        if (blk0 > 256) {
            blk0 = 256;
        }
        blkk = blk_bytes / (blk0 * type_size);
        blkk -= blkk % lanes;
        // vgather halfword offsets are unsigned 16-bit (f16 blkk up to 512); f16 cap is 256 since larger tiles measured slower
        const uint32_t max_blkk = type_size == 2 ? 256 : 1024;
        if (blkk > max_blkk) {
            blkk = max_blkk;
        }
        if (blkk > hex_round_up(nek, lanes)) {
            blkk = hex_round_up(nek, lanes);
        }
        if (blkk < lanes) {
            return false;
        }
    }

    tp->packed       = packed;
    tp->blk0         = blk0;
    tp->blkk         = blkk;
    tp->n0b          = (ne0 + blk0 - 1) / blk0;
    tp->nkb          = (nek + blkk - 1) / blkk;
    tp->in_stride    = blkk * type_size;
    tp->out_stride   = blk0 * type_size;
    // The packed gather of the last vector reads up to one vector past the staged rows.
    tp->in_buf_size  = hex_round_up(blk0 * tp->in_stride + (tp->packed ? 128 : 0), 128);
    tp->out_buf_size = hex_round_up(blkk * tp->out_stride, 128);
    return true;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
