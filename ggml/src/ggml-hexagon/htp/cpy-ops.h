#ifndef HTP_CPY_OPS_H
#define HTP_CPY_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include <stdint.h>

enum htp_copy_kernel_type {
    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
    HTP_COPY_KERNEL_1D_CONTIG          = 1,
    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
    HTP_COPY_KERNEL_RESHAPE            = 4,
    HTP_COPY_KERNEL_SCALAR             = 5,
    HTP_COPY_KERNEL_TRANSPOSE          = 6,
};

struct htp_copy_convert_params {
    uint32_t              src0_buf_size;
    uint32_t              dst_buf_size;
    uint32_t              spad0_size_per_thread;
    uint32_t              spad1_size_per_thread;
    uint32_t              src0_row_stride;  // row pitch in the VTCM buffers
    uint32_t              dst_row_stride;
    uint32_t              blk_rows;         // rows moved per DMA descriptor
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02_ne01;
};

// Same-type copy whose (collapsed) shape is a batch of 2D transposes of elem_size-byte
// elements (the type size times the dims that are dense on both sides): dim a is dense
// in dst, dim b is dense in src. Tiles of [tile_a x tile_b] elements are DMAed into VTCM
// along b, transposed with HVX gathers, and DMAed out along a. ne2 / ne3 are the outer dims.
struct htp_copy_transpose_params {
    uint32_t elem_size;
    uint32_t ne_a;
    uint32_t ne_b;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t nb_a_src;  // src stride of dim a (dst stride is elem_size)
    uint32_t nb_b_dst;  // dst stride of dim b (src stride is elem_size)
    uint32_t nb2_src;
    uint32_t nb3_src;
    uint32_t nb2_dst;
    uint32_t nb3_dst;
    uint32_t tile_a;
    uint32_t tile_b;
    uint32_t n_tiles_a;
    uint32_t n_tiles_b;
    uint32_t a_pitch;       // row pitch of an input tile [tile_a][tile_b] in VTCM
    uint32_t b_pitch;       // row pitch of an output tile [tile_b][tile_a] in VTCM
    uint32_t period;        // gather offset vectors before the pattern repeats
    uint32_t period_elems;  // elements of dim a covered by one period
    uint32_t tab_size;      // offset table at the start of VTCM, shared by the threads
    uint32_t in_buf_size;
    uint32_t out_buf_size;
    uint32_t spad_size_per_thread;
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
        struct htp_copy_convert_params   convert;
        struct htp_copy_reshape_params   reshape;
        struct htp_copy_transpose_params transpose;
    } u;
};

// Size of one VTCM staging buffer: big enough that a DMA descriptor moves many short
// rows at once. The copy kernels keep four per thread (two in flight each way), about
// 1 MB of VTCM with 8 threads.
#define HTP_COPY_VTCM_BLK_BYTES (32 * 1024)

struct htp_copy_convert_vtcm_layout {
    uint32_t src0_row_stride;
    uint32_t dst_row_stride;
    uint32_t blk_rows;
    uint32_t src0_buf_size;
    uint32_t dst_buf_size;
    uint32_t spad0_size_per_thread;
    uint32_t spad1_size_per_thread;
    uint32_t total_bytes;
};

// ne01 bounds a block: rows of one block share the dim-1 stride.
static inline void htp_copy_convert_vtcm_layout_build(
    struct htp_copy_convert_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne01,
    uint32_t src_type_size,
    uint32_t dst_type_size,
    uint32_t n_threads) {

    layout->src0_row_stride = hex_round_up(ne00 * src_type_size, 256);
    layout->dst_row_stride  = hex_round_up(ne00 * dst_type_size, 256);

    const uint32_t max_row = layout->src0_row_stride > layout->dst_row_stride ? layout->src0_row_stride : layout->dst_row_stride;
    uint32_t blk_rows = HTP_COPY_VTCM_BLK_BYTES / max_row;
    if (blk_rows > ne01) {
        blk_rows = ne01;
    }
    if (blk_rows == 0) {
        blk_rows = 1;
    }
    layout->blk_rows = blk_rows;

    layout->src0_buf_size = blk_rows * layout->src0_row_stride;
    layout->dst_buf_size  = blk_rows * layout->dst_row_stride;
    layout->spad0_size_per_thread = 2 * layout->src0_buf_size;
    layout->spad1_size_per_thread = 2 * layout->dst_buf_size;
    layout->total_bytes = n_threads * (layout->spad0_size_per_thread + layout->spad1_size_per_thread);
}

struct htp_copy_transpose_vtcm_layout {
    uint32_t tile_a;
    uint32_t tile_b;
    uint32_t a_pitch;
    uint32_t b_pitch;
    uint32_t period;
    uint32_t period_elems;
    uint32_t tab_size;
    uint32_t in_buf_size;
    uint32_t out_buf_size;
    uint32_t spad_size_per_thread;
    uint32_t total_bytes;
};

// Gathers use 32-bit lanes when the element is a whole number of words, 16-bit lanes otherwise.
static inline uint32_t htp_copy_transpose_lane_size(uint32_t elem_size) {
    return elem_size % 4 == 0 ? 4 : 2;
}

// Conservative bank-conflict check: ignoring 128 B gather boundaries may reject safe pitches.
static inline bool htp_copy_transpose_pitch_ok(uint32_t pitch, uint32_t elem_size) {
    const int32_t e = (int32_t) elem_size;
    for (int32_t k = 1; (k - 1) * e < 128; k++) {
        const int32_t r = (int32_t) (((uint32_t) k * pitch + 128) % 256) - 128;  // signed residue modulo 256
        if (r > -e && r < e) {    // Elements k rows apart may share a bank.
            if (k * e - r < 128)  // Conflicting lanes may belong to the same 128 B gather.
            {
                return false;
            }
        }
    }
    return true;
}

static inline void htp_copy_transpose_vtcm_layout_build(struct htp_copy_transpose_vtcm_layout * layout,
                                                        uint32_t                                ne_a,
                                                        uint32_t                                ne_b,
                                                        uint32_t                                elem_size,
                                                        uint32_t                                n_threads) {
    const uint32_t lane = htp_copy_transpose_lane_size(elem_size);
    const uint32_t nl   = 128 / lane;
    const uint32_t w    = elem_size / lane;

    uint32_t g = nl;
    for (uint32_t x = w; x != 0;) {
        const uint32_t r = g % x;
        g                = x;
        x                = r;
    }
    const uint32_t period       = w / g;
    const uint32_t period_elems = nl / g;

    const uint32_t tile_b = ne_b < 256 / elem_size ? ne_b : 256 / elem_size;

    uint32_t a_pitch = tile_b * elem_size;
    while (!htp_copy_transpose_pitch_ok(a_pitch, elem_size)) {
        a_pitch += lane;
    }

    uint32_t tile_a = HTP_COPY_VTCM_BLK_BYTES / a_pitch;
    if (tile_a >= ne_a) {
        tile_a = ne_a;
    } else if (tile_a >= period_elems) {
        tile_a -= tile_a % period_elems;
    }

    layout->tile_a               = tile_a;
    layout->tile_b               = tile_b;
    layout->a_pitch              = a_pitch;
    layout->b_pitch              = hex_round_up(tile_a * elem_size, 128);
    layout->period               = period;
    layout->period_elems         = period_elems;
    layout->tab_size             = hex_round_up(period * 128, 256);
    layout->in_buf_size          = hex_round_up(tile_a * a_pitch, 256);
    layout->out_buf_size         = hex_round_up(tile_b * layout->b_pitch, 256);
    layout->spad_size_per_thread = 2 * (layout->in_buf_size + layout->out_buf_size);
    layout->total_bytes          = layout->tab_size + n_threads * layout->spad_size_per_thread;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CPY_OPS_H
