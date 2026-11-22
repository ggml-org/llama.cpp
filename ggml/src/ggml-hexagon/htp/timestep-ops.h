#ifndef HTP_TIMESTEP_OPS_H
#define HTP_TIMESTEP_OPS_H

#include "hex-common.h"
#include "htp-vtcm.h"
#include <stddef.h>
#include <stdint.h>

#define HTP_TIMESTEP_BLOCK 32 // f32 lanes per HVX vector

// Per-hvx-thread VTCM layout (all buffers are 128B aligned):
//   ts   - 2 x chunk_rows timesteps (ping-pong)
//   freq - blocks_per_hvx frequency vectors
//   out  - 2 x chunk_rows x [cos | sin] rows (ping-pong)
struct htp_timestep_vtcm_layout {
    size_t total_bytes;
    size_t bytes_per_hvx;
    size_t off_ts;
    size_t off_freq;
    size_t off_out;
    size_t ts_buf_size;
    size_t out_buf_size;
    size_t col_stride; // floats per cos (or sin) row in VTCM
};

static inline void htp_timestep_vtcm_layout_build(
    struct htp_timestep_vtcm_layout * L,
    uint32_t                          dim,
    uint32_t                          blocks_per_hvx,
    uint32_t                          chunk_rows,
    uint32_t                          n_hvx
) {
    // one extra slot holds the zero pad for odd dim
    L->col_stride   = hex_round_up((size_t) blocks_per_hvx * HTP_TIMESTEP_BLOCK + (dim & 1), HTP_TIMESTEP_BLOCK);
    L->ts_buf_size  = hex_round_up((size_t) chunk_rows * sizeof(float), 128);
    L->out_buf_size = (size_t) chunk_rows * 2 * L->col_stride * sizeof(float);

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_ts,   2 * L->ts_buf_size);
    VTCM_LAYOUT_ALLOC(off, off_freq, (size_t) blocks_per_hvx * HTP_TIMESTEP_BLOCK * sizeof(float));
    VTCM_LAYOUT_ALLOC(off, off_out,  2 * L->out_buf_size);

    L->bytes_per_hvx = off;
    L->total_bytes   = off * n_hvx;
}

struct htp_timestep_kernel_params {
    uint32_t n_hvx;
    uint32_t n_rows;
    uint32_t dim;
    uint32_t half;
    uint32_t n_blocks;
    uint32_t split_cols;     // 0: split rows across threads, 1: split freq blocks
    uint32_t rows_per_hvx;
    uint32_t blocks_per_hvx;
    uint32_t chunk_rows;
    uint32_t vtcm_size;
    float    freq_scale;     // -ln(max_period) / half
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_TIMESTEP_OPS_H
