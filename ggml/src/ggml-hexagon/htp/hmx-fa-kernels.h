#ifndef HMX_FA_KERNELS_H
#define HMX_FA_KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "hvx-utils.h"
#include "hmx-utils.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
static const int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
    0 * 136,  0 * 136 + 6,
    1 * 136,  1 * 136 + 6,
    2 * 136,  2 * 136 + 6,
    3 * 136,  3 * 136 + 6,
    4 * 136,  4 * 136 + 6,
    5 * 136,  5 * 136 + 6,
    6 * 136,  6 * 136 + 6,
    7 * 136,  7 * 136 + 6,
    8 * 136,  8 * 136 + 6,
    9 * 136,  9 * 136 + 6,
    10 * 136, 10 * 136 + 6,
    11 * 136, 11 * 136 + 6,
    12 * 136, 12 * 136 + 6,
    13 * 136, 13 * 136 + 6,
    14 * 136, 14 * 136 + 6,
    15 * 136, 15 * 136 + 6,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
};
// Inner HMX tile computation kernels

static inline void hmx_fa_qk_dot_tile(
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 *       out_tile,
    size_t         n_dot_tiles
) {
    for (size_t k = 0; k < n_dot_tiles; ++k) {
        Q6_activation_hf_mxmem_RR((unsigned int) row_tiles, 2047);
        Q6_weight_hf_mxmem_RR((unsigned int) col_tiles, 2047);
        row_tiles += HMX_FP16_TILE_N_ELMS;
        col_tiles += HMX_FP16_TILE_N_ELMS;
    }
    Q6_mxmem_AR_after_hf(out_tile, 0);
}

static inline void hmx_fa_o_update_tile(
    const __fp16 * d_diag,
    const __fp16 * o_rc,
    const __fp16 * p_tile_in,
    const __fp16 * v_tile_in,
    __fp16 *       o_tile_out,
    size_t         n_col_tiles
) {
    Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);

    for (size_t k = 0; k < n_col_tiles; ++k) {
        Q6_activation_hf_mxmem_RR((unsigned int) p_tile_in, 2047);
        Q6_weight_hf_mxmem_RR((unsigned int) v_tile_in, 2047);
        p_tile_in += HMX_FP16_TILE_N_ELMS;
        v_tile_in += HMX_FP16_TILE_N_ELMS;
    }

    Q6_mxmem_AR_after_hf(o_tile_out, 0);
}

static inline void hmx_fa_o_norm_tile(
    const __fp16 * d_diag,
    const __fp16 * o_rc,
    __fp16 *       o_out
) {
    Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);
    Q6_mxmem_AR_after_hf(o_out, 0);
}

#endif /* HMX_FA_KERNELS_H */
