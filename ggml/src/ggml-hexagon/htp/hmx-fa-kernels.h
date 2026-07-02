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

static void hmx_fa_qk_dot_tile(
    const __fp16 * row_tiles,
    const __fp16 * col_tiles,
    __fp16 *       out_tile,
    size_t         n_dot_tiles
) {
    if (n_dot_tiles == 2) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_dot_tiles == 4) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_dot_tiles == 8) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            HMX_LOAD_MPY_F16("%9", "%10", "%0")
            HMX_LOAD_MPY_F16("%11", "%12", "%0")
            HMX_LOAD_MPY_F16("%13", "%14", "%0")
            HMX_LOAD_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } else {
        for (size_t k = 0; k < n_dot_tiles; ++k) {
            asm volatile(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(row_tiles), "r"(col_tiles)
            );
            row_tiles += HMX_FP16_TILE_N_ELMS;
            col_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm volatile(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(out_tile), "r"(0)
        : "memory"
    );
}

static inline void hmx_fa_o_update_tile(
    const __fp16 * d_diag,
    const __fp16 * o_rc,
    const __fp16 * p_tile_in,
    const __fp16 * v_tile_in,
    __fp16 *       o_tile_out,
    size_t         n_col_tiles
) {
    asm volatile(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    if (n_col_tiles == 2) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_col_tiles == 4) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_col_tiles == 8) {
        asm volatile(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            HMX_LOAD_MPY_F16("%9", "%10", "%0")
            HMX_LOAD_MPY_F16("%11", "%12", "%0")
            HMX_LOAD_MPY_F16("%13", "%14", "%0")
            HMX_LOAD_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } else {
        for (size_t k = 0; k < n_col_tiles; ++k) {
            asm volatile(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_in), "r"(v_tile_in)
            );
            p_tile_in += HMX_FP16_TILE_N_ELMS;
            v_tile_in += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm volatile(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_out), "r"(0)
        : "memory"
    );
}

static inline void hmx_fa_o_norm_tile(
    const __fp16 * d_diag,
    const __fp16 * o_rc,
    __fp16 *       o_out
) {
    asm volatile(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm volatile(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_out), "r"(0)
        : "memory"
    );
}

#endif /* HMX_FA_KERNELS_H */
