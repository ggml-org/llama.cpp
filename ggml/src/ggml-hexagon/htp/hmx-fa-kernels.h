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
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            :
            : "r"(row_tiles), "r"(col_tiles), "r"(2047),
              "r"(row_tiles + HMX_FP16_TILE_N_ELMS), "r"(col_tiles + HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_dot_tiles == 4) {
        asm volatile(
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%5, %2)\n"
            "    weight.hf = mxmem(%6, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%7, %2)\n"
            "    weight.hf = mxmem(%8, %2)\n"
            "}\n"
            :
            : "r"(row_tiles), "r"(col_tiles), "r"(2047),
              "r"(row_tiles + HMX_FP16_TILE_N_ELMS), "r"(col_tiles + HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_dot_tiles == 8) {
        asm volatile(
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%5, %2)\n"
            "    weight.hf = mxmem(%6, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%7, %2)\n"
            "    weight.hf = mxmem(%8, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%9, %2)\n"
            "    weight.hf = mxmem(%10, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%11, %2)\n"
            "    weight.hf = mxmem(%12, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%13, %2)\n"
            "    weight.hf = mxmem(%14, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%15, %2)\n"
            "    weight.hf = mxmem(%16, %2)\n"
            "}\n"
            :
            : "r"(row_tiles), "r"(col_tiles), "r"(2047),
              "r"(row_tiles + HMX_FP16_TILE_N_ELMS), "r"(col_tiles + HMX_FP16_TILE_N_ELMS),
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
                "{\n"
                "    activation.hf = mxmem(%0, %2)\n"
                "    weight.hf = mxmem(%1, %2)\n"
                "}\n"
                :
                : "r"(row_tiles), "r"(col_tiles), "r"(2047)
            );
            row_tiles += HMX_FP16_TILE_N_ELMS;
            col_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
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
        "{\n"
        "    activation.hf = mxmem(%0, %2)\n"
        "    weight.hf = mxmem(%1, %2)\n"
        "}\n"
        :
        : "r"(d_diag), "r"(o_rc), "r"(2047)
    );
    if (n_col_tiles == 2) {
        asm volatile(
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            :
            : "r"(p_tile_in), "r"(v_tile_in), "r"(2047),
              "r"(p_tile_in + HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_col_tiles == 4) {
        asm volatile(
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%5, %2)\n"
            "    weight.hf = mxmem(%6, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%7, %2)\n"
            "    weight.hf = mxmem(%8, %2)\n"
            "}\n"
            :
            : "r"(p_tile_in), "r"(v_tile_in), "r"(2047),
              "r"(p_tile_in + HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } else if (n_col_tiles == 8) {
        asm volatile(
            "{\n"
            "    activation.hf = mxmem(%0, %2)\n"
            "    weight.hf = mxmem(%1, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%3, %2)\n"
            "    weight.hf = mxmem(%4, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%5, %2)\n"
            "    weight.hf = mxmem(%6, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%7, %2)\n"
            "    weight.hf = mxmem(%8, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%9, %2)\n"
            "    weight.hf = mxmem(%10, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%11, %2)\n"
            "    weight.hf = mxmem(%12, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%13, %2)\n"
            "    weight.hf = mxmem(%14, %2)\n"
            "}\n"
            "{\n"
            "    activation.hf = mxmem(%15, %2)\n"
            "    weight.hf = mxmem(%16, %2)\n"
            "}\n"
            :
            : "r"(p_tile_in), "r"(v_tile_in), "r"(2047),
              "r"(p_tile_in + HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + HMX_FP16_TILE_N_ELMS),
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
                "{\n"
                "    activation.hf = mxmem(%0, %2)\n"
                "    weight.hf = mxmem(%1, %2)\n"
                "}\n"
                :
                : "r"(p_tile_in), "r"(v_tile_in), "r"(2047)
            );
            p_tile_in += HMX_FP16_TILE_N_ELMS;
            v_tile_in += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
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
        "{\n"
        "    activation.hf = mxmem(%0, %2)\n"
        "    weight.hf = mxmem(%1, %2)\n"
        "}\n"
        :
        : "r"(d_diag), "r"(o_rc), "r"(2047)
    );
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
        :
        : "r"(o_out), "r"(0)
        : "memory"
    );
}

#endif /* HMX_FA_KERNELS_H */
