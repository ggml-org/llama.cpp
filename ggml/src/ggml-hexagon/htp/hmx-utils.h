// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include "hvx-base.h"

#include <assert.h>
#include <hexagon_types.h>
#include <stddef.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

// Initialise aligned 256-byte area with scale vector + zero padding.
static inline void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    volatile HVX_Vector *pv = (HVX_Vector *) out_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) — pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
static const int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
                                            const __fp16 * restrict vtcm_src,
                                            int n_cols,
                                            int k,
                                            int src_stride,
                                            int start_row,
                                            int end_row) {
    assert(k % HMX_FP16_TILE_N_COLS == 0);

    const int            n_k_tiles     = k / HMX_FP16_TILE_N_COLS;
    const HVX_Vector     v_scat_base   = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector     v_scat_step   = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    const bool           pair_scatter  = (n_k_tiles & 1) == 0;
    const size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_SIZE - 1);
    const size_t         single_region = (size_t) (HMX_FP16_TILE_SIZE - 1);
    __builtin_assume(k > 0);
    __builtin_assume(end_row > start_row);

    if (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        const int    c_step      = 2 * HMX_FP16_TILE_N_COLS;
        const size_t c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t dst_step    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        const int    n_c_iters   = k / c_step;

        for (int r = start_row; r < end_row; r += 2) {
            const int        ct             = r / HMX_FP16_TILE_N_ROWS;
            const int        local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            assert(hex_is_aligned(p0, 128));
            assert(hex_is_aligned(p1, 128));
            assert(c_byte_step % 128 == 0);

            if (p1) {
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmem(p1); p1 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, vzero);
                    tile_base += dst_step;
                }
            }
        }
    } else {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        const int    c_step      = HMX_FP16_TILE_N_COLS;
        const size_t c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t dst_step    = (size_t) HMX_FP16_TILE_N_ELMS;
        const int    n_c_iters   = k / c_step;

        for (int r = start_row; r < end_row; r += 2) {
            const int        ct             = r / HMX_FP16_TILE_N_ROWS;
            const int        local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            if (p1) {
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmemu(p1); p1 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, vzero);
                    tile_base += dst_step;
                }
            }
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
static inline void hmx_interleave_cols_to_tiles(__fp16 * restrict tiles_out,
                                            const __fp16 * restrict src,
                                            int n_rows,
                                            int head_dim,
                                            int src_stride,
                                            int n_row_tiles,
                                            int start_row,
                                            int end_row) {
    __builtin_assume(head_dim > 0);
    const size_t tile_stride_elms = (size_t) n_row_tiles * HMX_FP16_TILE_N_ELMS;

    for (int r = start_row; r < end_row; r += 2) {
        const bool next_row_valid = (r + 1) < end_row && (r + 1) < n_rows;

        const HVX_Vector * pv_in0 = (const HVX_Vector *) (src + r * src_stride);
        const HVX_Vector * pv_in1 = next_row_valid ? (const HVX_Vector *) (src + (r + 1) * src_stride) : NULL;

        // Row-pair invariants hoisted out of the c loop.
        const int r0      = r / HMX_FP16_TILE_N_ROWS;
        const int r1_half = (r % HMX_FP16_TILE_N_ROWS) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_out + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        const size_t tb_step = 2 * tile_stride_elms;

        if (pv_in1) {
            for (int c = 0; c < head_dim; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_Vector     v1             = *pv_in1++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        } else {
            const HVX_Vector vzero = Q6_V_vzero();
            for (int c = 0; c < head_dim; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        }
    }
}

#endif // HMX_UTILS_H
