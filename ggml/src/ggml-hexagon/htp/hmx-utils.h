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
// entries 16..31 cover the next adjacent tile (region 4095) - pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
static const int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// 5-stage vshuffvdd transpose of a 32x64 FP16 block into two contiguous 32x32 HMX tiles.
// Stages 1-3 (-4, -8, -16) mix groups of 8 rows into a 4KB stack scratch; stages 4-5 (-32, -64)
// Stage in-place in dst_tile0/dst_tile1 (each 4-vector slot holds g=0..3), then
// combine across the 4 groups in-place with zero stack scratch.
static inline void hmx_transpose_k_pair_tiles_vshuff(HVX_Vector * restrict dst_tile0,
                                                     HVX_Vector * restrict dst_tile1,
                                                     const uint8_t * restrict src_base,
                                                     size_t row_byte_stride,
                                                     uint32_t valid_rows) {
    if (valid_rows >= HMX_FP16_TILE_N_ROWS) {
        for (uint32_t g = 0; g < 4; ++g) {
            const uint8_t * p = src_base + (size_t) (8 * g) * row_byte_stride;
            HVX_Vector r0 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r1 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r2 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r3 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r4 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r5 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r6 = hvx_vmem(p); p += row_byte_stride;
            HVX_Vector r7 = hvx_vmem(p);

            HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -4);
            HVX_VectorPair p23 = Q6_W_vshuff_VVR(r3, r2, -4);
            HVX_VectorPair p45 = Q6_W_vshuff_VVR(r5, r4, -4);
            HVX_VectorPair p67 = Q6_W_vshuff_VVR(r7, r6, -4);
            r0 = Q6_V_lo_W(p01); r1 = Q6_V_hi_W(p01);
            r2 = Q6_V_lo_W(p23); r3 = Q6_V_hi_W(p23);
            r4 = Q6_V_lo_W(p45); r5 = Q6_V_hi_W(p45);
            r6 = Q6_V_lo_W(p67); r7 = Q6_V_hi_W(p67);

            HVX_VectorPair p02 = Q6_W_vshuff_VVR(r2, r0, -8);
            HVX_VectorPair p13 = Q6_W_vshuff_VVR(r3, r1, -8);
            HVX_VectorPair p46 = Q6_W_vshuff_VVR(r6, r4, -8);
            HVX_VectorPair p57 = Q6_W_vshuff_VVR(r7, r5, -8);
            r0 = Q6_V_lo_W(p02); r2 = Q6_V_hi_W(p02);
            r1 = Q6_V_lo_W(p13); r3 = Q6_V_hi_W(p13);
            r4 = Q6_V_lo_W(p46); r6 = Q6_V_hi_W(p46);
            r5 = Q6_V_lo_W(p57); r7 = Q6_V_hi_W(p57);

            HVX_VectorPair p04 = Q6_W_vshuff_VVR(r4, r0, -16);
            HVX_VectorPair p15 = Q6_W_vshuff_VVR(r5, r1, -16);
            HVX_VectorPair p26 = Q6_W_vshuff_VVR(r6, r2, -16);
            HVX_VectorPair p37 = Q6_W_vshuff_VVR(r7, r3, -16);

            dst_tile0[ 0 + g] = Q6_V_lo_W(p04); dst_tile0[ 4 + g] = Q6_V_hi_W(p04);
            dst_tile1[ 0 + g] = Q6_V_lo_W(p15); dst_tile1[ 4 + g] = Q6_V_hi_W(p15);
            dst_tile0[ 8 + g] = Q6_V_lo_W(p26); dst_tile0[12 + g] = Q6_V_hi_W(p26);
            dst_tile1[ 8 + g] = Q6_V_lo_W(p37); dst_tile1[12 + g] = Q6_V_hi_W(p37);
        }
    } else {
        const HVX_Vector vzero = Q6_V_vzero();
        for (uint32_t g = 0; g < 4; ++g) {
            const uint32_t base_r = 8 * g;
            const uint8_t * p     = src_base + (size_t) base_r * row_byte_stride;
            HVX_Vector r0 = (base_r + 0 < valid_rows) ? hvx_vmem(p + 0 * row_byte_stride) : vzero;
            HVX_Vector r1 = (base_r + 1 < valid_rows) ? hvx_vmem(p + 1 * row_byte_stride) : vzero;
            HVX_Vector r2 = (base_r + 2 < valid_rows) ? hvx_vmem(p + 2 * row_byte_stride) : vzero;
            HVX_Vector r3 = (base_r + 3 < valid_rows) ? hvx_vmem(p + 3 * row_byte_stride) : vzero;
            HVX_Vector r4 = (base_r + 4 < valid_rows) ? hvx_vmem(p + 4 * row_byte_stride) : vzero;
            HVX_Vector r5 = (base_r + 5 < valid_rows) ? hvx_vmem(p + 5 * row_byte_stride) : vzero;
            HVX_Vector r6 = (base_r + 6 < valid_rows) ? hvx_vmem(p + 6 * row_byte_stride) : vzero;
            HVX_Vector r7 = (base_r + 7 < valid_rows) ? hvx_vmem(p + 7 * row_byte_stride) : vzero;

            HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -4);
            HVX_VectorPair p23 = Q6_W_vshuff_VVR(r3, r2, -4);
            HVX_VectorPair p45 = Q6_W_vshuff_VVR(r5, r4, -4);
            HVX_VectorPair p67 = Q6_W_vshuff_VVR(r7, r6, -4);
            r0 = Q6_V_lo_W(p01); r1 = Q6_V_hi_W(p01);
            r2 = Q6_V_lo_W(p23); r3 = Q6_V_hi_W(p23);
            r4 = Q6_V_lo_W(p45); r5 = Q6_V_hi_W(p45);
            r6 = Q6_V_lo_W(p67); r7 = Q6_V_hi_W(p67);

            HVX_VectorPair p02 = Q6_W_vshuff_VVR(r2, r0, -8);
            HVX_VectorPair p13 = Q6_W_vshuff_VVR(r3, r1, -8);
            HVX_VectorPair p46 = Q6_W_vshuff_VVR(r6, r4, -8);
            HVX_VectorPair p57 = Q6_W_vshuff_VVR(r7, r5, -8);
            r0 = Q6_V_lo_W(p02); r2 = Q6_V_hi_W(p02);
            r1 = Q6_V_lo_W(p13); r3 = Q6_V_hi_W(p13);
            r4 = Q6_V_lo_W(p46); r6 = Q6_V_hi_W(p46);
            r5 = Q6_V_lo_W(p57); r7 = Q6_V_hi_W(p57);

            HVX_VectorPair p04 = Q6_W_vshuff_VVR(r4, r0, -16);
            HVX_VectorPair p15 = Q6_W_vshuff_VVR(r5, r1, -16);
            HVX_VectorPair p26 = Q6_W_vshuff_VVR(r6, r2, -16);
            HVX_VectorPair p37 = Q6_W_vshuff_VVR(r7, r3, -16);

            dst_tile0[ 0 + g] = Q6_V_lo_W(p04); dst_tile0[ 4 + g] = Q6_V_hi_W(p04);
            dst_tile1[ 0 + g] = Q6_V_lo_W(p15); dst_tile1[ 4 + g] = Q6_V_hi_W(p15);
            dst_tile0[ 8 + g] = Q6_V_lo_W(p26); dst_tile0[12 + g] = Q6_V_hi_W(p26);
            dst_tile1[ 8 + g] = Q6_V_lo_W(p37); dst_tile1[12 + g] = Q6_V_hi_W(p37);
        }
    }

    for (uint32_t t = 0; t < 2; ++t) {
        HVX_Vector * dst = t ? dst_tile1 : dst_tile0;
        for (uint32_t p0 = 0; p0 < 16; p0 += 4) {
            HVX_Vector a = dst[p0 + 0];
            HVX_Vector b = dst[p0 + 1];
            HVX_Vector c = dst[p0 + 2];
            HVX_Vector d = dst[p0 + 3];

            HVX_VectorPair pab  = Q6_W_vshuff_VVR(b, a, -32);
            HVX_VectorPair pcd  = Q6_W_vshuff_VVR(d, c, -32);
            HVX_VectorPair out0 = Q6_W_vshuff_VVR(Q6_V_lo_W(pcd), Q6_V_lo_W(pab), -64);
            HVX_VectorPair out1 = Q6_W_vshuff_VVR(Q6_V_hi_W(pcd), Q6_V_hi_W(pab), -64);

            dst[p0 + 0] = Q6_V_lo_W(out0);
            dst[p0 + 1] = Q6_V_hi_W(out0);
            dst[p0 + 2] = Q6_V_lo_W(out1);
            dst[p0 + 3] = Q6_V_hi_W(out1);
        }
    }
}

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
                                            const __fp16 * restrict vtcm_src,
                                            uint32_t n_cols,
                                            uint32_t k,
                                            size_t src_stride,
                                            uint32_t start_row,
                                            uint32_t end_row) {
    assert(k % HMX_FP16_TILE_N_COLS == 0);

    const uint32_t       n_k_tiles     = k / HMX_FP16_TILE_N_COLS;
    const bool           pair_scatter  = (n_k_tiles & 1) == 0;
    __builtin_assume(k > 0);
    __builtin_assume(end_row > start_row);

    if (pair_scatter && ((start_row & 31) == 0) && ((end_row & 31) == 0) &&
        hex_is_aligned(vtcm_src, 128) && ((src_stride & 63) == 0)) {
        const uint32_t n_c_iters       = n_k_tiles >> 1;
        const size_t   row_byte_stride = src_stride * sizeof(__fp16);
        const size_t   tile_pair_vecs  = 2 * (HMX_FP16_TILE_SIZE / sizeof(HVX_Vector));

        for (uint32_t r_base = start_row; r_base < end_row; r_base += HMX_FP16_TILE_N_ROWS) {
            const uint32_t ct         = r_base / HMX_FP16_TILE_N_ROWS;
            const uint32_t valid_rows = (r_base < n_cols) ? (n_cols - r_base) : 0;
            HVX_Vector *   tile_base  = (HVX_Vector *) (vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS);
            const uint8_t * row_src   = (const uint8_t *) (vtcm_src + (size_t) r_base * src_stride);

            for (uint32_t i = 0; i < n_c_iters; ++i) {
                hmx_transpose_k_pair_tiles_vshuff(tile_base, tile_base + 16, row_src + (size_t) i * 128,
                                                  row_byte_stride, valid_rows);
                tile_base += tile_pair_vecs;
            }
        }
        return;
    }

    const HVX_Vector     v_scat_base   = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector     v_scat_step   = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64      = Q6_Q_vsetq_R(64);
    const size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_SIZE - 1);
    const size_t         single_region = (size_t) (HMX_FP16_TILE_SIZE - 1);

    if (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        const uint32_t c_step      = 2 * HMX_FP16_TILE_N_COLS;
        const size_t   c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t   dst_step    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        const uint32_t n_c_iters   = k / c_step;

        for (uint32_t r = start_row; r < end_row; r += 2) {
            const uint32_t   ct             = r / HMX_FP16_TILE_N_ROWS;
            const uint32_t   local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       row0_valid     = r < n_cols;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = row0_valid ? (const uint8_t *) (vtcm_src + r * src_stride) : NULL;
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            assert(!p0 || hex_is_aligned(p0, 128));
            assert(!p1 || hex_is_aligned(p1, 128));
            assert(c_byte_step % 128 == 0);

            if (p0 && p1) {
                for (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmem(p1); p1 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = p0 ? hvx_vmem(p0) : vzero;
                    if (p0) p0 += c_byte_step;
                    HVX_Vector v1 = p1 ? hvx_vmem(p1) : vzero;
                    if (p1) p1 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                    tile_base += dst_step;
                }
            }
        }
    } else {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        const uint32_t c_step      = HMX_FP16_TILE_N_COLS;
        const size_t   c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t   dst_step    = (size_t) HMX_FP16_TILE_N_ELMS;
        const uint32_t n_c_iters   = k / c_step;

        for (uint32_t r = start_row; r < end_row; r += 2) {
            const uint32_t   ct             = r / HMX_FP16_TILE_N_ROWS;
            const uint32_t   local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       row0_valid     = r < n_cols;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = row0_valid ? (const uint8_t *) (vtcm_src + r * src_stride) : NULL;
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            if (p0 && p1) {
                for (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmemu(p1); p1 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = p0 ? hvx_vmemu(p0) : vzero;
                    if (p0) p0 += c_byte_step;
                    HVX_Vector v1 = p1 ? hvx_vmemu(p1) : vzero;
                    if (p1) p1 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, v1);
                    tile_base += dst_step;
                }
            }
        }
    }
    hvx_gather_sync(vtcm_dst);
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
static inline void hmx_interleave_cols_to_tiles(__fp16 * restrict tiles_out,
                                            const __fp16 * restrict src,
                                            uint32_t n_rows,
                                            uint32_t head_dim,
                                            size_t src_stride,
                                            uint32_t n_row_tiles,
                                            uint32_t start_row,
                                            uint32_t end_row) {
    __builtin_assume(head_dim > 0);
    const size_t tile_stride_elms = (size_t) n_row_tiles * HMX_FP16_TILE_N_ELMS;

    for (uint32_t r = start_row; r < end_row; r += 2) {
        const bool next_row_valid = (r + 1) < end_row && (r + 1) < n_rows;

        const HVX_Vector * pv_in0 = (const HVX_Vector *) (src + r * src_stride);
        const HVX_Vector * pv_in1 = next_row_valid ? (const HVX_Vector *) (src + (r + 1) * src_stride) : NULL;

        // Row-pair invariants hoisted out of the c loop.
        const uint32_t r0      = r / HMX_FP16_TILE_N_ROWS;
        const uint32_t r1_half = (r % HMX_FP16_TILE_N_ROWS) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_out + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        const size_t tb_step = 2 * tile_stride_elms;

        if (pv_in1) {
            for (uint32_t c = 0; c < head_dim; c += 64) {
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
            for (uint32_t c = 0; c < head_dim; c += 64) {
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

// --- HMX inline asm macros for load-store packetization ---
#define HMX_LOAD_MPY_F16(act, wt, range)              \
    "{\n"                                             \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"      \
    "}\n"

#define HMX_LOAD_MPY_DEEP_F16(act, wt, range)              \
    "{\n"                                                  \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"           \
    "}\n"

#define HMX_STORE_AFTER_F16(out, scale_reg) \
    "mxmem(" out ", " scale_reg "):after.hf = acc\n"

#define HMX_SET_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#define HMX_CLRACC_F16() \
    "mxclracc.hf\n"

// --- Fixed-point INT HMX inline asm macros and helpers (v73+) ---

#define HMX_CLRACC_INT() \
    "mxclracc\n"

#define HMX_SWAPACC() \
    "mxswapacc\n"

#define HMX_LOAD_MPY_DEEP_U8_S8_SM(act, act_rt, wt, wt_rt)  \
    "{\n"                                                   \
    "    activation.ub = mxmem(" act ", " act_rt "):deep\n" \
    "    weight.b = mxmem(" wt ", " wt_rt ")\n"             \
    "}\n"

#define HMX_LOAD_MPY_DEEP_U8_S8_SM_64C(act, act_rt, wt, wt_rt) \
    "{\n"                                                      \
    "    activation.ub = mxmem(" act ", " act_rt "):deep\n"    \
    "    weight.b = mxmem(" wt ", " wt_rt "):deep\n"           \
    "}\n"

#define HMX_LOAD_MPY_DEEP_U8_S8_CM(act, act_rt, wt, wt_rt)     \
    "{\n"                                                      \
    "    activation.ub = mxmem(" act ", " act_rt "):deep:cm\n" \
    "    weight.b = mxmem(" wt ", " wt_rt ")\n"                \
    "}\n"

#define HMX_LOAD_MPY_DEEP_U8_S8_CM_64C(act, act_rt, wt, wt_rt) \
    "{\n"                                                      \
    "    activation.ub = mxmem(" act ", " act_rt "):deep:cm\n" \
    "    weight.b = mxmem(" wt ", " wt_rt "):deep\n"           \
    "}\n"

#define HMX_CVT_UB_ACC(ctrl) \
    "cvt.ub = acc(" ctrl ")\n"

#define HMX_CVT_UH_ACC_2X1(ctrl) \
    "cvt.uh = acc(" ctrl "):2x1\n"

#define HMX_STORE_CVT_SM(out, range) \
    "mxmem(" out ", " range ") = cvt\n"

#define HMX_STORE_CVT_CM(out, range) \
    "mxmem(" out ", " range "):cm = cvt\n"

static inline uint32_t hmx_act_sm_range(uint32_t n_tiles) {
    return 2048u * n_tiles - 1u;
}

static inline uint32_t hmx_out_cm_range(void) {
    return 0x3Fu << 5;
}

static inline uint32_t hmx_wt_b_range(uint32_t n_bytes) {
    return ((n_bytes - 128u) & ~127u) | 0x7Fu;
}

static inline uint32_t hmx_cvt_ctrl(uint32_t bank, uint32_t frac_only, uint32_t sat_mode, uint32_t retain_acc) {
    return (1u << 15) | ((frac_only & 1u) << 14) | ((bank & 3u) << 12) | ((sat_mode & 1u) << 1) | (retain_acc & 1u);
}

static inline void hmx_load_bias_bank(const void * addr, uint32_t bank) {
    uint32_t rs = ((uint32_t) (uintptr_t) addr & ~255u) | (bank & 3u);
    asm volatile("bias = mxmem2(%0)\n" :: "r"(rs) : "memory");
}

#endif // HMX_UTILS_H
