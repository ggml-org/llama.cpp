// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include <assert.h>
#include <hexagon_types.h>
#include <stddef.h>

#include "hvx-base.h"

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
    Q6_bias_mxmem2_A((void *) scales);
}

// Initialise aligned 256-byte area with scale vector + zero padding.
static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    HVX_Vector *pv = (HVX_Vector *)out_scales;
    *pv++ = v_scale;
    *pv   = Q6_V_vzero();
}

// Load multiple contiguous tiles with :deep streaming.
// Rt = total region size - 1; the hardware streams through [Rs, Rs + Rt].
// IMPORTANT: the tile region [Rs, Rs + Rt] must NOT cross a VTCM 4 MB bank
// boundary, otherwise the mxmem instruction will raise a precise bus error.
// Callers must ensure their VTCM layout satisfies this constraint.
static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles,
                                                   const __fp16 *col_tiles,
                                                   size_t n_tiles) {
    // Activation ":deep" streams through a region; weight has no :deep variant.
    // Keep as a single inline-asm packet — the Q6 intrinsic mix of
    // Q6_activation_hf_mxmem_RR_deep + Q6_weight_hf_mxmem_RR trips the HMX
    // backend's "activate weight pair exceeds limit" constraint checker.
    size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.hf = mxmem(%2, %3) }\n"
        :: "r"(row_tiles), "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
}

// Load a single activation+weight tile pair (no :deep streaming).
// Rt defines the accessible region [Rs, Rs+Rt]. For a single tile Rt = 2047
// (covers all 16 HVX vectors at offsets 0..2047). An oversized Rt that would
// cross a VTCM 4 MB bank boundary triggers a precise bus error (0x2601).
static HMX_INLINE_ALWAYS void hmx_load_tile_pair_fp16(const __fp16 *act_tile,
                                                       const __fp16 *wt_tile) {
    Q6_activation_hf_mxmem_RR((unsigned int) act_tile, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int) wt_tile, 2047);
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
    Q6_mxmem_AR_after_hf(out, 0);
}

// Compute inner product of two vectors of tiles and store result.
static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out,
                                            const __fp16 *row_tiles,
                                            const __fp16 *col_tiles,
                                            size_t n_tiles) {
    hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
    hmx_consume_accumulator_fp16(out);
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128 in the tile.
// Column offset (n*4) is added at runtime.  Only entries 0..15 are used (masked by predicate).
static const int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0*128,  1*128,  2*128,  3*128,  4*128,  5*128,  6*128,  7*128,
    8*128,  9*128, 10*128, 11*128, 12*128, 13*128, 14*128, 15*128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
static inline void interleave_rows_to_tiles(__fp16 *restrict vtcm_dst,
                                            const __fp16 *restrict vtcm_src,
                                            int n_cols, int k, int src_stride,
                                            int start_row, int end_row) {
    assert(k % HMX_FP16_TILE_N_COLS == 0);

    const int n_k_tiles = k / HMX_FP16_TILE_N_COLS;
    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    for (int r = start_row; r < end_row; r += 2) {
        int ct = r / HMX_FP16_TILE_N_ROWS;       // N-dimension tile index
        int local_r = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row index
        const bool next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;

        // Offset vectors for N-columns local_r and local_r+1, reused across K-tiles.
        HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
        HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

        for (int c = 0; c < k; c += HMX_FP16_TILE_N_COLS) {
            int kt       = c / HMX_FP16_TILE_N_COLS;
            int tile_idx = ct * n_k_tiles + kt;
            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

            HVX_Vector v0 = hvx_vmemu(vtcm_src + r * src_stride + c);
            HVX_Vector v1 = next_row_valid ? hvx_vmemu(vtcm_src + (r + 1) * src_stride + c) : Q6_V_vzero();

            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off1, v1);
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
static inline void interleave_cols_to_tiles(__fp16 *restrict tiles_out,
                                            const __fp16 *restrict src,
                                            int n_rows, int head_dim, int src_stride,
                                            int n_row_tiles,
                                            int start_row, int end_row) {
    for (int r = start_row; r < end_row; r += 2) {
        const bool next_row_valid = (r + 1) < end_row && (r + 1) < n_rows;

        const HVX_Vector *pv_in0 = (const HVX_Vector *)(src + r * src_stride);
        const HVX_Vector *pv_in1 = next_row_valid ?
                                       (const HVX_Vector *)(src + (r + 1) * src_stride) : NULL;

        for (int c = 0; c < head_dim; c += 64) {
            HVX_Vector     v0 = pv_in0[c / 64];
            HVX_Vector     v1 = pv_in1 ? pv_in1[c / 64] : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;
            int c0 = c / HMX_FP16_TILE_N_COLS;

            // Transposed tile index: tile[dim_tile][row_tile]
            int      tile_idx0  = (c0 + 0) * n_row_tiles + r0;
            int      tile_idx1  = (c0 + 1) * n_row_tiles + r0;
            __fp16 *tile_base0 = tiles_out + tile_idx0 * HMX_FP16_TILE_N_ELMS;
            __fp16 *tile_base1 = tiles_out + tile_idx1 * HMX_FP16_TILE_N_ELMS;

            HVX_Vector *pv_out0 = ((HVX_Vector *)tile_base0) + r1 / 2;
            HVX_Vector *pv_out1 = ((HVX_Vector *)tile_base1) + r1 / 2;
            *pv_out0            = Q6_V_lo_W(vp);
            *pv_out1            = Q6_V_hi_W(vp);
        }
    }
}

// --- VTCM sequential allocator (from htp-ops-lib/include/dsp/vtcm_mgr.h) ---

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#endif // HMX_UTILS_H
