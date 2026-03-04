// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h.

#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include <hexagon_types.h>
#include <stddef.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
    asm volatile("bias = mxmem2(%0)" :: "r"(scales));
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
    size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.hf = mxmem(%2, %3) }\n"
        :: "r"(row_tiles), "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
}

// Load a single activation+weight tile pair (no :deep streaming).
// Rt defines the accessible region [Rs, Rs+Rt].  Following the reference formula
// (limit = n_tiles * HMX_FP16_TILE_SIZE - 1), for a single tile Rt = 2047.
// The original code used Rt=0x7FFF (32 KB region); when dynamic VTCM allocation
// places a tile near a 4 MB bank boundary, the oversized region crosses it and
// triggers a precise bus error (0x2601).  Rt=2047 confines accesses to exactly
// one 2048-byte tile while covering all 16 HVX vectors (offsets 0..2047).
static HMX_INLINE_ALWAYS void hmx_load_tile_pair_fp16(const __fp16 *act_tile,
                                                       const __fp16 *wt_tile) {
    asm volatile(
        "{ activation.hf = mxmem(%0, %1)\n"
        "weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_tile), "r"(2047),
           "r"(wt_tile),  "r"(2047)
        : "memory");
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
    // Use the combined convert-and-store instruction (matches the reference
    // Q6_mxmem_AR_after_hf intrinsic).  The previous two-instruction sequence
    // "cvt.hf = acc(2); mxmem = cvt" used an undocumented Rs=2 parameter.
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
        :: "r"(out), "r"(0)
        : "memory");
}

// Compute inner product of two vectors of tiles and store result.
static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out,
                                            const __fp16 *row_tiles,
                                            const __fp16 *col_tiles,
                                            size_t n_tiles) {
    hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
    hmx_consume_accumulator_fp16(out);
}

// --- Utility functions (from htp-ops-lib/include/dsp/utils.h) ---

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    return (num + den - 1) / den;
}

static inline size_t hmx_align_up(size_t v, size_t align) {
    return hmx_ceil_div(v, align) * align;
}

static inline size_t hmx_align_down(size_t v, size_t align) {
    return (v / align) * align;
}

static inline size_t hmx_smin(size_t a, size_t b) {
    return a < b ? a : b;
}

static inline size_t hmx_smax(size_t a, size_t b) {
    return a > b ? a : b;
}

// --- VTCM sequential allocator (from htp-ops-lib/include/dsp/vtcm_mgr.h) ---

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#endif // HMX_UTILS_H
