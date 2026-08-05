// ggml-quants-v2-dispatch.h
//
// Per-call dispatch for the two batched TILE640 v2 quant
// functions in ggml-quants-v2.c (apply_outlier_addback_v2,
// decode_per_row_meta_v2). The dispatch in ggml-ane.mm's
// GGML_OP_TILE640_MATMUL case decides per call whether to
// route to v2 or to the C reference, based on the cost
// model below.
//
// Cost model: per-function threshold derived from the
// bench (tests/bench-tessera-quants-v2.cpp) on M1 Pro.
//
//   apply_outlier_addback_v2:
//     v2 wins iff n_total in (0, 1024]. The v2's NEON path
//     is active for n_total <= 1024 (the 4 KB stack scratch
//     cap); above that the v2 falls back to a scalar convert
//     + scatter that is identical to the C ref. Calling the
//     v2 above the threshold wastes a function call + the
//     n_total > 1024 check, so the dispatch calls the C ref
//     directly. The per-row crossover from a linear fit is
//     noisy at large n (the operations are memory-bandwidth
//     bound; the v2 outlier at n_rows=1024 ranges 486-6416us
//     across runs on M1 Pro), so the threshold is pinned to
//     the v2's internal NEON path boundary.
//
//   decode_per_row_meta_v2:
//     v2 never wins. The v2's vDSP bulk calls (vDSP_vflt8 +
//     vDSP_vsdiv + NEON vcvt_f32_f16) are slower per element
//     than the C ref's scalar loop (0.41-0.65x of C across
//     all shapes on M1 Pro). The v2 has no internal threshold
//     that helps, so the dispatch always uses the C ref.
//
// The helpers here are static inline so the dispatch in
// ggml-ane.mm can call them without a function-call
// overhead. The C ref helpers (ts_decode_per_row_meta_ref,
// ts_apply_outlier_addback_ref) are also static inline; they
// are the batched scalar loops that the v2 functions replace
// in the steady state.

#pragma once

#include "ggml-common.h"
#include "ggml-impl.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Threshold: v2's NEON path scratch cap. Above this n_total
// the v2 falls back to scalar. The dispatch uses this as
// the crossover for the outlier addback cost model.
#define TS_V2_OUTLIER_NEON_PATH_MAX_N_TOTAL 1024

// ---------------------------------------------------------------------------
// Cost model: outlier addback
// ---------------------------------------------------------------------------
//
// Returns true if the dispatch should call apply_outlier_addback_v2
// for this call. Returns false if the dispatch should call the C
// ref (ts_apply_outlier_addback_ref) directly.
//
// n_total is the total number of outliers across all rows in the
// buffer (= outlier_row_offsets[n_rows] - outlier_row_offsets[0]).
// At n_total <= 1024 the v2's NEON bulk fp16->fp32 path is active
// and is faster than the C ref's per-element scalar convert. Above
// the threshold the v2 falls back to scalar and the C ref saves
// a function call.
static inline bool ts_v2_dispatch_should_use_v2_outlier(int64_t n_total) {
    return n_total > 0 && n_total <= TS_V2_OUTLIER_NEON_PATH_MAX_N_TOTAL;
}

// ---------------------------------------------------------------------------
// Cost model: meta decode
// ---------------------------------------------------------------------------
//
// Returns true if the dispatch should call decode_per_row_meta_v2
// for this call. Returns false if the dispatch should call the C
// ref (ts_decode_per_row_meta_ref) directly.
//
// v2 is 0.41-0.65x of C across all measured shapes on M1 Pro, so
// the cost model always returns false. The helper is kept for
// symmetry with the outlier cost model and to make the dispatch
// call site uniform.
static inline bool ts_v2_dispatch_should_use_v2_meta(int64_t n_rows, int64_t n_pages) {
    (void) n_rows;
    (void) n_pages;
    return false;
}

// ---------------------------------------------------------------------------
// C reference: per-tile meta decode
// ---------------------------------------------------------------------------
//
// Batch scalar decode of the per-row meta. The v2's
// decode_per_row_meta_v2 does the same work with vDSP + NEON
// bulk calls; this is the fallback the dispatch uses when the
// v2 cost model returns false.
//
// page_scales: flat uint16_t (fp16) array of size n_rows * n_pages
// lane_scales: flat int8_t array of size n_rows * n_pages * LANES_PER_PAGE
// page_max_out:  flat float array of size n_rows * n_pages
// lane_scale_out: flat float array of size n_rows * n_pages * LANES_PER_PAGE
static inline void ts_decode_per_row_meta_ref(
        const uint16_t * GGML_RESTRICT page_scales,
        const int8_t   * GGML_RESTRICT lane_scales,
        int64_t n_rows,
        int64_t n_pages,
        float  * GGML_RESTRICT page_max_out,
        float  * GGML_RESTRICT lane_scale_out) {
    const int64_t n_total_pages = n_rows * n_pages;
    const int64_t n_lanes_per_row = n_pages * TILE640_LANES_PER_PAGE;
    const int64_t n_total_lanes = n_rows * n_lanes_per_row;
    for (int64_t i = 0; i < n_total_pages; i++) {
        page_max_out[i] = GGML_FP16_TO_FP32(page_scales[i]);
    }
    for (int64_t i = 0; i < n_total_lanes; i++) {
        lane_scale_out[i] = ((float) lane_scales[i]) * (1.0f / 127.0f);
    }
}

// ---------------------------------------------------------------------------
// C reference: per-tile outlier addback
// ---------------------------------------------------------------------------
//
// Batch scalar addback of the outlier_vals (fp16) into the
// rows buffer at the outlier_cols positions. The v2's
// apply_outlier_addback_v2 does the same work with a NEON
// bulk fp16->fp32 convert + scalar scatter when n_total <=
// 1024; this is the fallback the dispatch uses when the v2
// cost model returns false.
//
// rows: contiguous buffer of n_rows * row_len floats
// row_len: number of floats per row
// n_rows: number of rows in the buffer
// outlier_row_offsets: CSR offsets, size n_rows + 1
// outlier_cols: column indices, size outlier_row_offsets[n_rows]
// outlier_vals: fp16 values, size outlier_row_offsets[n_rows]
static inline void ts_apply_outlier_addback_ref(
        float         * GGML_RESTRICT rows,
        int64_t row_len,
        int64_t n_rows,
        const int32_t * GGML_RESTRICT outlier_row_offsets,
        const int32_t * GGML_RESTRICT outlier_cols,
        const uint16_t* GGML_RESTRICT outlier_vals) {
    const uint16_t * vals = outlier_vals;
    if (n_rows <= 0) return;
    const int64_t base = (int64_t) outlier_row_offsets[0];
    const int64_t n_total = (int64_t) outlier_row_offsets[n_rows] - base;
    if (n_total <= 0) return;
    for (int64_t r = 0; r < n_rows; r++) {
        const int32_t lo = outlier_row_offsets[r];
        const int32_t hi = outlier_row_offsets[r + 1];
        float * GGML_RESTRICT row = rows + r * row_len;
        for (int32_t k = lo; k < hi; k++) {
            const int32_t col = outlier_cols[k];
            if (col >= 0 && col < row_len) {
                row[col] = GGML_FP16_TO_FP32(vals[k]);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
