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
// bench (tests/bench-tessera-quants-v2.cpp) on the host
// machine, which is an Apple M1 base (16 GB unified
// memory, ~68 GB/s bandwidth, 4P+4E cores, 8-core GPU,
// 11 TOPS ANE). The M1 base is the closest host to the
// iPhone 13 Pro Max A15 (the demo's target); an on-device
// re-bench on the A15 is a follow-up. The dispatch is
// table-driven so the per-target constants are easy to
// retune.
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
//     across runs on M1), so the threshold is pinned to the
//     v2's internal NEON path boundary. On M1 base: v2 wins
//     1.66x at n_total=51, 1.78x at n_total=204, 1.88x at
//     n_total=409, ties at 3264-52224, and recovers 1.23x at
//     208896 (where the v2's NEON path is no longer active
//     and both implementations are memory-bandwidth bound;
//     the v2 stays slightly ahead because of the function
//     call savings).
//
//   decode_per_row_meta_v2:
//     v2 wins iff n_total_pages (= n_rows * n_pages) >=
//     4096. The v2's vDSP bulk calls (vDSP_vflt8 + vDSP_vsdiv
//     for lane scales, NEON vcvt_f32_f16 for page scales)
//     have a per-call setup tax that is only amortised above
//     the threshold. On M1 base: v2 loses at small N
//     (0.80x at 528 elems, 0.92x at 8448 elems), ties at
//     33792 elems (0.99x), wins at 135168 and 540672 elems
//     (1.09x). The threshold of 4096 is conservative: at
//     n_pages=16 it maps to n_rows >= 256, at n_pages=64
//     to n_rows >= 64. Below the threshold the C ref's
//     scalar loop wins on M1 base.
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
#include "ggml-quants.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Threshold: v2's NEON path scratch cap. Above this n_total
// the v2 falls back to scalar. The dispatch uses this as
// the crossover for the outlier addback cost model.
//
// This is a HARD v2-internal boundary (4 KB stack scratch cap
// on the v2's NEON bulk path). The cost model threshold
// matches it exactly; the v2 has no other path that's
// faster than the C ref, so the dispatch is binary on this
// single value. The constant is named with the v2's
// internal property rather than the dispatch's policy
// label so the header documents the source.
#define TS_V2_OUTLIER_NEON_PATH_MAX_N_TOTAL 1024

// Threshold: minimum n_total_pages for the v2 meta decode
// cost model. n_total_pages = n_rows * n_pages. Below this
// the v2's vDSP + NEON setup tax dominates and the C ref
// scalar loop is faster. Above this the bulk calls amortise
// and v2 wins by ~9% on M1 base.
//
// On M1 base: n_total_pages=256 -> C ref (0.92x), 1024 -> C
// ref (0.99x tie), 4096 -> v2 (1.09x at n_pages=16 n_rows=256).
// 4096 is the first clean v2 win in the measured data; using
// 1024 here would route the tie case (33792 elems) to v2
// and lose 0.5%, which is a real cost on a per-tile dispatch
// hot path.
#define TS_V2_META_DECODE_MIN_N_TOTAL_PAGES 4096

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
// On M1 base the v2 wins by 1.09x at n_total_pages >= 4096 (the
// bulk vDSP + NEON calls amortise their per-call setup tax).
// Below the threshold the C ref's scalar loop is faster
// (0.80-0.99x). The threshold is conservative: 33792 elems
// (n_total_pages=1024) is a 0.99x tie on M1 base and the
// dispatch routes it to the C ref to avoid the per-call tax
// on the hot path.
static inline bool ts_v2_dispatch_should_use_v2_meta(int64_t n_rows, int64_t n_pages) {
    if (n_rows <= 0 || n_pages <= 0) return false;
    const int64_t n_total_pages = n_rows * n_pages;
    return n_total_pages >= TS_V2_META_DECODE_MIN_N_TOTAL_PAGES;
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
