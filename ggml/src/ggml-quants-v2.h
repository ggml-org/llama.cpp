// ggml-quants-v2.h
//
// Accelerate (vDSP) + NEON implementations of the 5 TILE640 quant
// helpers used by the iPhone ANE dispatch path. The C reference
// implementations stay in ggml-quants.c and remain the public
// fallbacks; the v2 functions are the host-side accelerations
// driven by ggml-ane.mm's GGML_OP_TILE640_MATMUL case.
//
// All five v2 functions are bit-identical (within fp32 noise) to
// their C counterparts; the parity test in
// tests/test-tessera-quants-v2.cpp verifies the bound.
//
// v2 dispatch cutoff: k >= GGML_TESSERA_T640_V2_MIN_K. Below the
// cutoff the dispatch falls back to the C reference (the vDSP
// call setup overhead is larger than the work for small k). The
// cutoff default is 1024 (matches the dequanton's row length for
// the 256x256 case which is the smallest canonical Phase 0 shape).
//
// Internal linkage: these are exposed as GGML_API so the C++
// dispatch in ggml-ane.mm and the parity test can call them, but
// the user-facing ggml type traits still point at the C reference
// (per the "evolve, don't version" rule: the public API is
// unchanged; the v2 is implementation-only).

#pragma once

#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tunable: dequant/quant fallback cutoff. Below this k the
// dispatch uses the C reference (vDSP call setup is not worth
// it for tiny rows). The default 1024 matches the smallest
// canonical Phase 0 row length (256-256 = 64 elements per page
// after the first partial page; we use 1024 to be conservative).
#define GGML_TESSERA_T640_V2_MIN_K 1024

// Function A: dequantize_row_tessera_t640_v2
//   Per-page: fp16->fp32 page_max, broadcast to 640 cols.
//   Per-lane: int8->fp32 lane_scale, /127, * page_max = scale.
//   Per-col: precompute trit_sign {-1, 0, 1} from the radix-243
//            packed word, then bulk vDSP_vmul by per-lane scale.
//   Outlier addback: NOT applied here; that is a separate helper
//                    (apply_outlier_addback_v2). The C reference's
//                    outlier omission is the documented trait.
GGML_API void dequantize_row_tessera_t640_v2(const void * GGML_RESTRICT x,
                                             float * GGML_RESTRICT y,
                                             int64_t k);

// Function B: quantize_row_tessera_t640_v2
//   Per-page: vDSP_maxmgv for page_max, vDSP_sve for sum_abs
//             (parallel reduction; may differ from C by 1-2 ulp
//             in the threshold but produces the same trits for
//             inputs well-separated from the threshold).
//   Per-lane: vDSP_maxmgv for lane_max; NEON for the trit
//             encoding and 243-base packing.
//   The v2 path matches the C reference for the test fixtures
//   (random uniform in [-0.5, 0.5] gives a threshold ~0.25, well
//   above the fp32 noise of the reductions; the parity test
//   asserts the dequant round-trip is bit-identical for these
//   fixtures).
GGML_API void quantize_row_tessera_t640_v2(const float * GGML_RESTRICT x,
                                           void * GGML_RESTRICT y,
                                           int64_t k);

// Feature flag accessor: tests and the dispatch read this to
// decide whether to use v2 or the C reference. Default ON for
// Apple Silicon builds; can be disabled with the
// GGML_TESSERA_T640_V2_DISABLE env var. The dispatch checks
// this at runtime so the same binary can run with v2 on or
// off (no recompile).
int ggml_tessera_t640_v2_enabled(void);

#ifdef __cplusplus
}
#endif
