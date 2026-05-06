#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

// Parallel version of ggml_quantize_chunk.
//
// Splits [start, start + nrows * n_per_row) across n_threads worker threads,
// each calling ggml_quantize_chunk on its row range.  Threads write to
// non-overlapping regions of dst, so no locking is required.
//
// Falls back to the single-threaded ggml_quantize_chunk when n_threads <= 1
// or nrows <= 1.  The primary motivation is iq4_nl, whose per-block NL search
// makes single-threaded throughput ~95x slower than other 4-bit types; this
// function recovers near-linear scaling with thread count.
//
// imatrix may be NULL for types that do not require it.
// Returns total bytes written (same contract as ggml_quantize_chunk).
GGML_API size_t ggml_quantize_chunk_mt(
        enum ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix,
                   int   n_threads);

#ifdef __cplusplus
}
#endif
