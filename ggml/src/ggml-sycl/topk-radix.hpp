#pragma once

#include "common.hpp"

// The legacy implementation uses SLM to implement sorting and top_k selection.
// SLM is limited to 128KB on Xe, which limits how much can be sorted to k<32.
// After a k=8, the radix selection becomes beneficial for most cases, because
// scan-merge has (block + 1) * k pairs of (value, index). Given normal sorting of nlog(n),
// radix-select becomes beneficial quite early. This sets it to 8 - however, the other parameters
// (columns and rows) may also be a driving factor.
// We select the legacy implementation for k below this constant because the overhead of radix select
// exceeds the benefit for very small problems
constexpr int SYCL_TOP_K_SCAN_MERGE_MAX_K = 8;

// Top-k of every row of src, k indices per row into dst_indices, in no particular order.
// Picks between the one-group-per-row and the split-row kernel from the shape and the device.
void ggml_sycl_top_k_radix(
    ggml_backend_sycl_context & ctx,
    const float *   src,
    int32_t *       dst_indices,
    const int64_t   ncols,
    const int64_t   nrows,
    const int       k,
    dpct::queue_ptr main_stream);
