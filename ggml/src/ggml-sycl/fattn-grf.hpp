//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//
// GGML_SYCL_FA_LARGE_GRF: per-kernel 256-entry register file for the flash-attention tile
// kernels. At 128 GRF the FA tile kernels spill 8-15 KB per thread on DG2 (IGC 2.41 shader
// dumps, 2026-09-30); the vec kernels spill less and lose more from the halved occupancy
// that comes with 256 GRF, so a tile-and-vec mode was measured and dropped. Mode TILE
// applies to tile launches with more than one query row (prefill). The 256-GRF
// instantiations exist only for the tile kernels and only in builds with CMake
// GGML_SYCL_FA_LARGE_GRF=ON (GGML_SYCL_FA_LARGE_GRF_VARIANTS=1).
//
// This header is SYCL-free: the parse and the launch decision are pure functions so a
// CPU-only test can cover them (tests/test-sycl-fa-large-grf.cpp).

#pragma once

#include <cstdint>

enum ggml_sycl_fa_grf_mode : int {
    GGML_SYCL_FA_GRF_OFF  = 0,
    GGML_SYCL_FA_GRF_TILE = 1,
};

enum ggml_sycl_fa_grf_parse_result : int {
    GGML_SYCL_FA_GRF_PARSE_UNSET,        // variable absent or empty: mode OFF
    GGML_SYCL_FA_GRF_PARSE_OK,           // exactly "0" or "1"
    GGML_SYCL_FA_GRF_PARSE_INVALID,      // anything else: mode OFF, caller warns with the raw text
    GGML_SYCL_FA_GRF_PARSE_NO_VARIANTS,  // valid request on a build without the variants: mode OFF
};

// Whole-value parse of the GGML_SYCL_FA_LARGE_GRF text; *mode receives the effective mode.
ggml_sycl_fa_grf_parse_result ggml_sycl_fa_large_grf_parse(const char * raw, bool variants_compiled,
                                                           ggml_sycl_fa_grf_mode * mode);

// Launch-side decision before the device check: mode TILE, a tile launch, more than one
// query row. Single-row decode that routes to TILE stays at 128 GRF, where it was measured.
bool ggml_sycl_fa_large_grf_wanted(ggml_sycl_fa_grf_mode mode, bool tile_route, int64_t q_rows);
