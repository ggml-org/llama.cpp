#pragma once

#include "ggml-ifairy-lut.h"

enum ggml_ifairy_lut_layout {
    GGML_IFAIRY_LUT_LAYOUT_LEGACY   = 0,  // 4x64 int16 per-group tables
    GGML_IFAIRY_LUT_LAYOUT_COMPACT  = 1,  // int8 per-position tables (3 positions x 4 codes x 4 channels)
    GGML_IFAIRY_LUT_LAYOUT_TBL64    = 2,  // 4ch x 64pat x int8 tables per group (decode-first)
    GGML_IFAIRY_LUT_LAYOUT_MERGED64 = 3,  // 64pat x 4ch x int8 tables per group (decode-first, one lookup)
    GGML_IFAIRY_LUT_LAYOUT_SYM16    = 4,  // 16pat x 4ch x int8 tables per group + factor transform (decode-first)
};

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_ifairy_lut_layout ggml_ifairy_lut_layout_from_env(int n);

enum ggml_ifairy_lut_kernel {
    GGML_IFAIRY_LUT_KERNEL_AUTO     = 0,
    GGML_IFAIRY_LUT_KERNEL_SDOT     = 1,
    GGML_IFAIRY_LUT_KERNEL_TBL      = 2,
    GGML_IFAIRY_LUT_KERNEL_MERGED64 = 3,
};

enum ggml_ifairy_lut_kernel ggml_ifairy_lut_kernel_from_env(void);

#ifdef __cplusplus
}
#endif

static const int k_ifairy_lut_patterns = 64;  // legacy table size
static const int k_ifairy_lut_codes    = 4;
static const int k_ifairy_lut_channels = 4;

static const size_t k_ifairy_lut_pos_bytes   = (size_t) k_ifairy_lut_codes * (size_t) k_ifairy_lut_channels;  // 16
// Compact layout per-group payload is 3 * 16B = 48B.
static const size_t k_ifairy_lut_group_bytes = GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES;
#ifdef __cplusplus
static_assert(GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES >= (3 * 4 * 4), "compact LUT group size must fit 3 position tables");
#endif

// 64-pattern int8 tables for experimental decode-first kernels.
static const size_t k_ifairy_lut_tbl64_group_bytes =
    (size_t) k_ifairy_lut_channels * (size_t) k_ifairy_lut_patterns;  // 256
static const size_t k_ifairy_lut_merged64_group_bytes =
    (size_t) k_ifairy_lut_channels * (size_t) k_ifairy_lut_patterns;  // 256
static const size_t k_ifairy_lut_sym16_patterns = 16;
static const size_t k_ifairy_lut_sym16_group_bytes =
    (size_t) k_ifairy_lut_channels * k_ifairy_lut_sym16_patterns * sizeof(int16_t);  // 128
