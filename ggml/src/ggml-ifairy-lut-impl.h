#pragma once

#include "ggml-ifairy-lut.h"

enum ggml_ifairy_lut_layout {
    GGML_IFAIRY_LUT_LAYOUT_LEGACY  = 0, // 4x64 int16 per-group tables
    GGML_IFAIRY_LUT_LAYOUT_COMPACT = 1, // int8 per-position tables (3 positions x 4 codes x 4 channels)
};

ggml_ifairy_lut_layout ggml_ifairy_lut_layout_from_env(int n);

enum ggml_ifairy_lut_kernel {
    GGML_IFAIRY_LUT_KERNEL_AUTO     = 0,
    GGML_IFAIRY_LUT_KERNEL_SDOT     = 1,
    GGML_IFAIRY_LUT_KERNEL_TBL      = 2,
    GGML_IFAIRY_LUT_KERNEL_MERGED64 = 3,
};

ggml_ifairy_lut_kernel ggml_ifairy_lut_kernel_from_env(void);

static const int k_ifairy_lut_patterns = 64; // legacy table size
static const int k_ifairy_lut_codes     = 4;
static const int k_ifairy_lut_channels  = 4;

static const size_t k_ifairy_lut_pos_bytes   = (size_t) k_ifairy_lut_codes    * (size_t) k_ifairy_lut_channels;  // 16
// Compact layout per-group payload is 3 * 16B = 48B.
static const size_t k_ifairy_lut_group_bytes = GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES;
static_assert(k_ifairy_lut_group_bytes >= 3 * k_ifairy_lut_pos_bytes, "compact LUT group size must fit 3 position tables");
