#pragma once

#include "ggml-ifairy-lut.h"

// V2 core path uses a single LUT layout:
// - merged64: 64 patterns × 4 channels × int8 = 256 bytes per group
static const int    k_ifairy_lut_patterns = 64;
static const int    k_ifairy_lut_channels = 4;
static const size_t k_ifairy_lut_merged64_group_bytes =
    (size_t) k_ifairy_lut_channels * (size_t) k_ifairy_lut_patterns;  // 256
