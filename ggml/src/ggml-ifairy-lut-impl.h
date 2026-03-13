#pragma once

#include "ggml-ifairy-lut.h"

// V2 core path (lut_c-style):
// - 16-entry LUT × 4 channels × int8 = 64 bytes per group
// - Weight codes are packed per 16 output rows (see `struct ifairy_lut_wtile_16`)
static const int    k_ifairy_lut_entries     = 16;
static const int    k_ifairy_lut_channels    = 4;
static const size_t k_ifairy_lut_group_bytes = (size_t) k_ifairy_lut_channels * (size_t) k_ifairy_lut_entries;  // 64
