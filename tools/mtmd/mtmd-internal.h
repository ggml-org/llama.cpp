#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_internal_part {
    std::string text;
    const mtmd_bitmap * bitmap;
    // only used for text parts
    bool parse_special = false;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_internal_part> & parts, int n_merge);

// MiniCPM-V 4.7: an image chunk without pixel data, an overview of ov_nx x ov_ny tokens followed by a grid of
// n_col x n_row slices of nx x ny tokens (no slices if n_col * n_row == 0), for testing the decoder positions
mtmd_input_chunk * mtmd_test_create_canvas_chunk(int ov_nx, int ov_ny, int n_col, int n_row, int nx, int ny);
