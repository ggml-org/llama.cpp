#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_input_part {
    std::string text;
    const mtmd_bitmap * bitmap;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_input_part> & parts, int n_merge);
