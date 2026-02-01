//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_ONEDNN_WOQ_HPP
#define GGML_SYCL_ONEDNN_WOQ_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace ggml_sycl::onednn_woq {

struct packed_weights {
    std::vector<uint8_t> s4;
    std::vector<float>   scales;
    std::vector<int8_t>  zero_points;
    int64_t              group_size = 0;
    int                  scales_mask = 0;
    int                  zero_points_mask = 0;
};

bool pack_q4_0_aos_to_s4(const void * src,
                         int64_t      m,
                         int64_t      k,
                         packed_weights & out,
                         std::string * error);

bool supports_dequant_fp16(ggml_type type);

} // namespace ggml_sycl::onednn_woq

#endif
