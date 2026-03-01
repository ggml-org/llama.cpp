//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <functional>

namespace ggml_sycl {

// Key identifying a unique MoE expert (layer, expert_id pair).
// Used across expert-prefetch and expert placement subsystems.
struct expert_key {
    int layer;
    int expert_id;

    bool operator==(const expert_key & o) const { return layer == o.layer && expert_id == o.expert_id; }
};

// Hash function for expert_key.
struct expert_key_hash {
    size_t operator()(const expert_key & k) const {
        return std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_id) << 16);
    }
};

}  // namespace ggml_sycl
