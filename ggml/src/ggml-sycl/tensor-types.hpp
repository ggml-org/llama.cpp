//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TENSOR_TYPES_HPP
#define GGML_SYCL_TENSOR_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if GGML_SYCL_DEBUG
#    include <unordered_set>
#endif

#include "ggml.h"

namespace ggml_sycl {

// Tensor classification for tiered memory placement
enum class tensor_class {
    EMBEDDING,  // token_embd - used every input token
    OUTPUT,     // lm_head/output - used every generated token
    ATTENTION,  // attn_q/k/v/o - every layer, every token
    FFN,        // ffn_up/down/gate (non-MoE) - every layer
    ROUTER,     // ffn_gate_inp - every MoE layer
    EXPERT,     // *_exps - conditional, ~10-20% usage
    NORM,       // *_norm - tiny, fast from host
    OTHER       // unknown tensors
};

// Memory tier for tensor placement
enum class memory_tier {
    VRAM,         // Device VRAM (fastest)
    PINNED_HOST,  // Pinned host memory (GPU-accessible via PCIe)
    MMAP          // File-backed mmap (slowest, fallback)
};

// Information about a single tensor
struct tensor_info {
    std::string  name;
    size_t       size;
    int          layer_id;         // -1 if not layer-specific
    int          expert_id;        // -1 if not a single expert (exps tensors contain all experts)
    tensor_class type;
    int          static_priority;  // 0=highest, computed from type
};

// Inventory of all tensors in a model
struct tensor_inventory {
    std::vector<tensor_info> tensors;
    size_t                   total_size  = 0;
    size_t                   dense_size  = 0;  // embeddings + attention + ffn + router + norms
    size_t                   expert_size = 0;  // all expert weights
    int                      num_layers  = 0;
    int                      num_experts = 0;
};

// Classify a tensor by name pattern
inline tensor_class classify_tensor(const char * name) {
    if (!name) {
        return tensor_class::OTHER;
    }

    // Embeddings (priority 0)
    if (strstr(name, "token_embd")) {
        return tensor_class::EMBEDDING;
    }

    // Output head (priority 0)
    if (strstr(name, "lm_head") || strstr(name, "output.weight")) {
        return tensor_class::OUTPUT;
    }

    // Experts must be checked before FFN (priority 3)
    if (strstr(name, "_exps")) {
        return tensor_class::EXPERT;
    }

    // Router (priority 2)
    if (strstr(name, "ffn_gate_inp")) {
        return tensor_class::ROUTER;
    }

    // Attention (priority 1)
    if (strstr(name, "attn_")) {
        return tensor_class::ATTENTION;
    }

    // Dense FFN (priority 2)
    if (strstr(name, "ffn_")) {
        return tensor_class::FFN;
    }

    // Norms (priority 4)
    if (strstr(name, "_norm")) {
        return tensor_class::NORM;
    }

#if GGML_SYCL_DEBUG
    // Log unknown tensors (once per unique name)
    static std::unordered_set<std::string> warned_tensors;
    if (warned_tensors.insert(name).second) {
        GGML_LOG_WARN("[SYCL] Unknown tensor type: %s\n", name);
    }
#endif

    return tensor_class::OTHER;
}

// Get static priority from tensor class (lower = higher priority)
inline int priority_from_type(tensor_class type) {
    switch (type) {
        case tensor_class::EMBEDDING:
            return 0;
        case tensor_class::OUTPUT:
            return 0;
        case tensor_class::ATTENTION:
            return 1;
        case tensor_class::FFN:
            return 2;
        case tensor_class::ROUTER:
            return 2;
        case tensor_class::EXPERT:
            return 3;
        case tensor_class::NORM:
            return 4;
        case tensor_class::OTHER:
            return 5;
    }
    return 5;
}

// Extract layer ID from tensor name (returns -1 if not found)
inline int extract_layer_id(const char * name) {
    if (!name) {
        return -1;
    }

    // Look for "blk.N." pattern
    const char * blk = strstr(name, "blk.");
    if (!blk) {
        return -1;
    }

    int          layer_id = 0;
    const char * p        = blk + 4;  // Skip "blk."
    while (*p >= '0' && *p <= '9') {
        layer_id = layer_id * 10 + (*p - '0');
        p++;
    }
    return layer_id;
}

// Extract expert ID from tensor name
// Note: Most expert tensors (ffn_*_exps) contain ALL experts in a single tensor
// This returns -1 for such tensors; actual expert ID comes from runtime indexing
inline int extract_expert_id(const char * name) {
    if (!name) {
        return -1;
    }

    // Check for per-expert tensor pattern (rare, model-specific)
    // Most models use *_exps which contains all experts
    const char * exp = strstr(name, ".expert.");
    if (exp) {
        int          expert_id = 0;
        const char * p         = exp + 8;  // Skip ".expert."
        while (*p >= '0' && *p <= '9') {
            expert_id = expert_id * 10 + (*p - '0');
            p++;
        }
        return expert_id;
    }

    return -1;  // Not a single-expert tensor
}

// Build tensor info from name and size
inline tensor_info make_tensor_info(const char * name, size_t size) {
    tensor_info info;
    info.name            = name ? name : "";
    info.size            = size;
    info.layer_id        = extract_layer_id(name);
    info.expert_id       = extract_expert_id(name);
    info.type            = classify_tensor(name);
    info.static_priority = priority_from_type(info.type);
    return info;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_TENSOR_TYPES_HPP
