//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Tensor heat classification for SYCL memory management
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.11)
//
// Classifies tensors into heat categories based on tensor name patterns:
// - HOT: attention tensors (attn, q_proj, k_proj, v_proj, o_proj) - never evict
// - WARM: FFN tensors (ffn, mlp, gate, up_proj, down_proj) - LRU eviction
// - COLD: MoE expert tensors (expert, moe) - stream on-demand
// - FROZEN: embedding tensors (embed, token_embd, output) - one-time load
//
// Priority values (higher = evict later):
//   HOT: 1000, WARM: 100, FROZEN: 50, COLD: 10
//

#ifndef GGML_SYCL_HEAT_CLASSIFIER_HPP
#define GGML_SYCL_HEAT_CLASSIFIER_HPP

#include <cstring>
#include <string>

// Include ggml.h for ggml_tensor definition when used in production
#ifdef GGML_TENSOR_STRUCT_VERSION
// Already included
#else
#    include "ggml.h"
#endif

// =============================================================================
// Heat Classification Types
// =============================================================================

// Heat classification for tensors based on their role in the model
// Used by the eviction policy to prioritize what to keep in memory
enum class TensorHeat {
    HOT,    // Attention tensors - never evict (priority 1000)
    WARM,   // FFN tensors - LRU eviction (priority 100)
    COLD,   // MoE expert tensors - stream on-demand (priority 10)
    FROZEN  // Embedding tensors - one-time load (priority 50)
};

// Policy for tensors based on their heat classification
struct TensorHeatPolicy {
    TensorHeat heat;               // Heat classification
    float      eviction_priority;  // Higher = evict later (keep longer)
    bool       can_stream;         // True for COLD/FROZEN (stream on-demand)
};

// =============================================================================
// Heat Policy Lookup
// =============================================================================

// Get the policy for a given heat classification
// Priority ordering: HOT (1000) > WARM (100) > FROZEN (50) > COLD (10)
inline TensorHeatPolicy get_heat_policy(TensorHeat heat) {
    switch (heat) {
        case TensorHeat::HOT:
            return TensorHeatPolicy{ heat, 1000.0f, false };
        case TensorHeat::WARM:
            return TensorHeatPolicy{ heat, 100.0f, false };
        case TensorHeat::COLD:
            return TensorHeatPolicy{ heat, 10.0f, true };
        case TensorHeat::FROZEN:
            return TensorHeatPolicy{ heat, 50.0f, true };
        default:
            // Unknown heat - default to WARM policy
            return TensorHeatPolicy{ TensorHeat::WARM, 100.0f, false };
    }
}

// =============================================================================
// Tensor Name Classification
// =============================================================================

// Helper to check if a string contains a substring
inline bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

// Classify a tensor based on its name pattern
//
// Classification Rules (order matters - more specific patterns first):
// 1. COLD: "expert", "moe", "exps" (MoE expert tensors - stream on-demand)
// 2. HOT: "attn", "q_proj", "k_proj", "v_proj", "o_proj" (attention - never evict)
//    Note: HOT must be checked before FROZEN because "attn_output" contains "output"
// 3. FROZEN: "embed", "token_embd", "output", "lm_head" (embeddings - one-time load)
// 4. WARM: "ffn", "mlp", "gate", "up_proj", "down_proj" (FFN - LRU eviction)
// 5. Default: WARM (safe fallback)
//
// Examples:
//   "blk.0.attn_q.weight" -> HOT
//   "blk.0.attn_output.weight" -> HOT (attention takes priority over output)
//   "blk.0.ffn_up.weight" -> WARM
//   "blk.0.ffn_gate_exps.3.weight" -> COLD (expert pattern takes priority)
//   "token_embd.weight" -> FROZEN
//   "output.weight" -> FROZEN (standalone output = embedding)
//   "random_name" -> WARM (default)
//
inline TensorHeat classify_tensor(const ggml_tensor * t) {
    // Null tensor or empty name -> default to WARM (safe fallback)
    if (!t || !t->name[0]) {
        return TensorHeat::WARM;
    }

    std::string name(t->name);

    // 1. Check for MoE expert patterns first (most specific)
    // These tensors can be streamed on-demand for MoE models
    if (contains(name, "expert") || contains(name, "moe") || contains(name, "exps")) {
        return TensorHeat::COLD;
    }

    // 2. Check for attention patterns BEFORE FROZEN
    // This is important because "attn_output" contains "output"
    // These are heavily accessed during inference - never evict
    if (contains(name, "attn") || contains(name, "q_proj") || contains(name, "k_proj") || contains(name, "v_proj") ||
        contains(name, "o_proj")) {
        return TensorHeat::HOT;
    }

    // 3. Check for embedding patterns
    // These are loaded once and accessed infrequently
    if (contains(name, "embed") || contains(name, "token_embd") || contains(name, "output") ||
        contains(name, "lm_head")) {
        return TensorHeat::FROZEN;
    }

    // 4. Check for FFN/MLP patterns
    // Moderately accessed, can be evicted if needed
    if (contains(name, "ffn") || contains(name, "mlp") || contains(name, "gate") || contains(name, "up_proj") ||
        contains(name, "down_proj")) {
        return TensorHeat::WARM;
    }

    // 5. Default to WARM for unknown tensors
    // This is a safe fallback - tensors will be subject to LRU eviction
    return TensorHeat::WARM;
}

// Classify by name string directly (convenience for testing)
inline TensorHeat classify_tensor_name(const char * name) {
    if (!name || !name[0]) {
        return TensorHeat::WARM;
    }

    std::string name_str(name);

    // Same logic as classify_tensor - order matters!
    // 1. MoE expert patterns first
    if (contains(name_str, "expert") || contains(name_str, "moe") || contains(name_str, "exps")) {
        return TensorHeat::COLD;
    }
    // 2. Attention patterns BEFORE FROZEN (attn_output contains "output")
    if (contains(name_str, "attn") || contains(name_str, "q_proj") || contains(name_str, "k_proj") ||
        contains(name_str, "v_proj") || contains(name_str, "o_proj")) {
        return TensorHeat::HOT;
    }
    // 3. Embedding patterns
    if (contains(name_str, "embed") || contains(name_str, "token_embd") || contains(name_str, "output") ||
        contains(name_str, "lm_head")) {
        return TensorHeat::FROZEN;
    }
    // 4. FFN/MLP patterns
    if (contains(name_str, "ffn") || contains(name_str, "mlp") || contains(name_str, "gate") ||
        contains(name_str, "up_proj") || contains(name_str, "down_proj")) {
        return TensorHeat::WARM;
    }

    return TensorHeat::WARM;
}

// =============================================================================
// Debug/Logging Utilities
// =============================================================================

// Convert heat classification to string for debugging
inline const char * heat_to_string(TensorHeat heat) {
    switch (heat) {
        case TensorHeat::HOT:
            return "HOT";
        case TensorHeat::WARM:
            return "WARM";
        case TensorHeat::COLD:
            return "COLD";
        case TensorHeat::FROZEN:
            return "FROZEN";
        default:
            return "UNKNOWN";
    }
}

// Get description of what each heat level means
inline const char * heat_description(TensorHeat heat) {
    switch (heat) {
        case TensorHeat::HOT:
            return "Attention tensors - never evict";
        case TensorHeat::WARM:
            return "FFN tensors - LRU eviction";
        case TensorHeat::COLD:
            return "MoE expert tensors - stream on-demand";
        case TensorHeat::FROZEN:
            return "Embedding tensors - one-time load";
        default:
            return "Unknown classification";
    }
}

#endif  // GGML_SYCL_HEAT_CLASSIFIER_HPP
