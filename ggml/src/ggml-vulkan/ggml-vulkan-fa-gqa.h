#pragma once

#include <cstdint>

// Host-side predicate for Vulkan FA GQA packing (N := gqa_ratio, rows = heads).
// Only valid for single-token decode (neq1 == 1). Multi-token speculative verify
// must use the normal FA path.
inline bool ggml_vk_fa_should_pack_gqa(
        uint32_t neq1,
        uint32_t N,
        uint32_t qk_ratio,
        uint32_t max_gqa,
        uint32_t nek2,
        uint32_t neq2,
        uint32_t nev2,
        uint32_t nem2) {
    return neq1 == 1 && N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
           qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1;
}
