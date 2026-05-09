// llama.cpp-PoC/src/llama-split-info.h
// spec §3.3 — split.* metadata types
#pragma once
#include <cstdint>
#include <string>

enum split_type_t {
    SPLIT_NONE      = 0,   // full model (default)
    SPLIT_ATTENTION = 1,   // attention slice
    SPLIT_FFN       = 2,   // FFN slice
    SPLIT_EMBED     = 3,   // embed-only (future)
};

struct split_info_t {
    split_type_t type             = SPLIT_NONE;
    std::string  source_sha256    = "";
    uint32_t     layer_first      = 0;
    uint32_t     layer_last       = 0;
    uint32_t     n_embd           = 0;
    uint32_t     wire_version     = 0;
    std::string  ffn_norm_placement = "ffn";  // "ffn" or "attention"
};