// llama.cpp-PoC/src/llama-ffn-local.h
// spec §4.1, §4.2
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "llama-split-info.h"
#include "llama.h"

struct ffn_layer_ptrs_t {
    const float* ffn_norm;   // [n_embd]
    const float* gate;       // [n_ffn * n_embd]
    const float* up;         // [n_ffn * n_embd]
    const float* down;       // [n_embd * n_ffn]
    uint32_t     n_ffn;
    uint32_t     n_embd;
    uint32_t     type;
    uint32_t     type_norm;
};

struct ffn_mmap_t {
    int    fd       = -1;
    void*  base     = nullptr;
    size_t size     = 0;
    std::vector<ffn_layer_ptrs_t> layers;
    split_info_t info;
};

LLAMA_API ffn_mmap_t* ffn_mmap_load(const char* path);
LLAMA_API void ffn_mmap_prefetch(const ffn_mmap_t* ffn, int il);
LLAMA_API void ffn_mmap_free(ffn_mmap_t* ffn);

LLAMA_API void llm_compute_ffn_cpu(
    const ffn_mmap_t* ffn,
    int               layer,
    float*            hidden,
    int               n_tokens,
    int               n_embd
);

LLAMA_API void llama_swap_ffn(
    struct llama_context* ctx,
    int         layer_first,
    int         layer_last,
    const char* new_ffn_path
);
