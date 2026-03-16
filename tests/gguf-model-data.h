#pragma once

#include "ggml.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct gguf_remote_tensor {
    std::string  name;
    ggml_type    type    = GGML_TYPE_F32;
    int64_t      ne[4]   = {1, 1, 1, 1};  // dimensions, unused dims = 1
    uint32_t     n_dims  = 0;
};

struct gguf_remote_model {
    // Selected KV metadata
    std::string architecture;               // general.architecture
    uint32_t    n_embd          = 0;        // <arch>.embedding_length
    uint32_t    n_ff            = 0;        // <arch>.feed_forward_length
    uint32_t    n_vocab         = 0;        // inferred from token_embd.weight ne[1]
    uint32_t    n_layer         = 0;        // <arch>.block_count
    uint32_t    n_head          = 0;        // <arch>.attention.head_count
    uint32_t    n_head_kv       = 0;        // <arch>.attention.head_count_kv
    uint32_t    n_expert        = 0;        // <arch>.expert_count (0 if absent)
    uint32_t    n_embd_head_k   = 0;        // <arch>.attention.key_length
    uint32_t    n_embd_head_v   = 0;        // <arch>.attention.value_length
    uint16_t    n_split         = 0;        // split.count (0 = not split)
    uint32_t    n_split_tensors = 0;        // split.tensors.count (0 if not split)

    std::vector<gguf_remote_tensor> tensors;
};

// Fetch model metadata from HuggingFace with local caching.
// repo: e.g., "ggml-org/Qwen3-32B-GGUF"
// quant: e.g., "Q8_0" -- auto-detects filename (including first shard of split models)
// Returns nullopt if download fails or network is unavailable.
std::optional<gguf_remote_model> gguf_fetch_model_meta(
    const std::string & repo,
    const std::string & quant = "Q8_0",
    const std::string & cache_dir = "");  // empty = default
