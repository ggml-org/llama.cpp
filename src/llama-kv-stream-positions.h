#pragma once

#include "llama.h"
#include "llama-kv-cells.h"

struct llama_kv_stream_append_coordinates {
    size_t first_cell = 0;
    size_t capacity = 0;
    llama_pos previous_position = -1;
    uint32_t position_channels = 1;
    bool decode = false;
    bool linear_required = false;
};

// Model positions drive RoPE/masking; only first_cell and row count address the KV store.
bool llama_kv_stream_validate_append(const llama_batch & batch,
    const llama_kv_stream_append_coordinates & coordinates) noexcept;
// Check a checkpoint's dense physical prefix without changing its stored position metadata.
bool llama_kv_stream_validate_cells(const llama_kv_cells & cells, size_t count, bool linear_required) noexcept;
// Translate an interval in a validated monotone prefix into a suffix, without scanning the whole cache.
bool llama_kv_stream_suffix(const llama_kv_cells & cells, size_t count,
    llama_pos begin, llama_pos end, size_t & retained) noexcept;
