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
struct llama_batch_ext;
bool llama_kv_stream_validate_append(const llama_batch_ext & batch,
    const llama_kv_stream_append_coordinates & coordinates) noexcept;
// Check a checkpoint's dense physical prefix without changing its stored position metadata.
bool llama_kv_stream_validate_cells(const llama_kv_cells & cells, size_t count, bool linear_required) noexcept;
// Translate an interval in a validated monotone prefix into a suffix, without scanning the whole cache.
bool llama_kv_stream_suffix(const llama_kv_cells & cells, size_t count,
    llama_pos begin, llama_pos end, size_t & retained) noexcept;
// Match a required prefix against local cells; position-only checkpoints do not carry unused extended metadata.
bool llama_kv_stream_bounded_contains(const llama_kv_cells & source, size_t first, size_t end,
    const llama_kv_cells & local, bool match_extended = true) noexcept;
struct llama_ubatch;
// Derive exact serial causal visibility from cache positions, including repeated M-RoPE positions.
bool llama_kv_stream_mask_bounds(const llama_kv_cells & cells, size_t count, size_t padded,
    const llama_ubatch & batch, bool causal, std::vector<int32_t> & bounds);
