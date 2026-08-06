#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;

// per-slice CPU store for the MoE expert weights. when the tier is active
// (--expert-hot-s set) and the model is loaded without mmap, the loader routes
// each exps tensor's bytes into per-expert allocations here instead of the
// model buffer, so a slice can be physically freed when its expert moves to
// the GPU. mmap loads keep the file-backed weights and use logical removal
// instead (the store stays inactive).
//
// lifecycle:
//   common/arg.cpp  -> mark_requested() when -ehs is parsed
//   loader          -> register_tensor() + fill_tensor() per exps tensor
//   hotstore ctor   -> attach(), then get/free/set slices for moves
namespace llama_expert_pool {

    void mark_requested();
    bool requested();

    // true if `name` matches an exps weight tensor; sets layer_idx on match
    bool match_exps(const char * name, int & layer_idx);

    // one registered exps weight tensor (e.g. blk.3.ffn_gate_exps.weight)
    struct entry {
        const ggml_tensor * tensor;
        int layer_idx;
        size_t slice_bytes; // bytes of a single expert slice
    };

    bool register_tensor(const ggml_tensor * tensor, int layer_idx, int n_experts, size_t slice_bytes);
    // split `data` (nbytes = n_experts * slice_bytes) into per-expert slices
    bool fill_tensor(const ggml_tensor * tensor, const uint8_t * data, size_t nbytes);

    bool active();
    size_t num_entries();
    const entry & get_entry(size_t idx);

    // slice access; returns nullptr if the slice is currently freed (on GPU)
    uint8_t * get_slice(size_t entry_idx, int expert);
    void free_slice(size_t entry_idx, int expert);
    void set_slice(size_t entry_idx, int expert, const uint8_t * data);

} // namespace llama_expert_pool
