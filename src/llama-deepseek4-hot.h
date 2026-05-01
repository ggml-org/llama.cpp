// DeepSeek4 hot-expert pinning manager.
//
// Reads a per-layer hot-expert-ID profile (produced by ds4-expert-profile +
// ds4-hot-experts.py), extracts the K hot experts of each layer's
// `ffn_gate_up_exps` and `ffn_down_exps` tensors into a separate GPU buffer
// after model load, and exposes those subset tensors to the deepseek4 graph
// builder so build_moe_v4 / build_expert_mix can issue dual mul_mat_id
// dispatches (hot subset on GPU, cold subset on CPU).
//
// Activation: set DS4_HOT_PROFILE_JSON=path.json before starting llama-server
// or llama-cli. The JSON shape matches what ds4-hot-experts.py extract emits:
//   { "n_layer": 43, "n_expert": 256, "k": 32,
//     "category": "code",
//     "hot": { "0": [12, 47, ...], ... } }
//
// This is Phase 1 (load-time extraction). Phase 2 (graph dispatch) lives in
// src/models/deepseek4.cpp.
#pragma once

#include "ggml.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

struct llama_model;
struct llama_context;

namespace ds4_hot {

struct layer_hot_state {
    int                          il        = -1;
    int                          k         = 0;
    std::vector<int32_t>         hot_ids;            // size K, sorted by frequency desc
    std::unordered_set<int32_t>  hot_set;            // for O(1) membership
    std::vector<int32_t>         cold_ids;           // size n_expert - K
    std::unordered_set<int32_t>  cold_set;
    std::vector<int32_t>         remap_hot;          // size n_expert: original -> 0..K-1 or -1
    std::vector<int32_t>         remap_cold;         // size n_expert: original -> 0..(n_expert-K)-1 or -1

    // Pinned hot tensor data: contiguous K rows from the original tensor.
    // These live on a GPU device buffer once allocated.
    // For models with combined gate+up (DS-V3 style): hot_gate_up_exps is set, hot_gate_exps and hot_up_exps are null.
    // For models with separate gate/up (DS4-Flash style): hot_gate_exps and hot_up_exps are set, hot_gate_up_exps is null.
    ggml_tensor *                hot_gate_up_exps = nullptr;
    ggml_tensor *                hot_gate_exps    = nullptr;
    ggml_tensor *                hot_up_exps      = nullptr;
    ggml_tensor *                hot_down_exps    = nullptr;

    // Phase 2 graph-time lookup tables (live on the same GPU buffer as the hot tensors).
    // Each is shape [1, n_expert] (use a 1D flatten of selected_experts when calling ggml_get_rows).
    // hot_remap_table[0, e] = remap_hot[e] if e in hot_set else 0 (sentinel; masked out).
    // cold_remap_table[0, e] = e if e in cold_set else cold_sentinel (a guaranteed cold expert id).
    // is_hot_mask[0, e]  = 1.0 if e in hot_set else 0.0
    // is_cold_mask[0, e] = 1.0 if e not in hot_set else 0.0
    ggml_tensor *                hot_remap_table  = nullptr; // i32
    ggml_tensor *                cold_remap_table = nullptr; // i32
    ggml_tensor *                is_hot_mask      = nullptr; // f32
    ggml_tensor *                is_cold_mask     = nullptr; // f32

    // Returns true if all tensors required for Phase 2 dual dispatch are non-null.
    bool ready_for_dispatch() const {
        const bool gate_up_ok = hot_gate_up_exps || (hot_gate_exps && hot_up_exps);
        return gate_up_ok && hot_down_exps && hot_remap_table && cold_remap_table
               && is_hot_mask && is_cold_mask;
    }
};

class hot_manager {
public:
    hot_manager() = default;
    ~hot_manager();

    // Returns true if a profile path was provided and successfully loaded.
    // Idempotent. Pulls path from DS4_HOT_PROFILE_JSON env var if path is empty.
    bool load_profile(std::string path = {});

    // Allocate per-layer hot subset tensors on the same device as the model's
    // GPU split would prefer. Reads the original ffn_*_exps host data from
    // each layer (which must already be loaded into CPU memory) and copies the
    // K hot rows into a new GPU tensor.
    //
    // Must be called AFTER the model has been loaded and BEFORE inference.
    bool allocate(const llama_model & model);

    bool   is_active() const { return active; }
    int    k_per_layer() const { return k; }
    size_t profile_n_layer() const { return n_layer; }
    int    profile_n_expert() const { return n_expert; }

    // Per-layer accessors. il is the layer index. Returns nullptr if no hot
    // state was allocated for that layer (e.g., layer is fully on GPU already
    // and we skipped it).
    const layer_hot_state * get(int il) const;

    // Total bytes pinned to GPU buffers across all layers (for reporting).
    size_t total_gpu_bytes() const;

private:
    bool                                  active   = false;
    std::string                           category = {};
    int                                   k        = 0;
    size_t                                n_layer  = 0;
    int                                   n_expert = 0;
    std::vector<std::unique_ptr<layer_hot_state>> layers;

    struct ggml_buffers;
    std::unique_ptr<ggml_buffers> bufs;
};

// Singleton accessor; convenient for plumbing through llama-context without
// changing the C API. The instance is created on first call and persists for
// the program lifetime.
hot_manager & instance();

} // namespace ds4_hot
