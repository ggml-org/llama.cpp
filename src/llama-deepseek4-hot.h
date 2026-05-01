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
    int                          n_picks   = 0;       // n_expert_used (P), e.g. 6 for DS4
    std::vector<int32_t>         hot_ids;            // size K, sorted by frequency desc
    std::unordered_set<int32_t>  hot_set;            // for O(1) membership
    std::vector<int32_t>         cold_ids;           // size n_expert - K
    std::unordered_set<int32_t>  cold_set;
    std::vector<int32_t>         remap_hot;          // size n_expert: original -> 0..K-1 or -1
    std::vector<int32_t>         remap_cold;         // size n_expert: original -> 0..(n_expert-K)-1 or -1

    // Pinned hot tensor data: extracted K hot expert rows + P zero-weighted
    // dummy expert rows (one per pick index) + 1 trailing prefetch padding row.
    // Total ne[2] = K + P + 1. The dummy experts at positions [K, K+P) let
    // each pick within a token get a unique remapped ID even when most picks
    // are cold, which is required by the CUDA mm_ids_helper kernel: it
    // dedups (token, expert) pairs and produces fewer compacted rows when
    // multiple picks share the same id, leaving the tail of ids_src1
    // uninitialized -> illegal memory access in quantize_mmq_mxfp4_cuda.
    // Per-pick unique dummy experts (id = K + pick_idx) keep the helper
    // emitting exactly P*T rows.
    // For models with combined gate+up (DS-V3 style): hot_gate_up_exps is set, hot_gate_exps and hot_up_exps are null.
    // For models with separate gate/up (DS4-Flash style): hot_gate_exps and hot_up_exps are set, hot_gate_up_exps is null.
    ggml_tensor *                hot_gate_up_exps = nullptr;
    ggml_tensor *                hot_gate_exps    = nullptr;
    ggml_tensor *                hot_up_exps      = nullptr;
    ggml_tensor *                hot_down_exps    = nullptr;

    // Phase 2 graph-time lookup tables.
    //
    // hot_remap_table_f32[0, e] = remap_hot[e] (in [0, K)) if hot, K (base
    //   sentinel) if cold. Combined with a per-pick offset arange [0..P-1]
    //   in the graph: hot_ids = hot_remap + is_cold * arange so each cold
    //   pick gets a unique dummy id in [K, K+P).
    // cold_remap_table_f32[0, e] = e if cold, 0 if hot. Combined with a
    //   per-pick cold sentinel arange [cold_ids[0]..cold_ids[P-1]] so each
    //   hot pick gets a different cold sentinel within the token (avoids
    //   the same dedup bug on the CPU mul_mat_id, defensively).
    // is_hot_mask[0, e] / is_cold_mask[0, e] = 1.0 / 0.0. Used for the
    //   cold-path output mask (hot path no longer needs an output mask
    //   because the dummy experts produce zero output by construction).
    // hot_pick_arange = [0, 1, ..., P-1] f32, length P.
    // cold_pick_sentinel = [cold_ids[0], ..., cold_ids[P-1]] f32, length P.
    ggml_tensor *                hot_remap_table   = nullptr; // f32
    ggml_tensor *                cold_remap_table  = nullptr; // f32
    ggml_tensor *                is_hot_mask       = nullptr; // f32
    ggml_tensor *                is_cold_mask      = nullptr; // f32
    ggml_tensor *                hot_pick_arange   = nullptr; // f32 [P]
    ggml_tensor *                cold_pick_sentinel = nullptr; // f32 [P]

    // Returns true if all tensors required for Phase 2 dual dispatch are non-null.
    bool ready_for_dispatch() const {
        const bool gate_up_ok = hot_gate_up_exps || (hot_gate_exps && hot_up_exps);
        return gate_up_ok && hot_down_exps && hot_remap_table && cold_remap_table
               && is_hot_mask && is_cold_mask
               && hot_pick_arange && cold_pick_sentinel;
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
    int    n_picks() const { return n_picks_; }

    void   set_n_picks(int p) { n_picks_ = p; }

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
    int                                   n_picks_ = 6; // n_expert_used (P); set from model hparams via set_n_picks
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
