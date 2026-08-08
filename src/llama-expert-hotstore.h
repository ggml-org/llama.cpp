#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ggml-cpp.h"

struct llama_model;
struct llama_expert_heatmap;

// stores per-layer sizing for the Mixture of Experts GPU hot store.
// one "slot" holds a single expert's weights for one layer.
struct llama_expert_hotstore {
    int n_layers;
    int n_experts;
    int hot_s;

    // bytes of a single expert slot per layer, summed over that layer's
    // expert weight tensors (gate/up/down, incl. chexps variants)
    std::vector<size_t> bytes_per_slot;

    // one hot tensor per expert weight tensor, shape {ne0, ne1, hot_s}
    struct entry {
        int          layer_idx;
        ggml_tensor* src; // model tensor holding all n_experts slices
        ggml_tensor* dst; // hot tensor holding hot_s slots
    };
    std::vector<entry> entries;

    // per-layer index into entries (built once in ctor, entries stable after)
    std::vector<std::vector<entry *>> entries_by_layer;

    // slot_to_expert[il][p] = expert id held in slot p of layer il, or -1 if empty.
    // stable across re-syncs: an expert that stays hot keeps its slot.
    std::vector<std::vector<int>> slot_to_expert;

    // per-layer LUT and mask for in-graph routing.
    // hot_lut[e]   = slot index [0..hot_s-1] if e is hot, or hot_s (sentinel) if cold.
    // cold_mask[e] = 1.0f if e is cold, else 0.0f (passed to mul_mat_id_cold).
    struct layer_lut {
        ggml_tensor * hot_lut   = nullptr; // i32[n_experts]
        ggml_tensor * cold_mask = nullptr; // f32[n_experts]
    };
    std::vector<layer_lut> luts; // size n_layers

    // bumped on every resync that swapped >0 slots; build_moe_ffn_tiered
    // compares to its own cached counter to know whether H2D is needed.
    int64_t luts_version = 0;

    // keeps the GPU buffer (and its no_alloc context) alive
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    // true once the first copy of the top-S experts landed (once per session)
    bool is_filled = false;

    // re-sync cadence in tokens; 0 disables periodic re-sync
    int sync_period = 0;
    // tokens_total at the last sync (fill or re-sync) for boundary-cross check
    int64_t last_sync_tokens = 0;

    // hysteresis gate (Trick 6): a resident slot is only swapped when a cold
    // expert scores >= hyst * the incumbent AND the slot has dwelled long enough
    float hyst  = 0.0f; // 0 = gate off (swap freely)
    int   dwell = 0;    // minimum syncs a resident must keep; 0 = off
    // dwell_count[il][p] = syncs since slot p last changed (0 = fresh/empty)
    std::vector<std::vector<int>> dwell_count;

llama_expert_hotstore(const llama_model * model, int n_layers,
                      int n_experts, int hot_s, int sync_period = 0,
                      float hyst = 0.0f, int dwell = 0);

    ~llama_expert_hotstore();

    // allocate the GPU hot store for `hot_s` slots. returns false (and
    // leaves the store disabled) on failure or shortage of VRAM.
    bool allocate(ggml_backend_buffer_type_t gpu_buft);

    // copy the top-S expert slices for every layer into the GPU hot store,
    // using the given heatmap for the ranking. one-shot (guarded by is_filled).
    void copy_top_s(const llama_expert_heatmap & heatmap);

    // static plant: fill slots 0..hot_s-1 with experts 0..hot_s-1 per layer,
    // build LUTs/masks accordingly, no heatmap, no resync. Diagnostic only:
    // isolates the dual-path graph from the heat/dynamic-copy path.
    void plant_static();

    // re-sync the hot store to the current heatmap ranking, swapping only
    // the experts that changed (stable slots; unchanged experts not re-copied).
    void resync_top_s(const llama_expert_heatmap & heatmap);

    // cadence-gated wrapper: re-sync only if tokens_total crossed sync_period;
    // multi_slot freezes the hot store (static slots, no swapping)
    void maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot);

    // returns the GPU slot index holding expert_id in layer il, or -1 if none
    int slot_of(int layer_idx, int expert_id) const;

    // diagnostic: count how many router-selected expert ids hit a hot slot.
    // reads the selected_experts tensors (call after synchronize).
    void log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel);

    // rebuild hot_lut/cold_mask from slot_to_expert for every layer
    // and H2D-copy them into the GPU tensors. bumps luts_version.
    // called from copy_top_s (initial fill) and resync_top_s (swaps).
    void update_luts();

    void log() const;
};
