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

    // one hot tensor per expert weight tensor per device, shape {ne0, ne1,
    // local_slots_g + 1}; the last plane is the zeroed sentinel slot
    struct entry {
        int          layer_idx;
        ggml_tensor* src; // model tensor holding all n_experts slices
        std::vector<ggml_tensor *> dst; // per-device hot tensors
    };
    std::vector<entry> entries;

    // per-layer index into entries (built once in ctor, entries stable after)
    std::vector<std::vector<entry *>> entries_by_layer;

    // slot_to_expert[il][p] = expert id held in slot p of layer il, or -1 if empty.
    // stable across re-syncs: an expert that stays hot keeps its slot.
    std::vector<std::vector<int>> slot_to_expert;

    // per-layer LUT and mask for in-graph routing.
    // hot_lut[g][e]   = LOCAL slot index if e is hot on device g, else the
    //                   device's local sentinel slot (zero contribution).
    // cold_mask[e]    = 1 if e is cold, else 0 (read as int zero-check by
    //                   mul_mat_id_cold).
    struct layer_lut {
        std::vector<ggml_tensor *> hot_lut; // per-device i32[n_experts]
        std::vector<ggml_tensor *> mask_lut; // per-device f32[local_slots+1], 0 at sentinel
        ggml_tensor * cold_mask = nullptr;  // i32[n_experts]
    };
    std::vector<layer_lut> luts; // size n_layers

    // per-device hot store: each device owns a contiguous slot range and its
    // own no_alloc context + GPU buffer (dst tensors and hot_luts inside).
    int n_devices = 1;
    std::vector<int>                  slot_start; // per-device slot range start (inclusive)
    std::vector<int>                  slot_end;   // per-device slot range end (exclusive)
    std::vector<ggml_context_ptr>        ctx_dev;
    std::vector<ggml_backend_buffer_ptr> buf_dev;

    // CPU context and buffer for host-side tensors (like cold_mask)
    ggml_context_ptr        ctx_cpu;
    ggml_backend_buffer_ptr buf_cpu;

    // true once the first copy of the top-S experts landed (once per session)
    bool is_filled = false;

    // re-sync cadence in tokens; 0 disables periodic re-sync
    int sync_period = 0;
    // tokens_total at the last sync (fill or re-sync) for boundary-cross check
    int64_t last_sync_tokens = 0;

    // hysteresis gate: a resident slot is only swapped when a cold
    // expert scores >= hyst * the incumbent AND the slot has dwelled long enough
    float hyst  = 0.0f; // 0 = gate off (swap freely)
    int   dwell = 0;    // minimum syncs a resident must keep; 0 = off
    // dwell_count[il][p] = syncs since slot p last changed (0 = fresh/empty)
    std::vector<std::vector<int>> dwell_count;

llama_expert_hotstore(const llama_model * model, int n_layers,
                      int n_experts, int hot_s, int sync_period = 0,
                      float hyst = 0.0f, int dwell = 0);

    ~llama_expert_hotstore();

    // allocate the GPU hot store for `hot_s` slots, split across the given
    // device buffer types by tensor_split (fractions, one per device). returns
    // false (and leaves the store disabled) on failure or shortage of VRAM.
    bool allocate(const std::vector<ggml_backend_buffer_type_t> & bufts,
                  const float * tensor_split, int n_split);

    // copy the top-S expert slices for every layer into the GPU hot store,
    // using the given heatmap for the ranking. one-shot (guarded by is_filled).
    // copy the top-S expert slices for every layer into the GPU hot store,
    // using the given heatmap for the ranking. one-shot (guarded by is_filled).
    // returns true if a fill happened (caller should synchronize the GPU).
    bool copy_top_s(const llama_expert_heatmap & heatmap);

    // re-sync the hot store to the current heatmap ranking, swapping only
    // the experts that changed (stable slots; unchanged experts not re-copied).
    // returns true if any slot changed (caller should synchronize the GPU).
    bool resync_top_s(const llama_expert_heatmap & heatmap);

    // cadence-gated wrapper: re-sync only if tokens_total crossed sync_period;
    // multi_slot freezes the hot store (static slots, no swapping). returns
    // true if a re-sync ran and swapped slots.
    bool maybe_resync(const llama_expert_heatmap & heatmap, bool multi_slot);

    // returns the GPU slot index holding expert_id in layer il, or -1 if none
    int slot_of(int layer_idx, int expert_id) const;

    // diagnostic: count how many router-selected expert ids hit a hot slot.
    // reads the selected_experts tensors (call after synchronize).
    void log_hit_rate(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel);

    // rebuild hot_lut/cold_mask from slot_to_expert for every layer
    // and copy them into the tensors. called from copy_top_s (initial
    // fill) and resync_top_s (swaps).
    void update_luts();

    void log() const;
};
