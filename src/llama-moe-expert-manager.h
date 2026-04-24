#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct llama_mmap;

struct llama_expert_slice {
    llama_mmap * mapping;        // which mmap region (nullptr = GPU/skip)
    size_t       offset_in_mmap; // byte offset of expert slice start within mmap
    size_t       size;           // size of the expert slice in bytes
};

struct llama_expert_state {
    uint64_t last_used_at; // query counter when expert was last used (0 = never)
    bool     resident;     // whether we believe pages are in RAM
};

class llama_moe_expert_manager {
public:
    llama_moe_expert_manager() = default;
    ~llama_moe_expert_manager() = default;

    // initialize from a model. returns true if MoE lazy loading is active.
    bool init(llama_model & model);

    bool is_active() const { return active; }

    // called at start of each decode() - resets per-query tracking
    void begin_query();

    // called from eval callback when ffn_moe_topk-{il} is computed
    void record_topk(int layer_id, const int32_t * ids, int n_expert_used, int n_tokens);

    // called at end of each decode() - runs eviction
    void end_query();

    // static eval callback that wraps user callback + captures expert selections
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

    // user callback to chain
    ggml_backend_sched_eval_callback user_cb      = nullptr;
    void *                           user_cb_data = nullptr;

private:
    void evict_stale_experts();

    bool     active        = false;
    uint64_t query_counter = 0;

    static constexpr uint64_t EVICTION_THRESHOLD = 4;

    struct moe_layer_data {
        int layer_id;
        int n_expert;
        // [expert_id][tensor_idx] - memory slices for each expert's weight tensors
        std::vector<std::vector<llama_expert_slice>> expert_slices;
        // [expert_id]
        std::vector<llama_expert_state> expert_states;
    };

    std::vector<moe_layer_data> layers;

    // map from model layer_id to index in layers[]
    std::vector<int> layer_id_to_idx;

    // per-query usage: [moe_layer_index][expert_id]
    std::vector<std::vector<bool>> current_usage;
};
