#pragma once

#include <vector>
#include <cstdint>

struct ggml_tensor;

struct llama_expert_heatmap {
    int n_layers;
    int n_experts;
    int hot_s;
    float decay_rate;
    int   log_period;
    int64_t tokens_total; // real tokens seen (not multiplied by layers)
    int64_t generated_tokens_count; // decode tokens seen; drives first-fill deferral
    int update_counter = 1; // batched decay every 2 updates; start at 1 for early stability
    std::vector<float> heat;

    llama_expert_heatmap(int n_layers, int n_experts,
                         float decay_rate = 0.99f,
                         int log_period = 100,
                         int hot_s = 0);

    // cold-op counts path: per-layer tallies of the selected experts (host
    // memory). advances tokens_total; the first-3-token boost lives in get_score.
    void update_counts(const std::vector<const int32_t *> & per_layer, int n_tokens);

    // standalone path (no cold op): per-expert increment from graph readback
    void update_ids(int layer_idx, const int32_t * expert_ids, int n_ids, int n_tokens);
    void log() const;

    float get_score(int layer_idx, int expert_id) const;
    std::vector<int> get_top_s(int layer_idx, int s) const;
};
