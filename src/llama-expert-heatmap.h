#pragma once

#include <vector>
#include <cstdint>
#include <utility>

struct ggml_tensor;

struct llama_expert_heatmap {
    int n_layers;
    int n_experts;
    int hot_s;
    float decay_rate;
    int   log_period;
    int64_t tokens_total; // real tokens seen (not multiplied by layers)

    std::vector<float> heat;

    llama_expert_heatmap(int n_layers, int n_experts,
                         float decay_rate = 0.99f,
                         int log_period = 100,
                         int hot_s = 0);

    void update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens);
    void update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts);
    void decay_all();
    void log() const;

    std::vector<int> get_top_s(int layer_idx, int s) const;
};
