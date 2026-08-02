#pragma once

#include <vector>
#include <cstdint>

struct llama_expert_heatmap {
    int n_layers;
    int n_experts;
    int hot_s;
    float decay_rate;
    int   log_period;
    int   update_count;

    std::vector<float> heat;

    llama_expert_heatmap(int n_layers, int n_experts,
                         float decay_rate = 0.99f,
                         int log_period = 100,
                         int hot_s = 0);

    void update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens);
    void decay_all();
    void log() const;

    std::vector<int> get_top_s(int layer_idx, int s) const;
};
