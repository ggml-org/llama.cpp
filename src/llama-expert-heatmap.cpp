#include "llama-expert-heatmap.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdio>
#include <cmath>

llama_expert_heatmap::llama_expert_heatmap(
        int n_layers, int n_experts,
        float decay_rate, int log_period) :
    n_layers(n_layers),
    n_experts(n_experts),
    decay_rate(decay_rate),
    log_period(log_period),
    update_count(0),
    heat(n_layers * n_experts, 0.0f) {
}

void llama_expert_heatmap::update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens) {
    decay_all();

    float * layer_heat = heat.data() + layer_idx * n_experts;

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int32_t id = expert_ids[t * n_expert_used + e];
            if (id >= 0 && id < n_experts) {
                layer_heat[id] += 1.0f;
            }
        }
    }

    update_count++;
    if (log_period > 0 && update_count % log_period == 0) {
        log();
    }
}

void llama_expert_heatmap::decay_all() {
    for (int i = 0; i < n_layers * n_experts; i++) {
        heat[i] *= decay_rate;
    }
}

void llama_expert_heatmap::log() const {
    LLAMA_LOG("=== Expert heatmap (update %d) ===\n", update_count);

    for (int l = 0; l < n_layers; l++) {
        const float * layer_heat = heat.data() + l * n_experts;
        int active_count = 0;
        float max_heat = 0.0f;
        int max_id = -1;

        for (int e = 0; e < n_experts; e++) {
            if (layer_heat[e] > 0.01f) {
                active_count++;
            }
            if (layer_heat[e] > max_heat) {
                max_heat = layer_heat[e];
                max_id = e;
            }
        }

        if (active_count > 0) {
            LLAMA_LOG("  layer %3d: %d warm experts, max heat=%.2f (expert %d)\n",
                l, active_count, max_heat, max_id);
        }
    }
}
