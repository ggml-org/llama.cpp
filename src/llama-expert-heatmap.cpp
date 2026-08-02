#include "llama-expert-heatmap.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cmath>

llama_expert_heatmap::llama_expert_heatmap(
        int n_layers, int n_experts,
        float decay_rate, int log_period, int hot_s) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    decay_rate(decay_rate),
    log_period(log_period),
    tokens_total(0),
    heat(n_layers * n_experts, 0.0f) {
}

void llama_expert_heatmap::update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens) {
    float * layer_heat = heat.data() + layer_idx * n_experts;

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int32_t id = expert_ids[t * n_expert_used + e];
            if (id >= 0 && id < n_experts) {
                layer_heat[id] += 1.0f;
            }
        }
    }
}
void llama_expert_heatmap::update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts) {
    if (moe_sel_experts.empty()) {
        return;
    }

    decay_all();

    int64_t n_tokens = 0;
    for (const auto & [il, tensor] : moe_sel_experts) {
        n_tokens = tensor->ne[1];

        if (!tensor->data) {
            continue;
        }

        std::vector<int32_t> expert_ids(tensor->ne[0] * n_tokens);
        ggml_backend_tensor_get(tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));

        update(il, expert_ids.data(), tensor->ne[0], n_tokens);
    }

    tokens_total += n_tokens;
    if (log_period > 0 && tokens_total / log_period > (tokens_total - n_tokens) / log_period) {
        log();
    }
}

void llama_expert_heatmap::decay_all() {
    for (int i = 0; i < n_layers * n_experts; i++) {
        heat[i] *= decay_rate;
    }
}

void llama_expert_heatmap::log() const {
    LLAMA_LOG("=== Expert heatmap (tokens %" PRId64 ") ===\n", tokens_total);

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
            LLAMA_LOG("  layer %3d: %d warm experts, max heat=%.2f (expert %d)",
                l, active_count, max_heat, max_id);

            auto top = get_top_s(l, 8);
            LLAMA_LOG("  top-8=");
            for (size_t i = 0; i < top.size(); i++) {
                LLAMA_LOG("%s%d", i > 0 ? "," : "{", top[i]);
            }
            LLAMA_LOG("}\n");
        }
    }
}

std::vector<int> llama_expert_heatmap::get_top_s(int layer_idx, int s) const {
    std::vector<int> result;
    if (layer_idx < 0 || layer_idx >= n_layers || s <= 0) {
        return result;
    }

    const float * layer_heat = heat.data() + layer_idx * n_experts;

    std::vector<int> indices(n_experts);
    for (int i = 0; i < n_experts; i++) {
        indices[i] = i;
    }

    int k = std::min(s, n_experts);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [layer_heat](int a, int b) {
            return layer_heat[a] > layer_heat[b];
        });

    result.assign(indices.begin(), indices.begin() + k);
    return result;
}
