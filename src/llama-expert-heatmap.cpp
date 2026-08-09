#include "llama-expert-heatmap.h"
#include "llama-impl.h"

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
    generated_tokens_count(0),
    heat(n_layers * n_experts, 0.0f) {
}

void llama_expert_heatmap::update_counts(const std::vector<const int32_t *> & per_layer, int n_tokens) {
    tick(n_tokens);
    for (int il = 0; il < n_layers && il < (int) per_layer.size(); il++) {
        const int32_t * cnt = per_layer[il];
        if (!cnt) {
            continue;
        }
        float * layer_heat = &heat[il * n_experts];
        for (int e = 0; e < n_experts; e++) {
            if (cnt[e] > 0) {
                layer_heat[e] += (float) cnt[e];
            }
        }
    }
}

void llama_expert_heatmap::tick(int n_tokens) {
    // batched decay: apply decay_rate^2 every two updates instead of
    // decay_rate every update (identical long-run weighting, half the work)
    ++update_counter;
    if (update_counter % 2 == 0) {
        const float rate = decay_rate * decay_rate;
        for (int i = 0; i < n_layers * n_experts; i++) {
            heat[i] *= rate;
        }
    }
    if (n_tokens == 1) {
        generated_tokens_count++;
    }
    tokens_total += n_tokens;
}

void llama_expert_heatmap::update_ids(int layer_idx, const int32_t * expert_ids, int n_ids, int n_tokens) {
    if (layer_idx < 0 || layer_idx >= n_layers || !expert_ids) {
        return;
    }
    float * layer_heat = &heat[layer_idx * n_experts];
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_ids; e++) {
            const int32_t id = expert_ids[t * n_ids + e];
            if (id >= 0 && id < n_experts) {
                layer_heat[id] += 1.0f;
            }
        }
    }
}

void llama_expert_heatmap::log() const {
    LLAMA_LOG("expert_heatmap: tokens %" PRId64 "\n", tokens_total);

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

float llama_expert_heatmap::get_score(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || expert_id < 0 || expert_id >= n_experts) {
        return 0.0f;
    }
    float s = heat[layer_idx * n_experts + expert_id];
    if (generated_tokens_count <= 3) {
        s *= 5.0f; // start-up boost: force the store toward the true hot set
    }
    return s;
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
