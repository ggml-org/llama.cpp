#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_model;

// stores per-layer sizing for the Mixture of Experts GPU hot store.
// one "slot" holds a single expert's weights for one layer.
struct llama_expert_hotstore {
    int n_layers;
    int n_experts;
    int hot_s;

    // bytes of a single expert slot per layer, summed over that layer's
    // expert weight tensors (gate/up/down, incl. chexps variants)
    std::vector<size_t> bytes_per_slot;

    llama_expert_hotstore(const llama_model * model, int n_layers,
                          int n_experts, int hot_s);

    void log() const;
};