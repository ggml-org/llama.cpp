#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

struct deepseek4_mmq_model_config {
    int64_t n_layer;
    int64_t n_embd;
    int64_t n_ff_exp;
    int64_t n_expert;
    int64_t n_expert_used;
    bool tensor_parallel;
    size_t n_devices;
    const float * tensor_split;
};

inline bool deepseek4_use_auto_rdna2_mmq_j16(const deepseek4_mmq_model_config & config) {
    if (config.n_layer != 43 || config.n_embd != 4096 || config.n_ff_exp != 2048 ||
            config.n_expert != 256 || config.n_expert_used != 6 || !config.tensor_parallel ||
            config.n_devices != 4 || config.tensor_split == nullptr) {
        return false;
    }

    const float first = config.tensor_split[0];
    if (!std::isfinite(first) || first <= 0.0f) {
        return false;
    }
    for (size_t i = 1; i < config.n_devices; ++i) {
        if (!std::isfinite(config.tensor_split[i]) || config.tensor_split[i] <= 0.0f ||
                config.tensor_split[i] != first) {
            return false;
        }
    }
    return true;
}