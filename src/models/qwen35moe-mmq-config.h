#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

struct qwen35moe_mmq_model_config {
    bool is_35b_a3b;
    bool is_122b_a10b;
    int64_t n_embd;
    int64_t n_ff_exp;
    int64_t n_expert;
    int64_t n_expert_used;
    bool tensor_parallel;
    bool layer_split;
    size_t n_devices;
    const float * tensor_split;
};

inline bool qwen35moe_use_auto_rdna2_q4_k_j16(const qwen35moe_mmq_model_config & config) {
    const bool qwen35_122b = config.is_122b_a10b && !config.is_35b_a3b &&
        config.n_embd == 3072 && config.n_ff_exp == 1024 && config.n_expert == 256 &&
        config.n_expert_used == 8 && config.tensor_parallel && !config.layer_split;
    const bool qwen36_35b = config.is_35b_a3b && !config.is_122b_a10b &&
        config.n_embd == 2048 && config.n_ff_exp == 512 && config.n_expert == 256 &&
        config.n_expert_used == 8 && !config.tensor_parallel && config.layer_split;
    if ((!qwen35_122b && !qwen36_35b) || config.n_devices != 4 || config.tensor_split == nullptr) {
        return false;
    }

    const float split = config.tensor_split[0];
    if (!std::isfinite(split) || split <= 0.0f) {
        return false;
    }
    for (size_t i = 1; i < config.n_devices; ++i) {
        if (!std::isfinite(config.tensor_split[i]) || config.tensor_split[i] != split) {
            return false;
        }
    }
    return true;
}