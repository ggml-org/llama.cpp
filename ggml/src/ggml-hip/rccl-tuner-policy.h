#pragma once

#include <cstddef>

enum class ggml_rdna2_rccl_tune_mode {
    off,
    automatic,
    force,
};

struct ggml_rdna2_rccl_policy_input {
    ggml_rdna2_rccl_tune_mode mode = ggml_rdna2_rccl_tune_mode::off;
    size_t ranks = 0;
    size_t nodes = 0;
    bool all_v620 = false;
    bool pcie_hop2 = false;
    bool conflicting_policy = false;
};

inline bool ggml_rdna2_rccl_policy_eligible(const ggml_rdna2_rccl_policy_input & input) {
    if (input.nodes != 1 || !input.all_v620 || input.conflicting_policy) {
        return false;
    }
    if (input.mode == ggml_rdna2_rccl_tune_mode::automatic) {
        return input.ranks == 4 && input.pcie_hop2;
    }
    if (input.mode == ggml_rdna2_rccl_tune_mode::force) {
        return input.ranks >= 2 && input.ranks <= 8;
    }
    return false;
}
