#pragma once

#include "moe-prune.h"

#include <cstdint>
#include <map>
#include <string>

struct llama_prune_hard_prune_report {
    uint64_t source_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t expert_bytes_removed = 0;
    std::map<int32_t, std::map<int32_t, int32_t>> original_to_new;
};

llama_prune_hard_prune_report llama_prune_hard_prune_gemma4_q4_0(
        const std::string & model_path,
        const common_moe_prune_profile & profile,
        const common_moe_prune_model_info & model_info,
        const std::string & output_path);
