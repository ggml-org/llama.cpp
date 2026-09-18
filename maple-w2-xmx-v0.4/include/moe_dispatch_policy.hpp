// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <stdexcept>
namespace maple_w2 {
enum class MoeSchedule { inherit_v04, direct, grouped, auto_select };
// No Vulkan crossover is silently installed on XMX. min_tokens==0 means
// uncalibrated: auto stays direct. Thresholds belong to a device/build/shape
// calibration, not the kernel format. jobs/experts is a load estimate, NOT a
// measurement of routing skew. No ID readback or hidden host synchronization.
struct MoeAutoPolicy {
    uint32_t min_tokens=0;
    uint32_t min_mean_assignments=0;
};
struct MoeScheduleDecision { bool grouped=false; const char* reason="legacy_direct"; };
inline MoeScheduleDecision choose_moe_schedule(MoeSchedule requested,bool legacy,
    uint32_t tokens,uint32_t topk,uint32_t experts,bool supported,MoeAutoPolicy policy={}) {
    if(!tokens||!topk||!experts||topk>experts)throw std::invalid_argument("invalid dispatch shape");
    if(requested==MoeSchedule::inherit_v04) {
        if(legacy&&!supported)throw std::invalid_argument("unsupported forced legacy grouping");
        return {legacy,legacy?"legacy_grouped":"legacy_direct"};
    }
    if(requested==MoeSchedule::direct)return {false,"forced_direct"};
    if(requested==MoeSchedule::grouped) {
        if(!supported)throw std::invalid_argument("unsupported forced grouping");
        return {true,"forced_grouped"};
    }
    if(requested!=MoeSchedule::auto_select)throw std::invalid_argument("invalid schedule enum");
    if(!supported)return {false,"auto_unsupported"};
    if(tokens==1)return {false,"auto_single_token"};
    if(!policy.min_tokens)return {false,"auto_uncalibrated"};
    if(tokens<policy.min_tokens)return {false,"auto_below_token_threshold"};
    if(uint64_t(tokens)*topk<uint64_t(policy.min_mean_assignments)*experts)
        return {false,"auto_below_mean_assignment_threshold"};
    return {true,"auto_threshold_grouped"};
}
inline void check_tokens_per_tile(uint32_t n) {
    if(n!=1&&n!=2&&n!=4)throw std::invalid_argument("tokens per tile must be 1, 2 or 4");
}
} // namespace maple_w2
