#include "tiered-memory.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace {

bool is_expert_stack(const std::string & name) {
    return name.find("_exps") != std::string::npos ||
           name.find(".experts.") != std::string::npos;
}

double normalize_active_fraction(double value) {
    if (!std::isfinite(value)) {
        return 1.0;
    }
    return std::max(0.0, std::min(1.0, value));
}

void account_placement(common_tiered_memory_plan & plan, const common_tiered_memory_placement & placement) {
    const double active_bytes = static_cast<double>(placement.size) * placement.active_fraction;

    switch (placement.tier) {
        case common_tiered_memory_tier::VRAM:
            plan.vram_bytes += placement.size;
            plan.active_vram_bytes += active_bytes;
            break;
        case common_tiered_memory_tier::DRAM:
            plan.dram_bytes += placement.size;
            plan.active_dram_bytes += active_bytes;
            break;
        case common_tiered_memory_tier::SSD:
            plan.ssd_bytes += placement.size;
            plan.active_ssd_bytes += active_bytes;
            break;
    }
}

} // namespace

double common_tiered_memory_active_fraction(
        const std::string & tensor_name,
        uint32_t n_expert,
        uint32_t n_expert_used) {
    if (!is_expert_stack(tensor_name) || n_expert == 0 || n_expert_used == 0) {
        return 1.0;
    }

    return normalize_active_fraction(
            static_cast<double>(n_expert_used) / static_cast<double>(n_expert));
}

common_tiered_memory_plan common_tiered_memory_make_plan(
        const std::vector<common_tiered_memory_item> & items,
        const common_tiered_memory_config & config) {
    common_tiered_memory_plan plan;
    plan.placements.reserve(items.size());

    for (const auto & item : items) {
        plan.placements.push_back({
            item.name,
            item.size,
            normalize_active_fraction(item.active_fraction),
            common_tiered_memory_tier::SSD,
        });
    }

    std::vector<size_t> priority(plan.placements.size());
    std::iota(priority.begin(), priority.end(), 0);
    std::stable_sort(priority.begin(), priority.end(), [&](size_t lhs, size_t rhs) {
        const auto & a = plan.placements[lhs];
        const auto & b = plan.placements[rhs];
        if (a.active_fraction != b.active_fraction) {
            return a.active_fraction > b.active_fraction;
        }
        return a.name < b.name;
    });

    size_t vram_remaining = config.vram_budget;
    for (const size_t index : priority) {
        auto & placement = plan.placements[index];
        if (placement.size <= vram_remaining) {
            placement.tier = common_tiered_memory_tier::VRAM;
            vram_remaining -= placement.size;
        }
    }

    size_t dram_remaining = config.dram_budget;
    for (const size_t index : priority) {
        auto & placement = plan.placements[index];
        if (placement.tier != common_tiered_memory_tier::SSD) {
            continue;
        }
        if (placement.size <= dram_remaining) {
            placement.tier = common_tiered_memory_tier::DRAM;
            dram_remaining -= placement.size;
        }
    }

    for (const auto & placement : plan.placements) {
        account_placement(plan, placement);
    }

    std::stable_sort(plan.placements.begin(), plan.placements.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.tier != rhs.tier) {
            return static_cast<int>(lhs.tier) < static_cast<int>(rhs.tier);
        }
        if (lhs.active_fraction != rhs.active_fraction) {
            return lhs.active_fraction > rhs.active_fraction;
        }
        return lhs.name < rhs.name;
    });

    return plan;
}

const char * common_tiered_memory_tier_name(common_tiered_memory_tier tier) {
    switch (tier) {
        case common_tiered_memory_tier::VRAM: return "VRAM";
        case common_tiered_memory_tier::DRAM: return "DRAM";
        case common_tiered_memory_tier::SSD:  return "SSD";
    }
    return "unknown";
}
