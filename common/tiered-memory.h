#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class common_tiered_memory_tier {
    VRAM,
    DRAM,
    SSD,
};

struct common_tiered_memory_item {
    std::string name;
    size_t size = 0;
    double active_fraction = 1.0;
};

struct common_tiered_memory_config {
    size_t vram_budget = 0;
    size_t dram_budget = 0;
};

struct common_tiered_memory_placement {
    std::string name;
    size_t size = 0;
    double active_fraction = 1.0;
    common_tiered_memory_tier tier = common_tiered_memory_tier::SSD;
};

struct common_tiered_memory_plan {
    std::vector<common_tiered_memory_placement> placements;

    size_t vram_bytes = 0;
    size_t dram_bytes = 0;
    size_t ssd_bytes = 0;

    double active_vram_bytes = 0.0;
    double active_dram_bytes = 0.0;
    double active_ssd_bytes = 0.0;
};

// Estimate the fraction of a tensor read for one token. Dense tensors have an
// active fraction of 1.0. A stacked MoE expert tensor has
// n_expert_used / n_expert when both values are known.
double common_tiered_memory_active_fraction(
        const std::string & tensor_name,
        uint32_t n_expert,
        uint32_t n_expert_used);

// Assign whole tensors to VRAM, DRAM, or SSD. Items are prioritized by active
// fraction, which is the expected transfer benefit per byte of faster storage.
// The planner is deterministic and never splits a tensor between tiers.
common_tiered_memory_plan common_tiered_memory_make_plan(
        const std::vector<common_tiered_memory_item> & items,
        const common_tiered_memory_config & config);

const char * common_tiered_memory_tier_name(common_tiered_memory_tier tier);
