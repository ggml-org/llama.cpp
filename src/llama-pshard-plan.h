#pragma once

#include "llama.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

enum llama_pshard_strategy {
    LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM        = 0,
    LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM           = 1,
    LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM           = 2,
    LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS           = 3,
    LLAMA_PSHARD_COUNT
};

// strategy name for logging
inline const char * llama_pshard_strategy_name(llama_pshard_strategy s) {
    switch (s) {
        case LLAMA_PSHARD_GPUONLY_LAYERPIN_LAYERSTREAM: return "GPUONLY_LAYERPIN_LAYERSTREAM";
        case LLAMA_PSHARD_GPUONLY_ATTNPIN_FFNSTREAM:    return "GPUONLY_ATTNPIN_FFNSTREAM";
        case LLAMA_PSHARD_DYNAMIC_FFNCPU_ATTNSTREAM:    return "DYNAMIC_FFNCPU_ATTNSTREAM";
        case LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS:    return "STATIC_ATTNPRIO_ALLMODELS";
        default:                                        return "UNKNOWN";
    }
}

inline bool llama_pshard_strategy_delegates_compute(llama_pshard_strategy s) {
    return s == LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
}

// PSHARD_STRATEGY accepts a name or numeric id
inline int pshard_strategy_from_env() {
    const char * env = getenv("PSHARD_STRATEGY");
    if (!env || !*env) return -1;
    for (int i = 0; i < LLAMA_PSHARD_COUNT; i++) {
        if (strcmp(env, llama_pshard_strategy_name((llama_pshard_strategy)i)) == 0) {
            return i;
        }
    }
    char * end = nullptr;
    long v = strtol(env, &end, 10);
    if (end != env && *end == '\0' && v >= 0 && v < LLAMA_PSHARD_COUNT) {
        return (int)v;
    }
    return -1;
}

// cached tensor override entry
struct llama_pshard_override {
    std::string                pattern;
    ggml_backend_buffer_type_t buft;
    int32_t                    backend_id;
};

// saved allocator and backend ids for plan switches
struct llama_pshard_alloc_state {
    std::vector<uint8_t> node_allocs;
    std::vector<uint8_t> leaf_allocs;
    std::vector<int>     node_backend_ids;
    std::vector<int>     leaf_backend_ids;
    int  n_nodes = 0;
    int  n_leafs = 0;
    bool valid   = false;
};

struct llama_pshard_plan {
    llama_pshard_strategy strategy       = LLAMA_PSHARD_STATIC_ATTNPRIO_ALLMODELS;
    uint32_t             batch_size      = 0;
    uint32_t             n_pinned        = 0;   // fully pinned layers (all tensors on GPU)
    uint32_t             n_attn_pinned   = 0;   // attention priority layers on GPU (>= n_pinned)
    int                  overflow        = 0;   // llama_layer_fraction
    bool                 pin_from_back   = false;
    bool                 output_on_gpu   = false;

    std::vector<llama_pshard_override> overrides;

    size_t total_vram_req   = 0;
    size_t scratch_measured = 0;
    size_t cache_measured   = 0;
    float  tps              = 0.0f;  // predicted tokens/sec (0 = no benchmark data)
    bool   is_viable        = false;

    // cached maps and offsets from first apply
    mutable std::unordered_map<std::string, int32_t> cached_tensor_bids;
    mutable std::unordered_map<int, int32_t>         cached_layer_bids;
    mutable std::unordered_map<std::string, size_t>  cached_weight_offsets;
    mutable size_t cached_scratch_off = 0;
    mutable bool   maps_cached       = false;
    mutable bool   addrs_cached      = false;

    mutable llama_pshard_alloc_state alloc_state;
};

enum llama_layer_fraction {
    LLAMA_LAYER_FRACTION_NONE = 0,
    LLAMA_LAYER_FRACTION_ATTN = 1,
    LLAMA_LAYER_FRACTION_UP   = 2,
    LLAMA_LAYER_FRACTION_GATE = 3,
    LLAMA_LAYER_FRACTION_MOE  = 4,
};

const char * llama_get_overflow_pattern(size_t il, llama_layer_fraction lf);

void llama_pshard_generate_overrides(
        uint32_t n_pinned,
        uint32_t n_layers,
        ggml_backend_buffer_type_t gpu_buft,
        ggml_backend_buffer_type_t host_buft,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        llama_layer_fraction overflow_type,
        llama_pshard_strategy strategy,
        const pshard_dev_layout & layout,
        bool pin_from_back = false,
        bool output_on_gpu = false,
        uint32_t n_attn_pinned = 0);

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

std::vector<llama_device_memory_data> llama_get_device_memory_data(
        const char * path_model, const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        std::vector<llama_device> & devs, uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert,
        enum ggml_log_level log_level);

// plan cache serialization. fingerprint covers only runtime plan-compatibility params
// so the planner binary and the runtime binary can share the same cache file.
// keep this in sync with planner save
uint64_t pshard_registry_fingerprint(
        const struct llama_model_params * mparams,
        const struct llama_context_params * cparams,
        int64_t model_file_size);

bool pshard_registry_load(
        struct llama_pshard_plan_registry * registry, uint64_t fingerprint,
        const char * cache_path, ggml_backend_buffer_type_t host_buft,
        size_t current_budget, bool require_exact_budget = false);

struct llama_pshard_plan_registry {
    std::vector<uint32_t>                tier_sizes;
    std::vector<llama_pshard_plan>       best_plans;  // one best plan per tier
    llama_pshard_plan *                  active_plan = nullptr;
    uint32_t                             budget_mib = 0;
    uint32_t                             cache_ubatch = 0;

    // variant marker for a baseline load that fits
    // runtime still checks baseline_vram_req against the current budget
    bool                                 pshard_disabled = false;
    size_t                               baseline_vram_req = 0;

    void init(uint32_t n_ubatch, uint32_t n_parallel = 1, uint32_t n_draft = 0) {
        tier_sizes.clear();
        best_plans.clear();

        if (n_ubatch == 0) {
            cache_ubatch = 0;
            return;
        }

        // decode tiers
        if (n_parallel <= 1) {
            tier_sizes.push_back(1);
            tier_sizes.push_back(16);
            if (n_draft > 0) {
                uint32_t verify_tier = n_draft + 1;
                if (verify_tier > 16) {
                    tier_sizes.push_back(verify_tier);
                }
            }
        } else {
            for (uint32_t t = 1; t <= 64 && t < 512; t *= 4) {
                tier_sizes.push_back(t);
                if (t == 16) {
                    tier_sizes.push_back(32);
                }
            }
        }

        // prefill tiers: x2 growth from 512
        for (uint32_t t = 512; t < n_ubatch; t *= 2) {
            if (tier_sizes.empty() || tier_sizes.back() < t) {
                tier_sizes.push_back(t);
            }
        }

        if (tier_sizes.empty() || tier_sizes.back() != n_ubatch) {
            tier_sizes.push_back(n_ubatch);
        }

        cache_ubatch = tier_sizes.empty() ? 0 : tier_sizes.back();
        best_plans.resize(tier_sizes.size());
    }

    size_t tier_index(uint32_t batch_size) const {
        for (size_t i = 0; i < tier_sizes.size(); i++) {
            if (tier_sizes[i] >= batch_size) return i;
        }
        return tier_sizes.size() - 1;
    }

    llama_pshard_plan * get_best(size_t tier) {
        return best_plans[tier].is_viable ? &best_plans[tier] : nullptr;
    }

    // pick the prefill ubatch with the lowest predicted ttft
    // use max_ubatch when TPS data is missing
    uint32_t find_optimal_ubatch(uint32_t n_prompt, uint32_t max_ubatch) const {
        uint32_t best_ub  = max_ubatch;
        double   best_time = 1e30;

        for (size_t t = 0; t < tier_sizes.size(); t++) {
            uint32_t ts = tier_sizes[t];
            if (ts < 512 || ts > max_ubatch) continue;

            const auto & plan = best_plans[t];
            if (!plan.is_viable || plan.tps <= 0.0f) continue;

            double per_iter = (double)ts / (double)plan.tps;
            uint32_t n_iters = (n_prompt + ts - 1) / ts;
            double total = n_iters * per_iter;

            if (total < best_time) {
                best_time = total;
                best_ub   = ts;
            }
        }

        return best_ub;
    }
};
