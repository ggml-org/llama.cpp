#pragma once

#include "common.h"

enum class server_spec_target_backend_profile_kind {
    NONE,
    MTP,
    DFLASH,
};

struct server_spec_target_backend_profile {
    server_spec_target_backend_profile_kind kind = server_spec_target_backend_profile_kind::NONE;
    bool has_ngram_mod = false;

    explicit operator bool() const {
        return kind != server_spec_target_backend_profile_kind::NONE;
    }
};

// The validated DFlash target-backend path is greedy-only. Its universal
// stochastic target path was slower on the realistic qualification suite and
// is deliberately excluded from automatic activation; MTP remains eligible.
inline bool server_spec_target_backend_profile_allows_stochastic_auto(
        const server_spec_target_backend_profile & profile) {
    return profile && profile.kind == server_spec_target_backend_profile_kind::MTP;
}

// Return true only for the practical stateless stochastic chain that can reduce
// the target output transfer. Active history/stateful or unsupported samplers
// are deliberately rejected; neutral controls are represented by empty
// backend-capable samplers in common_sampler_init().
inline bool server_spec_target_backend_sampling_stochastic_eligible(
        const common_params_sampling & sampling) {
    // logit_bias_eog is only applied when ignore_eos is active; the guard above
    // rejects that active path, so the precomputed inactive table is harmless.
    if (sampling.samplers.empty() || sampling.temp <= 0.0f || sampling.dynatemp_range != 0.0f ||
            sampling.mirostat != 0 || sampling.adaptive_target >= 0.0f || sampling.min_keep != 0 ||
            sampling.ignore_eos || sampling.n_probs > 0 || !sampling.logit_bias.empty() ||
            !sampling.grammar.empty()) {
        return false;
    }

    const bool has_reasoning_budget =
        !sampling.reasoning_budget_start.empty() && !sampling.reasoning_budget_end.empty() &&
        (sampling.grammar_lazy || sampling.reasoning_budget_tokens >= 0 || sampling.reasoning_control);
    if (has_reasoning_budget) {
        return false;
    }

    bool has_active_top_k = false;
    for (common_sampler_type sampler : sampling.samplers) {
        switch (sampler) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                // The graph-level backend_accept hook is not active between
                // multi-output rows. Do not freeze active history penalties
                // across a speculative verification batch.
                if (sampling.penalty_repeat != 1.0f || sampling.penalty_freq != 0.0f ||
                        sampling.penalty_present != 0.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (sampling.dry_multiplier != 0.0f && sampling.dry_base >= 1.0f &&
                        sampling.dry_penalty_last_n != 0) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (sampling.top_n_sigma > 0.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
                // Keep the automatic path on the compact ROCm top-k route.
                if (sampling.top_k > 256) {
                    return false;
                }
                has_active_top_k = sampling.top_k > 0;
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (sampling.typ_p < 1.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (sampling.top_p <= 0.0f || sampling.top_p > 1.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (sampling.min_p < 0.0f || sampling.min_p > 1.0f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (sampling.xtc_probability > 0.0f && sampling.xtc_threshold <= 0.5f) {
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                break;
            default:
                return false;
        }
    }

    // Without an active top-k stage, the backend distribution path returns a
    // full-vocabulary probability/candidate row, so it does not provide the
    // intended transfer reduction for automatic E2E optimization.
    return has_active_top_k;
}

// Select the proposal profiles validated for automatic gfx1030 target backend
// sampling. Stacking is a priority cascade: ngram-mod may produce the draft for
// a cycle, otherwise the one neural drafter does. Target verification receives
// the same flat draft either way, so an optional ngram-mod does not change this
// target-side profile.
inline server_spec_target_backend_profile server_spec_target_backend_profile_select(
        const common_params_speculative & params) {
    server_spec_target_backend_profile result;
    bool has_neural = false;

    for (common_speculative_type type : params.types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                if (result.has_ngram_mod) {
                    return {};
                }
                result.has_ngram_mod = true;
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                if (has_neural) {
                    return {};
                }
                has_neural = true;
                result.kind = server_spec_target_backend_profile_kind::MTP;
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
                if (has_neural) {
                    return {};
                }
                has_neural = true;
                result.kind = server_spec_target_backend_profile_kind::DFLASH;
                break;
            default:
                return {};
        }
    }

    if (!has_neural) {
        return {};
    }

    switch (result.kind) {
        case server_spec_target_backend_profile_kind::MTP:
            return params.draft.n_max == 4 ? result : server_spec_target_backend_profile{};
        case server_spec_target_backend_profile_kind::DFLASH:
            return params.draft.n_max == 5 || params.draft.n_max == 7
                ? result
                : server_spec_target_backend_profile{};
        case server_spec_target_backend_profile_kind::NONE:
            return {};
    }

    return {};
}

// A stacked n-gram map has its own value width and can otherwise override the
// neural drafter's much smaller cycle. K4V m=48 can turn a new hit into a
// 49-row target pass and make the 248K-vocabulary output visibly pause. Reuse
// the existing request-level cap to chunk those hits at the validated neural
// drafter width. The server applies this policy only after separately
// qualifying the automatic hardware profile; an explicit request-level
// speculative.n_max continues to take precedence.
inline int32_t server_spec_neural_k4v_cycle_cap(
        const common_params_speculative & params) {
    int n_dflash = 0;
    int n_mtp = 0;
    int n_k4v = 0;

    for (common_speculative_type type : params.types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
                ++n_dflash;
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                ++n_mtp;
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                ++n_k4v;
                break;
            default:
                return -1;
        }
    }

    if (n_k4v != 1 || n_dflash + n_mtp != 1 ||
            params.ngram_map_k4v.size_m <= params.draft.n_max) {
        return -1;
    }
    if (n_dflash == 1 && params.draft.n_max == 5) {
        return 5;
    }
    if (n_mtp == 1 && params.draft.n_max >= 2 && params.draft.n_max <= 4) {
        return params.draft.n_max;
    }
    return -1;
}
