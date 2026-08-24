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
