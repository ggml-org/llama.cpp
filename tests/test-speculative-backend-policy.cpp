#include "server-speculative-sampling.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "policy test failure: %s\n", message);
        std::abort();
    }
}

static common_params_speculative make_spec(
        std::initializer_list<common_speculative_type> types,
        int32_t n_max) {
    common_params_speculative params;
    params.types.assign(types.begin(), types.end());
    params.draft.n_max = n_max;
    return params;
}

static void expect_profile(
        const common_params_speculative & params,
        server_spec_target_backend_profile_kind kind,
        bool has_ngram) {
    const auto profile = server_spec_target_backend_profile_select(params);
    require(static_cast<bool>(profile) == (kind != server_spec_target_backend_profile_kind::NONE), "profile enabled state");
    require(profile.kind == kind, "profile kind");
    require(profile.has_ngram_mod == has_ngram, "profile ngram flag");
}

static void expect_stochastic(bool expected, const common_params_sampling & sampling) {
    require(server_spec_target_backend_sampling_stochastic_eligible(sampling) == expected, "stochastic eligibility");
}

int main() {
    common_params_sampling practical;
    practical.temp = 1.0f;
    practical.top_k = 20;
    practical.top_p = 0.95f;
    practical.min_p = 0.05f;
    expect_stochastic(true, practical);

    common_params_sampling no_compact_filter = practical;
    no_compact_filter.top_k = 0;
    expect_stochastic(false, no_compact_filter);

    common_params_sampling greedy = practical;
    greedy.temp = 0.0f;
    expect_stochastic(false, greedy);

    common_params_sampling large_top_k = practical;
    large_top_k.top_k = 512;
    expect_stochastic(false, large_top_k);

    common_params_sampling active_penalty = practical;
    active_penalty.penalty_repeat = 1.1f;
    expect_stochastic(false, active_penalty);

    common_params_sampling active_dry = practical;
    active_dry.dry_multiplier = 0.5f;
    expect_stochastic(false, active_dry);

    common_params_sampling active_typical = practical;
    active_typical.typ_p = 0.95f;
    expect_stochastic(false, active_typical);

    common_params_sampling active_xtc = practical;
    active_xtc.xtc_probability = 0.1f;
    expect_stochastic(false, active_xtc);

    common_params_sampling requested_probs = practical;
    requested_probs.n_probs = 5;
    expect_stochastic(false, requested_probs);

    // The server precomputes EOG biases for every model, but they are inactive
    // unless ignore_eos is requested. The inactive table must not disable the
    // compact backend chain; active EOS suppression remains rejected below.
    common_params_sampling eog_metadata = practical;
    eog_metadata.logit_bias_eog.push_back({2, -1.0f});
    expect_stochastic(true, eog_metadata);

    common_params_sampling active_eos = practical;
    active_eos.ignore_eos = true;
    expect_stochastic(false, active_eos);

    common_params_sampling min_keep = practical;
    min_keep.min_keep = 1;
    expect_stochastic(false, min_keep);

    expect_profile(
        make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_MTP}, 4),
        server_spec_target_backend_profile_kind::MTP, false);
    expect_profile(
        make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 5),
        server_spec_target_backend_profile_kind::DFLASH, false);
    expect_profile(
        make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 7),
        server_spec_target_backend_profile_kind::DFLASH, false);

    expect_profile(
        make_spec({COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_DRAFT_MTP}, 4),
        server_spec_target_backend_profile_kind::MTP, true);
    expect_profile(
        make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, COMMON_SPECULATIVE_TYPE_NGRAM_MOD}, 7),
        server_spec_target_backend_profile_kind::DFLASH, true);

    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_MTP}, 5),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 4),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_NGRAM_MOD}, 4),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_NGRAM_MOD}, 4),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 4),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 7),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}, 4),
            server_spec_target_backend_profile_kind::NONE, false);
    expect_profile(make_spec({COMMON_SPECULATIVE_TYPE_NONE}, 4),
            server_spec_target_backend_profile_kind::NONE, false);

    return 0;
}
