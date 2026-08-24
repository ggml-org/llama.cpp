#include "server-speculative-sampling.h"

#include <cassert>
#include <initializer_list>

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
    assert(static_cast<bool>(profile) == (kind != server_spec_target_backend_profile_kind::NONE));
    assert(profile.kind == kind);
    assert(profile.has_ngram_mod == has_ngram);
}

int main() {
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
            server_spec_target_backend_profile_kind::NONE, true);
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
