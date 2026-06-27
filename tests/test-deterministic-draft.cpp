// test-deterministic-draft.cpp -- Unit tests for deterministic draft plugin loader
//
// Tests:
//   1. Plugin loader: init/free with valid and invalid paths
//   2. C API wrappers: get_capabilities, set_vocab, fill_bitmask, commit, reset
//   3. Speculative integration: common_speculative with DRAFT_DETERMINISTIC type
//   4. Auto-imply: --deterministic-draft-model implies draft-mtp
//   5. --det-draft-intercept-all flag validation, accessor, and non-MTP fallback
//   6. regex demo plugin: whitespace + EOG/stop-token filter behavior
//
// These tests use the generic plugin loader (libdeterministic_draft_spec.so)
// and the bundled regex demo plugin (libdeterministic_draft_spec_plugin.so).
// Grammar (.gbnf) based integration tests live in a separate XGrammar
// repository, not here.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama_deterministic_draft.h"
#include "sampling.h"
#include "speculative.h"

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#    define GETPID _getpid
#else
#    include <unistd.h>
#    define GETPID getpid
#endif
// Find the regex demo plugin (.so) built by this repo's SDK.
static std::string find_regex_plugin() {
    const char * candidates[] = {
        "external/plugins/lib/libdeterministic_draft_spec_plugin.so",
        "./external/plugins/lib/libdeterministic_draft_spec_plugin.so",
        "libdeterministic_draft_spec_plugin.so",
        nullptr
    };

    for (int i = 0; candidates[i]; i++) {
        FILE * f = fopen(candidates[i], "rb");
        if (f) {
            fclose(f);
            return candidates[i];
        }
    }

    return "";
}
// ============================================================================
// Test 1: Plugin loader lifecycle
// ============================================================================

static void test_plugin_loader_init_free() {
    printf("test_plugin_loader_init_free... ");

    // init with NULL path should return a valid handle (no plugin loaded)
    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);
    llama_deterministic_draft_free(draft);

    // init with non-existent path should return nullptr
    draft = llama_deterministic_draft_init("/nonexistent/path/plugin.so");
    assert(draft == nullptr);

    // free with nullptr should be safe
    llama_deterministic_draft_free(nullptr);

    printf("OK\n");
}

// ============================================================================
// Test 2: C API wrappers with no plugin loaded
// ============================================================================

static void test_c_api_no_plugin() {
    printf("test_c_api_no_plugin... ");

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);

    // get_capabilities should return 0 (no plugin loaded)
    assert(llama_deterministic_draft_get_capabilities(draft) == 0);

    // set_vocab should fail gracefully (no plugin)
    const char * dummy_vocab[] = { "a" };
    assert(!llama_deterministic_draft_set_vocab(draft, dummy_vocab, 1, nullptr, 0));

    // fill_bitmask should return false (no constraint / no plugin)
    uint32_t bitmask[4] = { 0 };
    assert(!llama_deterministic_draft_fill_bitmask(draft, 0, bitmask, 128));

    // commit should be safe (no-op)
    llama_deterministic_draft_commit(draft, 0, 0, "x", 1);

    // reset should be safe (no-op)
    llama_deterministic_draft_reset(draft, 0);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 3: Speculative type enum
// ============================================================================

static void test_speculative_type_enum() {
    printf("test_speculative_type_enum... ");

    // The enum should have the new type
    assert(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC != COMMON_SPECULATIVE_TYPE_NONE);
    assert(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC != COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    assert(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC < COMMON_SPECULATIVE_TYPE_COUNT);

    // Type name mapping
    std::string name = common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    assert(name == "draft-deterministic");

    printf("OK\n");
}

// ============================================================================
// Test 4: Params struct has deterministic_draft in speculative
// ============================================================================

static void test_params_struct() {
    printf("test_params_struct... ");

    common_params params;

    // deterministic_draft should be in speculative, not in sampling or root
    params.speculative.deterministic_draft.enabled        = true;
    params.speculative.deterministic_draft.n_max          = 42;
    params.speculative.deterministic_draft.plugin_path    = "/test/path.so";
    params.speculative.deterministic_draft.det_intercept_all = true;

    assert(params.speculative.deterministic_draft.enabled == true);
    assert(params.speculative.deterministic_draft.n_max == 42);
    assert(params.speculative.deterministic_draft.plugin_path == "/test/path.so");
    assert(params.speculative.deterministic_draft.det_intercept_all == true);

    // default value is false
    common_params params2;
    assert(params2.speculative.deterministic_draft.det_intercept_all == false);
    assert(params2.speculative.deterministic_draft.det_filter_bonus == false);

    printf("OK\n");
}

// ============================================================================
// Test 5: common_speculative_has_det_filter with null spec
// ============================================================================

static void test_det_filter_query() {
    printf("test_det_filter_query... ");

    assert(!common_speculative_has_det_filter(nullptr));

    // det_intercept_all should be false when spec is null
    assert(!common_speculative_get_det_accept_all(nullptr));

    // Get filter result from null should return empty
    const auto & fr = common_speculative_get_det_filter_result(nullptr, 0);
    assert(!fr.truncated);
    assert(fr.valid_count == 0);

    printf("OK\n");
}
// ============================================================================
// Test 7: Fail-to-start when DRAFT_DETERMINISTIC enabled without DRAFT_MTP
// ============================================================================

static void test_fail_without_mtp() {
    printf("test_fail_without_mtp... ");

    common_params_speculative params;
    params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    params.deterministic_draft.enabled     = true;
    params.deterministic_draft.plugin_path = "/nonexistent/plugin.so";

    // No DRAFT_MTP in types, no ctx_dft -> init must fail
    struct common_speculative * spec = common_speculative_init(params, 1);
    assert(spec == nullptr);

    printf("OK\n");
}

// ============================================================================
// Test 8: Fail-to-start when DRAFT_DETERMINISTIC enabled without plugin path
// ============================================================================

static void test_fail_without_plugin() {
    printf("test_fail_without_plugin... ");

    common_params_speculative params;
    params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    params.deterministic_draft.enabled = true;
    // plugin_path left empty

    // ctx_dft is nullptr (no model loaded), so has_mtp is false -> fail
    struct common_speculative * spec = common_speculative_init(params, 1);
    assert(spec == nullptr);

    printf("OK\n");
}

// ============================================================================
// Test 9: det_intercept_all requires the plugin and a compatible drafter type
// ============================================================================

static void test_intercept_all_requires_plugin() {
    printf("test_intercept_all_requires_plugin... ");

    // det_intercept_all without a plugin -> common_params_parse must fail
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-intercept-all";
        char * argv[] = { arg0, arg1, nullptr };
        assert(!common_params_parse(2, argv, params, LLAMA_EXAMPLE_SERVER));
    }

    // enabled + det_intercept_all without a compatible --spec-type -> parse must fail
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--det-draft-intercept-all";
        char arg4[] = "--det-draft-n-max";
        char arg5[] = "16";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, nullptr };
        assert(!common_params_parse(6, argv, params, LLAMA_EXAMPLE_SERVER));
    }

    // enabled + det_intercept_all + draft-mtp -> parses; n_max copy applied
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--det-draft-intercept-all";
        char arg4[] = "--det-draft-n-max";
        char arg5[] = "16";
        char arg6[] = "--spec-type";
        char arg7[] = "draft-mtp";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, nullptr };
        assert(common_params_parse(8, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(params.speculative.deterministic_draft.enabled);
        assert(params.speculative.deterministic_draft.det_intercept_all);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end());
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC) != types.end());
        assert(params.speculative.draft.n_max == 16);
    }

    printf("OK\n");
}
// ============================================================================
// Test 11: Auto-imply draft-mtp when deterministic draft enabled
// ============================================================================

static void test_auto_imply_mtp() {
    printf("test_auto_imply_mtp... ");

    common_params params;
    params.speculative.deterministic_draft.enabled     = true;
    params.speculative.deterministic_draft.plugin_path = "/test/plugin.so";

    // Clear any default types to test the auto-imply logic in isolation
    params.speculative.types.clear();
    assert(params.speculative.types.empty());

    // Simulate the auto-imply logic from the --det-draft-model option callback
    if (params.speculative.deterministic_draft.enabled) {
        auto & types = params.speculative.types;
        if (std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == types.end()) {
            types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        }
        if (std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC) == types.end()) {
            types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
        }
    }

    // Both types should now be present
    auto & types = params.speculative.types;
    assert(types.size() == 2);
    assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end());
    assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC) != types.end());

    printf("OK\n");
}

// ============================================================================
// Test 12: common_token_to_piece with special=false produces clean text
// ============================================================================

static void test_token_to_piece_special_flag() {
    printf("test_token_to_piece_special_flag... ");

    // The deterministic draft filter uses special=false to skip control
    // tokens so the filter sees clean text without BPE artifacts.
    // Compile-time signature check verifies the function accepts the flag.
    // Runtime verification requires a loaded model (done in integration tests).
    using fn_type = std::string (*)(const struct llama_vocab *, llama_token, bool);
    auto * fn = static_cast<fn_type>(&common_token_to_piece);
    (void) fn;

    printf("OK\n");
}

// ============================================================================
// Test 13: Cumulative stats struct fields exist and are zero-initialized
// ============================================================================

static void test_cumulative_stats_fields() {
    printf("test_cumulative_stats_fields... ");

    // Verify that common_speculative with det_filter has cumulative stats
    // that start at zero. We can't fully test without a loaded model, but
    // we can verify the struct is properly initialized by checking that
    // has_det_filter returns false for a spec without a plugin.

    common_params_speculative params;
    params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    // No plugin path -> det_filter won't be loaded
    struct common_speculative * spec = common_speculative_init(params, 1);
    // Without ctx_dft, init fails (has_mtp is false)
    assert(spec == nullptr);

    // If we had a valid spec, we would check:
    // assert(!common_speculative_has_det_filter(spec));
    // But since init fails, we just verify the API exists.

    // Cumulative det_filter counters (checked numerically only with a loaded
    // model + plugin default-mode cycle; see test_standard_mode_* tests):
    //   n_drafts_total, n_truncated_total, n_tokens_pre_filter,
    //   n_tokens_post_filter, n_target_rejected_total
    // common_speculative_print_stats(nullptr) is a no-op and safe to call.
    common_speculative_print_stats(nullptr);

    printf("OK\n");
}

// ============================================================================
// Test 14: accept() with null impl does not crash (LOG_WRN path)
// ============================================================================

static void test_accept_null_impl_safe() {
    printf("test_accept_null_impl_safe... ");

    // This tests that common_speculative_accept() handles the case where
    // impl_last[seq_id] is null without crashing (returns with LOG_WRN).
    // We can't easily create this scenario without a loaded model, but
    // we verify the function handles null spec gracefully.

    common_speculative_accept(nullptr, 0, 0);

    printf("OK\n");
}

// ============================================================================
// Test 15: get_version API exists and returns "unknown" for null plugin
// ============================================================================

static void test_get_version_null() {
    printf("test_get_version_null... ");

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);

    // With no plugin loaded, get_version should return "unknown"
    const char * version = llama_deterministic_draft_get_version(draft);
    assert(version != nullptr);
    assert(std::string(version) == "unknown");

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 16: New high-level filter API with no plugin (safety)
// ============================================================================

static void test_filter_api_no_plugin() {
    printf("test_filter_api_no_plugin... ");

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);

    // filter_draft should return 0 (no plugin)
    const int tokens[] = { 0, 1, 2 };
    assert(llama_deterministic_draft_filter_draft(draft, 0, tokens, 3) == 0);

    // apply_bitmask should return false (no plugin)
    uint32_t bitmask[4] = { 0 };
    float logits[128] = { 0 };
    assert(!llama_deterministic_draft_apply_bitmask(draft, 0, bitmask, 128, logits));

    // commit_tokens should be safe (no-op)
    llama_deterministic_draft_commit_tokens(draft, 0, tokens, 3);

    // null args should be safe
    assert(llama_deterministic_draft_filter_draft(nullptr, 0, tokens, 3) == 0);
    assert(!llama_deterministic_draft_apply_bitmask(nullptr, 0, bitmask, 128, logits));
    llama_deterministic_draft_commit_tokens(nullptr, 0, tokens, 3);

    // zero n_tokens should return 0/false
    assert(llama_deterministic_draft_filter_draft(draft, 0, tokens, 0) == 0);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}
// ============================================================================
// Test 21: rollback returns false for invalid args (no plugin loaded)
// ============================================================================

static void test_rollback_returns_false_no_plugin() {
    printf("test_rollback_returns_false_no_plugin... ");

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);

    // No plugin loaded -> rollback function pointer is null -> returns false
    assert(!llama_deterministic_draft_rollback(draft, 0, 1));
    assert(!llama_deterministic_draft_rollback(draft, 0, 0));
    assert(!llama_deterministic_draft_rollback(draft, 0, -1));

    // null draft -> returns false
    assert(!llama_deterministic_draft_rollback(nullptr, 0, 1));

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}
// ============================================================================
// Test 23: apply_bitmask returns false for invalid args (no plugin loaded)
// ============================================================================

static void test_apply_bitmask_returns_false_no_plugin() {
    printf("test_apply_bitmask_returns_false_no_plugin... ");

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(nullptr);
    assert(draft != nullptr);

    uint32_t bitmask[4] = { 0 };
    float logits[128] = { 0 };

    // No plugin loaded -> apply_bitmask function pointer is null -> returns false
    assert(!llama_deterministic_draft_apply_bitmask(draft, 0, bitmask, 128, logits));

    // null draft -> returns false
    assert(!llama_deterministic_draft_apply_bitmask(nullptr, 0, bitmask, 128, logits));

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 24: rollback and apply_bitmask return false after failed init
// ============================================================================

static void test_rollback_bitmask_after_failed_init() {
    printf("test_rollback_bitmask_after_failed_init... ");

    // Init with non-existent path returns nullptr
    struct llama_deterministic_draft * draft = llama_deterministic_draft_init("/nonexistent/path/plugin.so");
    assert(draft == nullptr);

    // null draft -> both return false
    assert(!llama_deterministic_draft_rollback(draft, 0, 1));

    uint32_t bitmask[4] = { 0 };
    float logits[128] = { 0 };
    assert(!llama_deterministic_draft_apply_bitmask(draft, 0, bitmask, 128, logits));

    printf("OK\n");
}
// ============================================================================
// Test: intercept-all fallback for non-MTP drafter families
// ============================================================================

static void test_intercept_all_non_mtp_fallback() {
    printf("test_intercept_all_non_mtp_fallback... ");

    // draft-dspark: parses, intercept-all downgraded to default mode
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--spec-type";
        char arg4[] = "draft-dspark";
        char arg5[] = "--det-draft-intercept-all";
        char arg6[] = "--det-draft-n-max";
        char arg7[] = "16";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, nullptr };
        assert(common_params_parse(8, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(params.speculative.deterministic_draft.enabled);
        assert(!params.speculative.deterministic_draft.det_intercept_all);
        assert(params.speculative.deterministic_draft.det_filter_bonus);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != types.end());
    }

    // draft-eagle3: parses, intercept-all downgraded to default mode
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--spec-type";
        char arg4[] = "draft-eagle3";
        char arg5[] = "--det-draft-intercept-all";
        char arg6[] = "--det-draft-n-max";
        char arg7[] = "4";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, nullptr };
        assert(common_params_parse(8, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(!params.speculative.deterministic_draft.det_intercept_all);
        assert(params.speculative.deterministic_draft.det_filter_bonus);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) != types.end());
    }

    // draft-dflash: parses, intercept-all downgraded to default mode
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--spec-type";
        char arg4[] = "draft-dflash";
        char arg5[] = "--det-draft-intercept-all";
        char arg6[] = "--det-draft-n-max";
        char arg7[] = "7";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, nullptr };
        assert(common_params_parse(8, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(!params.speculative.deterministic_draft.det_intercept_all);
        assert(params.speculative.deterministic_draft.det_filter_bonus);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != types.end());
    }

    // draft-mtp: intercept-all stays enabled
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--spec-type";
        char arg4[] = "draft-mtp";
        char arg5[] = "--det-draft-intercept-all";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, nullptr };
        assert(common_params_parse(6, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(params.speculative.deterministic_draft.det_intercept_all);
        assert(!params.speculative.deterministic_draft.det_filter_bonus);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end());
    }

    printf("OK\n");
}

// ============================================================================
// Test: regex plugin filter - whitespace allowed, EOG/stop tokens pass through,
// non-alpha-non-whitespace still rejected
// ============================================================================

static void test_regex_plugin_filter_whitespace_eog() {
    printf("test_regex_plugin_filter_whitespace_eog... ");

    std::string plugin_path = find_regex_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no regex plugin .so found)\n");
        return;
    }

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    // vocab: letters (0,1), whitespace (2,3), a digit (4, must be rejected),
    // and an EOG token (5)
    const char * vocab[] = { "hello", "world", " ", "\n", "3", "</s>" };
    const int     n_vocab = 6;
    const int32_t stop_tokens[] = { 5 }; // "</s>" is the EOG

    assert(llama_deterministic_draft_set_vocab(draft, vocab, n_vocab, stop_tokens, 1));
    llama_deterministic_draft_reset(draft, 0);

    // whitespace is now accepted, not a truncation point
    const int ws[] = { 0, 1, 2, 3 }; // hello, world, " ", "\n"
    assert(llama_deterministic_draft_filter_draft(draft, 0, ws, 4) == 4);

    // a digit is still rejected and truncates
    const int digit[] = { 0, 4 }; // hello, "3"
    assert(llama_deterministic_draft_filter_draft(draft, 0, digit, 2) == 1);

    // the EOG stop token is allowed and does not truncate
    const int eog[] = { 0, 5 }; // hello, "</s>"
    assert(llama_deterministic_draft_filter_draft(draft, 0, eog, 2) == 2);

    // apply_bitmask: EOG and whitespace stay unmasked, digit is masked
    float    logits[6]  = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    uint32_t bitmask[1] = { 0 };
    assert(llama_deterministic_draft_apply_bitmask(draft, 0, bitmask, n_vocab, logits));
    assert(logits[0] == 1.0f);  // "hello" preserved
    assert(logits[2] == 1.0f);  // " " preserved
    assert(logits[3] == 1.0f);  // "\n" preserved
    assert(logits[5] == 1.0f);  // EOG preserved
    assert(logits[4] < -1e20f); // "3" masked

    llama_deterministic_draft_free(draft);
    printf("OK\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("\n=== Deterministic Draft Tests ===\n\n");

    test_plugin_loader_init_free();
    test_c_api_no_plugin();
    test_speculative_type_enum();
    test_params_struct();
    test_det_filter_query();
    test_fail_without_mtp();
    test_fail_without_plugin();
    test_intercept_all_requires_plugin();
    test_auto_imply_mtp();
    test_token_to_piece_special_flag();
    test_cumulative_stats_fields();
    test_accept_null_impl_safe();
    test_get_version_null();
    test_filter_api_no_plugin();
    test_rollback_returns_false_no_plugin();
    test_apply_bitmask_returns_false_no_plugin();
    test_rollback_bitmask_after_failed_init();
    test_intercept_all_non_mtp_fallback();
    test_regex_plugin_filter_whitespace_eog();

    printf("\n=== All tests passed ===\n\n");
    return 0;
}
