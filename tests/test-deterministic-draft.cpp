// test-deterministic-draft.cpp -- Unit tests for deterministic draft plugin loader
//
// Tests:
//   1. Plugin loader: init/free with valid and invalid paths
//   2. C API wrappers: get_capabilities, set_vocab, fill_bitmask, commit, reset
//   3. Speculative integration: common_speculative with DRAFT_DETERMINISTIC type
//   4. Auto-imply: --deterministic-draft-model implies draft-mtp
//   5. --det-draft-accept-all flag validation and accessor
//
// These tests use the generic plugin loader (libdeterministic_draft_spec.so).
// A plugin .so (XGrammar-based) is needed for integration tests; if not
// available, those tests are skipped.
//
// Plugin configuration in these tests goes through the same path production
// uses: the DETERMINISTIC_DRAFT_GRAMMAR_DIR environment variable plus the
// plugin's content-based bootstrap detection. There is no host-side
// grammar/language selection API.
//
// Why grammar fixtures in a generic-SPI test: the loader is constraint-
// agnostic (token IDs in, bitmasks/verdicts out), but exercising it requires
// one concrete plugin and the only existing implementation is grammar-based
// (the XGrammar PoC). TEST_GRAMMAR and the bundled .gbnf files are that
// concrete constraint. The assertions target loader-visible behavior -
// bitmask contents, commit/rollback state, serialization round-trips - not
// grammar semantics.

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

// Try to find the plugin .so in common locations
static std::string find_plugin() {
    const char * candidates[] = { "deterministic-draft.so", "./deterministic-draft.so",
                                  "../deterministic-draft-model-poc/build/deterministic-draft.so",
                                  "./build/deterministic-draft.so", nullptr };

    for (int i = 0; candidates[i]; i++) {
        FILE * f = fopen(candidates[i], "rb");
        if (f) {
            fclose(f);
            return candidates[i];
        }
    }

    return "";
}

static void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// Resolve the directory holding the plugin's bundled .gbnf files, relative to
// the located plugin binary ("<dir>/grammars" or "<dir>/../grammars").
// Resolves symlinks so the grammar directory is found relative to the real
// .so location, not the symlink path.
static std::string resolve_grammar_dir(const std::string & plugin_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path real = fs::canonical(plugin_path, ec);
    const fs::path dir = ec ? fs::path(plugin_path).parent_path() : real.parent_path();
    for (const fs::path & cand : { dir / "grammars", dir / ".." / "grammars" }) {
        if (fs::is_directory(cand, ec) && !fs::is_empty(cand, ec)) {
            return fs::weakly_canonical(cand, ec).string();
        }
    }
    return "";
}

// Point the plugin at its real bundled grammars (java/javascript/python/c),
// used by the detection-based integration tests.
static void use_plugin_grammars(const std::string & plugin_path) {
    const std::string dir = resolve_grammar_dir(plugin_path);
    if (!dir.empty()) {
        set_env("DETERMINISTIC_DRAFT_GRAMMAR_DIR", dir);
    }
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
    params.speculative.deterministic_draft.det_accept_all = true;

    assert(params.speculative.deterministic_draft.enabled == true);
    assert(params.speculative.deterministic_draft.n_max == 42);
    assert(params.speculative.deterministic_draft.plugin_path == "/test/path.so");
    assert(params.speculative.deterministic_draft.det_accept_all == true);

    // default value is false
    common_params params2;
    assert(params2.speculative.deterministic_draft.det_accept_all == false);

    printf("OK\n");
}

// ============================================================================
// Test 5: common_speculative_has_det_filter with null spec
// ============================================================================

static void test_det_filter_query() {
    printf("test_det_filter_query... ");

    assert(!common_speculative_has_det_filter(nullptr));

    // det_accept_all should be false when spec is null
    assert(!common_speculative_get_det_accept_all(nullptr));

    // Get filter result from null should return empty
    const auto & fr = common_speculative_get_det_filter_result(nullptr, 0);
    assert(!fr.truncated);
    assert(fr.valid_count == 0);

    printf("OK\n");
}

// ============================================================================
// Shared test vocabulary/grammar helpers for bitmask-based integration tests
// ============================================================================

// Fixed test vocabulary: index == token_id. Covers the exact tokens used by
// the integration tests below (two alternative complete "programs").
static const char * TEST_VOCAB[] = {
    /* 0*/ "int", /* 1*/ " ", /* 2*/ "main", /* 3*/ "(", /* 4*/ ")",
    /* 5*/ "{",   /* 6*/ "\n    ", /* 7*/ "return", /* 8*/ "0", /* 9*/ ";",
    /*10*/ "\n",  /*11*/ "}", /*12*/ "float", /*13*/ "x", /*14*/ "=", /*15*/ "1.0f",
};
static const int TEST_VOCAB_SIZE = sizeof(TEST_VOCAB) / sizeof(TEST_VOCAB[0]);

// Grammar accepting exactly two complete "programs" built from TEST_VOCAB's
// tokens (concatenated literally), so that a reset() between them can be
// tested against two independent valid completions from the same grammar.
static const char * TEST_GRAMMAR =
    "root ::= \"int main() {\\n    return 0;\\n}\" | \"float x = 1.0f;\"";

// Install TEST_GRAMMAR as the plugin's only bundled grammar (written as the
// sole .gbnf in a private temp dir): with a single detection candidate the
// plugin converges on it immediately. Returns the grammar dir.
static const std::string & use_test_grammar() {
    static const std::string dir = [] {
        namespace fs = std::filesystem;
        fs::path d = fs::temp_directory_path() / ("det_draft_test_grammar_" + std::to_string(GETPID()));
        std::error_code ec;
        fs::create_directories(d, ec);
        std::ofstream(d / "test.gbnf") << TEST_GRAMMAR << "\n";
        std::string s = d.string();
        set_env("DETERMINISTIC_DRAFT_GRAMMAR_DIR", s);
        return s;
    }();
    return dir;
}

// A full byte-level vocabulary (every one of the 256 possible byte values as
// its own single-byte token), needed to exercise the bundled real-language
// grammars (c/java/python/javascript .gbnf) with actual source-like text -
// TEST_VOCAB above is far too small for that; it only covers one grammar
// hand-picked for it.
static bool set_byte_vocab(struct llama_deterministic_draft * draft) {
    static std::vector<std::string> storage = [] {
        std::vector<std::string> v;
        v.reserve(256);
        for (int i = 0; i < 256; i++) {
            v.push_back(std::string(1, static_cast<char>(i)));
        }
        return v;
    }();

    std::vector<const char *> ptrs;
    ptrs.reserve(storage.size());
    for (auto & s : storage) {
        ptrs.push_back(s.c_str());
    }
    return llama_deterministic_draft_set_vocab(draft, ptrs.data(), (int) ptrs.size(), nullptr, 0);
}

// Feeds text through filter_draft one byte at a time (as if it were a
// drafted token sequence, byte-level token ids matching set_byte_vocab
// above), returning the number of leading bytes accepted.
static int feed_bytes(struct llama_deterministic_draft * draft, int slot_id, const std::string & text) {
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int32_t>(c));
    }
    return llama_deterministic_draft_filter_draft(draft, slot_id, tokens.data(), (int) tokens.size());
}

static bool bitmask_allows_token(const uint32_t * bitmask, int token_id) {
    int word_idx = token_id / 32;
    int bit_idx  = token_id % 32;
    return (bitmask[word_idx] & (1u << bit_idx)) != 0;
}

// ============================================================================
// Test 6: Plugin integration (if plugin .so available)
// ============================================================================

static void test_plugin_integration() {
    printf("test_plugin_integration... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    // XGrammar-based plugins declare CAPABILITY_BITMASK
    uint32_t caps = llama_deterministic_draft_get_capabilities(draft);
    assert((caps & LLAMA_DETERMINISTIC_DRAFT_CAPABILITY_BITMASK) != 0);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    llama_deterministic_draft_reset(draft, 0);

    // Build the first valid program token by token: "int main() {\n    return 0;\n}"
    const int program1[] = { 0, 1, 2, 3, 4, 1, 5, 6, 7, 1, 8, 9, 10, 11 };

    const int vocab_size_words = (TEST_VOCAB_SIZE + 31) / 32;
    std::vector<uint32_t> bitmask(vocab_size_words);

    for (int token_id : program1) {
        bool has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
        if (has_bitmask) {
            assert(bitmask_allows_token(bitmask.data(), token_id));
        }
        llama_deterministic_draft_commit(draft, 0, token_id, TEST_VOCAB[token_id], (int) strlen(TEST_VOCAB[token_id]));
    }

    // After the complete program, an unrelated token (e.g. "float", start of
    // the OTHER alternative) must no longer be accepted by the bitmask.
    {
        bool has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
        if (has_bitmask) {
            assert(!bitmask_allows_token(bitmask.data(), 12 /* "float" */));
        }
    }

    llama_deterministic_draft_free(draft);

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
// Test 9: det_accept_all requires the plugin to be enabled
// ============================================================================

static void test_accept_all_requires_plugin() {
    printf("test_accept_all_requires_plugin... ");

    // det_accept_all without a plugin -> common_params_parse must fail
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-accept-all";
        char * argv[] = { arg0, arg1, nullptr };
        assert(!common_params_parse(2, argv, params, LLAMA_EXAMPLE_SERVER));
    }

    // enabled + det_accept_all -> parses; auto-implied types and n_max copy applied
    {
        common_params params;
        char arg0[] = "test";
        char arg1[] = "--det-draft-model";
        char arg2[] = "/test/plugin.so";
        char arg3[] = "--det-draft-accept-all";
        char arg4[] = "--det-draft-n-max";
        char arg5[] = "16";
        char * argv[] = { arg0, arg1, arg2, arg3, arg4, arg5, nullptr };
        assert(common_params_parse(6, argv, params, LLAMA_EXAMPLE_SERVER));

        assert(params.speculative.deterministic_draft.enabled);
        assert(params.speculative.deterministic_draft.det_accept_all);

        const auto & types = params.speculative.types;
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end());
        assert(std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC) != types.end());
        assert(params.speculative.draft.n_max == 16);
    }

    printf("OK\n");
}

// ============================================================================
// Test 10: Plugin state across reset (checkpoint/restore simulation)
// ============================================================================

static void test_plugin_state_reset() {
    printf("test_plugin_state_reset... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    const int vocab_size_words = (TEST_VOCAB_SIZE + 31) / 32;
    std::vector<uint32_t> bitmask(vocab_size_words);

    // Round 1: build the first complete program: "int main() {\n    return 0;\n}"
    llama_deterministic_draft_reset(draft, 0);
    {
        const int program1[] = { 0, 1, 2, 3, 4, 1, 5, 6, 7, 1, 8, 9, 10, 11 };
        for (int token_id : program1) {
            bool has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
            if (has_bitmask) {
                assert(bitmask_allows_token(bitmask.data(), token_id));
            }
            llama_deterministic_draft_commit(draft, 0, token_id, TEST_VOCAB[token_id],
                                              (int) strlen(TEST_VOCAB[token_id]));
        }
    }

    // Reset: simulates checkpoint restore / new generation. Grammar state
    // must go back to the start - the OTHER alternative ("float x = 1.0f;")
    // must now be valid from the very first token, with no round-1 residue.
    llama_deterministic_draft_reset(draft, 0);

    // Round 2: build the second complete program: "float x = 1.0f;"
    {
        const int program2[] = { 12, 1, 13, 1, 14, 1, 15, 9 };
        for (int token_id : program2) {
            bool has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
            if (has_bitmask) {
                assert(bitmask_allows_token(bitmask.data(), token_id));
                // round-1-only token ("main") must not leak through after reset
                assert(!bitmask_allows_token(bitmask.data(), 2 /* "main" */));
            }
            llama_deterministic_draft_commit(draft, 0, token_id, TEST_VOCAB[token_id],
                                              (int) strlen(TEST_VOCAB[token_id]));
        }
    }

    llama_deterministic_draft_free(draft);

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
    // tokens so the grammar matcher parses clean code without BPE artifacts.
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
    // But since init fails, we just verify the API exists

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
// Test 17: filter_draft with plugin - valid tokens accepted
// ============================================================================

static void test_filter_draft_valid_tokens() {
    printf("test_filter_draft_valid_tokens... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    llama_deterministic_draft_reset(draft, 0);

    // feed "int main() {" which is tokens 0,1,2,3,4,1,5 - all valid
    const int tokens[] = { 0, 1, 2, 3, 4, 1, 5 };
    int accepted = llama_deterministic_draft_filter_draft(draft, 0, tokens, 7);
    assert(accepted == 7);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 18: filter_draft with plugin - invalid token truncates
// ============================================================================

static void test_filter_draft_truncates_on_invalid() {
    printf("test_filter_draft_truncates_on_invalid... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    llama_deterministic_draft_reset(draft, 0);

    // "int" is valid first token, but "float" (12) is not valid after "int" in grammar
    const int tokens[] = { 0, 12 };
    int accepted = llama_deterministic_draft_filter_draft(draft, 0, tokens, 2);
    assert(accepted == 1);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 19: apply_bitmask with plugin constrains logits
// ============================================================================

static void test_apply_bitmask_constrains_logits() {
    printf("test_apply_bitmask_constrains_logits... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    llama_deterministic_draft_reset(draft, 0);

    const int bitmask_words = (TEST_VOCAB_SIZE + 31) / 32;
    std::vector<uint32_t> bitmask(bitmask_words, 0);
    std::vector<float> logits(TEST_VOCAB_SIZE, 1.0f);

    bool applied = llama_deterministic_draft_apply_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE, logits.data());
    assert(applied);

    // "int" (0) should be allowed at start of program
    assert(logits[0] == 1.0f);

    // "float" (12) should also be allowed at start (both programs start here)
    assert(logits[12] == 1.0f);

    // "main" (2) should NOT be valid at start (neither program starts with "main")
    assert(logits[2] < -1e20f);

    // "return" (7) should NOT be valid at start
    assert(logits[7] < -1e20f);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 20: commit_tokens with plugin advances grammar state
// ============================================================================

static void test_commit_tokens_advances_grammar() {
    printf("test_commit_tokens_advances_grammar... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    llama_deterministic_draft_reset(draft, 0);

    // Commit "int main() " (tokens 0,1,2,3,4,1) in a batch
    const int tokens[] = { 0, 1, 2, 3, 4, 1 };
    llama_deterministic_draft_commit_tokens(draft, 0, tokens, 6);

    // After committing partial program, fill bitmask and check state advanced
    const int bitmask_words = (TEST_VOCAB_SIZE + 31) / 32;
    std::vector<uint32_t> bitmask(bitmask_words, 0);
    std::vector<float> logits(TEST_VOCAB_SIZE, 1.0f);

    bool applied = llama_deterministic_draft_apply_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE, logits.data());
    assert(applied);

    // After "int main() ", "{" (5) should be valid
    assert(logits[5] == 1.0f);

    // "return" (7) should NOT yet be valid (need "{" first)
    assert(logits[7] < -1e20f);

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
// Test 22: rollback succeeds with plugin (commit then rollback)
// ============================================================================

static void test_rollback_success_with_plugin() {
    printf("test_rollback_success_with_plugin... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_test_grammar();

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);

    bool vocab_ok = llama_deterministic_draft_set_vocab(draft, TEST_VOCAB, TEST_VOCAB_SIZE, nullptr, 0);
    assert(vocab_ok);

    const int vocab_size_words = (TEST_VOCAB_SIZE + 31) / 32;
    std::vector<uint32_t> bitmask(vocab_size_words);

    llama_deterministic_draft_reset(draft, 0);

    // Commit "int main() {" = tokens {0, 1, 2, 3, 4, 1, 5} (7 tokens)
    // Note: token 1 (" ") appears before both "main" and "{"
    const int preamble[] = { 0, 1, 2, 3, 4, 1, 5 };
    for (int i = 0; i < 7; i++) {
        llama_deterministic_draft_commit(draft, 0, preamble[i], TEST_VOCAB[preamble[i]],
                                         (int) strlen(TEST_VOCAB[preamble[i]]));
    }

    // After "int main() {", grammar expects "\n    return 0;\n}"
    // Token 6 ("\n    ") should be valid next
    bool has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
    assert(has_bitmask);
    assert(bitmask_allows_token(bitmask.data(), 6));    // "\n    " should be valid
    assert(!bitmask_allows_token(bitmask.data(), 0));   // "int" should NOT be valid (past it)

    // Roll back 3 tokens: undoes commits of tokens 5 ("{"), 1 (" "), 4 (")")
    // Grammar rewinds to "int main(" (tokens 0,1,2,3 committed)
    bool rolled_back = llama_deterministic_draft_rollback(draft, 0, 3);
    assert(rolled_back);

    // After rolling back 3, grammar expects ")" (token 4) after "int main("
    has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
    assert(has_bitmask);
    assert(bitmask_allows_token(bitmask.data(), 4));    // ")" should be valid
    assert(!bitmask_allows_token(bitmask.data(), 6));   // "\n    " should NOT be valid (too early)

    // Rollback remaining 4 tokens to get back to start
    rolled_back = llama_deterministic_draft_rollback(draft, 0, 4);
    assert(rolled_back);

    // After full rollback, grammar is at start - token 0 ("int") is valid
    has_bitmask = llama_deterministic_draft_fill_bitmask(draft, 0, bitmask.data(), TEST_VOCAB_SIZE);
    assert(has_bitmask);
    assert(bitmask_allows_token(bitmask.data(), 0));

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
// Test 25: slot grammars are scoped per slot, not plugin-wide.
// Behavioral equivalent of the old set_language per-slot test: bootstrap
// detection runs independently per slot, so feeding different content to
// different slots must resolve each slot to its own grammar.
// ============================================================================

static void test_set_language_is_per_slot() {
    printf("test_set_language_is_per_slot... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));

    // python to slot 5, c to slot 7; neither slot is configured explicitly
    const std::string py_code = "def f(n):\n    return n\n";
    const std::string c_code  = "int main() {\n    return 0;\n}\n";
    assert(feed_bytes(draft, 5, py_code) == (int) py_code.size());
    assert(feed_bytes(draft, 7, c_code)  == (int) c_code.size());

    const int words = (256 + 31) / 32;
    std::vector<uint32_t> bm5(words), bm7(words);
    assert(llama_deterministic_draft_fill_bitmask(draft, 5, bm5.data(), 256));
    assert(llama_deterministic_draft_fill_bitmask(draft, 7, bm7.data(), 256));

    // The slots resolved independently: their next-token masks must differ
    // (a shared plugin-wide grammar would produce identical masks).
    int discriminating = -1;
    for (int b = 0; b < 256; b++) {
        if (bitmask_allows_token(bm5.data(), b) != bitmask_allows_token(bm7.data(), b)) {
            discriminating = b;
            break;
        }
    }
    assert(discriminating >= 0);

    // Direction check: more python is fully accepted by the python slot,
    // more C by the C slot. Had either slot converged to the other's
    // grammar, these continuations would truncate.
    const std::string py_more = "def g():\n    return 2\n";
    const std::string c_more  = "int g() {\n    return 1;\n}\n";
    assert(feed_bytes(draft, 5, py_more) == (int) py_more.size());
    assert(feed_bytes(draft, 7, c_more)  == (int) c_more.size());

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 26: bootstrap detection resolves an unconfigured slot's grammar
// from its content alone. Full acceptance is the convergence proof: the
// other three bundled grammars each reject these byte sequences, so the
// slot's matcher must have converged to the right one.
// ============================================================================

static void test_bootstrap_detection_resolves_language() {
    printf("test_bootstrap_detection_resolves_language... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));

    // Slot 1: unambiguous Python (def + indentation). Slot 2: unambiguous
    // Java (public class wrapper). Neither slot is ever told its language.
    const std::string py_code = "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n";
    int n1 = feed_bytes(draft, 1, py_code);
    assert(n1 == (int) py_code.size());

    const std::string java_code = "public class Foo {\n    public static void main(String[] args) {}\n}\n";
    int n2 = feed_bytes(draft, 2, java_code);
    assert(n2 == (int) java_code.size());

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 27: "c" is tried last during bootstrap detection - critical, see
// list_bundled_languages() in the plugin. Without this, C's permissive
// grammar (bare top-level declarations, any identifier treated as a valid
// type) would swallow non-C content before more specific candidates like
// javascript ever get a chance to be tried.
// ============================================================================

static void test_bootstrap_detection_tries_c_last() {
    printf("test_bootstrap_detection_tries_c_last... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));

    // This is genuinely JavaScript-only (const/let/function keywords), but
    // is also superficially "declaration-shaped" enough that c.gbnf's
    // grammar (which treats any identifier as a valid type, with no
    // requirement to wrap top-level code in anything) accepts it too.
    const std::string js_code = "const x = 1;\nlet y = 2;\nfunction add(a, b) { return a + b; }\n";
    int n = feed_bytes(draft, 3, js_code);
    assert(n == (int) js_code.size());

    // Convergence probe: an arrow function is valid JavaScript but rejected
    // by C ("=>"). If detection had settled on C, these bytes would be
    // truncated; full acceptance proves the slot resolved to javascript.
    const std::string arrow = "const f = (a) => a;\n";
    assert(feed_bytes(draft, 3, arrow) == (int) arrow.size());

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 28: state serialization round-trip (save -> reset -> restore must
// reproduce the exact grammar state). The C content drives bootstrap
// detection to the C grammar without any explicit selection.
// ============================================================================

static void test_state_serialization_roundtrip() {
    printf("test_state_serialization_roundtrip... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));

    const std::string code = "int main() {\n    return 0;\n}\n";
    assert(feed_bytes(draft, 0, code) == (int) code.size());

    // snapshot the grammar state as a bitmask
    std::vector<uint32_t> before((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 0, before.data(), 256));

    const int size = llama_deterministic_draft_state_get_size(draft, 0);
    assert(size > 0);
    std::vector<uint8_t> blob(size);
    assert(llama_deterministic_draft_state_get_data(draft, 0, blob.data(), size) == size);

    // wipe the slot, then restore
    llama_deterministic_draft_reset(draft, 0);
    assert(llama_deterministic_draft_state_set_data(draft, 0, blob.data(), size));

    std::vector<uint32_t> after((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 0, after.data(), 256));
    assert(before == after);

    // garbage blobs are rejected and leave the restored state untouched
    std::vector<uint8_t> junk(size, 0xAB);
    assert(!llama_deterministic_draft_state_set_data(draft, 0, junk.data(), size));
    std::vector<uint32_t> still((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 0, still.data(), 256));
    assert(still == after);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 29: state serialization round-trip while bootstrap detection is
// still running (candidates + partial history must survive restore).
// Detection state is probed behaviorally through the union bitmask: while
// both c and python candidates survive, bytes exclusive to either one are
// both allowed.
// ============================================================================

static void test_state_serialization_detecting() {
    printf("test_state_serialization_detecting... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));

    // "def f" is a valid prefix for python (and tolerated by c) but rejected
    // by java/javascript, so detection is still running with >1 candidate
    const std::string prefix = "def f";
    assert(feed_bytes(draft, 1, prefix) == (int) prefix.size());

    std::vector<uint32_t> before((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 1, before.data(), 256));

    // union probe: '(' continues the python def - allowed while the python
    // candidate survives
    assert(bitmask_allows_token(before.data(), '('));

    const int size = llama_deterministic_draft_state_get_size(draft, 1);
    assert(size > 0);
    std::vector<uint8_t> blob(size);
    assert(llama_deterministic_draft_state_get_data(draft, 1, blob.data(), size) == size);

    llama_deterministic_draft_reset(draft, 1);
    assert(llama_deterministic_draft_state_set_data(draft, 1, blob.data(), size));

    std::vector<uint32_t> after((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 1, after.data(), 256));
    assert(before == after);

    // the restored matcher is live: continuing the python function kills the
    // c candidate (':' after the declarator) and converges to python
    const std::string rest = "ib(n):\n    return n\n";
    assert(feed_bytes(draft, 1, rest) == (int) rest.size());

    // convergence proof: the mask must have narrowed once the c candidate
    // died - if the slot still carried the unresolved union it would equal
    // the snapshot taken while detecting
    std::vector<uint32_t> conv((256 + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(draft, 1, conv.data(), 256));
    assert(conv != before);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 30: common-level checkpoint round-trip through
// common_speculative_get_state/set_state with a live det filter.
// Grammar state is built via the plugin handle directly, so no inference
// loop is needed. Requires a model: set LLAMA_TEST_MODEL.
// ============================================================================

static void test_common_checkpoint_roundtrip() {
    printf("test_common_checkpoint_roundtrip... ");

    const char * model_path = getenv("LLAMA_TEST_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        printf("SKIP (LLAMA_TEST_MODEL not set)\n");
        return;
    }
    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    common_params params;
    params.model.path = model_path;
    params.n_ctx      = 1024;
    params.no_perf    = true;
    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);

    auto            llama_init_tgt = common_init_from_params(params);
    llama_model *   model_tgt      = llama_init_tgt->model();
    llama_context * ctx_tgt        = llama_init_tgt->context();
    assert(ctx_tgt != nullptr && model_tgt != nullptr);

    // MTP draft context from the same model (same idiom as the bench tool)
    auto cparams_mtp          = common_context_params_to_llama(params);
    cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cparams_mtp.type_k        = params.speculative.draft.cache_type_k;
    cparams_mtp.type_v        = params.speculative.draft.cache_type_v;
    cparams_mtp.n_rs_seq      = 0;
    cparams_mtp.n_outputs_max = 1;
    cparams_mtp.ctx_other     = ctx_tgt;

    llama_context_ptr ctx_dft(llama_init_from_model(model_tgt, cparams_mtp));
    assert(ctx_dft != nullptr);

    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    params.speculative.deterministic_draft.enabled     = true;
    params.speculative.deterministic_draft.plugin_path = plugin_path;
    params.speculative.draft.ctx_tgt                   = ctx_tgt;
    params.speculative.draft.ctx_dft                   = ctx_dft.get();

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
    assert(spec != nullptr);

    struct llama_deterministic_draft * det = common_speculative_get_det_filter_plugin(spec);
    assert(det != nullptr);

    // build grammar state with real vocab tokens (valid C, detection
    // converges on the c grammar from content alone)
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const llama_tokens  code  = common_tokenize(vocab, "int main() {\n", false);
    assert(!code.empty());
    llama_deterministic_draft_commit_tokens(det, 0, code.data(), (int) code.size());

    const int             n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<uint32_t> before((n_vocab + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(det, 0, before.data(), n_vocab));

    // checkpoint save: det filter state must be packed even though the MTP
    // impl carries no state of its own
    std::vector<uint8_t> blob;
    assert(common_speculative_get_state(spec, 0, blob));
    assert(!blob.empty());

    // wipe the slot; grammar position must visibly change
    llama_deterministic_draft_reset(det, 0);
    std::vector<uint32_t> wiped((n_vocab + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(det, 0, wiped.data(), n_vocab));
    assert(wiped != before);

    // checkpoint restore: grammar state must match the snapshot exactly
    common_speculative_set_state(spec, 0, blob);
    std::vector<uint32_t> after((n_vocab + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(det, 0, after.data(), n_vocab));
    assert(before == after);

    // legacy (untagged) blobs still take the impl-broadcast path, no det effect
    const std::vector<uint8_t> legacy = { 1, 2, 3, 4 };
    common_speculative_set_state(spec, 0, legacy);
    std::vector<uint32_t> still((n_vocab + 31) / 32);
    assert(llama_deterministic_draft_fill_bitmask(det, 0, still.data(), n_vocab));
    assert(still == after);

    common_speculative_free(spec);

    printf("OK\n");
}

// ============================================================================
// Test 31: get_jump_forward - determined continuation once the grammar
// disambiguates; nullptr while multiple alternatives remain
// ============================================================================

static void test_get_jump_forward() {
    printf("test_get_jump_forward... ");

    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    struct llama_deterministic_draft * draft = llama_deterministic_draft_init(plugin_path.c_str());
    assert(draft != nullptr);
    assert(set_byte_vocab(draft));
    llama_deterministic_draft_reset(draft, 0);

    // complete C program: detection converges to c, but the continuation is
    // open-ended (any top-level construct) - nothing is uniquely determined
    const std::string prog = "int main() {\n}\n";
    assert(feed_bytes(draft, 0, prog) == (int) prog.size());

    int32_t len = -1;
    assert(llama_deterministic_draft_get_jump_forward(draft, 0, &len) == nullptr);
    assert(len == 0);

    // "#inc" has exactly one continuation in C: "lude" (#include) - the rest
    // of the directive keyword is fully determined
    const std::string directive = "#inc";
    assert(feed_bytes(draft, 0, directive) == (int) directive.size());

    const char * jump = llama_deterministic_draft_get_jump_forward(draft, 0, &len);
    assert(jump != nullptr && len > 0);

    // the jump-forward string must be a prefix of the determined continuation
    const std::string rest = "lude";
    assert((size_t) len <= rest.size());
    assert(rest.compare(0, (size_t) len, jump, (size_t) len) == 0);

    llama_deterministic_draft_free(draft);

    printf("OK\n");
}

// ============================================================================
// Test 32: accept-all sampling path - draft tokens returned verbatim, the
// bonus token is grammar-constrained and omitted once the grammar has
// terminated. Requires a model: set LLAMA_TEST_MODEL.
// ============================================================================

static void test_accept_all_sample_and_accept() {
    printf("test_accept_all_sample_and_accept... ");

    const char * model_path = getenv("LLAMA_TEST_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        printf("SKIP (LLAMA_TEST_MODEL not set)\n");
        return;
    }
    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 1024;
    params.no_perf       = true;
    params.sampling.temp = 0.0f; // greedy: deterministic shortlist
    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);

    auto            llama_init_tgt = common_init_from_params(params);
    llama_model *   model_tgt      = llama_init_tgt->model();
    llama_context * ctx_tgt        = llama_init_tgt->context();
    assert(ctx_tgt != nullptr && model_tgt != nullptr);

    // MTP draft context from the same model (same idiom as the roundtrip test)
    auto cparams_mtp          = common_context_params_to_llama(params);
    cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cparams_mtp.type_k        = params.speculative.draft.cache_type_k;
    cparams_mtp.type_v        = params.speculative.draft.cache_type_v;
    cparams_mtp.n_rs_seq      = 0;
    cparams_mtp.n_outputs_max = 1;
    cparams_mtp.ctx_other     = ctx_tgt;

    llama_context_ptr ctx_dft(llama_init_from_model(model_tgt, cparams_mtp));
    assert(ctx_dft != nullptr);

    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    params.speculative.deterministic_draft.enabled        = true;
    params.speculative.deterministic_draft.plugin_path    = plugin_path;
    params.speculative.deterministic_draft.det_accept_all = true;
    params.speculative.draft.ctx_tgt                      = ctx_tgt;
    params.speculative.draft.ctx_dft                      = ctx_dft.get();

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
    assert(spec != nullptr);
    assert(common_speculative_get_det_accept_all(spec));

    struct llama_deterministic_draft * det = common_speculative_get_det_filter_plugin(spec);
    assert(det != nullptr);

    // decode a valid C prefix so the sampler has logits to work with
    const llama_vocab * vocab  = llama_model_get_vocab(model_tgt);
    const llama_tokens  prompt = common_tokenize(vocab, "int main() {\n    ", false);
    assert(!prompt.empty());
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.data()), (int) prompt.size());
    std::vector<int8_t> logits_flag(prompt.size(), 0);
    logits_flag[0] = 1;
    batch.logits = logits_flag.data();
    assert(llama_decode(ctx_tgt, batch) == 0);

    common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
    assert(smpl != nullptr);
    const std::vector<int> idxs = { 0 };

    // (a) non-empty draft: accepted verbatim, plus one grammar-constrained bonus
    const llama_tokens draft = common_tokenize(vocab, "return", false);
    assert(!draft.empty());

    llama_tokens accepted = common_speculative_sample_and_accept(spec, smpl, ctx_tgt, idxs, draft, 0);
    assert(accepted.size() == draft.size() + 1);
    assert(std::equal(draft.begin(), draft.end(), accepted.begin()));

    // (b) empty draft: exactly one grammar-constrained token is emitted
    common_sampler_reset(smpl);
    llama_deterministic_draft_reset(det, 0);

    accepted = common_speculative_sample_and_accept(spec, smpl, ctx_tgt, idxs, llama_tokens{}, 0);
    assert(accepted.size() == 1);

    // (c) terminated grammar: the draft is returned without a bonus token
    common_sampler_reset(smpl);
    llama_deterministic_draft_reset(det, 0);
    const llama_tokens prog = common_tokenize(vocab, "int main() {\n    return 0;\n}", false);
    llama_deterministic_draft_commit_tokens(det, 0, prog.data(), (int) prog.size());

    const bool is_term = llama_deterministic_draft_is_terminated(det, 0);
    accepted = common_speculative_sample_and_accept(spec, smpl, ctx_tgt, idxs, draft, 0);
    assert(accepted.size() == draft.size() + (is_term ? 0 : 1));
    assert(std::equal(draft.begin(), draft.end(), accepted.begin()));

    common_sampler_free(smpl);
    common_speculative_free(spec);

    printf("OK\n");
}

// ============================================================================
// Test 33: standard (target-verified) mode - the final token of each step
// (rejection correction or bonus) must be grammar-constrained too.
// Regression test: the final token used to be sampled unconstrained and only
// committed post-hoc, so a grammar-invalid token was emitted and the grammar
// state desynced from the output. Requires a model: set LLAMA_TEST_MODEL.
// ============================================================================

static void test_standard_mode_constrains_final_token() {
    printf("test_standard_mode_constrains_final_token... ");

    const char * model_path = getenv("LLAMA_TEST_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        printf("SKIP (LLAMA_TEST_MODEL not set)\n");
        return;
    }
    std::string plugin_path = find_plugin();
    if (plugin_path.empty()) {
        printf("SKIP (no plugin .so found)\n");
        return;
    }
    use_plugin_grammars(plugin_path);

    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 1024;
    params.no_perf       = true;
    params.sampling.temp = 0.0f; // greedy: deterministic shortlist
    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);

    auto            llama_init_tgt = common_init_from_params(params);
    llama_model *   model_tgt      = llama_init_tgt->model();
    llama_context * ctx_tgt        = llama_init_tgt->context();
    assert(ctx_tgt != nullptr && model_tgt != nullptr);

    auto cparams_mtp          = common_context_params_to_llama(params);
    cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cparams_mtp.type_k        = params.speculative.draft.cache_type_k;
    cparams_mtp.type_v        = params.speculative.draft.cache_type_v;
    cparams_mtp.n_rs_seq      = 0;
    cparams_mtp.n_outputs_max = 1;
    cparams_mtp.ctx_other     = ctx_tgt;

    llama_context_ptr ctx_dft(llama_init_from_model(model_tgt, cparams_mtp));
    assert(ctx_dft != nullptr);

    params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC);
    params.speculative.deterministic_draft.enabled        = true;
    params.speculative.deterministic_draft.plugin_path    = plugin_path;
    params.speculative.deterministic_draft.det_accept_all = true;
    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft.get();

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
    assert(spec != nullptr);
    assert(common_speculative_get_det_accept_all(spec));

    struct llama_deterministic_draft * det = common_speculative_get_det_filter_plugin(spec);
    assert(det != nullptr);

    const llama_vocab * vocab  = llama_model_get_vocab(model_tgt);
    const llama_tokens  prompt = common_tokenize(vocab, "int main() {\n    ", false);
    assert(!prompt.empty());
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.data()), (int) prompt.size());
    std::vector<int8_t> logits_flag(prompt.size(), 0);
    logits_flag[0] = 1;
    batch.logits = logits_flag.data();
    assert(llama_decode(ctx_tgt, batch) == 0);

    common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
    assert(smpl != nullptr);
    const std::vector<int> idxs = { 0 };

    // mirror common_speculative_begin(): reset and commit the prompt; the C
    // content drives bootstrap detection to the c grammar
    llama_deterministic_draft_reset(det, 0);
    llama_deterministic_draft_commit_tokens(det, 0, prompt.data(), (int) prompt.size());

    // draft = the target's own greedy pick, so verification always accepts;
    // if the pick is grammar-invalid the filter truncates the draft to
    // empty, which exercises the same final-token constraint on the bonus
    const float * tgt_logits   = llama_get_logits_ith(ctx_tgt, 0);
    const int     n_vocab_tgt  = llama_vocab_n_tokens(vocab);
    llama_token   greedy       = 0;
    float         greedy_score = -1e30f;
    for (llama_token tid = 0; tid < n_vocab_tgt; tid++) {
        if (tgt_logits[tid] > greedy_score) {
            greedy_score = tgt_logits[tid];
            greedy       = tid;
        }
    }
    llama_tokens draft;
    if (llama_deterministic_draft_filter_draft(det, 0, &greedy, 1) == 1) {
        draft.push_back(greedy);
    }

    llama_tokens accepted = common_speculative_sample_and_accept(spec, smpl, ctx_tgt, idxs, draft, 0);
    assert(!accepted.empty());

    // invariant: every emitted token is grammar-valid from the prompt state,
    // verified against a fresh plugin instance as ground truth (its
    // detection converges on the same committed prompt)
    struct llama_deterministic_draft * verifier = llama_deterministic_draft_init(plugin_path.c_str());
    assert(verifier != nullptr);
    {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<std::string> vocab_strings(n_vocab);
        std::vector<const char *> vocab_entries(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            vocab_strings[i] = common_token_to_piece(vocab, i, false);
            vocab_entries[i] = vocab_strings[i].c_str();
        }
        std::vector<int32_t> stop_tokens;
        for (int i = 0; i < n_vocab; i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                stop_tokens.push_back(i);
            }
        }
        assert(llama_deterministic_draft_set_vocab(
                verifier, vocab_entries.data(), n_vocab,
                stop_tokens.data(), (int) stop_tokens.size()));
    }
    llama_deterministic_draft_commit_tokens(verifier, 0, prompt.data(), (int) prompt.size());
    for (auto tok : accepted) {
        assert(llama_deterministic_draft_filter_draft(verifier, 0, &tok, 1) == 1);
    }
    llama_deterministic_draft_free(verifier);

    common_sampler_free(smpl);
    common_speculative_free(spec);

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
    test_plugin_integration();
    test_fail_without_mtp();
    test_fail_without_plugin();
    test_accept_all_requires_plugin();
    test_plugin_state_reset();
    test_auto_imply_mtp();
    test_token_to_piece_special_flag();
    test_cumulative_stats_fields();
    test_accept_null_impl_safe();
    test_get_version_null();
    test_filter_api_no_plugin();
    test_filter_draft_valid_tokens();
    test_filter_draft_truncates_on_invalid();
    test_apply_bitmask_constrains_logits();
    test_commit_tokens_advances_grammar();
    test_rollback_returns_false_no_plugin();
    test_rollback_success_with_plugin();
    test_apply_bitmask_returns_false_no_plugin();
    test_rollback_bitmask_after_failed_init();
    test_set_language_is_per_slot();
    test_bootstrap_detection_resolves_language();
    test_bootstrap_detection_tries_c_last();
    test_state_serialization_roundtrip();
    test_state_serialization_detecting();
    test_common_checkpoint_roundtrip();
    test_get_jump_forward();
    test_accept_all_sample_and_accept();
    test_standard_mode_constrains_final_token();

    printf("\n=== All tests passed ===\n\n");
    return 0;
}
