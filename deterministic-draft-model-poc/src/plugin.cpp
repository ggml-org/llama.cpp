// plugins/deterministic-draft/plugin.cpp -- XGrammar-based deterministic draft plugin
//
// =============================================================================
// Overview
// =============================================================================
// Implements the deterministic_draft_plugin.h v3 C API using XGrammar for
// grammar-constrained decoding with jump-forward support:
//   - CAPABILITY_BITMASK: Pre-generation token constraint
//   - CAPABILITY_JUMP_FORWARD: Deterministic token skipping
//
// =============================================================================
// Architecture
// =============================================================================
//   deterministic-draft.so
//   ├── GrammarCompiler (shared, one per vocab)
//   ├── SlotState (per-inference-slot)
//   │   ├── GrammarMatcher (stateful parser)
//   │   └── jump_forward_cache (last computed string)
//   └── TokenizerInfo (shared vocabulary)
//
// =============================================================================
// Usage Flow
// =============================================================================
// 1. Host calls create() -> plugin instance
// 2. Host calls set_vocab() with tokenizer vocabulary
// 3. Host calls set_grammar() or set_language() to load a grammar
// 4. For each generation step:
//    a. Host calls fill_bitmask() -> constrained token IDs
//    b. Host samples from constrained distribution
//    c. Host calls commit() with sampled token
//    d. Host calls get_jump_forward() -> deterministic string (if any)
//    e. Host emits jump-forward tokens, calls commit() for each
// 5. On reset: reset() clears matcher state
// 6. On shutdown: destroy() frees resources
// =============================================================================

#include "deterministic_draft_plugin.h"

#include <xgrammar/xgrammar.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Environment variable names used for plugin configuration.
// These are set by the host process before loading the plugin.
#define DETERMINISTIC_DRAFT_ENV_GRAMMAR_DIR "DETERMINISTIC_DRAFT_GRAMMAR_DIR"
#define DETERMINISTIC_DRAFT_ENV_DEBUG "DETERMINISTIC_DRAFT_DEBUG"

// Debug tracing (DETERMINISTIC_DRAFT_DEBUG=1): logs detection decisions and
// per-token accept/reject to stderr. No-op otherwise.
static bool det_debug_enabled() {
    static const bool enabled = std::getenv(DETERMINISTIC_DRAFT_ENV_DEBUG) != nullptr;
    return enabled;
}
#define DET_DBG(...) do { if (det_debug_enabled()) fprintf(stderr, __VA_ARGS__); } while (0)

// XGrammar compiler configuration
#define XGRAMMAR_COMPILER_MAX_THREADS 8

// ══════════════════════════════════════════════════════════════════════
// Grammar file resolution (used both by set_language and by bootstrap
// language detection - see ensure_slot_grammar/try_eliminate_and_advance below)
// ══════════════════════════════════════════════════════════════════════

// Directory containing this plugin's own shared library file (.so/.dylib),
// or "." if it cannot be determined.
static std::string get_plugin_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&deterministic_draft_create), &info) && info.dli_fname) {
        std::string path = info.dli_fname;
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            return path.substr(0, pos);
        }
    }
    return ".";
}

// Resolve the bundled grammar directory: DETERMINISTIC_DRAFT_GRAMMAR_DIR
// environment variable if set, otherwise "<plugin_dir>/grammars".
static std::string get_grammar_dir() {
    const char* env_dir = std::getenv(DETERMINISTIC_DRAFT_ENV_GRAMMAR_DIR);
    if (env_dir && env_dir[0] != '\0') {
        return env_dir;
    }
    return get_plugin_dir() + "/grammars";
}

static bool read_file(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) {
        fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(fsize));
    size_t n_read = fsize > 0 ? fread(&out[0], 1, static_cast<size_t>(fsize), f) : 0;
    fclose(f);
    return n_read == static_cast<size_t>(fsize);
}

// Every bundled grammar file (<grammar_dir>/<name>.gbnf) is a language
// candidate for bootstrap detection - discovered dynamically so grammars
// can be added or removed without a plugin code change. Sorted
// alphabetically for a deterministic priority order, except "c" is always
// tried last: unlike the others, its grammar allows bare top-level
// declarations with no wrapping keyword, combined with treating any
// identifier as a valid type name, so it can accept non-C content that
// merely looks declaration-shaped. Fallthrough only triggers when the
// active candidate's matcher actually rejects a token, so if "c" is tried
// first and never rejects anything, the other, more specific candidates
// never get a chance to be considered at all. Trying it last lets those
// candidates correctly reject non-matching content and narrow things down
// first, falling back to "c" only once nothing more specific still fits.
static std::vector<std::string> list_bundled_languages() {
    std::vector<std::string> languages;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(get_grammar_dir(), ec)) {
        if (ec) {
            break;
        }
        if (entry.path().extension() == ".gbnf") {
            languages.push_back(entry.path().stem().string());
        }
    }
    std::sort(languages.begin(), languages.end());

    // Critical: "c" must always be tried last, never first - see the
    // function comment above for why. Do not remove this without also
    // removing the reason it exists.
    auto c_it = std::find(languages.begin(), languages.end(), "c");
    if (c_it != languages.end() && c_it + 1 != languages.end()) {
        std::rotate(c_it, c_it + 1, languages.end());
    }

    return languages;
}

// Per-slot state (defined before XGrammarPlugin, which references it).
struct SlotState {
    std::unique_ptr<xgrammar::GrammarMatcher> matcher;
    std::string jump_forward_cache;
    bool terminated = false;
    std::vector<uint32_t> bitmask;  // reusable bitmask buffer
    std::vector<int32_t> bitmask_scratch;  // reusable scratch for FillNextTokenBitmask

    // Grammar bound to this slot specifically, not shared plugin-wide, so
    // different slots can have different active languages at the same time.
    std::unique_ptr<xgrammar::CompiledGrammar> compiled_grammar;
    std::string current_language;

    // Bootstrap-detection candidates not yet ruled out, in priority order.
    // remaining_candidates.front() is always the language currently backing
    // compiled_grammar/matcher. Empty means this slot is not in detection
    // mode - either a language was explicitly configured/inherited, or
    // detection already resolved to a single winner (see try_eliminate_and_advance),
    // or there were no bundled grammars to detect among.
    std::vector<std::string> remaining_candidates;

    // Token ids committed to this slot's matcher so far while still in
    // detection mode, needed to replay into a freshly loaded candidate's
    // matcher when falling through - a matcher's internal state can't be
    // transplanted between different compiled grammars. Cleared once
    // detection resolves to a single winner.
    std::vector<int32_t> detection_history;

    // Every token id the current matcher has accepted since its grammar was
    // applied, including across candidate elimination (unlike
    // detection_history, never cleared while the matcher lives). This is the
    // replay log used to rebuild matcher state in state serialization.
    std::vector<int32_t> history;

    // True once bootstrap detection has been attempted for this slot at
    // least once, so ensure_slot_grammar does not keep re-scanning the
    // grammar directory on every call once nothing was found there.
    bool detection_attempted = false;

    void reset_matcher() {
        if (matcher) {
            matcher->Reset();
        }
        jump_forward_cache.clear();
        terminated = false;
        bitmask.clear();
        detection_history.clear();
        history.clear();
    }
};

// Plugin state (defined before precompile_bundled_grammars, which uses it).
struct XGrammarPlugin {
    // Shared components (initialized once)
    std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;

    // Grammar for DETERMINISTIC_DRAFT_SLOT_DEFAULT, inherited by any slot
    // created afterward (preserves single-slot behavior for legacy callers).
    // An explicit slot_id passed to set_language/set_grammar only touches
    // that slot's own SlotState::compiled_grammar and never affects this
    // default or other slots - see compile_and_apply_grammar.
    std::unique_ptr<xgrammar::CompiledGrammar> default_compiled_grammar;
    std::string default_language;

    // Per-slot state
    std::unordered_map<int, SlotState> slots;
    std::mutex slots_mutex;

    // Configuration
    int vocab_size = 0;
    std::vector<std::string> vocab_strings;
    std::vector<int32_t> stop_token_ids;

    // Compiled-grammar cache keyed by cache key (language name, or raw EBNF
    // text for set_grammar callers) - avoids recompiling when switching back
    // to a previously-seen grammar within the same process.
    std::unordered_map<std::string, std::shared_ptr<xgrammar::CompiledGrammar>> grammar_cache;

    // Capabilities
    uint32_t capabilities = DETERMINISTIC_DRAFT_CAPABILITY_BITMASK |
                            DETERMINISTIC_DRAFT_CAPABILITY_JUMP_FORWARD;

    // Get or create slot. A newly created slot inherits the default grammar.
    SlotState& get_slot(int32_t slot_id) {
        int key = (slot_id == DETERMINISTIC_DRAFT_SLOT_DEFAULT) ? 0 : slot_id;
        std::lock_guard<std::mutex> lock(slots_mutex);
        auto it = slots.find(key);
        if (it != slots.end()) {
            return it->second;
        }

        SlotState& slot = slots[key];
        if (default_compiled_grammar) {
            slot.compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(*default_compiled_grammar);
            slot.current_language = default_language;
            slot.matcher = std::make_unique<xgrammar::GrammarMatcher>(
                *slot.compiled_grammar,
                stop_token_ids.empty() ? std::nullopt : std::optional<std::vector<int32_t>>(stop_token_ids));
        }
        return slot;
    }

    bool has_slot(int32_t slot_id) const {
        int key = (slot_id == DETERMINISTIC_DRAFT_SLOT_DEFAULT) ? 0 : slot_id;
        return slots.find(key) != slots.end();
    }
};

// Pre-compile all bundled grammars into the shared cache. This moves the
// one-time compilation cost from the first request to plugin initialization,
// so bootstrap detection never pays it during generation. Does not apply any
// grammar to a slot; explicit set_language/set_grammar still controls the
// active matcher.
static bool precompile_bundled_grammars(XGrammarPlugin* plugin) {
    if (!plugin->compiler) {
        return false;
    }
    try {
        std::vector<std::string> languages = list_bundled_languages();
        for (const auto& lang : languages) {
            if (plugin->grammar_cache.find(lang) != plugin->grammar_cache.end()) {
                continue;
            }
            std::string ebnf_str;
            if (!read_file(get_grammar_dir() + "/" + lang + ".gbnf", ebnf_str)) {
                fprintf(stderr, "[xgrammar-draft] WARN: failed to read grammar '%s'\n", lang.c_str());
                continue;
            }
            auto grammar = xgrammar::Grammar::FromEBNF(ebnf_str, "root");
            auto compiled = std::make_shared<xgrammar::CompiledGrammar>(
                plugin->compiler->CompileGrammar(grammar));
            plugin->grammar_cache[lang] = compiled;
            fprintf(stderr, "[xgrammar-draft] Pre-compiled grammar '%s'\n", lang.c_str());
        }
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: failed to pre-compile grammars: %s\n", e.what());
        return false;
    }
}

// ══════════════════════════════════════════════════════════════════════
// C API Implementation
// ══════════════════════════════════════════════════════════════════════

extern "C" {

// ── Lifecycle ────────────────────────────────────────────────────────

DeterministicDraftPlugin* deterministic_draft_create(void) {
    auto* plugin = new (std::nothrow) XGrammarPlugin();
    if (!plugin) {
        fprintf(stderr, "[xgrammar-draft] ERROR: failed to allocate plugin\n");
        return nullptr;
    }
    return reinterpret_cast<DeterministicDraftPlugin*>(plugin);
}

void deterministic_draft_destroy(DeterministicDraftPlugin* state) {
    delete reinterpret_cast<XGrammarPlugin*>(state);
}

// ── Capabilities ─────────────────────────────────────────────────────

uint32_t deterministic_draft_get_capabilities(DeterministicDraftPlugin* state) {
    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin) {
        return 0;
    }
    return plugin->capabilities;
}

// ── Vocabulary setup ─────────────────────────────────────────────────

bool deterministic_draft_set_vocab(
        DeterministicDraftPlugin* state,
        const char** vocab_entries,
        int32_t vocab_size,
        const int32_t* stop_tokens,
        int32_t n_stop) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !vocab_entries || vocab_size <= 0) {
        return false;
    }

    // Store vocabulary
    plugin->vocab_size = vocab_size;
    plugin->vocab_strings.clear();
    plugin->vocab_strings.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        plugin->vocab_strings.push_back(vocab_entries[i] ? vocab_entries[i] : "");
    }

    // Store stop tokens
    plugin->stop_token_ids.clear();
    if (stop_tokens && n_stop > 0) {
        plugin->stop_token_ids.assign(stop_tokens, stop_tokens + n_stop);
    }

    // Create TokenizerInfo
    // XGrammar expects the encoded vocabulary (raw token bytes)
    try {
        plugin->tokenizer_info = std::make_unique<xgrammar::TokenizerInfo>(
            plugin->vocab_strings,
            xgrammar::VocabType::BYTE_LEVEL,
            vocab_size,
            plugin->stop_token_ids.empty() ? std::nullopt : std::optional<std::vector<int32_t>>(plugin->stop_token_ids),
            false  // add_prefix_space
        );

        // Create compiler
        plugin->compiler = std::make_unique<xgrammar::GrammarCompiler>(
            *plugin->tokenizer_info,
            XGRAMMAR_COMPILER_MAX_THREADS,
            true   // cache_enabled
        );

        // Pre-compile all bundled grammars so the first request does not
        // pay compilation cost during bootstrap detection.
        if (!precompile_bundled_grammars(plugin)) {
            fprintf(stderr, "[xgrammar-draft] WARN: grammar pre-compilation failed\n");
        }

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: failed to create tokenizer info: %s\n", e.what());
        return false;
    }
}

// ── Grammar configuration ────────────────────────────────────────────

// Compile (or reuse from cache) an EBNF grammar and apply it to a single
// slot's matcher. `cache_key` identifies this grammar for reuse (e.g. a
// language name, or the raw EBNF text itself); `language_name` is what
// get_language() will report for this slot afterward (empty for a raw
// set_grammar() call). DETERMINISTIC_DRAFT_SLOT_DEFAULT also updates the
// plugin's default grammar so slots created afterward inherit it; any other
// slot_id only affects that one slot.
static bool compile_and_apply_grammar(
        XGrammarPlugin* plugin,
        int32_t slot_id,
        const std::string& ebnf_str,
        const std::string& root_rule,
        const std::string& cache_key,
        const std::string& language_name) {

    if (!plugin->compiler) {
        fprintf(stderr, "[xgrammar-draft] ERROR: vocabulary not set, call set_vocab first\n");
        return false;
    }

    try {
        std::shared_ptr<xgrammar::CompiledGrammar> compiled;
        auto it = plugin->grammar_cache.find(cache_key);
        if (it != plugin->grammar_cache.end()) {
            compiled = it->second;
            fprintf(stderr, "[xgrammar-draft] Reusing cached grammar for '%s'\n", cache_key.c_str());
        } else {
            fprintf(stderr, "[xgrammar-draft] Compiling grammar '%s'...\n", cache_key.c_str());
            auto grammar = xgrammar::Grammar::FromEBNF(ebnf_str, root_rule);
            compiled = std::make_shared<xgrammar::CompiledGrammar>(
                plugin->compiler->CompileGrammar(grammar));
            plugin->grammar_cache[cache_key] = compiled;
            fprintf(stderr, "[xgrammar-draft] Grammar compiled and cached for '%s'\n", cache_key.c_str());
        }

        std::lock_guard<std::mutex> lock(plugin->slots_mutex);

        if (slot_id == DETERMINISTIC_DRAFT_SLOT_DEFAULT) {
            plugin->default_compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(*compiled);
            plugin->default_language = language_name;
        }

        const int key = (slot_id == DETERMINISTIC_DRAFT_SLOT_DEFAULT) ? 0 : slot_id;
        SlotState& slot = plugin->slots[key];
        slot.compiled_grammar = std::make_unique<xgrammar::CompiledGrammar>(*compiled);
        slot.current_language = language_name;
        slot.matcher = std::make_unique<xgrammar::GrammarMatcher>(
            *slot.compiled_grammar,
            plugin->stop_token_ids.empty() ? std::nullopt : std::optional<std::vector<int32_t>>(plugin->stop_token_ids));
        slot.terminated = false;
        slot.history.clear();

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: failed to compile grammar '%s': %s\n",
                cache_key.c_str(), e.what());
        return false;
    }
}

bool deterministic_draft_set_grammar(
        DeterministicDraftPlugin* state,
        const char* ebnf_str,
        const char* root_rule) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !ebnf_str) {
        return false;
    }

    std::string root = root_rule ? root_rule : "root";
    // set_grammar has no slot_id in its contract - it always targets the
    // default slot. Raw EBNF is not a named bundled language.
    return compile_and_apply_grammar(plugin, DETERMINISTIC_DRAFT_SLOT_DEFAULT, ebnf_str, root, ebnf_str, "");
}

// ── Bootstrap language detection ─────────────────────────────────────
//
// A slot with no configured language (no default inherited, no explicit
// set_language/set_grammar call) does not just sit unconstrained: it tries
// every bundled grammar as a candidate, one at a time in priority order,
// and narrows down as tokens are committed. There is only ever one active
// XGrammar matcher per slot - never several in parallel - so this reuses
// the exact same matcher/bitmask machinery as a normal fixed-language slot.

static bool load_candidate_grammar(XGrammarPlugin * plugin, int32_t slot_id, const std::string & lang) {
    std::string ebnf_str;
    return read_file(get_grammar_dir() + "/" + lang + ".gbnf", ebnf_str) &&
           compile_and_apply_grammar(plugin, slot_id, ebnf_str, "root", lang, lang);
}

// Lazily starts bootstrap detection for a slot that has no grammar at all.
// A no-op once the slot already has a matcher, or once detection has
// already been attempted and found no bundled grammars to try.
static void ensure_slot_grammar(XGrammarPlugin * plugin, int32_t slot_id, SlotState & slot) {
    if (slot.matcher || slot.detection_attempted) {
        return;
    }
    slot.detection_attempted = true;

    std::vector<std::string> languages = list_bundled_languages();
    if (det_debug_enabled()) {
        std::string order;
        for (const auto& l : languages) { order += l; order += " "; }
        DET_DBG("[det-debug] slot %d bootstrap candidates (priority order): %s\n", slot_id, order.c_str());
    }
    while (!languages.empty()) {
        const std::string lang = languages.front();
        if (load_candidate_grammar(plugin, slot_id, lang)) {
            DET_DBG("[det-debug] slot %d initial candidate: %s\n", slot_id, lang.c_str());
            slot.remaining_candidates = std::move(languages);
            return;
        }
        languages.erase(languages.begin());
    }
}

// When the current active candidate rejects token_id, try advancing to the
// next candidate: eliminate the rejected candidate, load the next one, replay
// detection history into it, and test the rejected token. If the next
// candidate also rejects, eliminate it too and keep going. The first
// candidate that accepts the full history + token_id wins. If none do, the
// token is genuinely invalid (not a language mismatch) and the original
// candidate is restored unchanged.
//
// Returns true if a new candidate was found and is now active.
// Returns false if the token is genuinely invalid (original restored).
static bool try_eliminate_and_advance(XGrammarPlugin * plugin, int32_t slot_id, SlotState & slot, int32_t token_id) {
    // Save the original candidate and its grammar in case we need to restore
    std::string original_lang = slot.remaining_candidates.front();

    // Try each remaining candidate after the current one
    for (size_t i = 1; i < slot.remaining_candidates.size(); ) {
        const std::string & candidate = slot.remaining_candidates[i];

        if (!load_candidate_grammar(plugin, slot_id, candidate)) {
            // Can't load this grammar - eliminate it and move on
            slot.remaining_candidates.erase(slot.remaining_candidates.begin() + i);
            continue;
        }

        // Replay history into this fresh matcher
        bool history_ok = true;
        for (int32_t id : slot.detection_history) {
            if (!slot.matcher->AcceptToken(id)) {
                history_ok = false;
                break;
            }
        }

        if (history_ok && slot.matcher->AcceptToken(token_id)) {
            // This candidate accepts everything - make it the winner
            // Eliminate all candidates before it (including the original)
            DET_DBG("[det-debug] slot %d eliminate '%s' -> '%s' now active (token %d), %zu candidates left\n",
                    slot_id, original_lang.c_str(), candidate.c_str(), token_id,
                    slot.remaining_candidates.size() - i);
            slot.remaining_candidates.erase(slot.remaining_candidates.begin(),
                                           slot.remaining_candidates.begin() + i);
            slot.detection_history.push_back(token_id);
            slot.history = slot.detection_history;
            if (slot.remaining_candidates.size() == 1) {
                // Detection resolved
                DET_DBG("[det-debug] slot %d detection RESOLVED -> '%s'\n", slot_id, candidate.c_str());
                slot.detection_history.clear();
            }
            return true;
        }

        // This candidate also rejected - eliminate it and try next
        DET_DBG("[det-debug] slot %d candidate '%s' also rejected token %d, eliminating\n",
                slot_id, candidate.c_str(), token_id);
        slot.remaining_candidates.erase(slot.remaining_candidates.begin() + i);
    }

    // No candidate accepted - token is genuinely invalid
    // Restore the original candidate
    DET_DBG("[det-debug] slot %d NO candidate accepted token %d, restoring '%s'\n",
            slot_id, token_id, original_lang.c_str());
    load_candidate_grammar(plugin, slot_id, original_lang);
    for (int32_t id : slot.detection_history) {
        slot.matcher->AcceptToken(id);
    }
    slot.history = slot.detection_history;
    return false;
}

// ── Bitmask (CAPABILITY_BITMASK) ─────────────────────────────────────

bool deterministic_draft_fill_bitmask(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint32_t* bitmask,
        int32_t vocab_size) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !bitmask || vocab_size <= 0) {
        return false;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    ensure_slot_grammar(plugin, slot_id, slot);
    if (!slot.matcher || slot.terminated) {
        return false;
    }

    try {
        // Allocate DLTensor for bitmask
        // XGrammar expects shape (GetBitmaskSize(),) with dtype int32
        int bitmask_size = xgrammar::GetBitmaskSize(vocab_size);
        std::vector<int32_t> bitmask_data(bitmask_size, 0);

        DLTensor bitmask_tensor;
        bitmask_tensor.data = bitmask_data.data();
        bitmask_tensor.device = DLDevice{kDLCPU, 0};
        bitmask_tensor.ndim = 1;
        bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
        bitmask_tensor.shape = new int64_t[1]{bitmask_size};
        bitmask_tensor.strides = nullptr;
        bitmask_tensor.byte_offset = 0;

        // Fill bitmask
        slot.matcher->FillNextTokenBitmask(&bitmask_tensor);

        delete[] bitmask_tensor.shape;

        // Copy to output (convert int32 to uint32)
        for (int i = 0; i < bitmask_size && i < (vocab_size + 31) / 32; i++) {
            bitmask[i] = static_cast<uint32_t>(bitmask_data[i]);
        }

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: fill_bitmask failed: %s\n", e.what());
        return false;
    }
}

// ── Jump-forward (CAPABILITY_JUMP_FORWARD) ───────────────────────────

const char* deterministic_draft_get_jump_forward(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t* out_length) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin) {
        if (out_length) *out_length = 0;
        return nullptr;
    }

    if (!plugin->has_slot(slot_id)) {
        if (out_length) *out_length = 0;
        return nullptr;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    if (!slot.matcher || slot.terminated) {
        if (out_length) *out_length = 0;
        return nullptr;
    }

    try {
        slot.jump_forward_cache = slot.matcher->FindJumpForwardString();

        if (slot.jump_forward_cache.empty()) {
            if (out_length) *out_length = 0;
            return nullptr;
        }

        if (out_length) {
            *out_length = static_cast<int>(slot.jump_forward_cache.size());
        }
        return slot.jump_forward_cache.c_str();
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: get_jump_forward failed: %s\n", e.what());
        if (out_length) *out_length = 0;
        return nullptr;
    }
}

// ── Commit ───────────────────────────────────────────────────────────

void deterministic_draft_commit(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t token_id,
        const char* token_text,
        int32_t token_length) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin) return;

    SlotState& slot = plugin->get_slot(slot_id);
    ensure_slot_grammar(plugin, slot_id, slot);
    if (!slot.matcher) return;

    // A terminated matcher has already reached a complete parse (e.g. an
    // empty function body is a grammatically complete program) and will
    // reject anything further. Feeding it more tokens anyway is not just
    // pointless - it produces a warning per call and, under sustained
    // load, real slowdown - so stop here instead of calling AcceptToken.
    if (slot.terminated) {
        return;
    }

    std::string token(token_text, static_cast<size_t>(token_length));

    try {
        bool accepted = slot.matcher->AcceptToken(token_id);
        if (accepted) {
            slot.history.push_back(token_id);
            if (!slot.remaining_candidates.empty()) {
                slot.detection_history.push_back(token_id);
            }
        } else if (slot.remaining_candidates.size() > 1) {
            // Still bootstrap-detecting: this candidate rejecting the token
            // may just mean it is the wrong language, not that the token
            // itself is bad. try_eliminate_and_advance already records history and
            // switches the active matcher on success.
            accepted = try_eliminate_and_advance(plugin, slot_id, slot, token_id);
        }

        if (!accepted) {
            fprintf(stderr, "[xgrammar-draft] WARNING: AcceptToken rejected token_id=%d '%s'\n",
                    token_id, token.c_str());
        }

        if (slot.matcher->IsTerminated()) {
            slot.terminated = true;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: commit failed: %s\n", e.what());
    }
}

bool deterministic_draft_rollback(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t n_tokens) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || n_tokens <= 0) return n_tokens == 0;

    SlotState& slot = plugin->get_slot(slot_id);
    if (!slot.matcher) return false;

    try {
        slot.matcher->Rollback(n_tokens);
        slot.terminated = false;
        slot.history.resize(n_tokens < (int) slot.history.size() ? slot.history.size() - n_tokens : 0);
        slot.detection_history.resize(n_tokens < (int) slot.detection_history.size() ? slot.detection_history.size() - n_tokens : 0);
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: rollback failed: %s\n", e.what());
        return false;
    }
}

// ── High-level filter helpers (CAPABILITY_BITMASK) ──────────────────

static void bitmask_apply_to_logits(
        const uint32_t* bitmask,
        int vocab_size,
        float* logits) {
    const int bitmask_words = (vocab_size + 31) / 32;
    for (int w = 0; w < bitmask_words; w++) {
        if (bitmask[w] == 0xFFFFFFFFu) {
            continue;
        }
        for (int b = 0; b < 32; b++) {
            const int i = w * 32 + b;
            if (i >= vocab_size) {
                break;
            }
            if (!(bitmask[w] & (1u << b))) {
                logits[i] = -1e30f;
            }
        }
    }
}

// Printable form of a token for debug traces (escapes control chars).
static std::string det_token_repr(const XGrammarPlugin* plugin, int32_t token_id) {
    std::string s;
    if (token_id >= 0 && token_id < (int) plugin->vocab_strings.size()) {
        s = plugin->vocab_strings[token_id];
    } else {
        s = "?";
    }
    std::string out;
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if ((unsigned char) c < 0x20) out += "?";
        else out += c;
    }
    return out;
}

int32_t deterministic_draft_filter_draft(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const int32_t* tokens,
        int32_t n_tokens) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !tokens || n_tokens <= 0) {
        return 0;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    ensure_slot_grammar(plugin, slot_id, slot);
    if (!slot.matcher || slot.terminated) {
        return 0;
    }

    const int vocab_size = plugin->vocab_size;
    const int bitmask_words = (vocab_size + 31) / 32;
    slot.bitmask.assign(bitmask_words, 0xFFFFFFFFu);

    int n_accepted = 0;
    for (int i = 0; i < n_tokens; i++) {
        const int32_t token_id = tokens[i];

        // Check grammar validity directly via AcceptToken (O(1) vs
        // O(vocab_size)). AcceptToken both validates the token against the
        // current grammar state AND advances the matcher, so it replaces the
        // old FillNextTokenBitmask + bitmask lookup roundtrip.
        bool token_valid = true;
        try {
            token_valid = slot.matcher->AcceptToken(token_id);
        } catch (const std::exception& e) {
            // Fail closed, not open: a token we couldn't validate (e.g. the
            // matcher already reached a complete parse and rejects everything
            // further) must not be treated as valid - that would let an
            // invalid token through silently. Stop this batch here instead
            // of guessing.
            fprintf(stderr, "[xgrammar-draft] ERROR: AcceptToken failed: %s\n", e.what());
            break;
        }

        if (!token_valid) {
            // Still bootstrap-detecting: this candidate rejecting the
            // token may just mean it is the wrong language, not that the
            // token itself is bad. Try falling through to another
            // still-live candidate before giving up on it.
            if (slot.remaining_candidates.size() > 1 &&
                try_eliminate_and_advance(plugin, slot_id, slot, token_id)) {
                n_accepted++;
                if (slot.matcher->IsTerminated()) {
                    slot.terminated = true;
                    break;
                }
                continue;
            }
            DET_DBG("[det-debug] filter_draft slot %d: REJECT token %d '%s' at pos %d/%d (lang=%s)\n",
                    slot_id, token_id, det_token_repr(plugin, token_id).c_str(), i, n_tokens,
                    slot.current_language.c_str());
            break;
        }

        DET_DBG("[det-debug] filter_draft slot %d: accept token %d '%s' (lang=%s)\n",
                slot_id, token_id, det_token_repr(plugin, token_id).c_str(), slot.current_language.c_str());

        // Token already committed via AcceptToken in the validity check above.

        slot.history.push_back(token_id);
        if (!slot.remaining_candidates.empty()) {
            slot.detection_history.push_back(token_id);
        }

        n_accepted++;

        // Once the grammar reaches a complete parse, it will reject
        // everything further (see the fail-closed comment above) - stop
        // this batch immediately instead of hitting that on every
        // remaining token.
        if (slot.matcher->IsTerminated()) {
            slot.terminated = true;
            break;
        }
    }

    return n_accepted;
}

bool deterministic_draft_apply_bitmask(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint32_t* bitmask,
        int32_t vocab_size,
        float* logits) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !bitmask || vocab_size <= 0 || !logits) {
        return false;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    ensure_slot_grammar(plugin, slot_id, slot);
    if (!slot.matcher || slot.terminated) {
        return false;
    }

    try {
        int bitmask_size = xgrammar::GetBitmaskSize(vocab_size);
        std::vector<int32_t> bitmask_data(bitmask_size, 0);

        DLTensor bitmask_tensor;
        bitmask_tensor.data = bitmask_data.data();
        bitmask_tensor.device = DLDevice{kDLCPU, 0};
        bitmask_tensor.ndim = 1;
        bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
        bitmask_tensor.shape = new int64_t[1]{bitmask_size};
        bitmask_tensor.strides = nullptr;
        bitmask_tensor.byte_offset = 0;

        const auto t0 = det_debug_enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        slot.matcher->FillNextTokenBitmask(&bitmask_tensor);
        if (det_debug_enabled()) {
            const auto t1 = std::chrono::steady_clock::now();
            DET_DBG("[det-debug] apply_bitmask: FillNextTokenBitmask took %.2f ms (lang=%s)\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), slot.current_language.c_str());
        }
        delete[] bitmask_tensor.shape;

        // Convert to uint32 bitmask
        const int bitmask_words = (vocab_size + 31) / 32;
        for (int i = 0; i < bitmask_size && i < bitmask_words; i++) {
            bitmask[i] = static_cast<uint32_t>(bitmask_data[i]);
        }

        // Apply to logits
        bitmask_apply_to_logits(bitmask, vocab_size, logits);

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: apply_bitmask failed: %s\n", e.what());
        return false;
    }
}

void deterministic_draft_commit_tokens(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const int32_t* tokens,
        int32_t n_tokens) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !tokens || n_tokens <= 0) {
        return;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    ensure_slot_grammar(plugin, slot_id, slot);
    if (!slot.matcher || slot.terminated) {
        return;
    }

    for (int i = 0; i < n_tokens; i++) {
        const int32_t token_id = tokens[i];
        try {
            bool accepted = slot.matcher->AcceptToken(token_id);
            if (accepted) {
                slot.history.push_back(token_id);
                if (!slot.remaining_candidates.empty()) {
                    slot.detection_history.push_back(token_id);
                }
            } else if (slot.remaining_candidates.size() > 1) {
                accepted = try_eliminate_and_advance(plugin, slot_id, slot, token_id);
            }

            if (det_debug_enabled() && !accepted) {
                DET_DBG("[det-debug] commit_tokens slot %d: REJECT token %d '%s' at %d/%d (lang=%s)\n",
                        slot_id, token_id, det_token_repr(plugin, token_id).c_str(), i, n_tokens,
                        slot.current_language.c_str());
            }

            if (slot.matcher->IsTerminated()) {
                slot.terminated = true;
                DET_DBG("[det-debug] commit_tokens slot %d: matcher TERMINATED at %d/%d (lang=%s)\n",
                        slot_id, i, n_tokens, slot.current_language.c_str());
                break;
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "[xgrammar-draft] ERROR: commit_tokens failed: %s\n", e.what());
            break;
        }
    }
}

// ── State access ─────────────────────────────────────────────────────

void deterministic_draft_reset(
        DeterministicDraftPlugin* state,
        int32_t slot_id) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin) return;

    if (!plugin->has_slot(slot_id)) return;

    SlotState& slot = plugin->get_slot(slot_id);
    slot.reset_matcher();
}

// ── State serialization (optional SPI) ─────────────────────────────
//
// XGrammar matchers cannot be dumped to bytes, so a slot's state is
// serialized as the inputs needed to rebuild it: the active grammar
// (language name, or the ordered candidate list while bootstrap detection
// is still running) plus SlotState::history, the replay log of every token
// id the matcher has accepted. Restore re-applies the grammar (served from
// the compiled-grammar cache) and replays the history with AcceptToken,
// reproducing the matcher state exactly.
//
// Slots configured via set_grammar() with raw EBNF have no language name
// and are not serializable - get_size returns 0 for them.

#define XDDS_STATE_MAGIC   0x58444453u // 'XDDS'
#define XDDS_STATE_VERSION 1u

// mode 1: bootstrap detection still running, 2: single language active

static int slot_state_serialized_size(const SlotState & slot) {
    if (!slot.matcher) {
        return 0;
    }
    if (slot.remaining_candidates.size() <= 1 && slot.current_language.empty()) {
        return 0;
    }

    int size = 4 /* magic */ + 4 /* version */ + 4 /* mode */;
    if (slot.remaining_candidates.size() > 1) {
        size += 4; // n_candidates
        for (const auto & c : slot.remaining_candidates) {
            size += 4 + (int) c.size();
        }
    } else {
        size += 4 + (int) slot.current_language.size();
    }
    size += 4 + 4 * (int) slot.history.size();
    return size;
}

int32_t deterministic_draft_state_get_size(
        DeterministicDraftPlugin* state,
        int32_t slot_id) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin) {
        return -1;
    }
    if (!plugin->has_slot(slot_id)) {
        return 0;
    }
    return slot_state_serialized_size(plugin->get_slot(slot_id));
}

int32_t deterministic_draft_state_get_data(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint8_t* buffer,
        int32_t buffer_size) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !buffer || buffer_size <= 0 || !plugin->has_slot(slot_id)) {
        return -1;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    const int need = slot_state_serialized_size(slot);
    if (need <= 0 || buffer_size < need) {
        return -1;
    }

    uint8_t* out = buffer;
    auto put_u32 = [&out](uint32_t v) { memcpy(out, &v, 4); out += 4; };

    const bool detecting = slot.remaining_candidates.size() > 1;

    put_u32(XDDS_STATE_MAGIC);
    put_u32(XDDS_STATE_VERSION);
    put_u32(detecting ? 1 : 2);

    if (detecting) {
        put_u32((uint32_t) slot.remaining_candidates.size());
        for (const auto & c : slot.remaining_candidates) {
            put_u32((uint32_t) c.size());
            memcpy(out, c.data(), c.size());
            out += c.size();
        }
    } else {
        put_u32((uint32_t) slot.current_language.size());
        memcpy(out, slot.current_language.data(), slot.current_language.size());
        out += slot.current_language.size();
    }

    put_u32((uint32_t) slot.history.size());
    if (!slot.history.empty()) {
        memcpy(out, slot.history.data(), slot.history.size() * 4);
        out += slot.history.size() * 4;
    }

    return (int) (out - buffer);
}

bool deterministic_draft_state_set_data(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const uint8_t* data,
        int32_t size) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !data || size < 16) {
        return false;
    }

    // Parse everything into locals first - the slot is only touched once
    // the whole blob has validated.
    const uint8_t* cur = data;
    const uint8_t* end = data + size;
    auto get_u32 = [&cur, end](uint32_t & v) {
        if (end - cur < 4) {
            return false;
        }
        memcpy(&v, cur, 4);
        cur += 4;
        return true;
    };
    auto get_str = [&cur, end, &get_u32](std::string & s) {
        uint32_t len;
        if (!get_u32(len) || (uint32_t) (end - cur) < len) {
            return false;
        }
        s.assign((const char*) cur, len);
        cur += len;
        return true;
    };

    uint32_t magic, version, mode;
    if (!get_u32(magic) || !get_u32(version) || !get_u32(mode) ||
            magic != XDDS_STATE_MAGIC || version != XDDS_STATE_VERSION ||
            (mode != 1 && mode != 2)) {
        return false;
    }

    std::vector<std::string> candidates;
    std::string language;
    if (mode == 1) {
        uint32_t n_candidates;
        if (!get_u32(n_candidates) || n_candidates < 2) {
            return false;
        }
        for (uint32_t i = 0; i < n_candidates; i++) {
            std::string name;
            if (!get_str(name)) {
                return false;
            }
            candidates.push_back(std::move(name));
        }
    } else if (!get_str(language) || language.empty()) {
        return false;
    }

    uint32_t n_history;
    if (!get_u32(n_history) || (uint32_t) (end - cur) != n_history * 4) {
        return false;
    }
    std::vector<int32_t> history(n_history);
    if (n_history > 0) {
        memcpy(history.data(), cur, n_history * 4);
    }

    try {
        SlotState& slot = plugin->get_slot(slot_id);

        // Fresh matcher for the saved grammar (cache hit in practice)
        if (!load_candidate_grammar(plugin, slot_id, mode == 1 ? candidates.front() : language)) {
            return false;
        }
        if (mode == 1) {
            slot.remaining_candidates = std::move(candidates);
            slot.detection_attempted   = true;
        }

        // Replay the token history; the current matcher accepted exactly
        // this sequence at save time, so every token must replay cleanly.
        for (int32_t id : history) {
            if (!slot.matcher->AcceptToken(id)) {
                fprintf(stderr, "[xgrammar-draft] ERROR: state replay rejected token %d\n", id);
                return false;
            }
        }

        slot.detection_history = history;
        slot.history           = std::move(history);
        slot.terminated        = slot.matcher->IsTerminated();
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xgrammar-draft] ERROR: state restore failed: %s\n", e.what());
        return false;
    }
}

// ── Termination query ───────────────────────────────────────────────

bool deterministic_draft_is_terminated(
        DeterministicDraftPlugin* state,
        int32_t slot_id) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !plugin->has_slot(slot_id)) {
        return false;
    }

    return plugin->get_slot(slot_id).terminated;
}

// ── Language control (bundled grammar loading) ────────────────────────

bool deterministic_draft_set_language(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const char* lang) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !lang || lang[0] == '\0') {
        return false;
    }

    std::string language = lang;
    std::string grammar_path = get_grammar_dir() + "/" + language + ".gbnf";

    std::string ebnf_str;
    if (!read_file(grammar_path, ebnf_str)) {
        fprintf(stderr, "[xgrammar-draft] ERROR: no bundled grammar for language '%s' (looked in %s)\n",
                language.c_str(), grammar_path.c_str());
        return false;
    }

    return compile_and_apply_grammar(plugin, slot_id, ebnf_str, "root", language, language);
}

const char* deterministic_draft_get_language(
        DeterministicDraftPlugin* state,
        int32_t slot_id) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !plugin->has_slot(slot_id)) {
        return "unknown";
    }

    SlotState& slot = plugin->get_slot(slot_id);
    if (slot.current_language.empty()) {
        return "unknown";
    }
    return slot.current_language.c_str();
}

bool deterministic_draft_is_detecting(
        DeterministicDraftPlugin* state,
        int32_t slot_id) {

    auto* plugin = reinterpret_cast<XGrammarPlugin*>(state);
    if (!plugin || !plugin->has_slot(slot_id)) {
        return false;
    }

    SlotState& slot = plugin->get_slot(slot_id);
    // More than one live candidate means bootstrap detection has not yet
    // converged to a single language.
    return slot.remaining_candidates.size() > 1;
}

// ── Metadata ─────────────────────────────────────────────────────────

const char* deterministic_draft_get_version(
        DeterministicDraftPlugin* /*state*/) {
    return "3.1.0";
}

} // extern "C"
