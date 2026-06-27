// deterministic_draft_plugin.h -- Public C API contract for deterministic draft plugins
//
// Plugin authors write a shared library (.so/.dylib/.dll) that exports the
// functions declared below.  The loader in llama.cpp uses dlopen/dlsym to
// bind to symbols prefixed with "deterministic_draft_".
//
// Lifecycle:
//   1. llama.cpp calls deterministic_draft_create() to obtain an opaque state
//   2. Calls deterministic_draft_set_vocab() once with tokenizer vocabulary
//   3. Calls deterministic_draft_set_grammar() or deterministic_draft_set_language()
//      to select a grammar for constrained generation
//   4. For each generation step:
//      a. Calls deterministic_draft_fill_bitmask() to get valid next-token IDs
//      b. Samples/drafts a token constrained to that bitmask
//      c. Calls deterministic_draft_commit() once the token is accepted
//      d. Optionally calls deterministic_draft_get_jump_forward() to skip
//         deterministic sequences
//   5. On reset/gen-start, calls deterministic_draft_reset() to clear state
//   6. At shutdown, calls deterministic_draft_destroy() to free the state
//
// Multi-slot isolation:
//   - Each inference slot is identified by an |int32_t slot_id| parameter
//   - Plugins maintain per-slot state maps internally
//   - Slot IDs are assigned by the host (0, 1, 2, ...)
//   - The sentinel value -1 means "default slot" for single-slot use
//
// Threading: the host serializes all calls on a given plugin instance;
// plugins do not need to be thread-safe per instance.
//
// Capability-based API:
//   Plugins declare capabilities via deterministic_draft_get_capabilities().
//   The host queries capabilities and uses available features:
//
//   CAPABILITY_BITMASK (bit 1):
//     Pre-generation token constraint via fill_bitmask().
//     Plugin provides a bitmask of valid token IDs before sampling.
//
//   CAPABILITY_JUMP_FORWARD (bit 2):
//     Deterministic token skipping via get_jump_forward().
//     Plugin returns strings that are uniquely determined by grammar.
//
// Plugins may implement any combination of capabilities. The host gracefully
// degrades when capabilities are missing.

#ifndef DETERMINISTIC_DRAFT_PLUGIN_H
#define DETERMINISTIC_DRAFT_PLUGIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel slot ID meaning "default/single slot" (backward-compatible).
// Plugins that only support a single slot can ignore the slot_id parameter
// and use their internal state directly.
#define DETERMINISTIC_DRAFT_SLOT_DEFAULT (-1)

// Compile-time version of this plugin API contract. Bump on any breaking
// SPI change (signatures, semantics); plugins can check it via #ifdef.
#define DETERMINISTIC_DRAFT_API_VERSION 1

// Capability flags (bitmask)
#include "deterministic_draft_capabilities.h"

// Opaque handle representing plugin instance state.
typedef struct DeterministicDraftPlugin DeterministicDraftPlugin;

// --- Lifecycle ---

// Create a new plugin state instance.  Returns NULL on failure.
DeterministicDraftPlugin* deterministic_draft_create(void);

// Destroy a plugin state instance previously returned by create().
void deterministic_draft_destroy(DeterministicDraftPlugin* state);

// --- Capabilities ---

// Return a bitmask of DETERMINISTIC_DRAFT_CAPABILITY_* flags indicating
// which features this plugin supports. The host uses this to determine
// which API functions are available.
//
// Plugins that don't implement this function are assumed to support no
// capabilities (0) - the host should treat them as unusable.
uint32_t deterministic_draft_get_capabilities(DeterministicDraftPlugin* state);

// --- Tokenizer setup (for BITMASK/JUMP_FORWARD plugins) ---

// Provide vocabulary information to the plugin. Called once after create()
// for plugins that need tokenizer info (e.g., XGrammar-based plugins).
//
// |vocab_entries| - array of token strings (encoded vocabulary)
// |vocab_size|   - number of entries in vocab_entries
// |stop_tokens|  - array of stop token IDs (may be NULL)
// |n_stop|       - number of stop token IDs
//
// Returns true on success, false on failure.
// Plugins that don't need vocab info can ignore this call and return true.
bool deterministic_draft_set_vocab(
        DeterministicDraftPlugin* state,
        const char** vocab_entries,
        int32_t vocab_size,
        const int32_t* stop_tokens,
        int32_t n_stop);

// --- Grammar configuration ---

// Load a grammar from an EBNF/GBNF string. Called when the user specifies
// a grammar for constrained generation.
//
// |ebnf_str|      - grammar in EBNF format (GBNF compatible)
// |root_rule|    - name of the root rule (typically "root")
//
// Returns true on success, false on failure.
bool deterministic_draft_set_grammar(
        DeterministicDraftPlugin* state,
        const char* ebnf_str,
        const char* root_rule);

// --- Bitmask (CAPABILITY_BITMASK) ---

// Fill a bitmask indicating which token IDs are valid for the next step.
// Called before sampling to constrain generation.
//
// |slot_id|   - identifies the inference slot
// |bitmask|   - pre-allocated array of uint32_t, size = (vocab_size + 31) / 32
//               Each bit represents a token ID. Bit set = valid, bit clear = invalid.
// |vocab_size|- total vocabulary size
//
// Returns true if the bitmask was filled and should be applied.
// Returns false if no constraint is needed (all tokens valid).
bool deterministic_draft_fill_bitmask(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint32_t* bitmask,
        int32_t vocab_size);

// --- Jump-forward (CAPABILITY_JUMP_FORWARD) ---

// Get the longest string that is uniquely determined by the current grammar state.
// Called after accepting tokens to skip deterministic sequences.
//
// |slot_id|      - identifies the inference slot
// |out_length|   - set to the length of the returned string in bytes
//
// Returns a pointer to the jump-forward string (valid until next plugin call),
// or NULL if no jump-forward is available.
// The returned string is owned by the plugin and must not be freed by the caller.
const char* deterministic_draft_get_jump_forward(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t* out_length);

// --- Commit ---

// Permanently commit |token_id| and |token_text| (length |token_length|) to the plugin's
// grammar state.  Called after a token has been fully accepted.
//
// |slot_id| - identifies the inference slot (0..N-1, or
//   DETERMINISTIC_DRAFT_SLOT_DEFAULT for single-slot use).
// |token_id| - the token ID (used by XGrammar's AcceptToken for tokenizer-aware matching).
void deterministic_draft_commit(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t token_id,
        const char* token_text,
        int32_t token_length);

// Undo the last |n_tokens| commit() calls for the given slot, restoring the
// grammar matcher to the state it was in before those commits. Used when a
// standard (non-accept-all) target-model verification accepts fewer tokens
// than the grammar already committed during draft filtering, so the grammar
// state stays consistent with what was actually emitted.
//
// Returns true on success, false if the plugin doesn't support rollback or
// n_tokens exceeds what can be rolled back.
bool deterministic_draft_rollback(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        int32_t n_tokens);

// --- High-level filter helpers (CAPABILITY_BITMASK) ---

// Filter a batch of draft tokens against the grammar bitmask.
// Commit-on-accept: every accepted token is committed to the grammar state
// as a side effect. Stops at the first invalid token.
//
// |tokens|   - array of token IDs to validate
// |n_tokens| - number of tokens
//
// Returns the number of leading valid tokens (committed to grammar state).
// Tokens beyond the first invalid token are NOT committed.
int32_t deterministic_draft_filter_draft(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const int32_t* tokens,
        int32_t n_tokens);

// Fill a bitmask and apply it to a logits array.
// Sets logits[i] = -1e30f for invalid tokens (bit clear in bitmask).
//
// |bitmask|   - pre-allocated bitmask array, size = (vocab_size + 31) / 32
// |vocab_size|- total vocabulary size
// |logits|    - logits array to constrain (modified in-place)
//
// Returns true if a bitmask was applied, false if no constraint needed.
bool deterministic_draft_apply_bitmask(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint32_t* bitmask,
        int32_t vocab_size,
        float* logits);

// Commit multiple tokens to the grammar state.
// Converts token IDs to text internally using the vocabulary from set_vocab().
//
// |tokens|   - array of token IDs
// |n_tokens| - number of tokens
void deterministic_draft_commit_tokens(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const int32_t* tokens,
        int32_t n_tokens);

// --- Termination query ---

// Return true if the slot's grammar has reached a complete, terminal parse
// and will reject any further tokens. The host treats this like
// end-of-sequence for that slot. Optional - plugins that don't implement
// this are treated as never terminated.
bool deterministic_draft_is_terminated(
        DeterministicDraftPlugin* state,
        int32_t slot_id);

// --- State access ---

// Reset the plugin state for the given slot to its initial (empty) state.
void deterministic_draft_reset(
        DeterministicDraftPlugin* state,
        int32_t slot_id);

// --- State serialization (optional) ---

// These functions let the host persist a slot's grammar state (e.g. server
// slot checkpointing on context overflow or pre-emption) and restore it
// later. A plugin implements either all three or none.

// Return the number of bytes needed to serialize the slot's state, 0 if the
// slot has no state to save, or -1 if serialization is not supported.
int32_t deterministic_draft_state_get_size(
        DeterministicDraftPlugin* state,
        int32_t slot_id);

// Serialize the slot's state into |buffer| of |buffer_size| bytes.
// Return the number of bytes written, or -1 on error.
int32_t deterministic_draft_state_get_data(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        uint8_t* buffer,
        int32_t buffer_size);

// Restore the slot's state from data previously produced by state_get_data.
// Return true on success.
bool deterministic_draft_state_set_data(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const uint8_t* data,
        int32_t size);

// --- Language control (multi-language / polyglot support) ---

// Select a bundled grammar for the given slot by language name (e.g. "python",
// "c", "java", "javascript"). The plugin resolves and loads the grammar itself -
// hosts never need to know grammar file paths or formats.
//
// Resolution order for the grammar bundle directory:
//   1. DETERMINISTIC_DRAFT_GRAMMAR_DIR environment variable, if set
//   2. "<plugin_directory>/grammars/" (directory containing this .so/.dylib/.dll)
//
// Grammars are loaded from "<grammar_dir>/<lang>.gbnf" as plain data files -
// they can be added or edited without rebuilding the plugin.
//
// Returns true on success, false if the language is unknown or the grammar
// failed to load/compile.
bool deterministic_draft_set_language(
        DeterministicDraftPlugin* state,
        int32_t slot_id,
        const char* lang);

// Return whether the given slot is still in bootstrap language auto-detection
// (i.e. more than one candidate grammar remains unresolved). Once detection
// converges to a single language this returns false. Useful for callers that
// want to detect a non-terminating detection oscillation.
bool deterministic_draft_is_detecting(
        DeterministicDraftPlugin* state,
        int32_t slot_id);

// Return the currently active language name for the given slot.
// Returns "unknown" if the state is invalid or no language has been set.
const char* deterministic_draft_get_language(
        DeterministicDraftPlugin* state,
        int32_t slot_id);

// --- Metadata ---

// Return a version string for this plugin (e.g. "3.0.0").
// The returned string must remain valid for the lifetime of the plugin.
const char* deterministic_draft_get_version(
        DeterministicDraftPlugin* state);

#ifdef __cplusplus
}
#endif

#endif // DETERMINISTIC_DRAFT_PLUGIN_H
