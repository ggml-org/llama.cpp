// llama_deterministic_draft.h -- Consumer API for deterministic draft plugins
//
// This is the canonical declaration of the C API used to load and interact
// with deterministic draft plugins (.so/.dylib/.dll). It is implemented by
// src/llama-deterministic-draft-serviceloader.cpp, both inside libllama and
// as the standalone libdeterministic_draft_spec SDK library (external/).
// external/CMakeLists.txt copies this header to external/include/ at build
// time; do not edit the copy.
//
// This header is self-contained - it does not require llama.h or any other
// llama.cpp headers.

#ifndef LLAMA_DETERMINISTIC_DRAFT_API_H
#define LLAMA_DETERMINISTIC_DRAFT_API_H

#include "deterministic_draft_capabilities.h"

#include <stdbool.h>
#include <stdint.h>

// Export macro, mirrors the LLAMA_API pattern in llama.h:
//   - in libllama builds, LLAMA_SHARED/LLAMA_BUILD come from the llama target
//   - the standalone SDK library target defines LLAMA_SHARED + LLAMA_BUILD
//     itself (external/CMakeLists.txt)
#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_DET_API __declspec(dllexport)
#        else
#            define LLAMA_DET_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_DET_API __attribute__((visibility("default")))
#    endif
#else
#    define LLAMA_DET_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the plugin loader instance.
struct llama_deterministic_draft;

// Threading: this API is not thread-safe. All calls on the same instance must
// be serialized by the caller (per-slot state is shared with the plugin).
// Distinct instances are independent.

// Aliases for the LLAMA_* names previously declared in llama.h
#ifndef LLAMA_DETERMINISTIC_DRAFT_CAPABILITY_BITMASK
#    define LLAMA_DETERMINISTIC_DRAFT_CAPABILITY_BITMASK      DETERMINISTIC_DRAFT_CAPABILITY_BITMASK
#    define LLAMA_DETERMINISTIC_DRAFT_CAPABILITY_JUMP_FORWARD DETERMINISTIC_DRAFT_CAPABILITY_JUMP_FORWARD
#endif

// --- Lifecycle ---

// Initialize a deterministic draft plugin from a shared library path.
// Loads the library, resolves the contract symbols, creates plugin state.
// Returns NULL on failure; the failure reason is reported through the log
// (llama log in libllama builds, stderr in the standalone SDK library).
LLAMA_DET_API struct llama_deterministic_draft * llama_deterministic_draft_init(const char * plugin_path);

// Free a deterministic draft plugin instance.
// Destroys plugin state and unloads the library.
LLAMA_DET_API void llama_deterministic_draft_free(struct llama_deterministic_draft * draft);

// --- Capabilities ---

// Query plugin capabilities. Returns a bitmask of DETERMINISTIC_DRAFT_CAPABILITY_*
// flags, or 0 if the plugin doesn't implement capability negotiation.
LLAMA_DET_API uint32_t llama_deterministic_draft_get_capabilities(struct llama_deterministic_draft * draft);

// --- Tokenizer setup (for BITMASK/JUMP_FORWARD plugins) ---

// Provide vocabulary information to the plugin. Called once after init for
// plugins that need tokenizer info. Returns true on success, false on failure.
// Plugins that don't need vocab info return true.
LLAMA_DET_API bool llama_deterministic_draft_set_vocab(struct llama_deterministic_draft * draft,
                                                       const char **                      vocab_entries,
                                                       int32_t                            vocab_size,
                                                       const int32_t *                    stop_tokens,
                                                       int32_t                            n_stop);

// --- Bitmask (DETERMINISTIC_DRAFT_CAPABILITY_BITMASK) ---

// Fill a bitmask indicating which token IDs are valid for the next step.
// |bitmask| must be pre-allocated with size (vocab_size + 31) / 32.
// Returns true if the bitmask was filled and should be applied, false if no
// constraint is needed.
LLAMA_DET_API bool llama_deterministic_draft_fill_bitmask(struct llama_deterministic_draft * draft,
                                                          int32_t                            slot_id,
                                                          uint32_t *                         bitmask,
                                                          int32_t                            vocab_size);

// --- Jump-forward (DETERMINISTIC_DRAFT_CAPABILITY_JUMP_FORWARD) ---

// Get the longest string uniquely determined by the current constraint state.
// Returns NULL if no jump-forward is available. The returned string is owned
// by the plugin and must not be freed by the caller.
LLAMA_DET_API const char * llama_deterministic_draft_get_jump_forward(struct llama_deterministic_draft * draft,
                                                                      int32_t                            slot_id,
                                                                      int32_t *                          out_length);

// --- Commit ---

// Commit an accepted token to the plugin's constraint state.
// |token_id| is used by tokenizer-aware plugins for token-aware matching (e.g. XGrammar's AcceptToken).
LLAMA_DET_API void llama_deterministic_draft_commit(struct llama_deterministic_draft * draft,
                                                    int32_t                            slot_id,
                                                    int32_t                            token_id,
                                                    const char *                       token_text,
                                                    int32_t                            token_length);

// Undo the last |n_tokens| commit() calls for the given slot, restoring the
// constraint matcher to its prior state. Used when standard (non-accept-all)
// target-model verification accepts fewer tokens than the filter already
// committed during draft filtering. Returns true on success, false if the
// plugin doesn't support rollback or n_tokens is invalid. A call with
// n_tokens == 0 is a no-op and must return true - hosts use it to probe
// whether the plugin implements the method.
LLAMA_DET_API bool llama_deterministic_draft_rollback(struct llama_deterministic_draft * draft,
                                                      int32_t                            slot_id,
                                                      int32_t                            n_tokens);

// --- High-level filter helpers ---

// Filter a batch of draft tokens against the constraint bitmask.
// Commit-on-accept: every accepted token is committed to the constraint state as
// a side effect. Stops at the first invalid token and returns the number of
// leading valid (now committed) tokens. Callers that accept fewer tokens than
// returned must restore the constraint state with llama_deterministic_draft_rollback().
LLAMA_DET_API int32_t llama_deterministic_draft_filter_draft(struct llama_deterministic_draft * draft,
                                                             int32_t                            slot_id,
                                                             const int32_t *                    tokens,
                                                             int32_t                            n_tokens);

// Fill a bitmask and apply it to a logits array.
// Sets logits[i] = -1e30f for invalid tokens (bit clear in bitmask).
// Returns true if a bitmask was applied, false if no constraint needed.
LLAMA_DET_API bool llama_deterministic_draft_apply_bitmask(struct llama_deterministic_draft * draft,
                                                           int32_t                            slot_id,
                                                           uint32_t *                         bitmask,
                                                           int32_t                            vocab_size,
                                                           float *                            logits);

// Commit multiple tokens to the constraint state.
// Converts token IDs to text internally using the vocabulary from set_vocab().
LLAMA_DET_API void llama_deterministic_draft_commit_tokens(struct llama_deterministic_draft * draft,
                                                           int32_t                            slot_id,
                                                           const int32_t *                    tokens,
                                                           int32_t                            n_tokens);

// --- Termination query ---

// Returns true if the slot's constraint has reached a complete, terminal state
// and will reject any further tokens. Callers should treat this like
// end-of-sequence for that slot: stop drafting instead of continuing to call
// fill_bitmask/commit/filter_draft, which will just keep rejecting everything.
// Returns false if the plugin doesn't implement this, the state is invalid,
// or the slot has not reached a terminal state.
LLAMA_DET_API bool llama_deterministic_draft_is_terminated(struct llama_deterministic_draft * draft, int32_t slot_id);

// --- State access ---

// Reset the plugin state for the given slot to its initial (empty) state.
LLAMA_DET_API void llama_deterministic_draft_reset(struct llama_deterministic_draft * draft, int32_t slot_id);

// --- State serialization (optional) ---

// Returns the number of bytes needed to serialize the slot's state, 0 if the
// slot has no state to save, or -1 if the plugin does not implement state
// serialization.
LLAMA_DET_API int32_t llama_deterministic_draft_state_get_size(struct llama_deterministic_draft * draft, int32_t slot_id);

// Serialize the slot's state into |buffer| of |buffer_size| bytes.
// Returns the number of bytes written, or -1 on error.
LLAMA_DET_API int32_t llama_deterministic_draft_state_get_data(struct llama_deterministic_draft * draft,
                                                               int32_t                            slot_id,
                                                               uint8_t *                          buffer,
                                                               int32_t                            buffer_size);

// Restore the slot's state from data previously produced by state_get_data.
// The serialized format starts with a magic and a u32 version, so mismatched
// data is rejected. Returns true on success.
LLAMA_DET_API bool llama_deterministic_draft_state_set_data(struct llama_deterministic_draft * draft,
                                                            int32_t                            slot_id,
                                                            const uint8_t *                    data,
                                                            int32_t                            size);

// --- Metadata ---

// Return the plugin version string (e.g. "3.0.0").
// The returned string is valid for the lifetime of the plugin.
LLAMA_DET_API const char * llama_deterministic_draft_get_version(struct llama_deterministic_draft * draft);

#ifdef __cplusplus
}
#endif

#endif  // LLAMA_DETERMINISTIC_DRAFT_API_H
