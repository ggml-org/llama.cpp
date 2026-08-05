// cllama_shim.h - thin, self-contained C bridge between Tessera Studio and
// libllama (the llama.cpp fork this repo ships).
//
// The implementation (cllama_shim.c) loads libllama.dylib at runtime with
// dlopen and resolves every symbol with dlsym, so the SwiftPM package links
// and runs even when no native library is present: cllama_load_library()
// simply reports failure and the Swift provider falls back to another backend.
//
// This header intentionally does NOT include llama.h. It exposes only an
// opaque handle and plain C types so the Swift-facing module stays decoupled
// from the (large) llama.cpp ABI. The llama.h structs are used only inside
// cllama_shim.c, which includes the real header for ABI correctness.
//
// Conventions:
//   - Functions returning int use non-zero for success / 0 for failure where
//     noted, matching the "is available" style used by tessera_ffi.h.
//   - cllama_last_error() returns a static, NUL-terminated UTF-8 string that
//     is valid until the next shim call on the same thread. "" means no error.

#ifndef CLLAMA_SHIM_H
#define CLLAMA_SHIM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque inference engine: owns a llama_model + llama_context + sampler chain.
typedef struct cllama_engine cllama_engine;

// Per-token streaming callback. `piece` is a NUL-terminated UTF-8 chunk for a
// single decoded token; `token_id` is the vocab token id. The string is only
// valid for the duration of the call - copy it if you need to keep it.
typedef void (*cllama_token_callback)(const char *piece, int32_t token_id, void *user_data);

// Load libllama.dylib and resolve the required symbols.
//   dylib_path_override: explicit path to libllama.dylib, or NULL/"" to search
//     the TESSERA_LLAMA_DYLIB env var and then the default loader paths.
// Returns non-zero on success. Idempotent: a successful load is cached.
int cllama_load_library(const char *dylib_path_override);

// Non-zero once cllama_load_library() has succeeded.
int cllama_is_available(void);

// Last error message for the calling thread ("" if none).
const char *cllama_last_error(void);

// Load a GGUF model, create a context (n_ctx tokens, n_threads worker threads;
// 0 picks a sensible default), and build a greedy sampler chain.
// n_gpu_layers < 0 offloads all layers, 0 keeps everything on the CPU.
// Returns NULL on error (see cllama_last_error()). Requires a loaded library.
cllama_engine *cllama_engine_load(const char *model_path,
                                  uint32_t n_ctx,
                                  int32_t n_threads,
                                  int32_t n_gpu_layers);

// Tokenize + decode `prompt`, then generate up to max_tokens tokens, invoking
// on_token for each decoded piece. Stops early on an end-of-generation token.
// Returns the number of tokens generated, or -1 on error.
int32_t cllama_engine_generate(cllama_engine *eng,
                               const char *prompt,
                               int32_t max_tokens,
                               cllama_token_callback on_token,
                               void *user_data);

// Free the context, sampler, and model. NULL-safe.
void cllama_engine_free(cllama_engine *eng);

// Opaque speculative-decoding engine: trunk + drafter models, contexts, and
// the spec handle, owned by libllama-common's tessera_rt runtime.
typedef struct cllama_spec_engine cllama_spec_engine;

// Per-step trace callback: `jsonl_line` is one NUL-terminated
// llama.tessera.spec.v1 record (provenance "runtime", session sid attached).
// The string is only valid for the duration of the call - copy it if you
// need to keep it.
typedef void (*cllama_trace_callback)(const char *jsonl_line, void *user_data);

// Load libllama-common.dylib and resolve the tessera_rt_* entry points.
// Candidates, in order: dylib_path_override, the TESSERA_LLAMA_COMMON_DYLIB
// env var, a sibling of the already-resolved libllama.dylib (both ship from
// the same build), then the default loader search.
// Returns non-zero on success. Idempotent: a successful load is cached.
// A failure here is not fatal for the app: the provider degrades to the
// single-model path (see cllama_is_spec_available()).
int cllama_load_spec_library(const char *dylib_path_override);

// Non-zero once cllama_load_spec_library() has succeeded.
int cllama_is_spec_available(void);

// Load trunk + drafter and build the runtime spec engine.
// draft_max: max drafted tokens per step.
// Returns NULL on error (see cllama_last_error()). Requires the spec library.
cllama_spec_engine *cllama_engine_load_spec(const char *trunk_path,
                                            const char *draft_path,
                                            uint32_t n_ctx,
                                            int32_t n_threads,
                                            int32_t n_gpu_layers,
                                            int32_t draft_max);

// Tokenize + decode `prompt` with speculative decoding, streaming accepted
// pieces through on_token. telemetry_topk: 0 = no trace emission (cheap
// path); > 0 = one spec.v1 record per spec step through on_trace (may be
// NULL). Returns the number of tokens generated, or -1 on error.
int32_t cllama_engine_generate_spec(cllama_spec_engine *eng,
                                    const char *prompt,
                                    int32_t max_tokens,
                                    int32_t telemetry_topk,
                                    cllama_token_callback on_token,
                                    cllama_trace_callback on_trace,
                                    void *user_data);

// Free the spec engine. NULL-safe.
void cllama_engine_free_spec(cllama_spec_engine *eng);

// Vocab size of the engine's model (llama_vocab_n_tokens). Used by the
// curation stage's compatibility check: captured token ids must fall inside
// the current trunk's vocab. Returns -1 on error.
int32_t cllama_engine_n_vocab(const cllama_engine *eng);

// Detokenize `n_tokens` token ids into UTF-8 text using the engine's vocab
// (stateless, no context needed - used by the trace curation stage to decode
// accepted token sequences). Returns the number of bytes written excluding
// the NUL on success; if out_buf is too small, returns the negative required
// size; -1 on any other error.
int32_t cllama_detokenize(const cllama_engine *eng,
                          const int32_t *tokens,
                          int32_t n_tokens,
                          char *out_buf,
                          int32_t out_len);

#ifdef __cplusplus
}
#endif

#endif // CLLAMA_SHIM_H
