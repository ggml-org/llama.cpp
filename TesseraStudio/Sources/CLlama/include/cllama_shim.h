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

#ifdef __cplusplus
}
#endif

#endif // CLLAMA_SHIM_H
