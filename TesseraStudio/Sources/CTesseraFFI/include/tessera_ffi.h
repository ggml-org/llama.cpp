// tessera_ffi.h - C FFI surface for the native Tessera engine.
//
// Declares the entry points implemented in TesseraStudio/ffi/tessera_ffi.cpp
// (compiled into tessera.xcframework by TesseraStudio/scripts/build-xcframework.sh)
// and in the standalone stub TesseraStudio/Sources/CTesseraFFI/tessera_ffi.c
// (compiled by SwiftPM when the xcframework is not linked).
//
// The stub returns "unavailable" for every call so SwiftPM builds and tests
// pass without the native engine. When the xcframework is linked into the
// Xcode app the real symbols from the C++ engine take over and
// tessera_ffi_is_available() returns 1.
//
// Conventions:
//   - Functions returning int use 0 for success and a non-zero code otherwise
//     (matches the CLI exit-code convention the Swift tools check). Positive
//     non-zero means "request valid but not runnable via FFI; use the CLI".
//   - Functions returning char* return a heap-allocated, NUL-terminated UTF-8
//     JSON string owned by the caller, which MUST be released with
//     tessera_free_string(). They return NULL on bad arguments.
//   - All char* inputs are NUL-terminated UTF-8. config_json is a JSON object.

#ifndef TESSERA_FFI_H
#define TESSERA_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Availability + version. tessera_ffi_is_available() returns 1 when a real
// engine is linked, 0 for the SwiftPM stub. tessera_ffi_version() returns a
// static, NUL-terminated version string (do not free it).
int         tessera_ffi_is_available(void);
const char *tessera_ffi_version(void);

// Quantize a GGUF model per config_json. Returns 0 on success.
int tessera_quantize(const char *model_path,
                     const char *output_path,
                     const char *config_json);

// Run imatrix calibration over corpus_path. Returns 0 on success.
int tessera_calibrate(const char *model_path,
                      const char *corpus_path,
                      const char *config_json);

// Run AWQ-evolve policy search. Returns 0 on success, 1 when the request is
// valid but the FFI cannot run it (no loaded model context); the caller
// should fall back to the CLI.
int tessera_evolve(const char *model_path, const char *config_json);

// Evaluate a model; returns a JSON result string (caller frees) or NULL.
char *tessera_evaluate(const char *model_path, const char *config_json);

// Convert a Tessera GGUF to the named format (e.g. "coreml"). Returns 0 on
// success, 1 when the request is valid but the FFI cannot run it.
int tessera_convert(const char *model_path,
                    const char *output_path,
                    const char *format);

// Inspect a sidecar binary; returns a JSON string (caller frees) or NULL.
char *tessera_inspect_sidecar(const char *sidecar_path);

// List .gguf models in dir; returns a JSON array string (caller frees) or NULL.
char *tessera_list_models(const char *dir);

// Opaque handle to a loaded model context. Wraps a llama_model* on the
// native side; the Swift side keeps it as an unowned OpaquePointer. The
// matching tessera_free_model() call is mandatory - the handle owns the
// underlying native object.
typedef struct tessera_model * tessera_model_handle_t;

// Load a model from a GGUF file. Returns NULL on failure (bad path, parse
// error, OOM). The caller owns the returned handle and must release it
// with tessera_free_model(). The optional n_gpu_layers parameter may be
// NULL; when non-NULL it points to an int32_t with the GPU layer count
// (0 = CPU only, matches llama.cpp's llama_model_default_params.n_gpu_layers
// default of 0 for safety on first cut).
tessera_model_handle_t tessera_load_model(const char *model_path,
                                          const int32_t *n_gpu_layers);

// Release a model context. NULL is a no-op. Calling with a handle already
// freed is undefined behaviour; the Swift actor centralises ownership to
// prevent double-free.
void tessera_free_model(tessera_model_handle_t handle);

// Run AWQ-evolve policy search against the loaded model. Returns 0 on
// success, 1 when the FFI implementation is not yet wired (TODO marker - the
// caller should fall back to the CLI), and a negative code on a hard error
// (bad handle, malformed config).
int tessera_evolve_model(tessera_model_handle_t handle,
                         const char *config_json);

// Run a perplexity / KL forward probe over the loaded model. Returns a
// heap-allocated JSON string (caller frees via tessera_free_string) on
// success, NULL on bad arguments, and a structured {"ok":false,...} JSON
// when the implementation is not yet wired.
char *tessera_evaluate_model(tessera_model_handle_t handle,
                             const char *config_json);

// Convert the loaded model to the named format (e.g. "coreml") and write
// to output_path. Returns 0 on success, 1 when the FFI implementation is
// not yet wired, and a negative code on a hard error.
int tessera_convert_model(tessera_model_handle_t handle,
                          const char *output_path,
                          const char *format);

// Free a string previously returned by this library. NULL is a no-op.
void tessera_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif // TESSERA_FFI_H
