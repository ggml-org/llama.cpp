// tessera_ffi_reference.h - REFERENCE ONLY, NOT COMPILED.
//
// This file documents the planned C FFI surface for the native Tessera
// engine (tessera.xcframework, see docs/tessera-studio-design.md section
// 3). It is intentionally NOT on any include path and is not part of any
// target: no source in this package imports it, and SwiftPM does not
// compile it. It exists only as a contract sketch for the future
// xcframework integration.
//
// The previous CTesseraFFI target (a link-time stub whose
// tessera_ffi_is_available() always returned 0) has been removed: every
// call site gated on it was statically dead, and the working on-device
// path is the CLlama runtime bridge (dlopen of libllama.dylib). When the
// native engine ships, build a real xcframework target against this
// contract and reintroduce the Swift wrapper.
//
// Conventions:
//   - Functions returning int use 0 for success and a non-zero error code
//     for failure (matches the CLI exit-code convention the tools check).
//   - Functions returning char* return a heap-allocated, NUL-terminated
//     UTF-8 JSON string owned by the caller, which MUST be released with
//     tessera_free_string(). They return NULL on error.
//   - All char* inputs are NUL-terminated UTF-8. config_json is a JSON
//     object.

#ifndef TESSERA_FFI_REFERENCE_H
#define TESSERA_FFI_REFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Availability + version. tessera_ffi_is_available() returns non-zero when a
// real engine is linked. tessera_ffi_version() returns a static,
// NUL-terminated version string (do not free it).
int tessera_ffi_is_available(void);
const char *tessera_ffi_version(void);

// Quantize a GGUF model per config_json. Returns 0 on success.
int tessera_quantize(const char *model_path,
                     const char *output_path,
                     const char *config_json);

// Run imatrix calibration over corpus_path. Returns 0 on success.
int tessera_calibrate(const char *model_path,
                      const char *corpus_path,
                      const char *config_json);

// Run AWQ-evolve policy search. Returns 0 on success.
int tessera_evolve(const char *model_path, const char *config_json);

// Evaluate a model; returns a JSON result string (caller frees) or NULL.
char *tessera_evaluate(const char *model_path, const char *config_json);

// Convert a Tessera GGUF to the named format (e.g. "coreml"). Returns 0 on
// success.
int tessera_convert(const char *model_path,
                    const char *output_path,
                    const char *format);

// Inspect a sidecar JSON; returns a JSON string (caller frees) or NULL.
char *tessera_inspect_sidecar(const char *sidecar_path);

// List models in dir; returns a JSON array string (caller frees) or NULL.
char *tessera_list_models(const char *dir);

// Free a string previously returned by this library. NULL is a no-op.
void tessera_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif // TESSERA_FFI_REFERENCE_H
