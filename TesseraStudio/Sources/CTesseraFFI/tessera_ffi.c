// tessera_ffi.c - SwiftPM stub for the Tessera FFI surface.
//
// This is the UNBUILT DEFAULT: it compiles when the package is built by
// SwiftPM (swift build / swift test) so the Swift wrapper links without the
// native engine present. Every entry point reports unavailable, and the
// int-returning operations return a positive code meaning "valid request,
// not runnable via FFI" so the Swift tools fall back to the CLI subprocess
// (matching the recovery's structured-error contract).
//
// When tessera.xcframework is linked into the Xcode app the REAL
// implementation in TesseraStudio/ffi/tessera_ffi.cpp (compiled by CMake,
// see TesseraStudio/scripts/build-xcframework.sh) supplies these symbols
// instead, and tessera_ffi_is_available() returns 1. The Xcode project does
// not compile this stub - it is on the SwiftPM source path only.

#include "include/tessera_ffi.h"

#include <stdint.h>
#include <stdlib.h>

// A small static JSON string for the char*-returning stubs: signals that the
// native engine is not linked and the caller should use the CLI. Caller frees
// the malloc'd copy via tessera_free_string.
static char * ts_stub_unavailable_json(void) {
    const char *kMsg =
        "{\"ok\":false,\"error\":\"tessera FFI not linked\","
        "\"backend\":\"cli-fallback\"}";
    char *p = (char *)malloc(64);
    if (p) {
        size_t i = 0;
        for (; kMsg[i] != '\0' && i < 63; ++i) p[i] = kMsg[i];
        p[i] = '\0';
    }
    return p;
}

int tessera_ffi_is_available(void) {
    return 0;
}

const char *tessera_ffi_version(void) {
    return "tessera-ffi-stub";
}

int tessera_quantize(const char *model_path,
                     const char *output_path,
                     const char *config_json) {
    // Arguments validated; signal "use the CLI" (positive, non-error).
    (void)config_json;
    if (!model_path || !output_path) return -1;
    return 1;
}

int tessera_calibrate(const char *model_path,
                      const char *corpus_path,
                      const char *config_json) {
    (void)model_path;
    (void)config_json;
    if (!corpus_path) return -1;
    return 1;
}

int tessera_evolve(const char *model_path, const char *config_json) {
    (void)model_path;
    (void)config_json;
    return 1;
}

char *tessera_evaluate(const char *model_path, const char *config_json) {
    (void)model_path;
    (void)config_json;
    return ts_stub_unavailable_json();
}

int tessera_convert(const char *model_path,
                    const char *output_path,
                    const char *format) {
    if (!model_path || !output_path || !format) return -1;
    return 1;
}

char *tessera_inspect_sidecar(const char *sidecar_path) {
    (void)sidecar_path;
    return ts_stub_unavailable_json();
}

char *tessera_list_models(const char *dir) {
    // list_models always returns a valid (possibly empty) JSON array in the
    // real impl; the stub returns the unavailable marker so the Swift bridge
    // can fall back to FileManager enumeration.
    (void)dir;
    return ts_stub_unavailable_json();
}

// --- model-context variants (header added 2026-08; see include/tessera_ffi.h) ---
//
// Stub semantics: loadModel returns NULL (so TesseraEngineContext throws
// TesseraError.engineUnavailable), freeModel is a no-op on NULL, and the
// *_model operations return the unavailable marker so the bridge falls back
// to the CLI subprocess. This keeps SwiftPM builds green without the
// xcframework and exercises the same error paths the real impl will hit
// when its engine wiring is incomplete.

tessera_model_handle_t tessera_load_model(const char *model_path,
                                          const int32_t *n_gpu_layers) {
    (void)model_path;
    (void)n_gpu_layers;
    return NULL;
}

void tessera_free_model(tessera_model_handle_t handle) {
    (void)handle;
}

int tessera_evolve_model(tessera_model_handle_t handle, const char *config_json) {
    (void)handle;
    (void)config_json;
    return 1;
}

char *tessera_evaluate_model(tessera_model_handle_t handle, const char *config_json) {
    (void)handle;
    (void)config_json;
    return ts_stub_unavailable_json();
}

int tessera_convert_model(tessera_model_handle_t handle,
                          const char *output_path,
                          const char *format) {
    (void)handle;
    if (!output_path || !format) return -1;
    return 1;
}

void tessera_free_string(char *s) {
    free(s);
}
