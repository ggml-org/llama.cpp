// tessera_ffi_stub.c - link-time placeholder for the Tessera engine FFI.
//
// The real implementations ship in tessera.xcframework. This stub lets the
// SwiftPM package link and run without the native engine: every operation
// reports unavailable / failure, so TesseraFFIBridge.isAvailable is false and
// the Swift factory falls back to the CLI bridge. Replace this translation
// unit with the xcframework link when the native engine is integrated.

#include "include/tessera_ffi.h"
#include <stdlib.h>

int tessera_ffi_is_available(void) {
    return 0; // stub: no native engine linked
}

const char *tessera_ffi_version(void) {
    return "stub-0.0.0";
}

int tessera_quantize(const char *model_path,
                     const char *output_path,
                     const char *config_json) {
    (void)model_path; (void)output_path; (void)config_json;
    return -1;
}

int tessera_calibrate(const char *model_path,
                      const char *corpus_path,
                      const char *config_json) {
    (void)model_path; (void)corpus_path; (void)config_json;
    return -1;
}

int tessera_evolve(const char *model_path, const char *config_json) {
    (void)model_path; (void)config_json;
    return -1;
}

char *tessera_evaluate(const char *model_path, const char *config_json) {
    (void)model_path; (void)config_json;
    return NULL;
}

int tessera_convert(const char *model_path,
                    const char *output_path,
                    const char *format) {
    (void)model_path; (void)output_path; (void)format;
    return -1;
}

char *tessera_inspect_sidecar(const char *sidecar_path) {
    (void)sidecar_path;
    return NULL;
}

char *tessera_list_models(const char *dir) {
    (void)dir;
    return NULL;
}

void tessera_free_string(char *s) {
    free(s);
}
