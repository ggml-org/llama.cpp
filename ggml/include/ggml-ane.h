// ggml-ane: Apple Neural Engine backend (Core ML public path).
//
// Backend registration + IOSurface-backed buffer type (slice 1) plus op
// dispatch and graph compute (slice 2). Composite transformer ops whose
// decompositions live in pre-compiled .mlmodelc bundles are routed to Core
// ML; simple element-wise native ops also have a host-mapped fallback path
// so the backend is exercisable without a bundle bound.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_ane_program;

#ifdef __cplusplus
extern "C" {
#endif

// Registry entry exposed to ggml-backend-reg.cpp. Returns the same singleton
// reg on every call; resolves to nullptr on non-Apple-Silicon builds.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ane_reg(void);

// True if backend was created by the ANE registry. Mirrors
// ggml_backend_is_metal so callers can identify ANE backends at runtime.
GGML_BACKEND_API bool ggml_backend_is_ane(ggml_backend_t backend);

// Load a pre-compiled Core ML program directory (.mlmodelc) and bind it to
// the ANE backend device. The bundle supplies the composite-op decompositions
// (RMS norm, RoPE, SDPA, TILE640 dequant) and the matmul kernels; until a
// program is bound the backend only accepts simple element-wise native ops.
//
// `function_name` selects a single entry point from a multifunction bundle
// (nullptr/"" loads the default function).
//
// Returns an opaque program handle (refcounted) or nullptr on load/warmup
// failure. The handle is independent of any specific ggml_backend_t; attach
// it to an ANE backend with ggml_backend_ane_set_program.
GGML_BACKEND_API struct ggml_backend_ane_program * ggml_backend_ane_program_load_from_dir(
        const char * mlmodelc_dir,
        const char * function_name);

GGML_BACKEND_API void ggml_backend_ane_program_free(
        struct ggml_backend_ane_program * program);

// Bind a loaded program to a specific backend instance. Pass nullptr to
// detach. Returns true on success.
GGML_BACKEND_API bool ggml_backend_ane_set_program(
        ggml_backend_t backend,
        struct ggml_backend_ane_program * program);

#ifdef __cplusplus
}
#endif
