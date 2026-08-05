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

// Test instrumentation for the TILE640_MATMUL inner-dim tiling path.
// The counter increments once per ANE sub-matmul dispatched (i.e. once
// per tile in the tiled path, once per op in the non-tiled path). The
// reset zeroes the counter. Used by tests/test-ane-tile640-matmul.cpp
// to assert the tile-vs-no-tile dispatch policy (4 dispatches for the
// 4096x4096 case under the 4096-threshold / 1024-tile-size constants).
GGML_BACKEND_API uint64_t ggml_backend_ane_tile640_dispatch_count(void);
GGML_BACKEND_API void ggml_backend_ane_tile640_dispatch_count_reset(void);

// Tiling policy constants (also exposed for tests / future tuning).
// The dispatch splits the inner-dim into tiles of kTile640InnerDimTileSize
// when in_dim >= kTile640InnerDimThreshold. Both knobs are at the top
// of the TILE640_MATMUL dispatch case in ggml-ane.mm.
GGML_BACKEND_API int64_t ggml_backend_ane_tile640_threshold(void);
GGML_BACKEND_API int64_t ggml_backend_ane_tile640_tile_size(void);

// Lock-free data plane: a cross-backend IOSurface-backed buffer.
//
// Allocates an IOSurface that the CPU, Metal, and ANE backends can all
// read and write without copies. This is the data-plane primitive for
// the cross-backend lock-free dispatch (per the prism-engine
// SharedEventContract / Arena pattern, mapped to llama.cpp / ggml).
//
//   bytes: minimum byte count. The actual allocation is rounded up to
//          the ANE-mandated 16 KB page boundary and clamped to the
//          64 KB IOSurface minimum (Orion #4).
//
// The returned buffer's `get_base()` returns the locked CVPixelBuffer
// base address (CPU view). The Metal view is exposed via
// ggml_backend_ane_iosurface_buffer_get_mtl_buffer (lazily created on
// first access; cached for the buffer's lifetime). The ANE view is
// the raw IOSurfaceRef via ggml_backend_ane_iosurface_buffer_get_iosurface.
//
// Returns nullptr on allocation failure. The caller owns the buffer
// and must free it via ggml_backend_buffer_free.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_ane_iosurface_buffer_alloc(
        size_t bytes);

// Returns true if the buffer is an ANE cross-backend IOSurface buffer.
GGML_BACKEND_API bool ggml_backend_ane_iosurface_buffer_check(
        ggml_backend_buffer_t buffer);

// Get the raw IOSurfaceRef (ANE view) for the buffer. The IOSurface
// is locked for the buffer's lifetime; the returned ref is retained
// by the buffer. Returns NULL if the buffer is not an ANE
// IOSurface-backed buffer.
//
// The IOSurfaceRef can be wrapped as a _ANEIOSurfaceObject (the ANE
// private framework's IOSurface handle) for direct ANE dispatch.
GGML_BACKEND_API void * ggml_backend_ane_iosurface_buffer_get_iosurface(
        ggml_backend_buffer_t buffer);

// Wrap the IOSurface as an MTLBuffer (Metal view). Lazily creates the
// MTLBuffer on first call and caches it. The MTLBuffer shares memory
// with the IOSurface (no copy). The returned MTLBuffer is owned by the
// buffer (released on free). Returns NULL on failure or if the buffer
// is not an ANE IOSurface buffer.
GGML_BACKEND_API void * ggml_backend_ane_iosurface_buffer_get_mtl_buffer(
        ggml_backend_buffer_t buffer);

#ifdef __cplusplus
}
#endif
