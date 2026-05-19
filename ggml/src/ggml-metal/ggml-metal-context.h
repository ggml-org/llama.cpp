#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

//
// Profiling
//

// Opaque profiler state, owned by the C++ backend layer (ggml-metal.cpp). Holds the std::vector of
// records returned via the ggml_backend_profiler interface. ggml_metal_t keeps a borrowed pointer
// so that graph_compute can push records when sampling is active.
struct ggml_metal_profiler_state;

// Inject (or clear, with NULL) the profiler state pointer. Called once at backend init.
void ggml_metal_set_profiler_state(ggml_metal_t ctx, struct ggml_metal_profiler_state * state);

// Bridge function implemented in ggml-metal.cpp. Used by graph_compute (in .m) to push records.
void ggml_metal_profiler_push_record(
        struct ggml_metal_profiler_state * state,
        const struct ggml_tensor * node,
        uint64_t start_ns,
        uint64_t end_ns);

// Query whether the injected profiler state is currently enabled.
// (Avoids exposing the C++ struct layout to the .m file.)
bool ggml_metal_profiler_is_enabled(struct ggml_metal_profiler_state * state);

// Query the split-id currently set on the profiler state.
int  ggml_metal_profiler_get_split_id(struct ggml_metal_profiler_state * state);

#ifdef __cplusplus
}
#endif
