// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

typedef struct ggml_backend_metal_event * ggml_backend_metal_event_t;

GGML_BACKEND_API ggml_backend_metal_event_t ggml_backend_metal_event_new(ggml_backend_t backend);
GGML_BACKEND_API void                       ggml_backend_metal_event_free(ggml_backend_metal_event_t event);
GGML_BACKEND_API void   ggml_backend_metal_event_signal(ggml_backend_metal_event_t event, uint64_t value);
GGML_BACKEND_API void * ggml_backend_metal_event_raw(ggml_backend_metal_event_t event);

struct ggml_metal_moe_intercept {
    int                        n;
    uint32_t                   seq;
    const struct ggml_tensor * msg_tensor;
    size_t                     off_req;
    size_t                     off_selected;
    size_t                     off_remapped;
    ggml_backend_metal_event_t event;
    bool                       reuse;
};

typedef bool (*ggml_metal_moe_query_fn)(void *                            user_data,
                                        const struct ggml_tensor *        src0,
                                        const struct ggml_tensor *        ids,
                                        struct ggml_metal_moe_intercept * out);

struct ggml_metal_moe_handler {
    ggml_metal_moe_query_fn fn;
    void *                  user_data;
};

GGML_BACKEND_API void ggml_backend_metal_set_moe_handler(ggml_backend_t backend, struct ggml_metal_moe_handler handler);

#ifdef __cplusplus
}
#endif
