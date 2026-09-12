#pragma once

#include "ggml-backend.h"

// Internal, optional CUDA registry extensions. A key is the fixed graph's first node pointer.
// Call release only after backend completion and before releasing graph storage; it does not synchronize.
// These hooks do not transfer backend ownership and must not race with graph execution or cache mutation.
using ggml_backend_cuda_graph_release_t = void (*)(ggml_backend_t backend, const void * key);
using ggml_backend_cuda_graph_is_captured_t = bool (*)(ggml_backend_t backend, const void * key);

// Retire all scheduler-created graph keys after completion; the caller exclusively owns the backend cache.
using ggml_backend_cuda_graph_release_all_t = void (*)(ggml_backend_t backend);
