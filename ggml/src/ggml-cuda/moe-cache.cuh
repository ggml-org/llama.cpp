#pragma once

// MoE expert cache — registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h) unless disabled with
// GGML_CUDA_MOE_CACHE=0 in the environment.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_moe_cache_register(void);

// surrender the device's cache VRAM under allocator pressure; returns bytes freed
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}
#endif
