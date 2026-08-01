#ifndef LLAMA_TIERED_H
#define LLAMA_TIERED_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Owns a llama_model together with the generated CUDA device list, tensor
// overrides, and tier plan required for the full lifetime of the model.
struct llama_tiered_model;

struct llama_tiered_memory_params {
    // Maximum model-weight bytes permanently resident in VRAM.
    // A value of 0 selects the currently free VRAM minus vram_reserve_bytes.
    uint64_t vram_budget_bytes;

    // Maximum model-weight bytes mapped as page-registered host memory.
    uint64_t dram_budget_bytes;

    // VRAM kept free for the KV cache, activations, graph work buffers, and
    // temporary SSD expert mappings. Used only when vram_budget_bytes is 0.
    uint64_t vram_reserve_bytes;

    // Reserved for a persistent expert cache. The current implementation uses
    // per-operation sparse VMM mappings and therefore accepts only 0 here.
    uint64_t ssd_cache_bytes;

    // Physical CUDA device index.
    int32_t main_gpu;

    // Fail instead of silently promoting an unsupported host/SSD tensor to
    // VRAM. Production callers should keep this enabled.
    bool strict;
};

struct llama_tiered_memory_stats {
    uint64_t vram_bytes;
    uint64_t dram_bytes;
    uint64_t ssd_bytes;

    double active_vram_bytes_per_token;
    double active_dram_bytes_per_token;
    double active_ssd_bytes_per_token;

    uint32_t tensor_count;
    uint32_t ssd_tensor_count;
};

LLAMA_API struct llama_tiered_memory_params llama_tiered_memory_default_params(void);

// Loads a normal or conventionally named split GGUF model using a dedicated
// CUDA tiered backend. The returned owner must be released with
// llama_tiered_model_free. Returns NULL on failure; call
// llama_tiered_last_error for a diagnostic string owned by the current thread.
LLAMA_API struct llama_tiered_model * llama_tiered_model_load_from_file(
        const char * path_model,
        struct llama_model_params model_params,
        struct llama_tiered_memory_params tiered_params);

// Borrowed pointer. Do not pass it to llama_model_free directly.
LLAMA_API struct llama_model * llama_tiered_model_get_model(
        struct llama_tiered_model * tiered_model);

LLAMA_API const struct llama_tiered_memory_stats * llama_tiered_model_get_stats(
        const struct llama_tiered_model * tiered_model);

LLAMA_API void llama_tiered_model_free(struct llama_tiered_model * tiered_model);

LLAMA_API const char * llama_tiered_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TIERED_H
