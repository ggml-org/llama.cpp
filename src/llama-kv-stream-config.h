#pragma once

#include "llama-arch.h"
#include "../ggml/src/ggml-kv-stream.h"

#include <limits>
#include <vector>

struct llama_kv_stream_layer {
    int32_t id = -1;
    ggml_kv_stream_shape shape;
    ggml_kv_stream_capabilities capabilities;
    bool offloaded = false;
};
struct llama_kv_stream_config {
    bool enabled = false;
    llm_arch arch = LLM_ARCH_UNKNOWN;
    bool target_context = false;
    bool flash_attention = false;
    bool kv_offload = false;
    bool cuda_backend = false;
    uint32_t sequences = 0, devices = 0;
    size_t context_tokens = 0;
    std::vector<llama_kv_stream_layer> layers;
};
enum class llama_kv_stream_config_status {
    success, unsupported_model, unsupported_context, unsupported_device, unsupported_attention,
    invalid_context_size, invalid_layers, unsupported_geometry, nonuniform_geometry, geometry_error,
};
struct llama_kv_stream_config_result {
    llama_kv_stream_config_status status = llama_kv_stream_config_status::success;
    // Index into config.layers, or SIZE_MAX for a configuration-wide error.
    size_t layer = std::numeric_limits<size_t>::max();
    ggml_kv_stream_result geometry;
};
struct llama_kv_stream_config_plan {
    bool enabled = false;
    size_t padded_context_tokens = 0, layers = 0;
    ggml_kv_stream_execution page;
    ggml_kv_stream_layout context;
    size_t all_layers_bytes = 0;
};

// Initial Qwen35 target-context contract only. Caller-reported backend capabilities are not hardware probes.
// Disabled configuration is a no-op; failure preserves output. No runtime, buffer, or dispatch is enabled here.
llama_kv_stream_config_result llama_kv_stream_config_validate(
        const llama_kv_stream_config & config, llama_kv_stream_config_plan & output);
