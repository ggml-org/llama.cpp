#include "llama-kv-stream-config.h"

// Uniform slot geometry includes the K/V split, not just their combined page size.
static bool same_shape(const ggml_kv_stream_shape & a, const ggml_kv_stream_shape & b) {
    return a.type_k == b.type_k && a.type_v == b.type_v && a.head_dim_k == b.head_dim_k &&
           a.head_dim_v == b.head_dim_v && a.heads == b.heads && a.page_tokens == b.page_tokens && a.alignment == b.alignment;
}

// Keep the initial integration restricted to the reference model/runtime contract.
llama_kv_stream_config_result llama_kv_stream_config_validate(
        const llama_kv_stream_config & config, llama_kv_stream_config_plan & output) {
    using status = llama_kv_stream_config_status;
    const size_t no_layer = std::numeric_limits<size_t>::max();
    if (!config.enabled) {
        output = {};
        return {};
    }
    if (config.arch != LLM_ARCH_QWEN35) return {status::unsupported_model, no_layer, {}};
    if (!config.target_context || config.sequences != 1) return {status::unsupported_context, no_layer, {}};
    if (!config.cuda_backend || config.devices != 1 || !config.kv_offload) return {status::unsupported_device, no_layer, {}};
    if (!config.flash_attention) return {status::unsupported_attention, no_layer, {}};
    constexpr size_t page_tokens = 256;
    constexpr size_t max_context = size_t(UINT32_MAX)/page_tokens*page_tokens;
    if (config.context_tokens == 0 || config.context_tokens > max_context) return {status::invalid_context_size, no_layer, {}};
    if (config.layers.empty() || config.layers.size() > UINT32_MAX) return {status::invalid_layers, no_layer, {}};
    llama_kv_stream_config_plan next;
    next.enabled = true;
    next.layers = config.layers.size();
    next.padded_context_tokens = (config.context_tokens/page_tokens + (config.context_tokens % page_tokens != 0))*page_tokens;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        const auto & layer = config.layers[i];
        if (layer.id < 0) return {status::invalid_layers, i, {}};
        for (size_t j = 0; j < i; ++j) if (config.layers[j].id == layer.id) return {status::invalid_layers, i, {}};
        if (!layer.offloaded) return {status::unsupported_device, i, {}};
        const auto & shape = layer.shape;
        if (shape.head_dim_k != 256 || shape.head_dim_v != 256 || shape.page_tokens != 256 || shape.alignment != 128 || shape.heads > INT32_MAX)
            return {status::unsupported_geometry, i, {}};
        ggml_kv_stream_execution page;
        auto result = ggml_kv_stream_resolve(shape, layer.capabilities, page_tokens, page);
        if (result.status != ggml_kv_stream_status::success) return {status::geometry_error, i, result};
        if (i == 0) {
            next.page = page;
            result = ggml_kv_stream_layout_make(shape, next.padded_context_tokens, next.context);
            if (result.status != ggml_kv_stream_status::success) return {status::geometry_error, i, result};
        } else if (!same_shape(shape, config.layers[0].shape) || page.attention != next.page.attention ||
                page.conversion.bytes != next.page.conversion.bytes) {
            return {status::nonuniform_geometry, i, {}};
        }
    }
    if (next.context.bytes > SIZE_MAX/next.layers) return {status::geometry_error, no_layer, {ggml_kv_stream_status::overflow}};
    next.all_layers_bytes = next.context.bytes*next.layers;
    output = next;
    return {};
}
