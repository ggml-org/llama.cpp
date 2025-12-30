#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml-webgpu-structs.hpp"

#include <string>
#include <vector>

extern const char * wgsl_flash_attn;

webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                            const char *                             shader_code,
                                            const char *                             label,
                                            const std::vector<wgpu::ConstantEntry> & constants = {});

static inline const char * ggml_webgpu_wgsl_kv_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return "f16";
        case GGML_TYPE_F32:
            return "f32";
        default:
            return nullptr;
    }
}

inline webgpu_pipeline ggml_webgpu_get_flash_attn_pipeline(webgpu_context & ctx,
                                                           ggml_tensor *    Q,
                                                           ggml_tensor *    K,
                                                           ggml_tensor *    V,
                                                           ggml_tensor *    mask,
                                                           ggml_tensor *    sinks,
                                                           ggml_tensor *    dst,
                                                           float            logit_softcap) {
    GGML_ASSERT(K->type == V->type);

    flash_attn_pipeline_key key = {
        .q_type             = Q->type,
        .kv_type            = K->type,
        .mask_type          = mask->type,
        .sinks_type         = sinks->type,
        .dst_type           = dst->type,
        .head_dim_q         = (uint32_t) Q->ne[0],
        .head_dim_v         = (uint32_t) V->ne[0],
        .n_heads            = (uint32_t) Q->ne[2],
        .has_mask           = true,
        .has_sinks          = true,
        .uses_logit_softcap = logit_softcap != 0.0f,
    };

    auto it = ctx->flash_attn_pipelines.find(key);
    if (it != ctx->flash_attn_pipelines.end()) {
        return it->second;
    }

    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
    it = ctx->flash_attn_pipelines.find(key);
    if (it != ctx->flash_attn_pipelines.end()) {
        return it->second;
    }

    const char *    kv_type  = ggml_webgpu_wgsl_kv_type(K->type);
    std::string     label    = std::string("flash_attn_kv_") + kv_type;
    std::string     shader   = ctx->p.preprocess(wgsl_flash_attn, { std::string("KV_TYPE=") + kv_type });
    webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(ctx->device, shader.c_str(), label.c_str());
    ctx->flash_attn_pipelines.emplace(key, pipeline);
    return pipeline;
}

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
