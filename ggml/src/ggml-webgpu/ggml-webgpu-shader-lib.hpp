#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "pre_wgsl.hpp"

#include <string>
#include <vector>

struct ggml_webgpu_flash_attn_shader_lib_context {
    const char * kv_type;
    uint32_t     head_dim_qk;
    uint32_t     head_dim_v;
    bool         has_mask;
    bool         has_sinks;
    bool         uses_logit_softcap;
    uint32_t     sg_mat_m;
    uint32_t     sg_mat_n;
    uint32_t     sg_mat_k;
};

struct ggml_webgpu_flash_attn_shader_decisions {
    int unused = 0;
};

struct ggml_webgpu_processed_shader {
    std::string                             wgsl;
    std::string                             variant;
    ggml_webgpu_flash_attn_shader_decisions decisions;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_flash_attn_shader(
    pre_wgsl::Preprocessor &                          preprocessor,
    const char *                                      shader_src,
    const ggml_webgpu_flash_attn_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "flash_attn";

    defines.push_back(std::string("KV_TYPE=") + context.kv_type);
    variant += std::string("_") + context.kv_type;

    if (context.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    if (context.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    if (context.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(context.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(context.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(context.head_dim_v));
    variant += std::string("_hsv") + std::to_string(context.head_dim_v);

    // For now these are not part of the variant name
    defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
    defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
    defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));

    ggml_webgpu_processed_shader result;
    result.wgsl    = preprocessor.preprocess(shader_src, defines);
    result.variant = variant;
    return result;
}

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
