#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml.h"
#include "pre_wgsl.hpp"

#include <string>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD                       256u

struct ggml_webgpu_flash_attn_shader_lib_context {
    ggml_type kv_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    bool      kv_direct;
    bool      has_mask;
    bool      has_sinks;
    bool      uses_logit_softcap;
    uint32_t  sg_mat_m;
    uint32_t  sg_mat_n;
    uint32_t  sg_mat_k;
    size_t    wg_mem_limit_bytes;
    uint32_t  max_subgroup_size;
};

struct ggml_webgpu_flash_attn_shader_decisions {
    uint32_t q_tile  = 0;
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

struct ggml_webgpu_processed_shader {
    std::string                             wgsl;
    std::string                             variant;
    ggml_webgpu_flash_attn_shader_decisions decisions;
};

// This is exposed because it's necessary in supports_op
inline size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                  uint32_t kv_tile,
                                                  uint32_t head_dim_qk,
                                                  uint32_t head_dim_v,
                                                  bool     has_mask,
                                                  bool     kv_direct) {
    const uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         f16_elems    = 0;
    size_t         f32_elems    = 0;
    f16_elems += q_tile * head_dim_qk;        // q_shmem
    if (!kv_direct) {
        f16_elems += kv_tile * max_head_dim;  // kv_shmem
    }
    f16_elems += q_tile * head_dim_v;         // o_shmem
    if (has_mask) {
        f16_elems += q_tile * kv_tile;        // mask_shmem
    }
    f16_elems += q_tile * kv_tile;            // inter_shmem
    f32_elems += q_tile;                      // row_max_shmem
    f32_elems += q_tile;                      // exp_sum_shmem
    return f16_elems * GGML_WEBGPU_F16_SIZE_BYTES + f32_elems * GGML_WEBGPU_F32_SIZE_BYTES;
}

static uint32_t ggml_webgpu_flash_attn_max_kv_tile(const ggml_webgpu_flash_attn_shader_lib_context & context) {
    const size_t limit_bytes  = context.wg_mem_limit_bytes;
    const size_t q_tile       = context.sg_mat_m;
    const size_t base_q_bytes = (context.head_dim_qk + context.head_dim_v) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
                                2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
    size_t bytes_per_kv = 0;
    if (!context.kv_direct) {
        bytes_per_kv += std::max(context.head_dim_qk, context.head_dim_v);
    }
    if (context.has_mask) {
        bytes_per_kv += q_tile;
    }
    bytes_per_kv += q_tile;
    bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
    const uint32_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    return (max_kv_tile / context.sg_mat_n) * context.sg_mat_n;
}

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_flash_attn_shader(
    pre_wgsl::Preprocessor &                          preprocessor,
    const char *                                      shader_src,
    const ggml_webgpu_flash_attn_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "flash_attn";

    switch (context.kv_type) {
        case GGML_TYPE_F32:
            defines.push_back("KV_F32");
            break;
        case GGML_TYPE_F16:
            defines.push_back("KV_F16");
            break;
        case GGML_TYPE_Q4_0:
            defines.push_back("KV_Q4_0");
            break;
        case GGML_TYPE_Q8_0:
            defines.push_back("KV_Q8_0");
            break;
        default:
            GGML_ABORT("Unsupported KV type for flash attention shader");
    }
    variant += std::string("_") + ggml_type_name(context.kv_type);

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

    if (context.kv_direct) {
        defines.push_back("KV_DIRECT");
        variant += "_kvdirect";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(context.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(context.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(context.head_dim_v));
    variant += std::string("_hsv") + std::to_string(context.head_dim_v);

    // For now these are not part of the variant name
    defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
    defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
    defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));

    // Add chosen Q/KV tile sizes
    uint32_t q_tile  = context.sg_mat_m;
    uint32_t kv_tile = std::min(ggml_webgpu_flash_attn_max_kv_tile(context),
                                context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
    if (context.kv_direct) {
        GGML_ASSERT(kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
        // Avoids having to use bounds-checks and decreasing performance for direct KV loads
        while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= context.sg_mat_n;
        }
    }

    defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
    defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));

    // workgroup size
    uint32_t wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);

    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

    ggml_webgpu_processed_shader result;
    result.wgsl              = preprocessor.preprocess(shader_src, defines);
    result.variant           = variant;
    result.decisions.q_tile  = q_tile;
    result.decisions.kv_tile = kv_tile;
    result.decisions.wg_size = wg_size;
    return result;
}

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
