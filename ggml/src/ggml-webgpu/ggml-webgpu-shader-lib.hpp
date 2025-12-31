#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "pre_wgsl.hpp"

#include <string>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES 2
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE 64u

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
    size_t       wg_mem_limit_bytes;
    uint32_t     max_subgroup_size;
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
                                                  bool     has_mask) {
    const uint32_t max_head_dim = std::max(head_dim_qk, head_dim_v);
    size_t         elems        = 0;
    elems += q_tile * head_dim_qk;    // q_shmem
    elems += kv_tile * max_head_dim;  // kv_shmem
    elems += q_tile * head_dim_v;     // o_shmem
    if (has_mask) {
        elems += q_tile * kv_tile;    // mask_shmem
    }
    elems += q_tile * kv_tile;        // inter_shmem
    elems += q_tile;                  // row_max_shmem
    elems += q_tile;                  // exp_sum_shmem
    return elems * GGML_WEBGPU_F16_SIZE_BYTES;
}

// Returns a pair of (q_tile, kv_tile) that best fits within the workgroup memory limit
// Currently set to prefer the configuration that comes closest to using half of the limit
// Assumes that the base minimum tile sizes fits within the limit
static std::pair<uint32_t, uint32_t> ggml_webgpu_flash_attn_tile_sizes(
    const ggml_webgpu_flash_attn_shader_lib_context & context) {
    std::pair<uint32_t, uint32_t> best_pair  = { 0, 0 };
    size_t                        best_delta = 0;

    const uint32_t min_q_tile   = context.sg_mat_m;
    const uint32_t min_kv_tile  = context.sg_mat_n;
    const size_t   limit_bytes  = context.wg_mem_limit_bytes;
    const size_t   target_bytes = limit_bytes / 2;
    const uint32_t max_head_dim = std::max(context.head_dim_qk, context.head_dim_v);

    // These sizes come from the equations for wg_mem_bytes, solving for q_tile or kv_tile respectively
    const size_t base_kv_bytes = min_kv_tile * max_head_dim * GGML_WEBGPU_F16_SIZE_BYTES;
    const size_t bytes_per_q =
        (context.head_dim_qk + context.head_dim_v + (context.has_mask ? min_kv_tile : 0) + min_kv_tile + 2) *
        GGML_WEBGPU_F16_SIZE_BYTES;
    const uint32_t max_q_tile = (limit_bytes - base_kv_bytes) / bytes_per_q;

    const size_t base_q_bytes =
        (context.head_dim_qk + context.head_dim_v + 2) * min_q_tile * GGML_WEBGPU_F16_SIZE_BYTES;
    const size_t   bytes_per_kv =
        (max_head_dim + (context.has_mask ? min_q_tile : 0) + min_q_tile) * GGML_WEBGPU_F16_SIZE_BYTES;
    const uint32_t max_kv_tile  = (limit_bytes - base_q_bytes) / bytes_per_kv;

    // step by minimum tile sizes
    for (uint32_t q = min_q_tile; q <= max_q_tile; q += min_q_tile) {
        for (uint32_t kv = min_kv_tile; kv <= max_kv_tile; kv += min_kv_tile) {
            size_t bytes =
                ggml_webgpu_flash_attn_wg_mem_bytes(q, kv, context.head_dim_qk, context.head_dim_v, context.has_mask);
            if (bytes <= limit_bytes) {
                size_t delta = bytes > target_bytes ? bytes - target_bytes : target_bytes - bytes;
                if (best_pair.first == 0 || delta < best_delta) {
                    best_pair  = { q, kv };
                    best_delta = delta;
                }
            }
        }
    }

    return best_pair;
}

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

    // Add chosen Q/KV tile sizes
    auto [q_tile, kv_tile] = ggml_webgpu_flash_attn_tile_sizes(context);
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
