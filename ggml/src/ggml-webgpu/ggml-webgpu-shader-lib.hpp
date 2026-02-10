#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml.h"
#include "pre_wgsl.hpp"

#include <memory>
#include <string>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_I32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD                       256u

#define GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE 512u

struct ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<void> decisions;
};

// Same hash combine function as in boost
template <typename T> inline void ggml_webgpu_hash_combine(size_t & seed, const T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/** FlashAttention */

struct ggml_webgpu_flash_attn_pipeline_key {
    ggml_type kv_type;
    uint32_t  head_dim_qk;
    uint32_t  head_dim_v;
    bool      kv_direct;
    bool      has_mask;
    bool      has_sinks;
    bool      uses_logit_softcap;

    bool operator==(const ggml_webgpu_flash_attn_pipeline_key & other) const {
        return kv_type == other.kv_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
               kv_direct == other.kv_direct && has_mask == other.has_mask && has_sinks == other.has_sinks &&
               uses_logit_softcap == other.uses_logit_softcap;
    }
};

struct ggml_webgpu_flash_attn_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.kv_type);
        ggml_webgpu_hash_combine(seed, key.head_dim_qk);
        ggml_webgpu_hash_combine(seed, key.head_dim_v);
        ggml_webgpu_hash_combine(seed, key.kv_direct);
        ggml_webgpu_hash_combine(seed, key.has_mask);
        ggml_webgpu_hash_combine(seed, key.has_sinks);
        ggml_webgpu_hash_combine(seed, key.uses_logit_softcap);
        return seed;
    }
};

struct ggml_webgpu_flash_attn_shader_lib_context {
    ggml_webgpu_flash_attn_pipeline_key key;
    uint32_t                            sg_mat_m;
    uint32_t                            sg_mat_n;
    uint32_t                            sg_mat_k;
    size_t                              wg_mem_limit_bytes;
    uint32_t                            max_subgroup_size;
};

struct ggml_webgpu_flash_attn_shader_decisions {
    uint32_t q_tile  = 0;
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
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
    const size_t limit_bytes = context.wg_mem_limit_bytes;
    const size_t q_tile      = context.sg_mat_m;
    const size_t base_q_bytes =
        (context.key.head_dim_qk + context.key.head_dim_v) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
        2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
    size_t bytes_per_kv = 0;
    if (!context.key.kv_direct) {
        bytes_per_kv += std::max(context.key.head_dim_qk, context.key.head_dim_v);
    }
    if (context.key.has_mask) {
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

    switch (context.key.kv_type) {
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
    variant += std::string("_") + ggml_type_name(context.key.kv_type);

    if (context.key.has_mask) {
        defines.push_back("MASK");
        variant += "_mask";
    }
    if (context.key.has_sinks) {
        defines.push_back("SINKS");
        variant += "_sinks";
    }
    if (context.key.uses_logit_softcap) {
        defines.push_back("LOGIT_SOFTCAP");
        variant += "_lgsc";
    }

    if (context.key.kv_direct) {
        defines.push_back("KV_DIRECT");
        variant += "_kvdirect";
    }

    defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(context.key.head_dim_qk));
    variant += std::string("_hsqk") + std::to_string(context.key.head_dim_qk);

    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(context.key.head_dim_v));
    variant += std::string("_hsv") + std::to_string(context.key.head_dim_v);
    // For now these are not part of the variant name
    defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
    defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
    defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));

    // Add chosen Q/KV tile sizes
    uint32_t q_tile  = context.sg_mat_m;
    uint32_t kv_tile = std::min(ggml_webgpu_flash_attn_max_kv_tile(context),
                                context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
    if (context.key.kv_direct) {
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
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_flash_attn_shader_decisions>();
    decisions->q_tile  = q_tile;
    decisions->kv_tile = kv_tile;
    decisions->wg_size = wg_size;
    result.decisions   = decisions;
    return result;
}

/** Generic **/

struct ggml_webgpu_generic_shader_lib_context {
    int      vec4;
    uint32_t max_wg_size;
};

struct ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_generic_shader(
    pre_wgsl::Preprocessor &                       preprocessor,
    const char *                                   shader_src,
    const ggml_webgpu_generic_shader_lib_context & context,
    const std::string &                            base_variant) {
    std::vector<std::string> defines;
    std::string              variant = base_variant;

    if (context.vec4) {
        defines.push_back("VEC4");
        variant += "_vec";
    }

    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

    ggml_webgpu_processed_shader result;
    result.wgsl    = preprocessor.preprocess(shader_src, defines);
    result.variant = variant;
    return result;
}

/** Pad **/

struct ggml_webgpu_pad_pipeline_key {
    bool circular;

    bool operator==(const ggml_webgpu_pad_pipeline_key & other) const { return circular == other.circular; }
};

struct ggml_webgpu_pad_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_pad_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.circular);
        return seed;
    }
};

struct ggml_webgpu_pad_shader_lib_context {
    ggml_webgpu_pad_pipeline_key key;
    uint32_t                     max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_pad_shader(
    pre_wgsl::Preprocessor &                   preprocessor,
    const char *                               shader_src,
    const ggml_webgpu_pad_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "pad";

    if (context.key.circular) {
        defines.push_back("CIRCULAR");
        variant += "_circular";
    }

    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
    decisions->wg_size = context.max_wg_size;
    result.decisions   = decisions;
    return result;
}

/** Argsort **/

struct ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

struct ggml_webgpu_argsort_shader_decisions {
    uint32_t wg_size = 0;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_argsort_shader(
    pre_wgsl::Preprocessor &                       preprocessor,
    const char *                                   shader_src,
    const ggml_webgpu_argsort_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "argsort";
    defines.push_back(std::string("ORDER=") + std::to_string(context.order));
    variant += std::string("_order") + std::to_string(context.order);
    uint32_t wg_size = 1;
    while (wg_size * 2 <= context.max_wg_size &&
           wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
        wg_size *= 2;
    }
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_argsort_shader_decisions>();
    decisions->wg_size = wg_size;
    result.decisions   = decisions;
    return result;
}

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_argsort_merge_shader(
    pre_wgsl::Preprocessor &                       preprocessor,
    const char *                                   shader_src,
    const ggml_webgpu_argsort_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "argsort_merge";
    defines.push_back(std::string("ORDER=") + std::to_string(context.order));
    variant += std::string("_order") + std::to_string(context.order);
    uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_argsort_shader_decisions>();
    decisions->wg_size = wg_size;
    result.decisions   = decisions;
    return result;
}

/** Set Rows **/

struct ggml_webgpu_set_rows_pipeline_key {
    int dst_type;
    int vec4;
    int i64_idx;

    bool operator==(const ggml_webgpu_set_rows_pipeline_key & other) const {
        return dst_type == other.dst_type && vec4 == other.vec4 && i64_idx == other.i64_idx;
    }
};

struct ggml_webgpu_set_rows_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_set_rows_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.dst_type);
        ggml_webgpu_hash_combine(seed, key.vec4);
        ggml_webgpu_hash_combine(seed, key.i64_idx);
        return seed;
    }
};

struct ggml_webgpu_set_rows_shader_lib_context {
    ggml_webgpu_set_rows_pipeline_key key;
    uint32_t                          max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_set_rows_shader(
    pre_wgsl::Preprocessor &                        preprocessor,
    const char *                                    shader_src,
    const ggml_webgpu_set_rows_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = "set_rows";

    switch (context.key.dst_type) {
        case GGML_TYPE_F32:
            defines.push_back("DST_F32");
            variant += "_dstf32";
            break;
        case GGML_TYPE_F16:
            defines.push_back("DST_F16");
            variant += "_dstf16";
            break;
        default:
            GGML_ABORT("Unsupported dst type for set_rows shader");
    }

    if (context.key.vec4) {
        defines.push_back("VEC4");
        variant += "_vec";
    }
    if (context.key.i64_idx) {
        defines.push_back("I64_IDX");
        variant += "_i64idx";
    }

    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
    decisions->wg_size = context.max_wg_size;
    result.decisions   = decisions;
    return result;
}

struct ggml_webgpu_unary_pipeline_key {
    int  type;
    int  op;
    bool is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    bool inplace;

    bool operator==(const ggml_webgpu_unary_pipeline_key & other) const {
        return type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace;
    }
};

struct ggml_webgpu_unary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_unary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.is_unary);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

struct ggml_webgpu_unary_shader_lib_context {
    ggml_webgpu_unary_pipeline_key key;
    uint32_t                       max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_unary_shader(
    pre_wgsl::Preprocessor &                     preprocessor,
    const char *                                 shader_src,
    const ggml_webgpu_unary_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              variant = context.key.is_unary ? ggml_unary_op_name((ggml_unary_op) context.key.op) :
                                                              ggml_op_name((ggml_op) context.key.op);
    // Operation-specific behavior
    defines.push_back(variant);

    switch (context.key.type) {
        case GGML_TYPE_F32:
            defines.push_back("TYPE_F32");
            variant += "_f32";
            break;
        case GGML_TYPE_F16:
            defines.push_back("TYPE_F16");
            variant += "_f16";
            break;
        default:
            GGML_ABORT("Unsupported type for unary shader");
    }

    if (context.key.inplace) {
        defines.push_back("INPLACE");
        variant += "_inplace";
    }

    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
    decisions->wg_size = context.max_wg_size;
    result.decisions   = decisions;
    return result;
}

/** Binary **/

struct ggml_webgpu_binary_pipeline_key {
    int  type;
    int  op;
    bool inplace;
    bool overlap;

    bool operator==(const ggml_webgpu_binary_pipeline_key & other) const {
        return type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap;
    }
};

struct ggml_webgpu_binary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_binary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.overlap);
        return seed;
    }
};

struct ggml_webgpu_binary_shader_lib_context {
    ggml_webgpu_binary_pipeline_key key;
    uint32_t                        max_wg_size;
};

inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_binary_shader(
    pre_wgsl::Preprocessor &                      preprocessor,
    const char *                                  shader_src,
    const ggml_webgpu_binary_shader_lib_context & context) {
    std::vector<std::string> defines;
    std::string              op_name = ggml_op_name((ggml_op) context.key.op);
    std::string              variant = op_name;

    defines.push_back(std::string("OP_") + op_name);

    switch (context.key.type) {
        case GGML_TYPE_F32:
            defines.push_back("TYPE_F32");
            variant += "_f32";
            break;
        case GGML_TYPE_F16:
            defines.push_back("TYPE_F16");
            variant += "_f16";
            break;
        default:
            GGML_ABORT("Unsupported type for binary shader");
    }

    if (context.key.inplace) {
        defines.push_back("INPLACE");
        variant += "_inplace";
    } else if (context.key.overlap) {
        defines.push_back("OVERLAP");
        variant += "_overlap";
    }

    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
    ggml_webgpu_processed_shader result;
    result.wgsl        = preprocessor.preprocess(shader_src, defines);
    result.variant     = variant;
    auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
    decisions->wg_size = context.max_wg_size;
    result.decisions   = decisions;
    return result;
}
#endif  // GGML_WEBGPU_SHADER_LIB_HPP
