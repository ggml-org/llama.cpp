#ifndef GGML_WEBGPU_SHADER_LIB_HPP
#define GGML_WEBGPU_SHADER_LIB_HPP

#include "ggml-wgsl-shaders.hpp"
#include "ggml.h"
#include "pre_wgsl.hpp"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_WEBGPU_F16_SIZE_BYTES                   2
#define GGML_WEBGPU_F32_SIZE_BYTES                   4
#define GGML_WEBGPU_I32_SIZE_BYTES                   4
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES 8u
#define GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE     128u
// Matches GGML_PAD(..., 256) in src/llama-context.cpp for KV cache sizing.
#define GGML_WEBGPU_KV_SEQ_PAD                       256u

#define GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE 512u

// Matrix multiplication parameters

// Register tiling parameters
#define WEBGPU_MUL_MAT_TILE_M    8
#define WEBGPU_MUL_MAT_TILE_N    8
#define WEBGPU_MUL_MAT_WG_SIZE_M 8
#define WEBGPU_MUL_MAT_WG_SIZE_N 8
#define WEBGPU_MUL_MAT_TILE_K    32

// Subgroup matrix parameters
// The number of subgroups in the M dimension
#define WEBGPU_MUL_MAT_SUBGROUP_M        2
// The number of subgroups in the N dimension
#define WEBGPU_MUL_MAT_SUBGROUP_N        2
// The number of subgroup matrices each subgroup accumulates over
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M 4
#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N 2

// Matrix-vector multiplication parameters
#define WEBGPU_MUL_MAT_VEC_WG_SIZE        256
// Must be multiple of 4 to work with vectorized paths, and must divide
// mul_mat_vec wg size
#define WEBGPU_MUL_MAT_VEC_OUTPUTS_PER_WG 64
#define WEBGPU_MUL_MAT_VEC_TILE_K         256

// default size for legacy matrix multiplication
#define WEBGPU_MUL_MAT_WG_SIZE 256

// Same hash combine function as in boost
template <typename T> inline void ggml_webgpu_hash_combine(size_t & seed, const T & value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct ggml_webgpu_shader_lib_context {
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * src2;
    ggml_tensor * src3;
    ggml_tensor * src4;
    ggml_tensor * dst;

    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes       = 0;
    bool     inplace                  = false;
    bool     overlap                  = false;
    bool     supports_subgroup_matrix = false;
    uint32_t sg_mat_m                 = 0;
    uint32_t sg_mat_n                 = 0;
    uint32_t sg_mat_k                 = 0;
    uint32_t max_subgroup_size        = 0;
};

struct webgpu_pipeline {
    wgpu::ComputePipeline pipeline;
    std::string           name;
    std::shared_ptr<void> context = nullptr;
};

struct ggml_webgpu_generic_shader_decisions {
    uint32_t wg_size = 0;
};

/** Argsort **/

struct ggml_webgpu_argsort_shader_lib_context {
    uint32_t max_wg_size;
    size_t   wg_mem_limit_bytes;
    int32_t  order;
};

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

struct ggml_webgpu_set_rows_shader_decisions {
    bool     vec4;
    bool     i64_idx;
    uint32_t wg_size;
};

/** Get Rows **/

struct ggml_webgpu_get_rows_pipeline_key {
    ggml_type src_type;
    int       vectorized;

    bool operator==(const ggml_webgpu_get_rows_pipeline_key & other) const {
        return src_type == other.src_type && vectorized == other.vectorized;
    }
};

struct ggml_webgpu_get_rows_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_get_rows_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

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

/** Scale **/

struct ggml_webgpu_scale_pipeline_key {
    int inplace;

    bool operator==(const ggml_webgpu_scale_pipeline_key & other) const { return inplace == other.inplace; }
};

struct ggml_webgpu_scale_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_scale_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

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

/** Unary **/

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

/** Matrix Multiplication **/

struct ggml_webgpu_legacy_mul_mat_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;

    bool operator==(const ggml_webgpu_legacy_mul_mat_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type;
    }
};

struct ggml_webgpu_legacy_mul_mat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_legacy_mul_mat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_vec_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;
    int       vectorized;

    bool operator==(const ggml_webgpu_mul_mat_vec_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized;
    }
};

struct ggml_webgpu_mul_mat_vec_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_vec_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_vec_shader_decisions {
    uint32_t wg_size;
    uint32_t tile_k;
    uint32_t outputs_per_wg;
    uint32_t vec_size;
};

struct ggml_webgpu_mul_mat_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;
    int       vectorized;
    int       use_subgroup_matrix;

    bool operator==(const ggml_webgpu_mul_mat_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
               use_subgroup_matrix == other.use_subgroup_matrix;
    }
};

struct ggml_webgpu_mul_mat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
        ggml_webgpu_hash_combine(seed, key.use_subgroup_matrix);
        return seed;
    }
};

struct ggml_webgpu_mul_mat_shader_decisions {
    uint32_t tile_k;
    uint32_t wg_size_m;
    uint32_t wg_size_n;
    uint32_t wg_size;
    uint32_t outputs_per_wg;
    int      use_subgroup_matrix;

    uint32_t tile_m;
    uint32_t tile_n;

    // Subgroup matrix parameters
    uint32_t subgroup_m;
    uint32_t subgroup_n;
    uint32_t subgroup_matrix_m;
    uint32_t subgroup_matrix_n;

    uint32_t mul_mat_wg_size;
};

class ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<int, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<int, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<int, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<int, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_get_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_key_hash>
        get_rows_pipelines;                                            // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash>
        unary_pipelines;                                               // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_key, webgpu_pipeline, ggml_webgpu_scale_pipeline_key_hash>
        scale_pipelines;                                               // inplace
    std::unordered_map<ggml_webgpu_pad_pipeline_key, webgpu_pipeline, ggml_webgpu_pad_pipeline_key_hash>
        pad_pipelines;                                                 // circular/non-circular
    std::unordered_map<ggml_webgpu_binary_pipeline_key, webgpu_pipeline, ggml_webgpu_binary_pipeline_key_hash>
        binary_pipelines;                                              // type/op/inplace/overlap
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_legacy_mul_mat_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_legacy_mul_mat_pipeline_key_hash>
        mul_mat_legacy_pipelines;  // legacy mul_mat (non-subgroup/non-regtile/non-vec)
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_key_hash>
        mul_mat_vec_pipelines;     // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_key_hash>
        mul_mat_fast_pipelines;    // fast mat-mat (reg-tile or subgroup)

    std::unordered_map<ggml_webgpu_set_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_key_hash>
        set_rows_pipelines;

  public:
    ggml_webgpu_shader_lib(wgpu::Device device) { this->device = device; }

    webgpu_pipeline get_sum_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = sum_rows_pipelines.find(1);
        if (it != sum_rows_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
        return sum_rows_pipelines[1];
    }

    webgpu_pipeline get_argmax_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool vec4 = context.src0->ne[0] % 4 == 0;

        auto it = argmax_pipelines.find(vec4);
        if (it != argmax_pipelines.end()) {
            return it->second;
        }
        std::string              variant = "argmax";
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        if (vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }

        auto processed         = preprocessor.preprocess(wgsl_argmax, defines);
        argmax_pipelines[vec4] = ggml_webgpu_create_pipeline(device, processed, variant);
        return argmax_pipelines.at(vec4);
    }

    webgpu_pipeline get_set_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_rows_pipeline_key key = { .dst_type = context.dst->type,
                                                  .vec4     = context.src0->ne[0] % 4 == 0,
                                                  .i64_idx  = context.src1->type == GGML_TYPE_I64 };

        auto it = set_rows_pipelines.find(key);
        if (it != set_rows_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set_rows";

        switch (context.dst->type) {
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

        if (key.vec4) {
            defines.push_back("VEC4");
            variant += "_vec4";
        }
        if (key.i64_idx) {
            defines.push_back("I64_IDX");
            variant += "_i64idx";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed                  = preprocessor.preprocess(wgsl_set_rows, defines);
        auto decisions                  = std::make_shared<ggml_webgpu_set_rows_shader_decisions>();
        decisions->vec4                 = key.vec4;
        decisions->i64_idx              = key.i64_idx;
        decisions->wg_size              = context.max_wg_size;
        set_rows_pipelines[key]         = ggml_webgpu_create_pipeline(device, processed, variant);
        set_rows_pipelines[key].context = decisions;
        return set_rows_pipelines[key];
    }

    webgpu_pipeline get_cumsum_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = cumsum_pipelines.find(1);
        if (it != cumsum_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed      = preprocessor.preprocess(wgsl_cumsum, defines);
        cumsum_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "cumsum");
        return cumsum_pipelines[1];
    }

    webgpu_pipeline get_argsort_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool          is_top_k = context.dst->op == GGML_OP_TOP_K;
        // ascending order is 0, descending order is 1
        const int32_t order =
            is_top_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_pipelines.find(order);
        if (it != argsort_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = 1;
        while (wg_size * 2 <= context.max_wg_size &&
               wg_size * GGML_WEBGPU_I32_SIZE_BYTES <= context.wg_mem_limit_bytes / 2) {
            wg_size *= 2;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed                   = preprocessor.preprocess(wgsl_argsort, defines);
        auto decisions                   = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size               = wg_size;
        argsort_pipelines[order]         = ggml_webgpu_create_pipeline(device, processed, variant);
        argsort_pipelines[order].context = decisions;
        return argsort_pipelines[order];
    }

    webgpu_pipeline get_argsort_merge_pipeline(const ggml_webgpu_shader_lib_context & context) {
        bool          is_top_k = context.dst->op == GGML_OP_TOP_K;
        // ascending order is 0, descending order is 1
        const int32_t order =
            is_top_k ? (int32_t) GGML_SORT_ORDER_DESC : (int32_t) ggml_get_op_params_i32(context.dst, 0);

        auto it = argsort_merge_pipelines.find(order);
        if (it != argsort_merge_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "argsort_merge";
        defines.push_back(std::string("ORDER=") + std::to_string(order));
        variant += std::string("_order") + std::to_string(order);
        uint32_t wg_size = std::min(GGML_WEBGPU_ARGSORT_MERGE_MAX_WG_SIZE, context.max_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed                 = preprocessor.preprocess(wgsl_argsort_merge, defines);
        argsort_merge_pipelines[order] = ggml_webgpu_create_pipeline(device, processed, variant);
        return argsort_merge_pipelines[order];
    }

    webgpu_pipeline get_get_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool vectorized                 = context.src0->type == GGML_TYPE_F32 && context.dst->ne[0] % 4 == 0;
        ggml_webgpu_get_rows_pipeline_key key = {
            .src_type   = context.src0->type,
            .vectorized = (int) vectorized,
        };

        auto it = get_rows_pipelines.find(key);
        if (it != get_rows_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "get_rows";

        const struct ggml_type_traits * type_traits = ggml_get_type_traits(key.src_type);
        const char *                    type_str    = type_traits->type_name;

        switch (key.src_type) {
            case GGML_TYPE_F32:
                if (key.vectorized) {
                    defines.push_back("F32_VEC");
                    defines.push_back("SRC_TYPE=vec4<f32>");
                    defines.push_back("DST_TYPE=vec4<f32>");
                    defines.push_back("BLOCK_SIZE=4u");
                } else {
                    defines.push_back("F32");
                    defines.push_back("SRC_TYPE=f32");
                    defines.push_back("DST_TYPE=f32");
                    defines.push_back("BLOCK_SIZE=1u");
                }
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                break;
            case GGML_TYPE_I32:
                defines.push_back("I32");
                defines.push_back("SRC_TYPE=i32");
                defines.push_back("DST_TYPE=i32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_i32";
                break;
            default:
                {
                    std::string type_upper = type_str;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");

                    variant += "_";
                    variant += type_str;

                    defines.push_back(std::string("SRC_TYPE=") + type_str);
                    defines.push_back("DST_TYPE=f32");

                    if ((key.src_type >= GGML_TYPE_Q4_0 && key.src_type <= GGML_TYPE_Q8_1) ||
                        key.src_type == GGML_TYPE_IQ4_NL) {
                        defines.push_back("BLOCK_SIZE=32u");
                    } else if (key.src_type >= GGML_TYPE_Q2_K) {
                        defines.push_back("BLOCK_SIZE=256u");
                    } else {
                        defines.push_back("BLOCK_SIZE=1u");
                    }
                    break;
                }
        }

        if (key.vectorized) {
            variant += "_vec";
        }

        defines.push_back("WG_SIZE=" + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_get_rows, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        get_rows_pipelines[key]  = pipeline;
        return get_rows_pipelines[key];
    }

    webgpu_pipeline get_scale_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_scale_pipeline_key key = { .inplace = context.inplace };

        auto it = scale_pipelines.find(key);
        if (it != scale_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "scale";

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_scale, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        scale_pipelines[key]     = pipeline;
        return scale_pipelines[key];
    }

    webgpu_pipeline get_pad_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_key key = { .circular = ggml_get_op_params_i32(context.dst, 8) != 0 };

        auto it = pad_pipelines.find(key);
        if (it != pad_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "pad";

        if (key.circular) {
            defines.push_back("CIRCULAR");
            variant += "_circular";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_pad, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        pad_pipelines[key]       = pipeline;
        return pad_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_vec_pipeline_key key = {
            .src0_type  = context.src0->type,
            .src1_type  = context.src1->type,
            // Quantized mat-vec path currently runs scalar; only allow vectorization when both inputs are float
            .vectorized = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 &&
                           (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                              1 :
                              0,
        };

        auto it = mul_mat_vec_pipelines.find(key);
        if (it != mul_mat_vec_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_vec";

        // src1 type (vector)
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat_vec shader");
        }

        // src0 type (matrix row)
        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                break;
            default:
                {
                    // Quantized types: use helpers but accumulate in f16
                    const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);

                    // For fast path we always dequantize from f16 inside the shader
                    defines.push_back("SRC0_INNER_TYPE=f16");
                    break;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t tile_k         = WEBGPU_MUL_MAT_VEC_TILE_K;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_OUTPUTS_PER_WG;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("TILE_K=") + std::to_string(tile_k));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));

        auto processed            = preprocessor.preprocess(wgsl_mul_mat_vec, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->tile_k         = tile_k;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = key.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[key] = pipeline;
        return mul_mat_vec_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_pipeline_key key = {
            .src0_type  = context.src0->type,
            .src1_type  = context.src1->type,
            .vectorized = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 && context.dst->ne[1] % 4 == 0 &&
                           (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                              1 :
                              0,
            .use_subgroup_matrix = context.supports_subgroup_matrix
        };

        auto it = mul_mat_fast_pipelines.find(key);
        if (it != mul_mat_fast_pipelines.end()) {
            return it->second;
        }

        const char * shader_src = key.use_subgroup_matrix ? wgsl_mul_mat_subgroup_matrix : wgsl_mul_mat_reg_tile;
        std::vector<std::string> defines;
        std::string              variant = key.use_subgroup_matrix ? "mul_mat_subgroup_matrix" : "mul_mat_reg_tile";

        // src1 type
        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_INNER_TYPE=f32");
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_INNER_TYPE=f16");
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat fast shader");
        }

        // src0 type
        const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        const char *                    src0_name   = src0_traits->type_name;

        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("FLOAT");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("FLOAT");
                defines.push_back("MUL_ACC_FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");

                    // Use f16 inside the shader for quantized types
                    defines.push_back("SRC0_INNER_TYPE=f16");

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(WEBGPU_MUL_MAT_TILE_K) + "u");

        // Subgroup matrix specifics
        if (key.use_subgroup_matrix) {
            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
            defines.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
            defines.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M) + "u");
            defines.push_back("SUBGROUP_MATRIX_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N) + "u");
            defines.push_back("SUBGROUP_MATRIX_M_SIZE=" + std::to_string(context.sg_mat_m) + "u");
            defines.push_back("SUBGROUP_MATRIX_N_SIZE=" + std::to_string(context.sg_mat_n) + "u");
            defines.push_back("SUBGROUP_MATRIX_K_SIZE=" + std::to_string(context.sg_mat_k) + "u");
        }

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");
        if (key.vectorized) {
            variant += "_vectorized";
        }

        if (!key.use_subgroup_matrix) {
            defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
            defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
        }

        auto processed = preprocessor.preprocess(shader_src, defines);

        auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k              = WEBGPU_MUL_MAT_TILE_K;
        decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
        decisions->use_subgroup_matrix = key.use_subgroup_matrix;
        if (key.use_subgroup_matrix) {
            decisions->subgroup_m        = WEBGPU_MUL_MAT_SUBGROUP_M;
            decisions->subgroup_n        = WEBGPU_MUL_MAT_SUBGROUP_N;
            decisions->subgroup_matrix_m = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M;
            decisions->subgroup_matrix_n = WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N;
            decisions->wg_size           = context.max_subgroup_size;
        } else {
            decisions->wg_size_m       = WEBGPU_MUL_MAT_WG_SIZE_M;
            decisions->wg_size_n       = WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->wg_size         = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;
            decisions->mul_mat_wg_size = WEBGPU_MUL_MAT_WG_SIZE;
        }

        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        mul_mat_fast_pipelines[key] = pipeline;
        return mul_mat_fast_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_legacy_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_legacy_mul_mat_pipeline_key key = { .src0_type = context.src0->type,
                                                        .src1_type = context.src1->type };

        auto it = mul_mat_legacy_pipelines.find(key);
        if (it != mul_mat_legacy_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat";

        switch (context.src1->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC1_TYPE=f32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC1_TYPE=f16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src1 type for mul_mat legacy shader");
        }

        const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
        const char *                    src0_name   = src0_traits->type_name;

        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_TYPE=f32");
                defines.push_back("FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_TYPE=f16");
                defines.push_back("FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    // quantized types
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back(std::string("SRC0_TYPE=") + src0_name);
                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        auto processed = preprocessor.preprocess(wgsl_mul_mat, defines);

        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = WEBGPU_MUL_MAT_WG_SIZE;

        webgpu_pipeline pipeline      = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context              = decisions;
        mul_mat_legacy_pipelines[key] = pipeline;
        return mul_mat_legacy_pipelines[key];
    }

    webgpu_pipeline get_unary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool                     is_unary = context.dst->op == GGML_OP_UNARY;
        const int                      op       = is_unary ? (int) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_key key      = {
                 .type     = context.dst->type,
                 .op       = op,
                 .is_unary = is_unary,
                 .inplace  = context.inplace,
        };

        auto it = unary_pipelines.find(key);
        if (it != unary_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant =
            key.is_unary ? ggml_unary_op_name((ggml_unary_op) key.op) : ggml_op_name((ggml_op) key.op);
        defines.push_back(variant);

        switch (key.type) {
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

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_unary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        unary_pipelines[key]     = pipeline;
        return unary_pipelines[key];
    }

    webgpu_pipeline get_binary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_binary_pipeline_key key = {
            .type    = context.dst->type,
            .op      = context.dst->op,
            .inplace = context.inplace,
            .overlap = context.overlap,
        };

        auto it = binary_pipelines.find(key);
        if (it != binary_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = ggml_op_name((ggml_op) key.op);
        std::string              variant = op_name;

        defines.push_back(std::string("OP_") + op_name);

        switch (key.type) {
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

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } else if (key.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_binary, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        binary_pipelines[key]    = pipeline;
        return binary_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool has_mask  = context.src3 != nullptr;
        const bool has_sinks = context.src4 != nullptr;

        bool kv_direct = (context.src1->type == GGML_TYPE_F16) && (context.src0->ne[0] % context.sg_mat_k == 0) &&
                         (context.src1->ne[1] % context.sg_mat_n == 0);

        ggml_webgpu_flash_attn_pipeline_key key = {
            .kv_type            = context.src1->type,
            .head_dim_qk        = (uint32_t) context.src0->ne[0],
            .head_dim_v         = (uint32_t) context.src2->ne[0],
            .kv_direct          = kv_direct,
            .has_mask           = has_mask,
            .has_sinks          = has_sinks,
            .uses_logit_softcap = (*(float *) &context.dst->op_params[2]) != 0.0f,
        };

        auto it = flash_attn_pipelines.find(key);
        if (it != flash_attn_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn";

        switch (key.kv_type) {
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
        variant += std::string("_") + ggml_type_name(key.kv_type);

        if (key.has_mask) {
            defines.push_back("MASK");
            variant += "_mask";
        }
        if (key.has_sinks) {
            defines.push_back("SINKS");
            variant += "_sinks";
        }
        if (key.uses_logit_softcap) {
            defines.push_back("LOGIT_SOFTCAP");
            variant += "_lgsc";
        }
        if (key.kv_direct) {
            defines.push_back("KV_DIRECT");
            variant += "_kvdirect";
        }

        defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(key.head_dim_qk));
        variant += std::string("_hsqk") + std::to_string(key.head_dim_qk);

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
        variant += std::string("_hsv") + std::to_string(key.head_dim_v);

        defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
        defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
        defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));

        uint32_t q_tile = context.sg_mat_m;
        uint32_t kv_tile =
            std::min(ggml_webgpu_flash_attn_max_kv_tile({ key, context.sg_mat_m, context.sg_mat_n, context.sg_mat_k,
                                                          context.wg_mem_limit_bytes, context.max_subgroup_size }),
                     context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
        if (key.kv_direct) {
            while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
                kv_tile -= context.sg_mat_n;
            }
        }

        defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
        defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));

        uint32_t wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));

        auto processed     = preprocessor.preprocess(wgsl_flash_attn, defines);
        auto decisions     = std::make_shared<ggml_webgpu_flash_attn_shader_decisions>();
        decisions->q_tile  = q_tile;
        decisions->kv_tile = kv_tile;
        decisions->wg_size = wg_size;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        flash_attn_pipelines[key] = pipeline;
        return flash_attn_pipelines[key];
    }

  private:
    static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                       std::string    shader_code,
                                                       std::string    label) {
        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_code.c_str();

        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.nextInChain = &shader_source;

        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label              = label.c_str();
        pipeline_desc.compute.module     = shader_module;
        pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
        pipeline_desc.layout             = nullptr;  // nullptr means auto layout
        return { device.CreateComputePipeline(&pipeline_desc), label };
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
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
