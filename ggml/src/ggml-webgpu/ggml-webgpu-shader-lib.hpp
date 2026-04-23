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
#define WEBGPU_MUL_MAT_VEC_WG_SIZE 256

#define WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG    4
#define WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG 4
#define WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG      4

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
    bool     src_overlap              = false;
    bool     supports_subgroups       = false;
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

struct ggml_webgpu_processed_shader {
    std::string           wgsl;
    std::string           variant;
    std::shared_ptr<void> decisions;
};

struct ggml_webgpu_ssm_conv_shader_decisions {
    uint32_t block_size;
    uint32_t tokens_per_wg;
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

/** Set **/

struct ggml_webgpu_set_pipeline_key {
    ggml_type type;
    bool      inplace;

    bool operator==(const ggml_webgpu_set_pipeline_key & other) const {
        return type == other.type && inplace == other.inplace;
    }
};

struct ggml_webgpu_set_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_set_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
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

/** Row Norm **/

struct ggml_webgpu_row_norm_pipeline_key {
    ggml_op op;
    bool    inplace;

    bool operator==(const ggml_webgpu_row_norm_pipeline_key & other) const {
        return op == other.op && inplace == other.inplace;
    }
};

struct ggml_webgpu_row_norm_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_row_norm_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

/** RMS_NORM + MUL **/

struct ggml_webgpu_rms_norm_mul_pipeline_key {
    bool inplace;      // rn_src == dst
    bool overlap;      // mul_src == dst
    bool src_overlap;  // rn_src == mul_src

    bool operator==(const ggml_webgpu_rms_norm_mul_pipeline_key & other) const {
        return inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
    }
};

struct ggml_webgpu_rms_norm_mul_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_rms_norm_mul_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.overlap);
        ggml_webgpu_hash_combine(seed, key.src_overlap);
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

/** Solve Tri **/
struct ggml_webgpu_solve_tri_pipeline_key {
    int type;
    int n;
    int k;

    bool operator==(const ggml_webgpu_solve_tri_pipeline_key & other) const {
        return type == other.type && n == other.n && k == other.k;
    }
};

struct ggml_webgpu_solve_tri_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_solve_tri_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.n);
        ggml_webgpu_hash_combine(seed, key.k);
        return seed;
    }
};

/** SSM Conv **/
struct ggml_webgpu_ssm_conv_pipeline_key {
    int type;
    int vectorized;

    bool operator==(const ggml_webgpu_ssm_conv_pipeline_key & other) const {
        return type == other.type && vectorized == other.vectorized;
    }
};

/** CONV 2D */
struct ggml_webgpu_conv2d_pipeline_key {
    ggml_type weight_type;
    ggml_type input_type;
    ggml_type output_type;

    bool operator==(const ggml_webgpu_conv2d_pipeline_key & other) const {
        return weight_type == other.weight_type && input_type == other.input_type && output_type == other.output_type;
    }
};

struct ggml_webgpu_conv2d_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_conv2d_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.weight_type);
        ggml_webgpu_hash_combine(seed, key.input_type);
        ggml_webgpu_hash_combine(seed, key.output_type);
        return seed;
    }
};

/** Im2Col **/
struct ggml_webgpu_im2col_pipeline_key {
    ggml_type input_type;
    ggml_type output_type;

    bool operator==(const ggml_webgpu_im2col_pipeline_key & other) const {
        return input_type == other.input_type && output_type == other.output_type;
    }
};

struct ggml_webgpu_im2col_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_im2col_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.input_type);
        ggml_webgpu_hash_combine(seed, key.output_type);
        return seed;
    }
};

/** Gated Delta Net **/
struct ggml_webgpu_gated_delta_net_pipeline_key {
    int type;
    int s_v;
    int kda;

    bool operator==(const ggml_webgpu_gated_delta_net_pipeline_key & other) const {
        return type == other.type && s_v == other.s_v && kda == other.kda;
    }
};

struct ggml_webgpu_gated_delta_net_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_gated_delta_net_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.s_v);
        ggml_webgpu_hash_combine(seed, key.kda);
        return seed;
    }
};

struct ggml_webgpu_ssm_conv_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_ssm_conv_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.vectorized);
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

/** Concat **/

struct ggml_webgpu_concat_pipeline_key {
    int type;

    bool operator==(const ggml_webgpu_concat_pipeline_key & other) const { return type == other.type; }
};

struct ggml_webgpu_concat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_concat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        return seed;
    }
};

/** Repeat **/

struct ggml_webgpu_repeat_pipeline_key {
    int type;

    bool operator==(const ggml_webgpu_repeat_pipeline_key & other) const { return type == other.type; }
};

struct ggml_webgpu_repeat_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_repeat_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        return seed;
    }
};

/** Binary **/

struct ggml_webgpu_binary_pipeline_key {
    int  type;
    int  op;
    bool inplace;
    bool overlap;
    bool src_overlap;

    bool operator==(const ggml_webgpu_binary_pipeline_key & other) const {
        return type == other.type && op == other.op && inplace == other.inplace && overlap == other.overlap &&
               src_overlap == other.src_overlap;
    }
};

struct ggml_webgpu_binary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_binary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.overlap);
        ggml_webgpu_hash_combine(seed, key.src_overlap);
        return seed;
    }
};

/** Unary **/

struct ggml_webgpu_unary_pipeline_key {
    int           type;
    int           op;
    bool          is_unary;  // many unary operators fall under the GGML_OP_UNARY umbrella
    bool          inplace;
    ggml_tri_type ttype;     // only used for GGML_OP_TRI

    bool operator==(const ggml_webgpu_unary_pipeline_key & other) const {
        return type == other.type && op == other.op && is_unary == other.is_unary && inplace == other.inplace &&
               ttype == other.ttype;
    }
};

struct ggml_webgpu_unary_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_unary_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.op);
        ggml_webgpu_hash_combine(seed, key.is_unary);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.ttype);
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

struct ggml_webgpu_flash_attn_decisions {
    uint32_t q_tile  = 0;
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

struct ggml_webgpu_flash_attn_vec_decisions {
    uint32_t kv_tile = 0;
    uint32_t wg_size = 0;
};

inline ggml_webgpu_flash_attn_pipeline_key ggml_webgpu_flash_attn_make_pipeline_key(
    const ggml_webgpu_shader_lib_context & context) {
    const bool has_mask  = context.src3 != nullptr;
    const bool has_sinks = context.src4 != nullptr;
    const bool kv_direct = (context.src1->type == GGML_TYPE_F16) && (context.src0->ne[0] % context.sg_mat_k == 0) &&
                           (context.src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);

    ggml_webgpu_flash_attn_pipeline_key key = {};
    key.kv_type                             = context.src1->type;
    key.head_dim_qk                         = (uint32_t) context.src0->ne[0];
    key.head_dim_v                          = (uint32_t) context.src2->ne[0];
    key.kv_direct                           = kv_direct;
    key.has_mask                            = has_mask;
    key.has_sinks                           = has_sinks;
    key.uses_logit_softcap                  = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
    return key;
}

struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key {
    uint32_t head_dim_v;
    uint32_t wg_size;
};

struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.head_dim_v);
        ggml_webgpu_hash_combine(seed, key.wg_size);
        return seed;
    }
};

inline bool operator==(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & lhs,
                       const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & rhs) {
    return lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size;
}

struct ggml_webgpu_flash_attn_blk_pipeline_key {
    uint32_t kv_tile;

    bool operator==(const ggml_webgpu_flash_attn_blk_pipeline_key & other) const { return kv_tile == other.kv_tile; }
};

struct ggml_webgpu_flash_attn_blk_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_flash_attn_blk_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.kv_tile);
        return seed;
    }
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

inline uint32_t ggml_webgpu_flash_attn_max_kv_tile(const ggml_webgpu_shader_lib_context &      context,
                                                   const ggml_webgpu_flash_attn_pipeline_key & key) {
    const size_t limit_bytes  = context.wg_mem_limit_bytes;
    const size_t q_tile       = context.sg_mat_m;
    const size_t base_q_bytes = (key.head_dim_qk + key.head_dim_v) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
                                2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
    size_t bytes_per_kv = 0;
    if (!key.kv_direct) {
        bytes_per_kv += std::max(key.head_dim_qk, key.head_dim_v);
    }
    if (key.has_mask) {
        bytes_per_kv += q_tile;
    }
    bytes_per_kv += q_tile;
    bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
    const uint32_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
    return (max_kv_tile / context.sg_mat_n) * context.sg_mat_n;
}

inline uint32_t ggml_webgpu_flash_attn_vec_get_kv_tile(const ggml_webgpu_shader_lib_context & context) {
    const ggml_webgpu_flash_attn_pipeline_key key         = ggml_webgpu_flash_attn_make_pipeline_key(context);
    const uint32_t                            min_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(context, key);
    uint32_t                                  kv_tile     = std::max(context.sg_mat_n, std::min(32u, min_kv_tile));
    kv_tile                                               = (kv_tile / context.sg_mat_n) * context.sg_mat_n;

    if (key.kv_direct) {
        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
        while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
            kv_tile -= context.sg_mat_n;
        }
    }

    return kv_tile;
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

/** MUL_MAT_ID **/

struct ggml_webgpu_mul_mat_id_pipeline_key {
    ggml_type src0_type;
    ggml_type src1_type;

    bool operator==(const ggml_webgpu_mul_mat_id_pipeline_key & other) const {
        return src0_type == other.src0_type && src1_type == other.src1_type;
    }
};

struct ggml_webgpu_mul_mat_id_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_mul_mat_id_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src0_type);
        ggml_webgpu_hash_combine(seed, key.src1_type);
        return seed;
    }
};

/** Cpy **/

struct ggml_webgpu_cpy_pipeline_key {
    ggml_type src_type;
    ggml_type dst_type;

    bool operator==(const ggml_webgpu_cpy_pipeline_key & other) const {
        return src_type == other.src_type && dst_type == other.dst_type;
    }
};

struct ggml_webgpu_cpy_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_cpy_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.src_type);
        ggml_webgpu_hash_combine(seed, key.dst_type);
        return seed;
    }
};

/** Glu **/

struct ggml_webgpu_glu_pipeline_key {
    ggml_glu_op glu_op;
    ggml_type   type;
    bool        split;

    bool operator==(const ggml_webgpu_glu_pipeline_key & other) const {
        return glu_op == other.glu_op && type == other.type && split == other.split;
    }
};

struct ggml_webgpu_glu_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_glu_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.glu_op);
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.split);
        return seed;
    }
};

/** Rope **/

struct ggml_webgpu_rope_pipeline_key {
    ggml_type type;
    bool      inplace;
    bool      has_ff;

    bool operator==(const ggml_webgpu_rope_pipeline_key & other) const {
        return type == other.type && inplace == other.inplace && has_ff == other.has_ff;
    }
};

struct ggml_webgpu_rope_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_rope_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.type);
        ggml_webgpu_hash_combine(seed, key.inplace);
        ggml_webgpu_hash_combine(seed, key.has_ff);
        return seed;
    }
};

/** SoftMax **/

struct ggml_webgpu_soft_max_pipeline_key {
    ggml_type mask_type;
    bool      has_mask;
    bool      has_sink;
    bool      inplace;

    bool operator==(const ggml_webgpu_soft_max_pipeline_key & other) const {
        return mask_type == other.mask_type && has_mask == other.has_mask && has_sink == other.has_sink &&
               inplace == other.inplace;
    }
};

struct ggml_webgpu_soft_max_pipeline_key_hash {
    size_t operator()(const ggml_webgpu_soft_max_pipeline_key & key) const {
        size_t seed = 0;
        ggml_webgpu_hash_combine(seed, key.mask_type);
        ggml_webgpu_hash_combine(seed, key.has_mask);
        ggml_webgpu_hash_combine(seed, key.has_sink);
        ggml_webgpu_hash_combine(seed, key.inplace);
        return seed;
    }
};

class ggml_webgpu_shader_lib {
    wgpu::Device           device;
    pre_wgsl::Preprocessor preprocessor;

    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
    std::unordered_map<int, webgpu_pipeline> argmax_pipelines;         // key is vec4
    std::unordered_map<int, webgpu_pipeline> argsort_pipelines;        // key is order
    std::unordered_map<int, webgpu_pipeline> argsort_merge_pipelines;  // key is order
    std::unordered_map<int, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
    std::unordered_map<ggml_webgpu_row_norm_pipeline_key, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_key_hash>
        row_norm_pipelines;                                            // op/inplace

    std::unordered_map<ggml_webgpu_get_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_key_hash>
        get_rows_pipelines;   // src_type, vectorized
    std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash>
        unary_pipelines;      // type/op/inplace
    std::unordered_map<ggml_webgpu_scale_pipeline_key, webgpu_pipeline, ggml_webgpu_scale_pipeline_key_hash>
        scale_pipelines;      // inplace
    std::unordered_map<ggml_webgpu_solve_tri_pipeline_key, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_key_hash>
        solve_tri_pipelines;  // type
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_key, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_key_hash>
        ssm_conv_pipelines;   // type/vectorized
    std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_gated_delta_net_pipeline_key_hash>
        gated_delta_net_pipelines;  // type/S_v/kda
    std::unordered_map<ggml_webgpu_pad_pipeline_key, webgpu_pipeline, ggml_webgpu_pad_pipeline_key_hash>
        pad_pipelines;              // circular/non-circular
    std::unordered_map<ggml_webgpu_binary_pipeline_key, webgpu_pipeline, ggml_webgpu_binary_pipeline_key_hash>
        binary_pipelines;           // type/op/inplace/overlap
    std::unordered_map<ggml_webgpu_concat_pipeline_key, webgpu_pipeline, ggml_webgpu_concat_pipeline_key_hash>
        concat_pipelines;           // type
    std::unordered_map<ggml_webgpu_repeat_pipeline_key, webgpu_pipeline, ggml_webgpu_repeat_pipeline_key_hash>
        repeat_pipelines;           // type
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
        flash_attn_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
        flash_attn_vec_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash>
        flash_attn_vec_reduce_pipelines;
    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_flash_attn_blk_pipeline_key_hash>
        flash_attn_blk_pipelines;
    std::unordered_map<ggml_webgpu_legacy_mul_mat_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_legacy_mul_mat_pipeline_key_hash>
        mul_mat_legacy_pipelines;  // legacy mul_mat (non-subgroup/non-regtile/non-vec)
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_vec_pipeline_key_hash>
        mul_mat_vec_pipelines;     // fast mat-vec (n==1)
    std::unordered_map<ggml_webgpu_mul_mat_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_pipeline_key_hash>
                                             mul_mat_fast_pipelines;       // fast mat-mat (reg-tile or subgroup)
    std::unordered_map<int, webgpu_pipeline> mul_mat_id_gather_pipelines;  // key is fixed
    std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key, webgpu_pipeline, ggml_webgpu_mul_mat_id_pipeline_key_hash>
        mul_mat_id_pipelines;                                              // src0_type/src1_type

    std::unordered_map<ggml_webgpu_set_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_set_rows_pipeline_key_hash>
        set_rows_pipelines;
    std::unordered_map<ggml_webgpu_set_pipeline_key, webgpu_pipeline, ggml_webgpu_set_pipeline_key_hash> set_pipelines;
    std::unordered_map<ggml_webgpu_cpy_pipeline_key, webgpu_pipeline, ggml_webgpu_cpy_pipeline_key_hash> cpy_pipelines;
    std::unordered_map<ggml_webgpu_glu_pipeline_key, webgpu_pipeline, ggml_webgpu_glu_pipeline_key_hash> glu_pipelines;
    std::unordered_map<ggml_webgpu_rope_pipeline_key, webgpu_pipeline, ggml_webgpu_rope_pipeline_key_hash>
        rope_pipelines;
    std::unordered_map<ggml_webgpu_soft_max_pipeline_key, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_key_hash>
        soft_max_pipelines;
    std::unordered_map<ggml_webgpu_conv2d_pipeline_key, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_key_hash>
        conv2d_pipelines;
    std::unordered_map<ggml_webgpu_im2col_pipeline_key, webgpu_pipeline, ggml_webgpu_im2col_pipeline_key_hash>
        im2col_pipelines;

    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_key,
                       webgpu_pipeline,
                       ggml_webgpu_rms_norm_mul_pipeline_key_hash>
        rms_norm_mul_pipelines;

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

    webgpu_pipeline get_row_norm_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_row_norm_pipeline_key key = {};
        key.op                                = context.dst->op;
        key.inplace                           = context.inplace;

        auto it = row_norm_pipelines.find(key);
        if (it != row_norm_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        std::string              variant;

        switch (key.op) {
            case GGML_OP_RMS_NORM:
                defines.push_back("RMS_NORM");
                variant = "rms_norm";
                break;
            case GGML_OP_L2_NORM:
                defines.push_back("L2_NORM");
                variant = "l2_norm";
                break;
            default:
                GGML_ABORT("Unsupported op for row_norm shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        const uint32_t row_norm_wg_size = 128u;
        uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        auto processed          = preprocessor.preprocess(wgsl_row_norm, defines);
        row_norm_pipelines[key] = ggml_webgpu_create_pipeline(device, processed, variant);
        return row_norm_pipelines[key];
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
        ggml_webgpu_set_rows_pipeline_key key = {};
        key.dst_type                          = context.dst->type;
        key.vec4                              = context.src0->ne[0] % 4 == 0;
        key.i64_idx                           = context.src1->type == GGML_TYPE_I64;

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

    webgpu_pipeline get_set_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_set_pipeline_key key = {};
        key.type                         = context.dst->type;
        key.inplace                      = context.inplace;

        auto it = set_pipelines.find(key);
        if (it != set_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "set";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported type for set shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_set, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        set_pipelines[key]       = pipeline;
        return set_pipelines[key];
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
        ggml_webgpu_get_rows_pipeline_key key = {};
        key.src_type                          = context.src0->type;
        key.vectorized                        = (int) vectorized;

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
                defines.push_back("FLOAT_PARALLEL");
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
                defines.push_back("FLOAT_PARALLEL");
                defines.push_back("F16");
                defines.push_back("SRC_TYPE=f16");
                defines.push_back("DST_TYPE=f32");
                defines.push_back("BLOCK_SIZE=1u");
                variant += "_f16";
                break;
            case GGML_TYPE_I32:
                defines.push_back("FLOAT_PARALLEL");
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

                    switch (key.src_type) {
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ4_NL:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                break;
                            }
                        default:
                            {
                                defines.push_back(std::string("SRC_TYPE=") + type_str);
                            }
                    }

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back(type_upper + "_T");
                    defines.push_back(type_upper);
                    defines.push_back(type_upper + "_SCALE_MIN");
                    defines.push_back(type_upper + "_TABLES");
                    defines.push_back(type_upper + "_GRID");

                    variant += "_";
                    variant += type_str;

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
        ggml_webgpu_scale_pipeline_key key = {};
        key.inplace                        = context.inplace;

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

    webgpu_pipeline get_solve_tri_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_solve_tri_pipeline_key key = {};
        key.type                               = context.dst->type;
        key.n                                  = (int) context.src0->ne[0];
        key.k                                  = (int) context.src1->ne[0];

        auto it = solve_tri_pipelines.find(key);
        if (it != solve_tri_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "solve_tri";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for solve_tri shader");
        }

        const uint32_t wg_size       = std::min((uint32_t) key.n, context.max_wg_size);
        const uint32_t k_tile        = wg_size;
        const uint32_t bytes_per_row = ((uint32_t) key.n + wg_size) * GGML_WEBGPU_F32_SIZE_BYTES;
        const uint32_t batch_n       = (uint32_t) (context.wg_mem_limit_bytes / bytes_per_row);

        defines.push_back(std::string("N=") + std::to_string(key.n));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("K_TILE=") + std::to_string(k_tile));
        defines.push_back(std::string("BATCH_N=") + std::to_string(batch_n));

        auto processed           = preprocessor.preprocess(wgsl_solve_tri, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        solve_tri_pipelines[key] = pipeline;
        return solve_tri_pipelines[key];
    }

    webgpu_pipeline get_ssm_conv_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_ssm_conv_pipeline_key key = {};
        key.type                              = context.dst->type;
        key.vectorized                        = context.src1->ne[0] == 4;

        auto it = ssm_conv_pipelines.find(key);
        if (it != ssm_conv_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "ssm_conv";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for ssm_conv shader");
        }

        if (key.vectorized) {
            defines.push_back("VECTORIZED");
            variant += "_vec4";
        }

        constexpr uint32_t block_size    = 32u;
        constexpr uint32_t tokens_per_wg = 8u;

        defines.push_back("BLOCK_SIZE=" + std::to_string(block_size) + "u");
        defines.push_back("TOKENS_PER_WG=" + std::to_string(tokens_per_wg) + "u");

        auto processed           = preprocessor.preprocess(wgsl_ssm_conv, defines);
        auto decisions           = std::make_shared<ggml_webgpu_ssm_conv_shader_decisions>();
        decisions->block_size    = block_size;
        decisions->tokens_per_wg = tokens_per_wg;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        ssm_conv_pipelines[key]  = pipeline;
        return ssm_conv_pipelines[key];
    }

    webgpu_pipeline get_gated_delta_net_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_gated_delta_net_pipeline_key key = {};
        key.type                                     = context.dst->type;
        key.s_v                                      = (int) context.src2->ne[0];
        key.kda                                      = context.src3->ne[0] == context.src2->ne[0];

        auto it = gated_delta_net_pipelines.find(key);
        if (it != gated_delta_net_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "gated_delta_net";

        switch (key.type) {
            case GGML_TYPE_F32:
                variant += "_f32";
                break;
            default:
                GGML_ABORT("Unsupported type for gated_delta_net shader");
        }

        if (key.kda) {
            defines.push_back("KDA");
            variant += "_kda";
        }

        defines.push_back("S_V=" + std::to_string(key.s_v) + "u");
        defines.push_back("WG_SIZE=" + std::to_string(key.s_v) + "u");

        auto            processed      = preprocessor.preprocess(wgsl_gated_delta_net, defines);
        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, variant);
        gated_delta_net_pipelines[key] = pipeline;
        return gated_delta_net_pipelines[key];
    }

    webgpu_pipeline get_pad_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_pad_pipeline_key key = {};
        key.circular                     = ggml_get_op_params_i32(context.dst, 8) != 0;

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
        ggml_webgpu_mul_mat_vec_pipeline_key key = {};
        key.src0_type                            = context.src0->type;
        key.src1_type                            = context.src1->type;
        key.vectorized                           = (context.src0->ne[0] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                                                       1 :
                                                       0;

        auto it = mul_mat_vec_pipelines.find(key);
        if (it != mul_mat_vec_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant    = "mul_mat_vec";
        const char *             shader_src = wgsl_mul_mat_vec;

        // src0 type (matrix row)
        switch (context.src0->type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC0_INNER_TYPE=f32");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("MUL_ACC_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    // Quantized types: use helpers but accumulate in f16
                    const struct ggml_type_traits * src0_traits = ggml_get_type_traits(context.src0->type);
                    std::string                     src0_name   = src0_traits->type_name;
                    std::string                     type_upper  = src0_name;
                    variant += "_" + src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("MUL_ACC_" + type_upper);
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");
                    break;
                }
        }

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

        // VEC/SCALAR controls
        defines.push_back(key.vectorized ? "VEC" : "SCALAR");

        uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
        uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;

        if (key.src0_type >= GGML_TYPE_Q2_K) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
        } else if (key.src0_type >= GGML_TYPE_Q4_0) {
            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
        defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
        variant += context.supports_subgroups ? "_sg_reduce" : "_wg_reduce";
        if (key.vectorized) {
            variant += "_vectorized";
        }

        auto processed            = preprocessor.preprocess(shader_src, defines);
        auto decisions            = std::make_shared<ggml_webgpu_mul_mat_vec_shader_decisions>();
        decisions->wg_size        = wg_size;
        decisions->outputs_per_wg = outputs_per_wg;
        decisions->vec_size       = key.vectorized ? 4 : 1;

        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context           = decisions;
        mul_mat_vec_pipelines[key] = pipeline;
        return mul_mat_vec_pipelines[key];
    }

    webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_pipeline_key key = {};
        key.src0_type                        = context.src0->type;
        key.src1_type                        = context.src1->type;
        key.vectorized = (context.src0->ne[0] % 4 == 0 && context.dst->ne[0] % 4 == 0 && context.dst->ne[1] % 4 == 0 &&
                          (context.src0->type == GGML_TYPE_F32 || context.src0->type == GGML_TYPE_F16)) ?
                             1 :
                             0;
        key.use_subgroup_matrix = context.supports_subgroup_matrix;

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
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

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
        ggml_webgpu_legacy_mul_mat_pipeline_key key = {};
        key.src0_type                               = context.src0->type;
        key.src1_type                               = context.src1->type;

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
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    switch (context.src0->type) {
                        case GGML_TYPE_Q4_0:
                        case GGML_TYPE_Q5_0:
                        case GGML_TYPE_Q8_0:
                        case GGML_TYPE_Q3_K:
                        case GGML_TYPE_Q6_K:
                        case GGML_TYPE_IQ2_XXS:
                        case GGML_TYPE_IQ2_XS:
                        case GGML_TYPE_IQ2_S:
                        case GGML_TYPE_IQ3_XXS:
                        case GGML_TYPE_IQ3_S:
                        case GGML_TYPE_IQ1_S:
                        case GGML_TYPE_IQ4_NL:
                            {
                                // Quantized types using u32 buffers for portability.
                                defines.push_back("SRC0_TYPE=u32");
                                defines.push_back("U32_DEQUANT_HELPERS");
                                break;
                            }
                        default:
                            {
                                defines.push_back(std::string("SRC0_TYPE=") + src0_name);
                            }
                    }

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

    webgpu_pipeline get_mul_mat_id_gather_pipeline(const ggml_webgpu_shader_lib_context & context) {
        auto it = mul_mat_id_gather_pipelines.find(1);
        if (it != mul_mat_id_gather_pipelines.end()) {
            return it->second;
        }
        std::vector<std::string> defines;
        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed     = preprocessor.preprocess(wgsl_mul_mat_id_gather, defines);
        auto decisions     = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size = context.max_wg_size;

        webgpu_pipeline pipeline       = ggml_webgpu_create_pipeline(device, processed, "mul_mat_id_gather");
        pipeline.context               = decisions;
        mul_mat_id_gather_pipelines[1] = pipeline;
        return pipeline;
    }

    webgpu_pipeline get_mul_mat_id_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_mul_mat_id_pipeline_key key = {};
        key.src0_type                           = context.src0->type;
        key.src1_type                           = context.src1->type;

        auto it = mul_mat_id_pipelines.find(key);
        if (it != mul_mat_id_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "mul_mat_id";
        defines.push_back("MUL_MAT_ID");

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
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC0_INNER_TYPE=f16");
                defines.push_back("FLOAT");
                defines.push_back("INIT_SRC0_SHMEM_FLOAT");
                defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                variant += "_f16";
                break;
            default:
                {
                    std::string type_upper = src0_name;
                    std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

                    defines.push_back("BYTE_HELPERS");
                    defines.push_back("INIT_SRC0_SHMEM_" + type_upper);
                    defines.push_back("INIT_SRC1_SHMEM_FLOAT");
                    defines.push_back("U32_DEQUANT_HELPERS");
                    defines.push_back("SRC0_INNER_TYPE=u32");

                    variant += std::string("_") + src0_name;
                    break;
                }
        }

        defines.push_back("SCALAR");

        // Tiles
        defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
        defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
        defines.push_back("TILE_K=" + std::to_string(WEBGPU_MUL_MAT_TILE_K) + "u");

        defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
        defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");

        // variant suffix for src1 type
        variant += std::string("_") + (context.src1->type == GGML_TYPE_F32 ? "f32" : "f16");

        auto processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);

        auto decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
        decisions->tile_k    = WEBGPU_MUL_MAT_TILE_K;
        decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
        decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
        decisions->wg_size_m = WEBGPU_MUL_MAT_WG_SIZE_M;
        decisions->wg_size_n = WEBGPU_MUL_MAT_WG_SIZE_N;
        decisions->wg_size   = WEBGPU_MUL_MAT_WG_SIZE_M * WEBGPU_MUL_MAT_WG_SIZE_N;

        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context          = decisions;
        mul_mat_id_pipelines[key] = pipeline;
        return mul_mat_id_pipelines[key];
    }

    webgpu_pipeline get_unary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const bool                     is_unary = context.dst->op == GGML_OP_UNARY;
        const int                      op       = is_unary ? (int) ggml_get_unary_op(context.dst) : context.dst->op;
        ggml_webgpu_unary_pipeline_key key      = {};
        key.type                                = context.dst->type;
        key.op                                  = op;
        key.is_unary                            = is_unary;
        key.inplace                             = context.inplace;
        key.ttype                               = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);

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

        if (op == GGML_OP_TRI) {
            switch (key.ttype) {
                case GGML_TRI_TYPE_LOWER:
                    defines.push_back("TRI_TYPE_LOWER");
                    variant += "_tri_type_lower";
                    break;
                case GGML_TRI_TYPE_LOWER_DIAG:
                    defines.push_back("TRI_TYPE_LOWER_DIAG");
                    variant += "_tri_type_lower_diag";
                    break;
                case GGML_TRI_TYPE_UPPER:
                    defines.push_back("TRI_TYPE_UPPER");
                    variant += "_tri_type_upper";
                    break;
                case GGML_TRI_TYPE_UPPER_DIAG:
                    defines.push_back("TRI_TYPE_UPPER_DIAG");
                    variant += "_tri_upper_diag";
                    break;
                default:
                    GGML_ABORT("Unsupported ggml_tri_type for unary shader");
            }
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

    webgpu_pipeline get_rms_norm_mul_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rms_norm_mul_pipeline_key key = {};
        key.inplace                               = context.inplace;
        key.overlap                               = context.overlap;
        key.src_overlap                           = context.src_overlap;

        auto it = rms_norm_mul_pipelines.find(key);
        if (it != rms_norm_mul_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              op_name = "RMS_NORM_MUL";
        std::string              variant = op_name;

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        } else if (key.overlap) {
            defines.push_back("OVERLAP");
            variant += "_overlap";
        } else if (key.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed              = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
        auto decisions              = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size          = context.max_wg_size;
        webgpu_pipeline pipeline    = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context            = decisions;
        rms_norm_mul_pipelines[key] = pipeline;
        return rms_norm_mul_pipelines[key];
    }

    webgpu_pipeline get_binary_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_binary_pipeline_key key = {};
        key.type                            = context.dst->type;
        key.op                              = context.dst->op;
        key.inplace                         = context.inplace;
        key.overlap                         = context.overlap;
        key.src_overlap                     = context.src_overlap;

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
        } else if (key.src_overlap) {
            defines.push_back("SRC_OVERLAP");
            variant += "_src_overlap";
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

    webgpu_pipeline get_concat_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_concat_pipeline_key key = {};
        key.type                            = context.dst->type;

        auto it = concat_pipelines.find(key);
        if (it != concat_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "concat";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported type for concat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_concat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        concat_pipelines[key]    = pipeline;
        return concat_pipelines[key];
    }

    webgpu_pipeline get_repeat_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_repeat_pipeline_key key = {};
        key.type                            = context.dst->type;

        auto it = repeat_pipelines.find(key);
        if (it != repeat_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "repeat";

        switch (key.type) {
            case GGML_TYPE_F32:
                defines.push_back("TYPE_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_I32:
                defines.push_back("TYPE_I32");
                variant += "_i32";
                break;
            case GGML_TYPE_I16:
                defines.push_back("TYPE_I16");
                variant += "_i16";
                break;
            default:
                GGML_ABORT("Unsupported type for repeat shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_repeat, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        repeat_pipelines[key]    = pipeline;
        return repeat_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context);
        auto                                      it  = flash_attn_pipelines.find(key);
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

        auto decisions    = std::make_shared<ggml_webgpu_flash_attn_decisions>();
        decisions->q_tile = context.sg_mat_m;

        const uint32_t min_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(context, key);
        uint32_t       kv_tile = std::min(min_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);

        if (key.kv_direct) {
            kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
            while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
                kv_tile -= context.sg_mat_n;
            }
        }

        decisions->kv_tile = kv_tile;
        decisions->wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);

        defines.push_back(std::string("Q_TILE=") + std::to_string(decisions->q_tile));
        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions->kv_tile));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions->wg_size));

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn, defines), variant);
        pipeline.context          = decisions;
        flash_attn_pipelines[key] = pipeline;
        return flash_attn_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
        const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context);
        auto                                      it  = flash_attn_vec_pipelines.find(key);
        if (it != flash_attn_vec_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec";

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
            defines.push_back("BLK");
            variant += "_mask_blk";
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
        defines.push_back("Q_TILE=1");

        auto decisions     = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>();
        decisions->kv_tile = ggml_webgpu_flash_attn_vec_get_kv_tile(context);
        decisions->wg_size = std::max(1u, std::min<uint32_t>(32u, context.max_subgroup_size));
        uint32_t vec_ne    = 1u;

        // Keep conservative defaults unless this is the f16 vec-split shape family.
        if (key.kv_type == GGML_TYPE_F16 && key.head_dim_qk == key.head_dim_v) {
            switch (key.head_dim_qk) {
                case 64:
                case 192:
                case 576:
                    vec_ne = 2u;
                    break;
                case 96:
                    vec_ne = 4u;
                    break;
                default:
                    break;
            }
        }

        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions->kv_tile));
        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions->wg_size));
        defines.push_back(std::string("VEC_NE=") + std::to_string(vec_ne) + "u");

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
        pipeline.context              = decisions;
        flash_attn_vec_pipelines[key] = pipeline;
        return flash_attn_vec_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_blk_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_blk_pipeline_key key = {};
        key.kv_tile                                 = ggml_webgpu_flash_attn_vec_get_kv_tile(context);
        auto it                                     = flash_attn_blk_pipelines.find(key);
        if (it != flash_attn_blk_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_blk";

        defines.push_back(std::string("KV_TILE=") + std::to_string(key.kv_tile));
        variant += std::string("_kvt") + std::to_string(key.kv_tile);

        uint32_t wg_size = 1;
        while ((wg_size << 1) <= context.max_wg_size) {
            wg_size <<= 1;
        }
        defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
        variant += std::string("_wg") + std::to_string(wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_blk, defines), variant);
        flash_attn_blk_pipelines[key] = pipeline;
        return flash_attn_blk_pipelines[key];
    }

    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_flash_attn_vec_reduce_pipeline_key key = {};
        key.head_dim_v                                     = (uint32_t) context.src2->ne[0];
        key.wg_size                                        = context.max_wg_size;
        auto it                                            = flash_attn_vec_reduce_pipelines.find(key);
        if (it != flash_attn_vec_reduce_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "flash_attn_vec_reduce";

        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
        variant += std::string("_hsv") + std::to_string(key.head_dim_v);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
        variant += std::string("_wg") + std::to_string(context.max_wg_size);

        webgpu_pipeline pipeline =
            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_reduce, defines), variant);
        flash_attn_vec_reduce_pipelines[key] = pipeline;
        return flash_attn_vec_reduce_pipelines[key];
    }

    webgpu_pipeline get_cpy_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_cpy_pipeline_key key = {};
        key.src_type                     = context.src0->type;
        key.dst_type                     = context.dst->type;

        auto it = cpy_pipelines.find(key);
        if (it != cpy_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "cpy";

        switch (key.src_type) {
            case GGML_TYPE_F32:
                defines.push_back("SRC_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("SRC_F16");
                variant += "_f16";
                break;
            default:
                GGML_ABORT("Unsupported src type for cpy shader");
        }

        switch (key.dst_type) {
            case GGML_TYPE_F32:
                defines.push_back("DST_F32");
                variant += "_f32";
                break;
            case GGML_TYPE_F16:
                defines.push_back("DST_F16");
                variant += "_f16";
                break;
            case GGML_TYPE_I32:
                defines.push_back("DST_I32");
                variant += "_i32";
                break;
            default:
                GGML_ABORT("Unsupported dst type for cpy shader");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_cpy, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        cpy_pipelines[key]       = pipeline;
        return cpy_pipelines[key];
    }

    webgpu_pipeline get_glu_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_glu_pipeline_key key = {};
        key.glu_op                       = ggml_get_glu_op(context.dst);
        key.type                         = context.dst->type;
        key.split                        = (context.src1 != nullptr);

        auto it = glu_pipelines.find(key);
        if (it != glu_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "glu";

        switch (key.glu_op) {
            case GGML_GLU_OP_REGLU:
                defines.push_back("OP_REGLU");
                variant += "_reglu";
                break;
            case GGML_GLU_OP_GEGLU:
                defines.push_back("OP_GEGLU");
                variant += "_geglu";
                break;
            case GGML_GLU_OP_SWIGLU:
                defines.push_back("OP_SWIGLU");
                variant += "_swiglu";
                break;
            case GGML_GLU_OP_SWIGLU_OAI:
                defines.push_back("OP_SWIGLU_OAI");
                variant += "_swiglu_oai";
                break;
            case GGML_GLU_OP_GEGLU_ERF:
                defines.push_back("OP_GEGLU_ERF");
                variant += "_geglu_erf";
                break;
            case GGML_GLU_OP_GEGLU_QUICK:
                defines.push_back("OP_GEGLU_QUICK");
                variant += "_geglu_quick";
                break;
            default:
                GGML_ABORT("Unsupported GLU op");
        }
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
                GGML_ABORT("Unsupported type for GLU shader");
        }

        if (key.split) {
            variant += "_split";
        } else {
            defines.push_back("NO_SPLIT");
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_glu, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        glu_pipelines[key]       = pipeline;
        return glu_pipelines[key];
    }

    webgpu_pipeline get_rope_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_rope_pipeline_key key = {};
        key.type                          = context.dst->type;
        key.inplace                       = context.inplace;
        key.has_ff                        = (context.src2 != nullptr);

        auto it = rope_pipelines.find(key);
        if (it != rope_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "rope";

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
                GGML_ABORT("Unsupported type for ROPE shader");
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        if (key.has_ff) {
            defines.push_back("FF_FUNC");
            variant += "_ff";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_rope, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        rope_pipelines[key]      = pipeline;
        return rope_pipelines[key];
    }

    webgpu_pipeline get_soft_max_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_soft_max_pipeline_key key = {};
        key.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
        key.has_mask                          = (context.src1 != nullptr);
        key.has_sink                          = (context.src2 != nullptr);
        key.inplace                           = context.inplace;

        auto it = soft_max_pipelines.find(key);
        if (it != soft_max_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "soft_max";

        if (key.has_mask) {
            defines.push_back("HAS_MASK");
            switch (key.mask_type) {
                case GGML_TYPE_F32:
                    defines.push_back("MASK_F32");
                    variant += "_mask_f32";
                    break;
                case GGML_TYPE_F16:
                    defines.push_back("MASK_F16");
                    variant += "_mask_f16";
                    break;
                default:
                    GGML_ABORT("Unsupported type for SOFT_MAX shader");
            }
        }

        if (key.has_sink) {
            defines.push_back("HAS_SINK");
            variant += "_sink";
        }

        if (key.inplace) {
            defines.push_back("INPLACE");
            variant += "_inplace";
        }

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        soft_max_pipelines[key]  = pipeline;
        return soft_max_pipelines[key];
    }

    webgpu_pipeline get_conv2d_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_conv2d_pipeline_key key = {};
        key.weight_type                     = context.src0->type;
        key.input_type                      = context.src1->type;
        key.output_type                     = context.dst->type;

        auto it = conv2d_pipelines.find(key);
        if (it != conv2d_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "conv_2d";

        auto push_type_defines = [&](const char * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            if (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "_F32");
            } else if (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "_F16");
            } else {
                GGML_ABORT("Unsupported type for CONV_2D shader");
            }
        };

        push_type_defines("WEIGHT", key.weight_type);
        push_type_defines("INPUT", key.input_type);
        push_type_defines("OUTPUT", key.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        conv2d_pipelines[key]    = pipeline;
        return conv2d_pipelines[key];
    }

    webgpu_pipeline get_im2col_pipeline(const ggml_webgpu_shader_lib_context & context) {
        ggml_webgpu_im2col_pipeline_key key = {};
        key.input_type                      = context.src1->type;
        key.output_type                     = context.dst->type;

        auto it = im2col_pipelines.find(key);
        if (it != im2col_pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> defines;
        std::string              variant = "im2col";

        auto push_type_defines = [&](const char * prefix, ggml_type type) {
            std::string s_prefix = prefix;
            if (type == GGML_TYPE_F32) {
                defines.push_back(s_prefix + "_F32");
            } else if (type == GGML_TYPE_F16) {
                defines.push_back(s_prefix + "_F16");
            } else {
                GGML_ABORT("Unsupported type for IM2COL shader");
            }
        };

        push_type_defines("INPUT", key.input_type);
        push_type_defines("OUTPUT", key.output_type);

        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));

        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
        decisions->wg_size       = context.max_wg_size;
        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
        pipeline.context         = decisions;
        im2col_pipelines[key]    = pipeline;
        return im2col_pipelines[key];
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
};

#endif  // GGML_WEBGPU_SHADER_LIB_HPP
