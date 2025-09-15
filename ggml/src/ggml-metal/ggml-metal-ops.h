#pragma once

#include "ggml-metal-context.h"

#ifdef __cplusplus
extern "C" {
#endif

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(const struct ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(const struct ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_get_pipeline(
        ggml_metal_t ctx,
        struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg);

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_vec_get_pipeline(
        ggml_metal_t ctx,
        struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg,
        int32_t nwg);

ggml_metal_pipeline_t ggml_metal_op_flash_attn_ext_vec_reduce_get_pipeline(
        ggml_metal_t ctx,
        struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

ggml_metal_pipeline_t ggml_metal_op_bin_get_pipeline(
        ggml_metal_t ctx,
        enum ggml_op op,
        int32_t n_fuse,
        bool row);

ggml_metal_pipeline_t ggml_metal_op_rms_norm_get_pipeline(
        ggml_metal_t ctx,
        struct ggml_tensor * op,
        int32_t n_fuse);

#ifdef __cplusplus
}
#endif
