#pragma once

#include "ggml-metal-context.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: move to ggml_metal_library_t
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_base              (ggml_metal_t ctx, enum ggml_op op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_cpy               (ggml_metal_t ctx, enum ggml_type tsrc, enum ggml_type tdst);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pool_2d           (ggml_metal_t ctx, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_get_rows          (ggml_metal_t ctx, enum ggml_type tsrc);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_set_rows          (ggml_metal_t ctx, enum ggml_type tdst);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_repeat            (ggml_metal_t ctx, enum ggml_type tsrc);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_unary             (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_glu               (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_sum_rows          (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_soft_max          (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_ssm_conv          (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_ssm_scan          (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rwkv              (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv_ext        (ggml_metal_t ctx, enum ggml_type tsrc0, enum ggml_type tsrc1, int r1ptg);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm            (ggml_metal_t ctx, enum ggml_type tsrc0, enum ggml_type tsrc1);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv            (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm_id_map0    (ggml_metal_t ctx, int ne02, int ne20);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mm_id         (ggml_metal_t ctx, enum ggml_type tsrc0, enum ggml_type tsrc1);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_mul_mv_id         (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_argmax            (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_argsort           (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_bin               (ggml_metal_t ctx, enum ggml_op op, int32_t n_fuse, bool row);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rms_norm          (ggml_metal_t ctx, const struct ggml_tensor * op, int32_t n_fuse);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_l2_norm           (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_group_norm        (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_norm              (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_rope              (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_im2col            (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_conv_transpose_1d (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_upscale           (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pad               (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_pad_reflect_1d    (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_arange            (ggml_metal_t ctx, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_op_get_pipeline_timestep_embedding(ggml_metal_t ctx, const struct ggml_tensor * op);

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext(
        ggml_metal_t ctx,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg);

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext_vec(
        ggml_metal_t ctx,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        int32_t nsg,
        int32_t nwg);

ggml_metal_pipeline_t ggml_metal_op_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_t ctx,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(const struct ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(const struct ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);

int ggml_metal_op_concat            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_repeat            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_acc               (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_scale             (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_clamp             (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_unary             (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_glu               (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_sum_rows          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_get_rows          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_set_rows          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_soft_max          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_ssm_conv          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_ssm_scan          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_rwkv              (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_cpy               (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_pool_2d           (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_mul_mat           (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_mul_mat_id        (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_add_id            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_flash_attn_ext    (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_bin               (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_rms_norm          (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_l2_norm           (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_group_norm        (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_norm              (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_rope              (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_im2col            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_conv_transpose_1d (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_upscale           (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_pad               (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_pad_reflect_1d    (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_arange            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_timestep_embedding(ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_argmax            (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_argsort           (ggml_metal_graph_encoder_t ctx_enc, int idx);
int ggml_metal_op_leaky_relu        (ggml_metal_graph_encoder_t ctx_enc, int idx);

#ifdef __cplusplus
}
#endif
