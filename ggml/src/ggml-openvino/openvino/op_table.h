#pragma once

#include "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext & context)

GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_concat);
GGML_OP_CONVERTER(translate_add_id);
GGML_OP_CONVERTER(translate_div);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_im2col);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_mul_mat_id);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_norm);
GGML_OP_CONVERTER(translate_l2_norm);
GGML_OP_CONVERTER(translate_sum_rows);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_unary_silu);
GGML_OP_CONVERTER(translate_unary_softplus);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_glu_geglu);
GGML_OP_CONVERTER(translate_set_rows);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_argsort);
GGML_OP_CONVERTER(translate_flash_attn_ext);
GGML_OP_CONVERTER(translate_clamp);
GGML_OP_CONVERTER(translate_pad);
GGML_OP_CONVERTER(translate_ssm_conv);
GGML_OP_CONVERTER(translate_gated_delta_net);
GGML_OP_CONVERTER(translate_repeat);

}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
