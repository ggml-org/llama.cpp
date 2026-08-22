#pragma once

#include "ggml.h"

#include <variant>

struct ggml_tensor;

namespace ggml::hrx {

struct RmsNormParams {
    float eps = 0.0f;
};

struct FlashAttnExtParams {
    float     scale         = 0.0f;
    float     max_bias      = 0.0f;
    float     logit_softcap = 0.0f;
    ggml_prec prec          = GGML_PREC_DEFAULT;
};

struct SoftMaxParams {
    float scale    = 0.0f;
    float max_bias = 0.0f;
};

struct ArgsortParams {
    ggml_sort_order order = GGML_SORT_ORDER_ASC;
};

struct ClampParams {
    float min = 0.0f;
    float max = 0.0f;
};

struct GluParams {
    ggml_glu_op op = GGML_GLU_OP_REGLU;
};

struct RopeParams {
    int   n_dims      = 0;
    int   mode        = 0;
    int   n_ctx_orig  = 0;
    float freq_base   = 0.0f;
    float freq_scale  = 0.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;
};

// clang-format off
using OpParams = std::variant<
    std::monostate,
    RmsNormParams,
    FlashAttnExtParams,
    SoftMaxParams,
    ArgsortParams,
    ClampParams,
    GluParams,
    RopeParams>;
// clang-format on

template <typename T> const T * op_params_as(const OpParams & params) {
    return std::get_if<T>(&params);
}

OpParams import_op_params(const ggml_tensor & tensor);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const ggml_tensor & rhs);

}  // namespace ggml::hrx
