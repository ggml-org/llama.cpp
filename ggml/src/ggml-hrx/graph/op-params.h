#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
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
    bool        swapped = false;
};

struct ScaleParams {
    float scale = 0.0f;
    float bias  = 0.0f;
};

enum class BinaryKind : uint32_t {
    Add        = 0,
    Sub        = 1,
    Mul        = 2,
    Div        = 3,
    SwiGLU     = 4,
    GeGLU      = 5,
    RegLU      = 6,
    GeGLUErf   = 7,
    GeGLUQuick = 8,
};

struct BinaryParams {
    BinaryKind op = BinaryKind::Add;
};

enum class UnaryKind : uint32_t {
    Neg         = 0,
    Abs         = 1,
    Relu        = 2,
    Step        = 3,
    Sqr         = 4,
    Sgn         = 5,
    Floor       = 6,
    Ceil        = 7,
    Round       = 8,
    Trunc       = 9,
    Tanh        = 10,
    Elu         = 11,
    Sigmoid     = 12,
    Gelu        = 13,
    GeluQuick   = 14,
    Silu        = 15,
    HardSwish   = 16,
    HardSigmoid = 17,
    Exp         = 18,
    Expm1       = 19,
    GeluErf     = 20,
    Sqrt        = 21,
    Log         = 22,
    Identity    = 23,

    // Parameterized or remaining exact-math unary routes need separate lowering work.
    SoftPlus = 100,
    Xielu,
    Sin,
    Cos,
};

struct UnaryParams {
    UnaryKind op = UnaryKind::Abs;
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
    std::array<int, GGML_MROPE_SECTIONS> sections    = {};
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
    ScaleParams,
    BinaryParams,
    UnaryParams,
    RopeParams>;
// clang-format on

template <typename T> const T * op_params_as(const OpParams & params) {
    return std::get_if<T>(&params);
}

OpParams import_op_params(const ggml_tensor & tensor);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const ggml_tensor & rhs);
bool     import_binary_kind(const ggml_tensor & tensor, BinaryKind & kind);
bool     binary_kind_supported(BinaryKind kind);
uint32_t binary_kind_config_value(BinaryKind kind);
bool     import_unary_kind(const ggml_tensor & tensor, UnaryKind & kind);
bool     unary_kind_supported(UnaryKind kind);
uint32_t unary_kind_config_value(UnaryKind kind);

}  // namespace ggml::hrx
