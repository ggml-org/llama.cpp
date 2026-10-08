#include "op-params.h"

#include "ggml-impl.h"

#include <cmath>

namespace ggml::hrx {
namespace {

static bool nearly_equal(float lhs, float rhs) {
    if (lhs == rhs) {
        return true;
    }
    return std::fabs(lhs - rhs) <= 1.0e-12f;
}

static bool rms_norm_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const RmsNormParams * lhs_params = op_params_as<RmsNormParams>(lhs);
    const RmsNormParams * rhs_params = op_params_as<RmsNormParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->eps, rhs_params->eps);
}

static bool flash_attn_ext_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const FlashAttnExtParams * lhs_params = op_params_as<FlashAttnExtParams>(lhs);
    const FlashAttnExtParams * rhs_params = op_params_as<FlashAttnExtParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->scale, rhs_params->scale) &&
           nearly_equal(lhs_params->max_bias, rhs_params->max_bias) &&
           nearly_equal(lhs_params->logit_softcap, rhs_params->logit_softcap) && lhs_params->prec == rhs_params->prec;
}

static bool soft_max_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const SoftMaxParams * lhs_params = op_params_as<SoftMaxParams>(lhs);
    const SoftMaxParams * rhs_params = op_params_as<SoftMaxParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->scale, rhs_params->scale) &&
           nearly_equal(lhs_params->max_bias, rhs_params->max_bias);
}

static bool argsort_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const ArgsortParams * lhs_params = op_params_as<ArgsortParams>(lhs);
    const ArgsortParams * rhs_params = op_params_as<ArgsortParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && lhs_params->order == rhs_params->order;
}

static bool clamp_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const ClampParams * lhs_params = op_params_as<ClampParams>(lhs);
    const ClampParams * rhs_params = op_params_as<ClampParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->min, rhs_params->min) &&
           nearly_equal(lhs_params->max, rhs_params->max);
}

static bool glu_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const GluParams * lhs_params = op_params_as<GluParams>(lhs);
    const GluParams * rhs_params = op_params_as<GluParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && lhs_params->op == rhs_params->op &&
           lhs_params->swapped == rhs_params->swapped;
}

static bool scale_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const ScaleParams * lhs_params = op_params_as<ScaleParams>(lhs);
    const ScaleParams * rhs_params = op_params_as<ScaleParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->scale, rhs_params->scale) &&
           nearly_equal(lhs_params->bias, rhs_params->bias);
}

static bool binary_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const BinaryParams * lhs_params = op_params_as<BinaryParams>(lhs);
    const BinaryParams * rhs_params = op_params_as<BinaryParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && lhs_params->op == rhs_params->op;
}

static bool unary_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const UnaryParams * lhs_params = op_params_as<UnaryParams>(lhs);
    const UnaryParams * rhs_params = op_params_as<UnaryParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && lhs_params->op == rhs_params->op;
}

static bool rope_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const RopeParams * lhs_params = op_params_as<RopeParams>(lhs);
    const RopeParams * rhs_params = op_params_as<RopeParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && lhs_params->n_dims == rhs_params->n_dims &&
           lhs_params->mode == rhs_params->mode && lhs_params->n_ctx_orig == rhs_params->n_ctx_orig &&
           nearly_equal(lhs_params->freq_base, rhs_params->freq_base) &&
           nearly_equal(lhs_params->freq_scale, rhs_params->freq_scale) &&
           nearly_equal(lhs_params->ext_factor, rhs_params->ext_factor) &&
           nearly_equal(lhs_params->attn_factor, rhs_params->attn_factor) &&
           nearly_equal(lhs_params->beta_fast, rhs_params->beta_fast) &&
           nearly_equal(lhs_params->beta_slow, rhs_params->beta_slow) && lhs_params->sections == rhs_params->sections;
}

}  // namespace

bool import_binary_kind(const ggml_tensor & tensor, BinaryKind & kind) {
    switch (tensor.op) {
        case GGML_OP_ADD:
            kind = BinaryKind::Add;
            return true;
        case GGML_OP_SUB:
            kind = BinaryKind::Sub;
            return true;
        case GGML_OP_MUL:
            kind = BinaryKind::Mul;
            return true;
        case GGML_OP_DIV:
            kind = BinaryKind::Div;
            return true;
        case GGML_OP_GLU:
            if (tensor.src[1] == nullptr) {
                return false;
            }
            switch (ggml_get_glu_op(&tensor)) {
                case GGML_GLU_OP_REGLU:
                    kind = BinaryKind::RegLU;
                    return true;
                case GGML_GLU_OP_SWIGLU:
                    kind = BinaryKind::SwiGLU;
                    return true;
                case GGML_GLU_OP_GEGLU:
                    kind = BinaryKind::GeGLU;
                    return true;
                case GGML_GLU_OP_GEGLU_ERF:
                    kind = BinaryKind::GeGLUErf;
                    return true;
                case GGML_GLU_OP_GEGLU_QUICK:
                    kind = BinaryKind::GeGLUQuick;
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

bool binary_kind_supported(BinaryKind kind) {
    return static_cast<uint32_t>(kind) <= static_cast<uint32_t>(BinaryKind::GeGLUQuick);
}

uint32_t binary_kind_config_value(BinaryKind kind) {
    return static_cast<uint32_t>(kind);
}

static bool unary_kind_from_ggml_unary_op(ggml_unary_op op, UnaryKind & kind) {
    switch (op) {
        case GGML_UNARY_OP_ABS:
            kind = UnaryKind::Abs;
            return true;
        case GGML_UNARY_OP_SGN:
            kind = UnaryKind::Sgn;
            return true;
        case GGML_UNARY_OP_NEG:
            kind = UnaryKind::Neg;
            return true;
        case GGML_UNARY_OP_STEP:
            kind = UnaryKind::Step;
            return true;
        case GGML_UNARY_OP_TANH:
            kind = UnaryKind::Tanh;
            return true;
        case GGML_UNARY_OP_ELU:
            kind = UnaryKind::Elu;
            return true;
        case GGML_UNARY_OP_RELU:
            kind = UnaryKind::Relu;
            return true;
        case GGML_UNARY_OP_SIGMOID:
            kind = UnaryKind::Sigmoid;
            return true;
        case GGML_UNARY_OP_GELU:
            kind = UnaryKind::Gelu;
            return true;
        case GGML_UNARY_OP_GELU_QUICK:
            kind = UnaryKind::GeluQuick;
            return true;
        case GGML_UNARY_OP_SILU:
            kind = UnaryKind::Silu;
            return true;
        case GGML_UNARY_OP_HARDSWISH:
            kind = UnaryKind::HardSwish;
            return true;
        case GGML_UNARY_OP_HARDSIGMOID:
            kind = UnaryKind::HardSigmoid;
            return true;
        case GGML_UNARY_OP_EXP:
            kind = UnaryKind::Exp;
            return true;
        case GGML_UNARY_OP_EXPM1:
            kind = UnaryKind::Expm1;
            return true;
        case GGML_UNARY_OP_SOFTPLUS:
            kind = UnaryKind::SoftPlus;
            return true;
        case GGML_UNARY_OP_GELU_ERF:
            kind = UnaryKind::GeluErf;
            return true;
        case GGML_UNARY_OP_XIELU:
            kind = UnaryKind::Xielu;
            return true;
        case GGML_UNARY_OP_FLOOR:
            kind = UnaryKind::Floor;
            return true;
        case GGML_UNARY_OP_CEIL:
            kind = UnaryKind::Ceil;
            return true;
        case GGML_UNARY_OP_ROUND:
            kind = UnaryKind::Round;
            return true;
        case GGML_UNARY_OP_TRUNC:
            kind = UnaryKind::Trunc;
            return true;
        default:
            return false;
    }
}

bool import_unary_kind(const ggml_tensor & tensor, UnaryKind & kind) {
    switch (tensor.op) {
        case GGML_OP_UNARY:
            return unary_kind_from_ggml_unary_op(ggml_get_unary_op(&tensor), kind);
        case GGML_OP_SQR:
            kind = UnaryKind::Sqr;
            return true;
        case GGML_OP_SQRT:
            kind = UnaryKind::Sqrt;
            return true;
        case GGML_OP_LOG:
            kind = UnaryKind::Log;
            return true;
        case GGML_OP_SIN:
            kind = UnaryKind::Sin;
            return true;
        case GGML_OP_COS:
            kind = UnaryKind::Cos;
            return true;
        default:
            return false;
    }
}

bool unary_kind_supported(UnaryKind kind) {
    return static_cast<uint32_t>(kind) <= static_cast<uint32_t>(UnaryKind::Identity);
}

uint32_t unary_kind_config_value(UnaryKind kind) {
    return static_cast<uint32_t>(kind);
}

OpParams import_op_params(const ggml_tensor & tensor) {
    BinaryKind binary_kind;
    if (import_binary_kind(tensor, binary_kind)) {
        return BinaryParams{ binary_kind };
    }

    UnaryKind unary_kind;
    if (import_unary_kind(tensor, unary_kind)) {
        return UnaryParams{ unary_kind };
    }

    switch (tensor.op) {
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return RmsNormParams{ ggml_get_op_params_f32(&tensor, 0) };
        case GGML_OP_SOFT_MAX:
            return SoftMaxParams{
                ggml_get_op_params_f32(&tensor, 0),
                ggml_get_op_params_f32(&tensor, 1),
            };
        case GGML_OP_FLASH_ATTN_EXT:
            return FlashAttnExtParams{
                ggml_get_op_params_f32(&tensor, 0),
                ggml_get_op_params_f32(&tensor, 1),
                ggml_get_op_params_f32(&tensor, 2),
                ggml_flash_attn_ext_get_prec(&tensor),
            };
        case GGML_OP_ARGSORT:
            return ArgsortParams{ static_cast<ggml_sort_order>(ggml_get_op_params_i32(&tensor, 0)) };
        case GGML_OP_CLAMP:
            return ClampParams{
                ggml_get_op_params_f32(&tensor, 0),
                ggml_get_op_params_f32(&tensor, 1),
            };
        case GGML_OP_GLU:
            return GluParams{ ggml_get_glu_op(&tensor), ggml_get_op_params_i32(&tensor, 1) != 0 };
        case GGML_OP_SCALE:
            return ScaleParams{
                ggml_get_op_params_f32(&tensor, 0),
                ggml_get_op_params_f32(&tensor, 1),
            };
        case GGML_OP_ROPE:
            return RopeParams{
                ggml_get_op_params_i32(&tensor, 1),
                ggml_get_op_params_i32(&tensor, 2),
                ggml_get_op_params_i32(&tensor, 4),
                ggml_get_op_params_f32(&tensor, 5),
                ggml_get_op_params_f32(&tensor, 6),
                ggml_get_op_params_f32(&tensor, 7),
                ggml_get_op_params_f32(&tensor, 8),
                ggml_get_op_params_f32(&tensor, 9),
                ggml_get_op_params_f32(&tensor, 10),
                {
                               ggml_get_op_params_i32(&tensor, 11),
                               ggml_get_op_params_i32(&tensor, 12),
                               ggml_get_op_params_i32(&tensor, 13),
                               ggml_get_op_params_i32(&tensor, 14),
                               },
            };
        default:
            return std::monostate{};
    }
}

bool op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs) {
    switch (op) {
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return rms_norm_params_equivalent(lhs, rhs);
        case GGML_OP_SOFT_MAX:
            return soft_max_params_equivalent(lhs, rhs);
        case GGML_OP_FLASH_ATTN_EXT:
            return flash_attn_ext_params_equivalent(lhs, rhs);
        case GGML_OP_ARGSORT:
            return argsort_params_equivalent(lhs, rhs);
        case GGML_OP_CLAMP:
            return clamp_params_equivalent(lhs, rhs);
        case GGML_OP_GLU:
            return binary_params_equivalent(lhs, rhs) || glu_params_equivalent(lhs, rhs);
        case GGML_OP_SCALE:
            return scale_params_equivalent(lhs, rhs);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return binary_params_equivalent(lhs, rhs);
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return unary_params_equivalent(lhs, rhs);
        case GGML_OP_ROPE:
            return rope_params_equivalent(lhs, rhs);
        default:
            return lhs.index() == rhs.index();
    }
}

bool op_params_equivalent(ggml_op op, const OpParams & lhs, const ggml_tensor & rhs) {
    return op_params_equivalent(op, lhs, import_op_params(rhs));
}

}  // namespace ggml::hrx
