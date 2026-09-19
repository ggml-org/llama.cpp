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
           nearly_equal(lhs_params->beta_slow, rhs_params->beta_slow);
}

}  // namespace

OpParams import_op_params(const ggml_tensor & tensor) {
    switch (tensor.op) {
        case GGML_OP_RMS_NORM:
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
            return GluParams{ ggml_get_glu_op(&tensor) };
        case GGML_OP_ROPE:
            return RopeParams{
                ggml_get_op_params_i32(&tensor, 1),  ggml_get_op_params_i32(&tensor, 2),
                ggml_get_op_params_i32(&tensor, 4),  ggml_get_op_params_f32(&tensor, 5),
                ggml_get_op_params_f32(&tensor, 6),  ggml_get_op_params_f32(&tensor, 7),
                ggml_get_op_params_f32(&tensor, 8),  ggml_get_op_params_f32(&tensor, 9),
                ggml_get_op_params_f32(&tensor, 10),
            };
        default:
            return std::monostate{};
    }
}

bool op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs) {
    switch (op) {
        case GGML_OP_RMS_NORM:
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
            return glu_params_equivalent(lhs, rhs);
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
