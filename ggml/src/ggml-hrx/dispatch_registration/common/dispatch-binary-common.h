#pragma once

#include "../dispatch-registry.h"
#include "graph/op-params.h"

#include <cstdint>
#include <string>

namespace ggml::hrx {

inline constexpr uint32_t common_binary_kind_bit(BinaryKind kind) {
    return uint32_t{ 1 } << static_cast<uint32_t>(kind);
}

inline constexpr uint32_t kCommonArithmeticBinaryKinds =
    common_binary_kind_bit(BinaryKind::Add) | common_binary_kind_bit(BinaryKind::Sub) |
    common_binary_kind_bit(BinaryKind::Mul) | common_binary_kind_bit(BinaryKind::Div);

inline constexpr uint32_t kCommonGatedBinaryKinds =
    common_binary_kind_bit(BinaryKind::SwiGLU) | common_binary_kind_bit(BinaryKind::GeGLU) |
    common_binary_kind_bit(BinaryKind::RegLU) | common_binary_kind_bit(BinaryKind::GeGLUErf) |
    common_binary_kind_bit(BinaryKind::GeGLUQuick);

inline bool common_binary_kind_allowed(BinaryKind kind, uint32_t allowed_kinds) {
    return binary_kind_supported(kind) && (allowed_kinds & common_binary_kind_bit(kind)) != 0;
}

inline bool common_binary_kind_requires_order(BinaryKind kind) {
    return kind == BinaryKind::Sub || kind == BinaryKind::Div;
}

inline bool common_binary_kind_matches_graph_op(BinaryKind kind, ggml_op op) {
    switch (kind) {
        case BinaryKind::Add:
            return op == GGML_OP_ADD;
        case BinaryKind::Sub:
            return op == GGML_OP_SUB;
        case BinaryKind::Mul:
            return op == GGML_OP_MUL;
        case BinaryKind::Div:
            return op == GGML_OP_DIV;
        case BinaryKind::SwiGLU:
        case BinaryKind::GeGLU:
        case BinaryKind::RegLU:
        case BinaryKind::GeGLUErf:
        case BinaryKind::GeGLUQuick:
            return op == GGML_OP_GLU;
    }
    return false;
}

inline bool common_arithmetic_kind_from_graph_op(ggml_op op, BinaryKind & kind) {
    switch (op) {
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
        default:
            return false;
    }
}

inline bool common_fused_binary_kind_from_params(const OpParams & params, BinaryKind & kind) {
    const BinaryParams * binary_params = op_params_as<BinaryParams>(params);
    if (binary_params != nullptr && binary_kind_supported(binary_params->op)) {
        kind = binary_params->op;
        return true;
    }

    const GluParams * glu_params = op_params_as<GluParams>(params);
    if (glu_params == nullptr) {
        return false;
    }

    switch (glu_params->op) {
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
}

struct CommonBinaryMatch {
    const GraphNode * node            = nullptr;
    ValueId           operand         = ValueId(-1);
    BinaryKind        kind            = BinaryKind::Add;
    bool              producer_is_lhs = true;

    bool matched() const { return node != nullptr && operand.value >= 0; }
};

inline CommonBinaryMatch common_match_binary_consumer(const GraphNode * node,
                                                      ValueId           producer,
                                                      uint32_t          allowed_kinds) {
    CommonBinaryMatch match;
    if (node == nullptr || node->inputs.size() != 2) {
        return match;
    }

    BinaryKind kind;
    const bool has_kind =
        common_fused_binary_kind_from_params(node->params, kind) ||
        (std::holds_alternative<std::monostate>(node->params) && common_arithmetic_kind_from_graph_op(node->op, kind));
    if (!has_kind || !common_binary_kind_matches_graph_op(kind, node->op) ||
        !common_binary_kind_allowed(kind, allowed_kinds)) {
        return match;
    }

    const bool producer_is_lhs = node->inputs[0] == producer;
    const bool producer_is_rhs = node->inputs[1] == producer;
    if (producer_is_lhs == producer_is_rhs) {
        return match;
    }

    match.node            = node;
    match.operand         = producer_is_lhs ? node->inputs[1] : node->inputs[0];
    match.kind            = kind;
    match.producer_is_lhs = producer_is_lhs;
    return match;
}

inline void common_set_binary_compile_parameters(KernelSpecialization & kernel,
                                                 const char *           op_key,
                                                 const char *           lhs_key,
                                                 BinaryKind             kind,
                                                 bool                   producer_is_lhs) {
    kernel.compile_parameters.emplace(op_key, std::to_string(binary_kind_config_value(kind)));
    if (lhs_key != nullptr) {
        kernel.compile_parameters.emplace(lhs_key, producer_is_lhs ? "1" : "0");
    }
}

}  // namespace ggml::hrx
