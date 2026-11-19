#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <memory>
#include <openvino/op/abs.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/elu.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/greater.hpp>
#include <openvino/op/hard_sigmoid.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/round.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/softplus.hpp>
#include <openvino/op/subtract.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_gelu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(input, ov::op::GeluApproximationMode::TANH);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(input, ov::op::GeluApproximationMode::ERF);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto scale = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, sig);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(input, 1.0);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto input = process_view_input_new(context, 0);
    const auto type = ggml_openvino_is_npu() ? input.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = input;
    if (type != input.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(input, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    if (type != input.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, input.get_element_type());
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto zero = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(input, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, input.get_element_type());
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(input, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto input = process_view_input_new(context, 0);
    const auto type = ggml_openvino_is_npu() ? input.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = input;
    if (type != input.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(input, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    if (type != input.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, input.get_element_type());
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        return translate_1to1_match_1_input<ov::op::v4::SoftPlus>(context);
    }

    auto input = process_view_input_new(context, 0);
    const auto element_type = input.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(input);
    auto abs = std::make_shared<ov::op::v0::Abs>(input);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
