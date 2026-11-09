#include "move_scale_before_matmul.h"

#include <cmath>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {
// True when output is computed from constants only: a weight, possibly behind its dequantization chain.
// The depth bound stops the walk early on activation inputs.
bool is_constant_path(const ov::Output<ov::Node> & output, int depth = 0) {
    const auto node = output.get_node_shared_ptr();
    if (ov::is_type<ov::op::v0::Constant>(node)) {
        return true;
    }
    if (depth >= 8 || node->get_input_size() == 0) {
        return false;
    }
    for (const auto & input : node->input_values()) {
        if (!is_constant_path(input, depth + 1)) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<ov::op::v0::Constant> as_scalar_constant(const ov::Output<ov::Node> & output) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 1 ? constant : nullptr;
}
}  // namespace

MoveScaleBeforeMatMul::MoveScaleBeforeMatMul() {
    auto m_mul = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});

    const auto callback = [](opp::Matcher & m) {
        const auto mul = m.get_match_root();
        for (size_t i = 0; i < 2; ++i) {
            const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(mul->get_input_node_shared_ptr(i));
            const auto scale = as_scalar_constant(mul->input_value(1 - i));
            if (!matmul || !scale || matmul->get_output_target_inputs(0).size() != 1) {
                continue;
            }
            // Only a scale that shrinks the output helps, and the Multiply must not broadcast the MatMul output.
            const float s = scale->cast_vector<float>()[0];
            if (!(std::fabs(s) < 1.0f) || s == 0.0f ||
                !mul->get_output_partial_shape(0).same_scheme(matmul->get_output_partial_shape(0))) {
                return false;
            }
            const bool weight_1 = is_constant_path(matmul->input_value(1));
            const bool weight_0 = is_constant_path(matmul->input_value(0));
            if (weight_0 == weight_1) {
                return false;
            }
            const size_t act_idx = weight_1 ? 0 : 1;

            const auto act = matmul->input_value(act_idx);
            auto scaled_act = std::make_shared<ov::op::v1::Multiply>(
                act, ov::op::v0::Constant::create(act.get_element_type(), ov::Shape{}, {s}));
            auto inputs = matmul->input_values();
            inputs[act_idx] = scaled_act;
            auto new_matmul = matmul->clone_with_new_inputs(inputs);
            new_matmul->set_friendly_name(mul->get_friendly_name());
            ov::copy_runtime_info({matmul, mul}, {scaled_act, new_matmul});
            ov::replace_node(mul, new_matmul);
            return true;
        }
        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_mul, "ov::frontend::ggml::pass::MoveScaleBeforeMatMul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
