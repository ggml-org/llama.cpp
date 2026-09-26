#include "fuse_to_conv.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/extractimagepatches.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/pad.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// This pass fuses an IM2COL + MatMul convolution into OpenVINO's Convolution op for performance gains.
// Reference the im2col.cpp translator for reference on the pattern being matched.

FuseToConv::FuseToConv() {
    const auto m_wei = opp::any_input();
    const auto m_act = opp::any_input();
    const auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_wei, m_act});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        const auto & pm = m.get_pattern_value_map();

        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(m_matmul).get_node_shared_ptr());
        if (!matmul_node || matmul_node->get_transpose_a() || !matmul_node->get_transpose_b()) {
            return false;
        }

        auto trace = matmul_node->input_value(1);

        // Optional Convert
        if (auto n = ov::as_type_ptr<ov::op::v0::Convert>(trace.get_node_shared_ptr())) {
            trace = n->input_value(0);
        }

        for (int i = 0; i < 2; ++i) {
            auto n = ov::as_type_ptr<ov::op::v1::Reshape>(trace.get_node_shared_ptr());
            if (!n) {
                return false;
            }
            trace = n->input_value(0);
        }

        if (auto n = ov::as_type_ptr<ov::op::v1::Transpose>(trace.get_node_shared_ptr())) {
            trace = n->input_value(0);
        } else {
            return false;
        }

        if (auto n = ov::as_type_ptr<ov::op::v1::Reshape>(trace.get_node_shared_ptr())) {
            trace = n->input_value(0);
        } else {
            return false;
        }

        if (auto n = ov::as_type_ptr<ov::op::v1::Transpose>(trace.get_node_shared_ptr())) {
            trace = n->input_value(0);
        } else {
            return false;
        }

        auto eip = ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(trace.get_node_shared_ptr());
        if (!eip) {
            return false;
        }
        const auto eip_strides = eip->get_strides();  // {stride_h, stride_w}
        const auto eip_rates = eip->get_rates();      // {dil_h, dil_w}

        auto pad = ov::as_type_ptr<ov::op::v1::Pad>(eip->input_value(0).get_node_shared_ptr());
        if (!pad) {
            return false;
        }
        auto pads_begin_const =
            ov::as_type_ptr<ov::op::v0::Constant>(pad->input_value(1).get_node_shared_ptr());

        const auto pads_begin_vals = pads_begin_const->cast_vector<int64_t>();  // {0, 0, pad_h, pad_w}
        const std::ptrdiff_t pad_h = static_cast<std::ptrdiff_t>(pads_begin_vals[2]);
        const std::ptrdiff_t pad_w = static_cast<std::ptrdiff_t>(pads_begin_vals[3]);

        auto image_input = pad->input_value(0);  // [N, IC, 1, IW] NCHW

        auto w_trace = matmul_node->input_value(0);
        if (auto n = ov::as_type_ptr<ov::op::v0::Convert>(w_trace.get_node_shared_ptr())) {
            w_trace = n->input_value(0);
        }
        for (int i = 0; i < 2; ++i) {
            auto n = ov::as_type_ptr<ov::op::v1::Reshape>(w_trace.get_node_shared_ptr());
            if (!n) {
                break;
            }
            w_trace = n->input_value(0);
        }

        auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_node_shared_ptr());
        if (!weight_const) {
            return false;
        }

        // Reshape weight to [OC, IC, 1, KW] (OIHW).
        const auto w_shape = weight_const->get_shape();
        ov::Shape conv_w_shape;
        if (w_shape.size() == 3) {
            conv_w_shape = {w_shape[0], w_shape[1], 1, w_shape[2]};
        } else if (w_shape.size() == 4) {
            conv_w_shape = {w_shape[1], w_shape[2], 1, w_shape[3]};
        } else {
            return false;
        }

        auto weight_reshaped = register_new_node<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());

        ov::Output<Node> weight_input = weight_reshaped;
        if (weight_reshaped->get_element_type() != image_input.get_element_type()) {
            weight_input = register_new_node<ov::op::v0::Convert>(weight_reshaped, image_input.get_element_type());
        }

        auto conv = register_new_node<ov::op::v1::Convolution>(
            image_input, weight_input,
            ov::Strides{static_cast<size_t>(eip_strides[0]), static_cast<size_t>(eip_strides[1])},
            ov::CoordinateDiff{pad_h, pad_w}, ov::CoordinateDiff{pad_h, pad_w},
            ov::Strides{static_cast<size_t>(eip_rates[0]), static_cast<size_t>(eip_rates[1])},
            ov::op::PadType::EXPLICIT);

        constexpr auto target_type = ov::element::f32;
        ov::Output<Node> conv_out = conv;
        if (conv_out.get_element_type() != target_type) {
            conv_out = register_new_node<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_node;
        ov::Output<Node> bias_input;
        for (const auto & consumer_in : matmul_node->output(0).get_target_inputs()) {
            auto cast = ov::as_type_ptr<ov::op::v0::Convert>(consumer_in.get_node()->shared_from_this());
            if (!cast) {
                continue;
            }
            for (const auto & add_in : cast->output(0).get_target_inputs()) {
                auto add = ov::as_type_ptr<ov::op::v1::Add>(add_in.get_node()->shared_from_this());
                if (!add) {
                    continue;
                }
                for (size_t i = 0; i < 2; ++i) {
                    if (ov::as_type_ptr<ov::op::v0::Constant>(add->input_value(i).get_node_shared_ptr())) {
                        bias_input = add->input_value(i);
                        add_node = add;
                        break;
                    }
                }
                if (add_node) {
                    break;
                }
            }
            if (add_node) {
                break;
            }
        }

        ov::Output<Node> final_out;
        std::shared_ptr<Node> target_node;

        if (add_node) {
            // Reshape bias [OC, 1] → [1, OC, 1, 1] for NCHW broadcasting.
            ov::Output<Node> bias = bias_input;
            if (bias.get_element_type() != target_type) {
                bias = register_new_node<ov::op::v0::Convert>(bias, target_type);
            }
            const auto oc = static_cast<int64_t>(conv_w_shape[0]);
            auto bias_shape = register_new_node<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                      std::vector<int64_t>{1, oc, 1, 1});
            bias = register_new_node<ov::op::v1::Reshape>(bias, bias_shape, false);
            final_out = register_new_node<ov::op::v1::Add>(conv_out, bias);
            target_node = add_node;
        } else {
            final_out = conv_out;
            target_node = matmul_node;
        }

        // Reshape final output back to the target node's original shape if needed.
        auto orig_shape = target_node->get_output_partial_shape(0);
        if (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_node<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_node<ov::op::v1::Reshape>(final_out, shape_const, false);
        }

        final_out.get_node_shared_ptr()->set_friendly_name(target_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_node_shared_ptr());
        ov::replace_node(target_node, final_out.get_node_shared_ptr());

        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
