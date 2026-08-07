#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

static bool is_static_one(const ov::Dimension & dim) {
    return dim.is_static() && dim.get_length() == 1;
}

static bool same_static_dim(const ov::Dimension & lhs, const ov::Dimension & rhs) {
    return lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

static bool is_attention_sinks_input_shape(const ov::PartialShape & candidate, const ov::PartialShape & logits_shape) {
    if (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        return false;
    }

    return is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(const NodeContext & context) {
    num_inputs_check(context, 1, 3);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&max_bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    ov::Output<ov::Node> logits = context.get_input(0);
    const bool second_input_is_sinks =
        context.get_input_size() == 2 && is_attention_sinks_input_shape(context.get_input_shape(1), context.get_output_shape());
    const bool has_mask = context.get_input_size() > 1 && !second_input_is_sinks;
    const bool has_sinks = second_input_is_sinks || context.get_input_size() > 2;
    const size_t sinks_input_idx = second_input_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    if (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    if (has_mask) {
        ov::Output<ov::Node> mask = context.get_input(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        if (context.get_input_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        if (context.has_input(mask_name)) {
            mask = context.get_input(mask_name);
        }

        if (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        if (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            const uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(n_head))));
            const float m0 = std::pow(2.0f, -(max_bias) / static_cast<float>(n_head_log2));
            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<float>(n_head_log2));

            std::vector<float> slopes(n_head);
            for (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1)) :
                                              std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            if (slope_node.get_element_type() != mask.get_element_type()) {
                slope_node = std::make_shared<ov::op::v0::Convert>(slope_node, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_node, slope_shape, false);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_input = logits;
    if (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_input(sinks_input_idx);
        if (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, false);

        auto logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
        auto sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto sink_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
        auto sink_column = std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        softmax_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_input, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        return rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_input, -1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
