#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext & context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    const int mode = (op_case & 0xFFFF0000) >> 16;
    op_case = (op_case & 0x0000FFFF);

    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[2] * output_shape[3];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        if (context.is_stateful()) {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {3}, std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        } else {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        }
    }

    if (mode == TYPE_NORMAL) {
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        Output<Node> even_slice;
        Output<Node> odd_slice;
        int32_t unsqueeze_dim = context.is_stateful() ? 3 : 4;
        even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, neg_one);
        odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, neg_one);

        Output<Node> first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        Output<Node> second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        first_half = std::make_shared<ov::op::v0::Unsqueeze>(first_half,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        second_half = std::make_shared<ov::op::v0::Unsqueeze>(second_half,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, unsqueeze_dim);

        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
    } else if (mode == TYPE_NEOX) {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, -1);
    } else if (mode == TYPE_IMROPE) {
        int64_t n_dims = data_node->get_shape()[3];
        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1,-1,1,(n_dims >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
