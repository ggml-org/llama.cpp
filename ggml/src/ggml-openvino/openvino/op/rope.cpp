#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto cos_theta_node = context.get_input("rope_cos");
    auto sin_theta_node = context.get_input("rope_sin");

    int32_t* op_params = context.get_output_op_params(0);
    const int mode = op_params[2];
    constexpr int GGML_ROPE_TYPE_NEOX = 2;
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    if (!is_neox) {
        auto input_shape = context.get_input_shape(0);

        auto begin_even = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {0, 0, 0});
        auto begin_odd = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {0, 0, 1});
        auto end = std::make_shared<ov::op::v0::ShapeOf>(data_node);
        auto stride = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {1, 1, 2});
        auto even_slice = std::make_shared<ov::op::v8::Slice>(data_node, begin_even, end, stride);
        auto odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, begin_odd, end, stride);

        auto first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        auto second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, 2);
        res = std::make_shared<ov::op::v1::Reshape>(stack, std::make_shared<ov::op::v0::ShapeOf>(data_node), false);
    } else {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, 2);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
