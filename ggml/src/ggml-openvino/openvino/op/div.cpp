#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/tile.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

ov::Output<ov::Node> repeat_input_to_match(const NodeContext & context,
                                           const ov::Output<ov::Node> & input,
                                           const ov::Output<ov::Node> & target,
                                           size_t input_index) {
    const auto input_shape = context.get_input_shape(input_index);
    const auto target_shape = context.get_input_shape(0);

    if (input_shape == target_shape) {
        return input;
    }

    if (input_shape.rank().is_static() && target_shape.rank().is_static()) {
        const auto rank = static_cast<size_t>(input_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        bool needs_repeat = false;

        for (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(input_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            const int64_t input_dim = input_shape[axis].get_length();
            const int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && target_dim > 0 && target_dim % input_dim == 0,
                                          "DIV input shape ", input_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / input_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        if (!needs_repeat) {
            return input;
        }

        auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        return std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    }

    auto input_shape_node = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    auto target_shape_node = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_node = std::make_shared<ov::op::v1::Divide>(target_shape_node, input_shape_node);
    return std::make_shared<ov::op::v0::Tile>(input, repeats_node);
}

}  // namespace

OutputVector translate_div(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);
    input_1 = repeat_input_to_match(context, input_1, input_0, 1);

    const auto output_type = context.get_output_type();
    const bool use_f32_compute = input_0.get_element_type() != ov::element::f32 ||
                                 input_1.get_element_type() != ov::element::f32 ||
                                 output_type != ov::element::f32;

    if (use_f32_compute) {
        input_0 = std::make_shared<ov::op::v0::Convert>(input_0, ov::element::f32);
        input_1 = std::make_shared<ov::op::v0::Convert>(input_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(input_0, input_1);
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov