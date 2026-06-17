#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/tile.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    auto input = process_view_input_new(context, 0);

    const auto input_shape = context.get_input_shape(0);
    const auto output_shape = context.get_output_shape();

    if (input_shape.rank().is_static() && output_shape.rank().is_static() &&
        input_shape.rank() == output_shape.rank()) {
        const auto rank = static_cast<size_t>(input_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        bool all_static = true;

        for (size_t axis = 0; axis < rank; ++axis) {
            if (!input_shape[axis].is_static() || !output_shape[axis].is_static()) {
                all_static = false;
                break;
            }

            const int64_t input_dim = input_shape[axis].get_length();
            const int64_t output_dim = output_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && output_dim > 0 && output_dim % input_dim == 0,
                                          "REPEAT input shape ", input_shape, " cannot tile to match ", output_shape);

            repeats[axis] = output_dim / input_dim;
        }

        if (all_static) {
            auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
            ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
    }

    // Dynamic fallback: tile by the ratio of output to input shape.
    auto input_shape_node = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    std::shared_ptr<ov::Node> target_shape_node;
    if (output_shape.rank().is_static() && output_shape.is_static()) {
        target_shape_node =
            ov::op::v0::Constant::create(ov::element::i64, {output_shape.to_shape().size()}, output_shape.to_shape());
    } else {
        target_shape_node = std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1), ov::element::i64);
    }
    auto repeats_node = std::make_shared<ov::op::v1::Divide>(target_shape_node, input_shape_node);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
