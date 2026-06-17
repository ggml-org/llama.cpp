#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_add_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto input = process_view_input_new(context, 0);
    auto bias = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    auto bias_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(bias, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    bias = std::make_shared<ov::op::v1::Reshape>(bias, get_dimensions(bias_shape_4d, {2, 3}), false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, get_dimensions(ids_shape_4d, {2, 3}), false);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64), false);

    if (selected_bias.get_element_type() != input.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, input.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(input, selected_bias);
    const auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
