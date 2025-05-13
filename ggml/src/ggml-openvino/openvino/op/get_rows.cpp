#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_get_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto data_node = context.get_input(0);
    auto indices_node = context.get_input(1);

    auto indices_shape = get_dimensions(indices_node.get_node_shared_ptr(), {2});
    Output<Node> indice_reshaped = std::make_shared<ov::op::v1::Reshape>(indices_node, indices_shape, false);

    auto axis_node = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});

    Output<Node> res = std::make_shared<ov::op::v8::Gather>(data_node, indice_reshaped, axis_node);
    if (res.get_element_type() != context.get_output_type(0)) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type(0));
    }

    return {res};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
