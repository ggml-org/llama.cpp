#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input(0).get_partial_shape().is_static() &&
        context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    const int op_case = context.get_op_case();
    const auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_node;
    switch (op_case) {
    case 0:
        break;
    case 1:
    case 9:
        shape[1] = -1;
        if (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        break;
    case 2:
    case 3:
        shape[2] = -1;
        if (op_case == 3) {
            shape[3] = 1;
        }
        break;
    case 4:
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};
    case 5:
    case 7:
        shape = {1, 1, -1, shape[3]};
        if (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        break;
    case 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        if (context.has_input("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_input("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_node =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } else {
            shape = {1, 1, -1, shape[3]};
        }
        break;
    case 8:
        shape[0] = -1;
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported RESHAPE case: ", op_case);
    }
    if (!new_shape_node) {
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
