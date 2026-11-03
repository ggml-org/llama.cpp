#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/shape_of.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float c;
    memcpy(&c, context.get_output_op_params(), sizeof(float));

    // the shape of the input, which keeps its dynamic token dim
    auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(process_view_input_new(context, 0), ov::element::i64);
    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
