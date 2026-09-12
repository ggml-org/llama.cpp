#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/mvn.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = process_view_input_new(context, 0);
    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(input_node, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
