
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    if (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto input = process_view_input_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        input, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
