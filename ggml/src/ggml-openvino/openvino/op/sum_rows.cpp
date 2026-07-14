#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/reduce_sum.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_sum_rows(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
