#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sqrt.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_sqr(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, input);

    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(input);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
