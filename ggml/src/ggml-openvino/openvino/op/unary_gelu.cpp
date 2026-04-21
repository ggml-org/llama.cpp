#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/core/node_output.hpp>
#include <openvino/op/gelu.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_gelu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto res = std::make_shared<ov::op::v7::Gelu>(input);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
