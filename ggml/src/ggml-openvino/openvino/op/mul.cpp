#include <openvino/op/multiply.hpp>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mul(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto res = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), context.get_input(1));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
