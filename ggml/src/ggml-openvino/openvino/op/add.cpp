#include <openvino/op/add.hpp>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_add(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto add = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    return {add};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
