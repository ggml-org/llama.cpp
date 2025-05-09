#include <cstdint>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mul(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto res = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), context.get_input(1));
    return {res};
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
