#include "../node_context.hpp"
#include "../utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {
OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // TODO: make this more general
    auto res = std::make_shared<ov::op::v1::Transpose>(context.get_input(0),
                                                       ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));

    return {res};
};
}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
