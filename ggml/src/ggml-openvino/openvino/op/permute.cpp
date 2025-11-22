#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported PERMUTE case");
    ov::Output<Node> res;
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

    auto src = context.get_input(0);
    res = std::make_shared<ov::op::v1::Transpose>(src, ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
