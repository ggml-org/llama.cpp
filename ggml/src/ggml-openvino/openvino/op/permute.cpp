#include <openvino/op/constant.hpp>
#include <openvino/op/transpose.hpp>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported CONT case");
    ov::Output<Node> res;

    if (op_case == 1) {
        auto perm = argsort_descend(context.get_output_stride(0));
        auto res = std::make_shared<ov::op::v1::Transpose>(context.get_input(0),
                                                           ov::op::v0::Constant::create(ov::element::i64, {3}, perm));
        return rename_outputs_with_suffix({res}, context.get_name());
    } else {
        auto res = context.get_input(0);
        return {res};
    }
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
