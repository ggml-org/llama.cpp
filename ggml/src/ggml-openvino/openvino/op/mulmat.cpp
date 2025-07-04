#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported MULMAT case");

    ov::Output<Node> res;

    if (op_case == 1) {
        auto src0 = context.get_input(0);
        auto src1 = std::make_shared<ov::op::v0::Convert>(context.get_input(1), context.get_input_type(0));
        auto result_lp = std::make_shared<ov::op::v0::MatMul>(src1, src0, false, true);
        res = std::make_shared<ov::op::v0::Convert>(result_lp, context.get_output_type(0));
    } else {
        ov::Output<ov::Node> B = context.get_input(0);
        ov::Output<ov::Node> A = std::make_shared<ov::op::v0::Convert>(context.get_input(1), context.get_input_type(0));

        int64_t num_heads = context.get_num_heads();
        int64_t num_heads_kv = context.get_num_heads_kv();
        int64_t kv_num_heads_factor = num_heads / num_heads_kv;
        if (kv_num_heads_factor > 1) {
            auto num_heads_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{num_heads});
            auto num_heads_kv_node =
                ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{num_heads_kv});
            auto factor_node =
                ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{kv_num_heads_factor});
            auto B_shape_last_two = get_dimensions(B.get_node_shared_ptr(), {1, 2});

            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {1});
            auto B_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(B, unsqueeze_axes);

            auto broadcast_shape = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{num_heads_kv_node, factor_node, B_shape_last_two}, 0);
            auto B_broadcasted = std::make_shared<ov::op::v3::Broadcast>(B_unsqueezed, broadcast_shape);

            auto new_B_shape =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{num_heads_node, B_shape_last_two}, 0);
            B = std::make_shared<ov::op::v1::Reshape>(B_broadcasted, new_B_shape, false);
        }

        auto result_lp = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);
        res = std::make_shared<ov::op::v0::Convert>(result_lp, context.get_output_type(0));
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
