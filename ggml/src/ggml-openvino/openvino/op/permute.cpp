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

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported PERMUTE case");
    ov::Output<Node> res;
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

    if (op_case == 1) {
        if (context.is_static()) {
            res = std::make_shared<ov::op::v1::Transpose>(context.get_input(0),
                                                          ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));
        } else {
            auto src = context.get_input(0);
            if (src.get_partial_shape().rank() == 3) {
                src = std::make_shared<ov::op::v0::Unsqueeze>(src, zero);
            }
            res = std::make_shared<ov::op::v1::Transpose>(src,
                                                          ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        }
    } else {
        auto src = context.get_input(0);
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        if (context.is_static()) {
            auto src_shape_ = context.get_input_shape(0).to_shape();
            std::vector<int64_t> src_shape(src_shape_.begin(), src_shape_.end());
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src,
                ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, src_shape[1], src_shape[2]}),
                false);
            res = std::make_shared<ov::op::v1::Transpose>(
                src_reshaped, ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));
        } else {
            if (src.get_partial_shape().rank() == 3) {
                src = std::make_shared<ov::op::v0::Unsqueeze>(src, zero);
            }
            res = std::make_shared<ov::op::v1::Transpose>(src,
                                                          ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        }
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
