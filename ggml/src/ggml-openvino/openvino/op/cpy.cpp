#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_nd_update.hpp>
#include <openvino/op/squeeze.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported CPY case");

    auto src0 = context.get_input(0);
    auto src1 = context.get_input(1);
    auto token_len = context.get_input("token_len");
    auto past_token_len = context.get_input("past_token_len");

    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(token_len, zero);
    auto past_token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(past_token_len, zero);

    src0 = std::make_shared<ov::op::v0::Convert>(src0, context.get_input_type(1));
    ov::Output<Node> res;

    if (context.is_static() && context.is_first_token()) {
        res = src0;
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto src0_shape = context.get_input_shape(0).to_shape();
    auto output_shape = context.get_output_shape(0).to_shape();

    std::vector<size_t> input0_strides = context.get_input_stride(0);
    std::vector<size_t> output_strides = context.get_output_stride(0);

    if (op_case == 1) {
        // Write K to cache_k
        auto indices = context.get_input("update_indices_k");
        auto updated = std::make_shared<ov::op::v3::ScatterNDUpdate>(src1, indices, src0);
        res = std::make_shared<ov::op::v1::Reshape>(updated, std::make_shared<ov::op::v0::ShapeOf>(src1), false);
    } else {
        // Write V to cache_v
        auto flattend_src0 =
            std::make_shared<ov::op::v1::Reshape>(src0,
                                                  ov::op::v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                  false);
        int64_t total_head_size = src0_shape[1];
        auto reshaped_src1 = std::make_shared<ov::op::v1::Reshape>(
            src1,
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{total_head_size, -1}),
            false);
        auto indices = context.get_input("update_indices_v");
        auto updated = std::make_shared<ov::op::v3::ScatterNDUpdate>(reshaped_src1, indices, flattend_src0);
        res = std::make_shared<ov::op::v1::Reshape>(updated, std::make_shared<ov::op::v0::ShapeOf>(src1), false);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
