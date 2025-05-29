#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_nd_update.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "../node_context.hpp"
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
    auto past_token_len_scalar = context.get_input("past_token_len");

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

    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto one_scalar = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    if (op_case == 1) {
        // Write K to cache_k
        int64_t head_size = src0_shape[2];
        int64_t num_heads = src0_shape[1];

        auto reshaped_src1_shape =
            ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, num_heads, head_size});
        auto reshaped_src1 = std::make_shared<ov::op::v1::Reshape>(src1, reshaped_src1_shape, false);

        auto token_len = get_dimensions(src0.get_node_shared_ptr(), {0});
        auto token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(token_len, zero);

        std::shared_ptr<ov::Node> indices;
        if (context.is_static()) {
            indices = past_token_len_scalar.get_node_shared_ptr();
            indices = std::make_shared<ov::op::v0::Unsqueeze>(
                indices,
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 1}));
        } else {
            auto total_token_len_scalar = std::make_shared<ov::op::v1::Add>(past_token_len_scalar, token_len_scalar);
            indices = std::make_shared<ov::op::v4::Range>(past_token_len_scalar,
                                                          total_token_len_scalar,
                                                          one_scalar,
                                                          ov::element::i64);
            indices = std::make_shared<ov::op::v0::Unsqueeze>(indices, one);
        }

        res = std::make_shared<ov::op::v3::ScatterNDUpdate>(reshaped_src1, indices, src0);
    } else {
        // Write V to cache_v
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto zero_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

        int64_t total_head_size = src0_shape[1];
        auto total_head_size_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {total_head_size});
        auto total_head_size_scalar = std::make_shared<ov::op::v0::Squeeze>(total_head_size_node, zero);

        auto token_len = get_dimensions(src0.get_node_shared_ptr(), {2});
        auto token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(token_len, zero);

        // 1D tensor of shape [total_head_size], values starting from 0
        auto range_row =
            std::make_shared<ov::op::v4::Range>(zero_scalar, total_head_size_scalar, one_scalar, ov::element::i64);
        auto range_row_reshaped =
            std::make_shared<ov::op::v0::Unsqueeze>(range_row,
                                                    ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 2}));
        auto row_indices = std::make_shared<ov::op::v3::Broadcast>(
            range_row_reshaped,
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{total_head_size_node, token_len, one}, 0));

        // 1D tensor of shape [token_len], values starting from past_token_len
        std::shared_ptr<ov::Node> range_col;
        if (context.is_static()) {
            range_col = past_token_len_scalar.get_node_shared_ptr();
            range_col = std::make_shared<ov::op::v0::Unsqueeze>(
                range_col,
                ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{0}));
        } else {
            auto total_token_len_scalar = std::make_shared<ov::op::v1::Add>(past_token_len_scalar, token_len_scalar);
            range_col = std::make_shared<ov::op::v4::Range>(past_token_len_scalar,
                                                            total_token_len_scalar,
                                                            one_scalar,
                                                            ov::element::i64);
        }
        auto range_col_reshaped =
            std::make_shared<ov::op::v0::Unsqueeze>(range_col,
                                                    ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 2}));
        auto col_indices = std::make_shared<ov::op::v3::Broadcast>(
            range_col_reshaped,
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{total_head_size_node, token_len, one}, 0));

        // Stack row_indices and col_indices along last axis: [total_head_size, token_len, 2]
        auto indices = std::make_shared<ov::op::v0::Concat>(OutputVector{row_indices, col_indices}, 2);
        auto indices_final = std::make_shared<ov::op::v1::Reshape>(
            indices,
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, 2}),
            false);

        auto flattend_src0 =
            std::make_shared<ov::op::v1::Reshape>(src0,
                                                  ov::op::v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                  false);
        auto reshaped_src1 = std::make_shared<ov::op::v1::Reshape>(
            src1,
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{total_head_size, -1}),
            false);

        auto updated = std::make_shared<ov::op::v3::ScatterNDUpdate>(reshaped_src1, indices_final, flattend_src0);
        res = std::make_shared<ov::op::v0::Unsqueeze>(updated, zero);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
