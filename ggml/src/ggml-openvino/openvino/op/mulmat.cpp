#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

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
#include <openvino/op/util/op_types.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_input_new(context, 0);
    ov::Output<ov::Node> A = process_view_input_new(context, 1);
    // the stateful V cache with `-fa off` is not transposed (see permute)
    const bool transpose_b = B.get_node()->get_rt_info().count(GGML_OV_V_TOKEN_MAJOR) == 0;

    if (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_input_type(1));
    }

    auto B_shape = context.get_input_shape(0).to_shape();
    auto A_shape = context.get_input_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    if (factor > 1 && batch_small > 1 && A_batch_larger && A_shape[0] == B_shape[0]) {
        // B is shared by groups of factor consecutive A matrices (grouped-query attention with `-fa off`: K and V
        // have fewer heads than Q and KQ). Stack each group along the rows of A instead of repeating B, which would
        // copy the whole K or V cache factor times per layer and token:
        //   A [n, batch_large, rows, k] -> [n, batch_small, factor * rows, k] (a pure reshape: the group is adjacent)
        //   A x B^T [n, batch_small, factor * rows, cols] -> [n, batch_large, rows, cols]
        // The zeros copy the input dimensions (special_zero), which may be dynamic.
        ov::Output<Node> A_grouped = std::make_shared<ov::op::v1::Reshape>(
            A, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, batch_small, -1, 0}), true);
        // A is often a Reshape of the grouped scores (see softmax): take them directly. With one Reshape less, the
        // OpenVINO SDPAFusion can match the attention
        const auto * A_reshape = ov::as_type<ov::op::v1::Reshape>(A.get_node());
        const auto * A_target =
            A_reshape != nullptr ? ov::as_type<ov::op::v0::Constant>(A_reshape->get_input_node_ptr(1)) : nullptr;
        if (context.is_stateful() && A_target != nullptr && A_reshape->get_special_zero() &&
            A_target->cast_vector<int64_t>() == std::vector<int64_t>{0, batch_large, -1, 0} &&
            A_reshape->get_input_partial_shape(0)[1] == batch_small) {
            A_grouped = A_reshape->input_value(0);
        }
        ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A_grouped, B, false, transpose_b);
        res = std::make_shared<ov::op::v1::Reshape>(
            res, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, batch_large, -1, 0}),
            true);
        // lets SOFT_MAX work on the grouped scores (see softmax)
        res.get_node()->get_rt_info()[GGML_OV_GROUPED_ROWS] = factor;
        const auto output_type = context.get_output_type();
        if (res.get_element_type() != output_type) {
            res = std::make_shared<ov::op::v0::Convert>(res, output_type);
        }
        return rename_outputs_with_suffix({res}, context.get_name());
    }
    if (factor > 1 && batch_small > 1) {
        auto batch_large_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        if (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    if (A_batch_larger) {
        B = Z;
    } else {
        A = Z;
    }

    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    const auto output_type = context.get_output_type();
    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
