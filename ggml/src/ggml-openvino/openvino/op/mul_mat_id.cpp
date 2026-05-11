#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mul_mat_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = process_view_input_new(context, 0);
    auto activations = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    // OpenVINO sees GGML tensors in reversed dimension order:
    //   weights: [1, n_expert, m, k]
    //   activations: [1, n_tokens, n_used_or_1, k]
    //   ids: [1, 1, n_tokens, n_used]
    auto squeeze_weights_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto squeeze_acts_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto squeeze_ids_axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1});

    expert_weights = std::make_shared<ov::op::v0::Squeeze>(expert_weights, squeeze_weights_axes);
    activations = std::make_shared<ov::op::v0::Squeeze>(activations, squeeze_acts_axes);
    ids = std::make_shared<ov::op::v0::Squeeze>(ids, squeeze_ids_axes);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);

    const auto output_type = context.get_output_type();
    if (selected_weights.get_element_type() != ov::element::f32) {
        selected_weights = std::make_shared<ov::op::v0::Convert>(selected_weights, ov::element::f32);
    }
    if (activations.get_element_type() != ov::element::f32) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, ov::element::f32);
    }

    auto selected_weights_shape = std::make_shared<ov::op::v3::ShapeOf>(selected_weights, ov::element::i64);
    auto acts_target_dims = get_dimensions(selected_weights_shape, {0, 1, 3});
    ov::Output<ov::Node> acts_broadcasted = std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims,
                                                                                     ov::op::BroadcastType::BIDIRECTIONAL);

    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, unsqueeze_axes);

    ov::Output<ov::Node> result = std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);
    result = std::make_shared<ov::op::v0::Squeeze>(result, unsqueeze_axes);

    auto restore_batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    result = std::make_shared<ov::op::v0::Unsqueeze>(result, restore_batch_axis);

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov