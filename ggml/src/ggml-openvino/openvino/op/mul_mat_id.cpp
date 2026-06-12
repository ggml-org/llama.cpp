#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
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
    // Rebuild the logical ranks explicitly from the 4D inputs instead of relying
    // on fixed squeeze axes: real graphs can arrive through VIEW/RESHAPE chains
    // where singleton axes are still represented differently at this point.
    auto expert_weights_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(expert_weights, ov::element::i64);
    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    auto expert_weights_shape_3d = get_dimensions(expert_weights_shape_4d, {1, 2, 3});
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, false);
    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

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

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims, ov::op::BroadcastType::BIDIRECTIONAL);

    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, unsqueeze_axes);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto output_shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(output_shape.rank().is_static() && output_shape.rank().get_length() == 4,
                                  "Unexpected MUL_MAT_ID output rank");
    FRONT_END_OP_CONVERSION_CHECK(output_shape[3].is_static(), "Expected static row dimension for MUL_MAT_ID output");
    const auto row_dim_value = output_shape[3].get_length();
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {row_dim_value});

    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            batch_dim,
            get_dimensions(ids_shape, {0, 1}),
            row_dim,
        },
        0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
