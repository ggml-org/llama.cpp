#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <openvino/op/bitwise_and.hpp>
#include <openvino/op/bitwise_right_shift.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t> & values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> slice_axis(const ov::Output<ov::Node> & input, int64_t axis, int64_t begin, int64_t end) {
    return std::make_shared<ov::op::v8::Slice>(input, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(const NodeContext & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    const int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    const int64_t rows = static_cast<int64_t>(packed_shape[2]);
    const int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    const int64_t qk = 32;
    const int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, false);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);
    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    static const std::vector<float> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<float> e8m0_lut(256);
    for (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        memcpy(&e8m0_lut[i], &bits, sizeof(float));
    }
    e8m0_lut[0] = std::numeric_limits<float>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<float>::quiet_NaN();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto low = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    auto high_shift = std::make_shared<ov::op::v15::BitwiseRightShift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{low, high_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, false);

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims, ov::op::BroadcastType::BIDIRECTIONAL);

    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    const auto output_type = context.get_output_type();
    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    return result;
}

}  // namespace

OutputVector translate_mul_mat_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = process_view_input_new(context, 0);
    auto activations = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    if (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        return rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
                                          context.get_name());
    }

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
