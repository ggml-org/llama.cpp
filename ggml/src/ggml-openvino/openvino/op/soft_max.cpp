
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_soft_max(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input_node = context.get_input(0);
    ov::Output<Node> res;

    float scale = 1.0f;
    float max_bias = 0.0f;
    auto op_params = context.get_output_op_params(0);
    memcpy(&scale, (float*)op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*)op_params + 1, sizeof(float));

    const uint32_t n_head = context.get_input_shape(0)[0].get_length();
    const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

    // const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    // const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const float slope = (max_bias > 0.0f) ? 1.0f : 1.0f;
    // const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)
    // : 1.0f;

    if (scale != 1.0f) {
        auto scale_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
        input_node = std::make_shared<ov::op::v1::Multiply>(input_node, scale_node);
    }

    if (context.get_input_size() == 2) {
        // Calculate mask then softmax
        auto mask_node = context.get_input(1);
        ov::element::Type mask_type = context.get_input_type(1);
        if (mask_type == ov::element::f16) {
            // Convert f16 to f32
            mask_node = std::make_shared<ov::op::v0::Convert>(mask_node, ov::element::f32);
        }

        // Stride slice mask node
        Output<Node> slice_start = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {0, 0, 0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto token_len = get_dimensions(input_node.get_node_shared_ptr(), {1});
        auto total_token_len = get_dimensions(mask_node.get_node_shared_ptr(), {2});
        auto slice_end = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{one, token_len, total_token_len}, 0);
        Output<Node> slice_stride = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {1, 1, 1});
        auto mask_node_sliced = std::make_shared<ov::op::v8::Slice>(mask_node, slice_start, slice_end, slice_stride);

        // slope * mask
        auto slope_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{slope});
        auto slope_mask_node = std::make_shared<ov::op::v1::Multiply>(mask_node_sliced, slope_node);

        // input + slope * mask
        auto input_slope_mask_node = std::make_shared<ov::op::v1::Add>(input_node, slope_mask_node);

        // Calculate softmax
        res = std::make_shared<ov::op::v8::Softmax>(input_slope_mask_node, 2);
    } else {
        // Directly softmax
        res = std::make_shared<ov::op::v8::Softmax>(input_node, 0);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
