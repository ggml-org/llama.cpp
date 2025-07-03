#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_soft_max(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input_node = context.get_input(0).get_node_shared_ptr();
    ov::Output<Node> res;

    float scale = 1.0f;
    float max_bias = 0.0f;
    auto* op_params = context.get_output_op_params(0);
    memcpy(&scale, (float*) op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*) op_params + 1, sizeof(float));
    const uint32_t h = context.get_head_size();

    const uint32_t n_head = context.get_input_shape(0)[0].get_length();
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const float slope =
        (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

    std::shared_ptr<ov::Node> scaled_input;
    if (scale != 1.0f) {
        auto scale_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
        scaled_input = std::make_shared<ov::op::v1::Multiply>(input_node, scale_node);
    }

    auto mask_node = context.get_input(1);

    // Use Q-cur to retrieve the token length, so that the translation of SOFT_MAX
    // does not depend on the result of the QK MatMul, so that QK matmul + softmax + qkv matmul
    // can be fused into SDPA.
    if (input_node->get_type_info() != ov::op::v0::Convert::get_type_info_static()) {
        throw std::runtime_error("Input of SOFT_MAX should be MatMul of qk followed by a Convert");
    }
    auto qk = input_node->get_input_node_shared_ptr(0);
    if (qk->get_type_info() != ov::op::v0::MatMul::get_type_info_static()) {
        throw std::runtime_error("Input of SOFT_MAX should be MatMul of qk followed by a Convert");
    }
    auto token_len = get_dimensions(qk->get_input_node_shared_ptr(0), {1});

    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto mask_node_sliced = std::make_shared<ov::op::v8::Slice>(mask_node, zero, token_len, one, one);

    Output<Node> slope_mask;
    if (slope != 1.0f) {
        auto slope_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{slope});
        slope_mask = std::make_shared<ov::op::v1::Multiply>(mask_node_sliced, slope_node);
        throw std::runtime_error("Slope != 1.0f in softmax has not been tested, verify it before use.");
    }
    slope_mask = mask_node_sliced;

    auto input_slope_mask_node = std::make_shared<ov::op::v1::Add>(scaled_input, slope_mask);

    res = std::make_shared<ov::op::v8::Softmax>(input_slope_mask_node, 2);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
