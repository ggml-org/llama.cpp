#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/subtract.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = process_view_input_new(context, 0);

    // Step 1: Calculate mean along the last dimension
    // mean = reduce_mean(input, axis=-1, keepdims=true)
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        input_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    // Step 2: Calculate (input - mean)
    auto centered = std::make_shared<ov::op::v1::Subtract>(input_node, mean);

    // Step 3: Calculate squared differences (input - mean)^2
    auto squared = std::make_shared<ov::op::v1::Power>(
        centered, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));

    // Step 4: Calculate variance = mean((input - mean)^2)
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    // Step 5: Get epsilon from op_params
    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    // Step 6: Calculate std = sqrt(variance + eps)
    auto std_dev = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(
        variance, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    // Step 7: Normalize: output = (input - mean) / std
    auto res = std::make_shared<ov::op::v1::Divide>(centered, std_dev);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
