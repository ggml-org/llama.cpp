#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/sqrt.hpp>

#include "../node_context.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rms_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    ov::Shape input_shape = context.get_input_shape(0).to_shape();
    auto input_node = context.get_input(0);
    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto reduce_sum =
        std::make_shared<ov::op::v1::ReduceSum>(square,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                true);

    auto mean = std::make_shared<ov::op::v1::Divide>(
        reduce_sum,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {static_cast<float>(input_shape[2])}));

    float eps;
    memcpy(&eps, context.get_output_op_params(0), sizeof(float));
    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {eps})));

    auto scale =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, scale);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
