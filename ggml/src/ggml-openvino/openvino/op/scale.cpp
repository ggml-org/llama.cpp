#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_scale(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float scale;
    float bias;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});

    if (is_recurrent_cache_reset_case(context)) {
        return translate_recurrent_cache_reset(context);
    }

    // Multiply and Add need matching types; compute f16 input in f32 and round once, as the CPU backend does.
    auto input = process_view_input_new(context, 0);
    const auto input_type = input.get_element_type();
    if (input_type != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(input, scale_node);
    if (bias != 0.0f) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(res, bias_node);
    }
    if (input_type != ov::element::f32) {
        res = std::make_shared<ov::op::v0::Convert>(res, input_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
