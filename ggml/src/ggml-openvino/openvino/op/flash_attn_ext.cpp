#include <memory>
#include <openvino/op/convert.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_flash_attn_ext(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto mask = context.get_input(3);

    float* params = reinterpret_cast<float*>(context.get_output_op_params(0));
    float scale         = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});
    auto res = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v , mask, scale_node, false);
    auto res_f32 = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    return rename_outputs_with_suffix({res_f32}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
