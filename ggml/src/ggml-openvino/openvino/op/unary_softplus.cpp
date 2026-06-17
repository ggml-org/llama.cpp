#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/abs.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_softplus(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);
    const auto element_type = input.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(input);
    auto abs = std::make_shared<ov::op::v0::Abs>(input);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
