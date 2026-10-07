#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/roll.hpp>
#include <openvino/op/constant.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_roll(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    auto input = context.get_input(0);

    auto shift = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    auto axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    auto roll = std::make_shared<ov::op::v7::Roll>(input, shift, axes);
    return rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
