#include <openvino/core/node_output.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_swiglu(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto src1 = context.get_input(0);
    auto src2 = context.get_input(1);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src1);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src1, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src2);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
