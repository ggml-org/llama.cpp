#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_geglu(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        auto combined = context.get_input(0);
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {-1});
        auto split = std::make_shared<ov::op::v1::Split>(combined, split_axis, 2);
        src0 = split->output(0);
        src1 = split->output(1);
    }

    int32_t * params = context.get_output_op_params();
    const int32_t swapped = params[1];
    if (swapped) {
        std::swap(src0, src1);
    }

    auto gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    auto res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
