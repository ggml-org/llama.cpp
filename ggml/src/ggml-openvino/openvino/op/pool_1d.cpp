#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/avg_pool.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/max_pool.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_pool_1d(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    const int k0 = params[1];
    const int s0 = params[2];
    const int p0 = params[3];

    const int op_case = context.get_op_case();
    FRONT_END_OP_CONVERSION_CHECK(op_case == 1 || op_case == 2, "Unsupported POOL_1D mode");

    ov::Output<Node> input = process_view_input_new(context, 0);

    // OV pools over the last two axes; pool only along last one
    ov::Strides strides{1, static_cast<size_t>(s0)};
    ov::Shape pads{0, static_cast<size_t>(p0)};
    ov::Shape kernel{1, static_cast<size_t>(k0)};
    ov::Output<Node> res;
    if (op_case == 1) {
        res = std::make_shared<ov::op::v1::MaxPool>(input, strides, pads, pads, kernel);
    } else {
        // ggml divides by the count of valid (not padded) cells
        res = std::make_shared<ov::op::v1::AvgPool>(input, strides, pads, pads, kernel, true);
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
