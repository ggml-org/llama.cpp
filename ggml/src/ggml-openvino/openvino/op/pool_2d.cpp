#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/avg_pool.hpp>
#include <openvino/op/max_pool.hpp>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_pool_2d(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    const int32_t * params = context.get_output_op_params();

    const int k0 = params[1];
    const int k1 = params[2];
    const int s0 = params[3];
    const int s1 = params[4];
    const int p0 = params[5];
    const int p1 = params[6];

    const int op_case = context.get_op_case();
    ov::Output<Node> input = context.get_input(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_end{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    switch (op_case) {
    case 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(input, strides, pads_begin, pads_end, kernel);
        break;
    }
    case 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(input, strides, pads_begin, pads_end, kernel, false);
        break;
    }
    default:
        break;
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
