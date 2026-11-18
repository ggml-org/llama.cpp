#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstring>
#include <openvino/op/constant.hpp>
#include <vector>

namespace ov::frontend::ggml::op {
OutputVector translate_arange(const NodeContext & context) {
    num_inputs_check(context, 0, 0);
    float params[3];
    memcpy(params, context.get_output_op_params(), sizeof(params));
    // GGML ARANGE has fixed op parameters, including for rollback slot order.
    std::vector<float> values(ov::shape_size(context.get_output_shape().to_shape()));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = params[0] + params[2] * i;
    }
    auto result = ov::op::v0::Constant::create(context.get_output_type(), context.get_output_shape().to_shape(), values);
    return rename_outputs_with_suffix({result}, context.get_name());
}
}  // namespace ov::frontend::ggml::op
