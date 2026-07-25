#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstring>
#include <openvino/op/clamp.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_clamp(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    const int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    float min;
    float max;
    std::memcpy(&min, reinterpret_cast<const float *>(op_params) + 0, sizeof(float));
    std::memcpy(&max, reinterpret_cast<const float *>(op_params) + 1, sizeof(float));

    auto res = std::make_shared<ov::op::v0::Clamp>(input, min, max);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
