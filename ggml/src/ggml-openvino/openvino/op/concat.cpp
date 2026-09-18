#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <algorithm>
#include <memory>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_concat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    const int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    const auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    const auto rank = output_shape.rank().get_length();
    const int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);
    const auto output_type = context.get_output_type();

    if (input_0.get_element_type() != output_type) {
        input_0 = std::make_shared<ov::op::v0::Convert>(input_0, output_type);
    }
    if (input_1.get_element_type() != output_type) {
        input_1 = std::make_shared<ov::op::v0::Convert>(input_1, output_type);
    }

    const auto & ps_0 = input_0.get_partial_shape();
    const auto & ps_1 = input_1.get_partial_shape();
    int64_t concat_rank = rank;
    if (ps_0.rank().is_static() && ps_1.rank().is_static()) {
        concat_rank = std::max(ps_0.rank().get_length(), ps_1.rank().get_length());
        input_0 = lift_to_rank(input_0, concat_rank);
        input_1 = lift_to_rank(input_1, concat_rank);
    }

    const auto axis = static_cast<int64_t>(concat_rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{input_0, input_1}, axis);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
