#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto input = process_view_input_new(context, 0);
    auto input_shape = context.get_input_shape(0);
    auto output_shape = context.get_output_shape();

    // Non-cast CPY may need a reshape (e.g. [3,192,1,1] -> [576,1,1,1])
    if (input_shape != output_shape) {
        auto new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_length())}, output_shape.to_shape());
        input = std::make_shared<ov::op::v1::Reshape>(input, new_shape, false);
    }

    auto res = std::make_shared<ov::op::v0::Convert>(input, context.get_output_type());
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
