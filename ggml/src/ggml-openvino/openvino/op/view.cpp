#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
