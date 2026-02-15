#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#include <memory>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto res = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_output_type());
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
