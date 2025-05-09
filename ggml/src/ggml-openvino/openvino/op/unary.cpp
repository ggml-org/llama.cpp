
#include <cstdint>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    return {context.get_input(0)};
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
