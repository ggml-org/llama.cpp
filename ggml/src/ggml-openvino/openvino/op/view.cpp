#include <cstdint>
#include <vector>

#include "../utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    return {context.get_input(0)};
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
