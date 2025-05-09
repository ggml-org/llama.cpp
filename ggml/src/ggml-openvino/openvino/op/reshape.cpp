#include "openvino/op/reshape.hpp"

#include <cstdint>
#include <vector>

#include "../node_context.hpp"
#include "../utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape(0)) {
        return {context.get_input(0)};
    }

    auto output_shape = context.get_output_shape(0).to_shape();
    auto new_shape_node =
        ov::op::v0::Constant::create(ov::element::i64,
                                     {3},
                                     std::vector<int64_t>{-1, (int64_t)output_shape[1], (int64_t)output_shape[2]});
    Output<Node> res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return {res};
};

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
