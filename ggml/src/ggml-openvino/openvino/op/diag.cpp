#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/eye.hpp>
#include <openvino/op/multiply.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
OutputVector translate_diag(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto x = process_view_input_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    auto n = get_dimensions(x.get_node_shared_ptr(), {3});
    auto zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    auto res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
