#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_ssm_conv(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    auto sx = context.get_input(0);                         // conv state + input: OV shape [1, n_s, d_inner, ncs]
    auto c = context.get_input(1);                          // conv1d weight:      OV shape [1, 1, d_inner, d_conv]

    auto sx_shape = context.get_input_shape(0).to_shape();  // [1, n_s, d_inner, ncs]
    auto c_shape = context.get_input_shape(1).to_shape();   // [1, 1, d_inner, d_conv]

    int64_t n_s = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    int64_t ncs = sx_shape[3];  // d_conv - 1 + n_t
    int64_t d_conv = c_shape[3];
    int64_t n_t = ncs - d_conv + 1;

    // Reshape sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, ncs] for 1D GroupConvolution
    auto sx_new_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{n_s, d_inner, ncs});
    auto sx_reshaped = std::make_shared<ov::op::v1::Reshape>(sx, sx_new_shape, false);

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kernel_size]
    auto c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    auto c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, false);

    // Depthwise 1D convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, ncs], Filter: [d_inner, 1, 1, d_conv]
    // Output: [n_s, d_inner, n_t]
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // Transpose from [n_s, d_inner, n_t] to [n_s, n_t, d_inner]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // Reshape to output shape [1, n_s, n_t, d_inner]
    auto out_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, n_s, n_t, d_inner});
    auto res = std::make_shared<ov::op::v1::Reshape>(transposed, out_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
