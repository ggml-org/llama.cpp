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

    int64_t d_inner = sx_shape[2];
    int64_t d_conv = c_shape[3];

    // The depthwise conv is expressed as a 2D GroupConvolution with a unit H dim: the GPU plugin
    // (OV 2026.4, ACCURACY mode) fails to compile the 1D form ("Weights/ifm mismatch") when its
    // input comes from RMSNorm -> Transpose -> Concat.
    // Transpose sx from [1, n_s, d_inner, ncs] to [n_s, d_inner, 1, ncs]
    auto sx_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 2, 0, 3});
    auto sx_reshaped = std::make_shared<ov::op::v1::Transpose>(sx, sx_perm);

    // Reshape c from [1, 1, d_inner, d_conv] to [d_inner, 1, 1, 1, d_conv]
    // GroupConvolution filter: [groups, out_channels/groups, in_channels/groups, kH, kW]
    auto c_new_shape =
        ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{d_inner, 1, 1, 1, d_conv});
    auto c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, false);

    // Depthwise convolution: groups=d_inner, stride=1, no padding, no dilation
    // Input: [n_s, d_inner, 1, ncs], Filter: [d_inner, 1, 1, 1, d_conv]
    // Output: [n_s, d_inner, 1, n_t]
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(sx_reshaped, c_reshaped, ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

    // Transpose from [n_s, d_inner, 1, n_t] to output shape [1, n_s, n_t, d_inner]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{2, 0, 3, 1});
    auto res = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
