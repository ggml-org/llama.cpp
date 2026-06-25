#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    // Compute permute order from input/output shape and stride information
    // so it adapts to different input and output layouts.
    auto input_shape = context.get_input_shape(0).to_shape();
    auto input_stride = context.get_input_stride(0);
    auto output_shape = context.get_output_shape().to_shape();
    auto output_stride = context.get_output_stride();

    // Compute permute order by matching output and input stride rankings.
    // Build <stride, dim_index> pairs.
    std::vector<std::pair<size_t, int>> output_stride_dims;
    std::vector<std::pair<size_t, int>> input_stride_dims;

    for (int i = 0; i < 4; ++i) {
        output_stride_dims.push_back({output_stride[i], i});
        input_stride_dims.push_back({input_stride[i], i});
    }

    // Sort by stride in descending order.
    std::sort(output_stride_dims.rbegin(), output_stride_dims.rend());
    std::sort(input_stride_dims.rbegin(), input_stride_dims.rend());

    // Build permute order.
    std::vector<int64_t> permute_order(4);
    for (int i = 0; i < 4; ++i) {
        int output_dim = output_stride_dims[i].second;
        int input_dim = input_stride_dims[i].second;
        permute_order[output_dim] = input_dim;
    }

    auto input = process_view_input_new(context, 0);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        input, ov::op::v0::Constant::create(ov::element::i64, {4}, permute_order));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
