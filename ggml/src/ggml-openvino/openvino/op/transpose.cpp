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
    auto input_stride = context.get_input_stride(0);
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

    // ggml always describes tensors as rank-4, but the stateful path can deliver rank-3;
    // project the permutation onto the surviving axes (dropped ones must be fixed points).
    const auto & in_ps = input.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(in_ps.rank().is_static(), "TRANSPOSE requires a static input rank, got ", in_ps);
    const int64_t in_rank = in_ps.rank().get_length();
    if (in_rank < static_cast<int64_t>(permute_order.size())) {
        const int64_t dropped = static_cast<int64_t>(permute_order.size()) - in_rank;
        for (int64_t i = 0; i < dropped; ++i) {
            FRONT_END_OP_CONVERSION_CHECK(permute_order[i] == i, "TRANSPOSE: operand is rank ", in_rank,
                                          " but leading axis ", i, " is permuted to ", permute_order[i],
                                          "; cannot project the rank-4 ggml permutation onto it");
        }
        std::vector<int64_t> projected(permute_order.begin() + dropped, permute_order.end());
        for (auto & axis : projected) {
            axis -= dropped;
        }
        permute_order = std::move(projected);
    }

    auto res = std::make_shared<ov::op::v1::Transpose>(
        input, ov::op::v0::Constant::create(ov::element::i64, {permute_order.size()}, permute_order));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
