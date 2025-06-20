
#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cont(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported CONT case");

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape(0).to_shape();
    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE
        dst_shape[1] = -1;
        res = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0),
            ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape),
            false);
    } else {
        // The input comes from a VIEW
        // Currently all cases are slicing at lowest dim
        int32_t* op_params = context.get_input_op_params(0);
        auto output_stride = context.get_output_stride(0);

        int64_t split_addr = op_params[0] / output_stride[2];
        std::vector<int64_t> begin = {0, 0, split_addr};
        std::vector<int64_t> end = {(int64_t)src_shape[0], INT_MAX, split_addr + (int64_t)src_shape[2]};
        std::vector<int64_t> strides = {1, 1, 1};

        auto begin_const = ov::op::v0::Constant::create(element::i64, {begin.size()}, begin);
        auto end_const = ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end);
        auto strides_const = ov::op::v0::Constant::create(ov::element::i64, {strides.size()}, strides);
        res = std::make_shared<ov::op::v8::Slice>(context.get_input(0), begin_const, end_const, strides_const);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
