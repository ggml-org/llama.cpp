#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (is_recurrent_cache_reset_case(context)) {
        return translate_recurrent_cache_reset(context);
    }

    float c;
    memcpy(&c, context.get_output_op_params(), sizeof(float));

    // the shape of the input, which keeps its dynamic token dim
    auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(process_view_input_new(context, 0), ov::element::i64);
    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    // In place on a contiguous part of a larger buffer: the result stands for the whole buffer (the in-place
    // source), so keep the elements outside the filled range.
    const auto base = context.get_input(0);
    const auto & base_ps = base.get_partial_shape();
    const auto out_ps = context.get_output_shape();
    if (!context.get_inplace_op_src().empty() && base_ps.is_static() && out_ps.is_static()) {
        const auto out_shape = out_ps.to_shape();
        const auto stride = context.get_output_stride();
        const size_t elem = context.get_output_type().size();
        bool contiguous = stride.size() == out_shape.size();
        size_t expected = elem;
        for (size_t i = out_shape.size(); contiguous && i-- > 0;) {
            contiguous = out_shape[i] == 1 || stride[i] == expected;
            expected *= out_shape[i];
        }
        const size_t total = ov::shape_size(base_ps.to_shape());
        const size_t count = ov::shape_size(out_shape);
        const size_t begin = context.get_output_op_offset() / elem;
        if (contiguous && count < total && begin <= total - count) {
            auto i64 = [](int64_t v) {
                return ov::op::v0::Constant::create(ov::element::i64, {1}, {v});
            };
            auto flat = std::make_shared<ov::op::v1::Reshape>(base, i64(-1), false);
            auto head = std::make_shared<ov::op::v8::Slice>(flat, i64(0), i64(begin), i64(1), i64(0));
            auto body = std::make_shared<ov::op::v1::Reshape>(res, i64(-1), false);
            auto tail = std::make_shared<ov::op::v8::Slice>(flat, i64(begin + count), i64(INT64_MAX), i64(1), i64(0));
            auto full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head, body, tail}, 0);
            res = std::make_shared<ov::op::v1::Reshape>(
                full, std::make_shared<ov::op::v3::ShapeOf>(base, ov::element::i64), false);
        }
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
