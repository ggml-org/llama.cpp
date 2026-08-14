#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <memory>
#include <vector>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto op_case = context.get_op_case();
    auto input_shape = context.get_input_shape(0);
    auto output_shape = context.get_input_shape(1);

    if (op_case == 4) {
        auto src = process_view_input_new(context, 0);
        auto base = context.get_input(1);

        int64_t n_elems = 1;
        for (const auto & dim : context.get_output_shape().to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        const auto output_stride = context.get_output_stride();
        const size_t elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "CPY conv state view update has invalid element size");

        const int64_t begin_val = static_cast<int64_t>(context.get_output_op_offset() / elem_size);
        const int64_t end_val = begin_val + n_elems;

        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, false);
        if (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, 3);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts and
    // where the copied data starts in the source are runtime inputs, so the cached model works for
    // any kv head, active sequence count and token count. The result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    const std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    const bool slice_assign =
        context.has_input(slot_begin_name) && !context.is_stateful() && (op_case >= 1 && op_case <= 3);
    if (slice_assign) {
        const int64_t slot_axis = 2;
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        auto feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_length()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin = context.get_input(slot_begin_name);
        auto base = context.get_input(1);
        if (op_case == 1) {
            // GDN packs [attn | state snapshots]; the state part runs from src_begin to the end.
            auto src_begin = context.get_input("rs_src_begin_" + context.get_name());
            auto state_part = std::make_shared<ov::op::v8::Slice>(context.get_input(0), src_begin, int_max, one, axis);
            src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, false);
        } else if (op_case == 2) {
            // conv_input is [previous conv state | new tokens]; copy the conv_kernel_size - 1 wide
            // window starting at src_begin, which is the snapshot this writeback corresponds to.
            auto window_size = (int64_t) input_shape[3].get_length();
            auto src_begin = context.get_input("rs_src_begin_" + context.get_name());
            auto src_end = std::make_shared<ov::op::v1::Add>(
                src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {window_size}));
            auto window = std::make_shared<ov::op::v8::Slice>(context.get_input(0), src_begin, src_end, one,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
            const auto base_shape = base.get_partial_shape();
            FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape.rank().get_length() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_END_OP_CONVERSION_CHECK(base_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_END_OP_CONVERSION_CHECK(input_shape.rank().is_static() && input_shape.rank().get_length() == 4 &&
                                              input_shape[2].is_static() && input_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            const int64_t full_feature_size = base_shape[3].get_length();
            const int64_t update_feature_size = input_shape[2].get_length() * input_shape[3].get_length();
            const auto output_stride = context.get_output_stride();
            const size_t elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
            FRONT_END_OP_CONVERSION_CHECK(elem_size > 0,
                                          "CPY conv state cache update has invalid element size");
            const int64_t feature_begin = static_cast<int64_t>(context.get_output_op_offset() / elem_size) %
                                          full_feature_size;
            const int64_t feature_end = feature_begin + update_feature_size;
            FRONT_END_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_end <= full_feature_size,
                                          "CPY conv state cache update feature range is out of bounds");

            auto partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_size});
            src = std::make_shared<ov::op::v1::Reshape>(window, partial_feature, false);
            if (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            auto src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            auto slot_end = std::make_shared<ov::op::v1::Add>(begin, src_len);
            auto active_slots = std::make_shared<ov::op::v8::Slice>(base, begin, slot_end, one, axis);

            auto feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto feature_begin_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            auto feature_end_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_end});
            auto feature_head = std::make_shared<ov::op::v8::Slice>(active_slots, zero, feature_begin_node, one,
                                                                    feature_axis);
            auto feature_tail = std::make_shared<ov::op::v8::Slice>(active_slots, feature_end_node, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{feature_head, src, feature_tail}, 3);
        } else {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_input(0);
        }

        if (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, slot_axis);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto input = process_view_input_new(context, 0);

    if (input_shape != output_shape) {
        auto new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_length())}, output_shape.to_shape());
        input = std::make_shared<ov::op::v1::Reshape>(input, new_shape, false);
    }

    ov::Output<Node> res;
    if (context.get_input_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(input, context.get_output_type());
    } else {
        res = input;
    }

    if (res.get_node_shared_ptr() == context.get_input(0).get_node_shared_ptr()) {
        return {res};
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
