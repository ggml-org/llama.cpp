#include "../op_table.h"
#include "../utils.h"
#include <openvino/op/reshape.hpp>
#include <set>
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    // op_case 3
    if (context.get_op_case() == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();

        auto input_llama_shape = context.get_input_shape(0).to_shape();

        // if the input ov shape size is different from the input llama shape size, it means the input is already reshaped and we need to reshape it back to the original shape before slicing
        if (input_ov_shape.size() != input_llama_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, {input_llama_shape.size()}, input_llama_shape), false);
        }

        auto dst_shape = context.get_output_shape().to_shape();

        std::vector<size_t> diff_dims;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_llama_shape[i]) {
                diff_dims.push_back(i);
            }
        }

        FRONT_END_CHECK_IMPLEMENTED(!diff_dims.empty(), "VIEW op_case 3 failed to infer changed dims");

        const size_t offset = context.get_output_op_offset();
        const auto input_stride = context.get_input_stride(0);
        FRONT_END_CHECK_IMPLEMENTED(input_stride.size() == dst_shape.size(),
                                    "VIEW op_case 3 shape/stride rank mismatch");

        // Multi-dim change: infer begin/end for each axis from shape/stride/offset directly.
        if (diff_dims.size() > 1) {
            std::vector<int64_t> begin(dst_shape.size(), 0);
            std::vector<int64_t> end(dst_shape.size(), 0);
            std::vector<int64_t> step(dst_shape.size(), 1);
            std::vector<int64_t> axes(dst_shape.size(), 0);

            size_t rem_offset = offset;
            for (size_t i = 0; i < dst_shape.size(); ++i) {
                FRONT_END_CHECK_IMPLEMENTED(input_stride[i] > 0, "VIEW op_case 3 invalid stride");
                begin[i] = static_cast<int64_t>(rem_offset / input_stride[i]);
                rem_offset %= input_stride[i];
                end[i] = begin[i] + static_cast<int64_t>(dst_shape[i]);
                axes[i] = static_cast<int64_t>(i);

                FRONT_END_CHECK_IMPLEMENTED(begin[i] >= 0 &&
                                                end[i] <= static_cast<int64_t>(input_llama_shape[i]),
                                            "VIEW op_case 3 multi-dim inferred slice out of bounds");
            }

            auto sliced = std::make_shared<ov::op::v8::Slice>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
                ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
                ov::op::v0::Constant::create(ov::element::i64, {step.size()}, step),
                ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));
            return {sliced};
        }

        // find the index of dst_shape that is different from input shape, and use that index to slice the input
        int slice_dim = -1;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_llama_shape[i]) {
                slice_dim = i;
                break;
            }
        }

        FRONT_END_CHECK_IMPLEMENTED(slice_dim >= 0, "VIEW op_case 3 failed to infer slice dim");

        FRONT_END_CHECK_IMPLEMENTED(input_stride[slice_dim] > 0, "VIEW op_case 3 invalid stride");

        const int64_t dim_size = static_cast<int64_t>(input_llama_shape[slice_dim]);

        if (offset % input_stride[slice_dim] == 0) {
            const int64_t begin_val = static_cast<int64_t>((offset / input_stride[slice_dim]) % static_cast<size_t>(dim_size));
            const int64_t end_val = begin_val + static_cast<int64_t>(dst_shape[slice_dim]);

            FRONT_END_CHECK_IMPLEMENTED(begin_val >= 0 &&
                                            end_val <= dim_size,
                                        "VIEW op_case 3 inferred slice out of bounds");

            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
            auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
            auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
            return {sliced};
        }

        // Fallback for offsets that cross lower dimensions: flatten tail dims, slice 1D range, then reshape.
        FRONT_END_CHECK_IMPLEMENTED(slice_dim + 1 < static_cast<int>(dst_shape.size()),
                                    "VIEW op_case 3 fallback requires lower dimensions");

        int64_t tail_src_elems = 1;
        int64_t tail_dst_elems = 1;
        for (size_t i = static_cast<size_t>(slice_dim); i < input_llama_shape.size(); ++i) {
            tail_src_elems *= static_cast<int64_t>(input_llama_shape[i]);
            tail_dst_elems *= static_cast<int64_t>(dst_shape[i]);
        }

        const auto elem_stride = input_stride.back();
        FRONT_END_CHECK_IMPLEMENTED(elem_stride > 0 && offset % elem_stride == 0,
                                    "VIEW op_case 3 fallback invalid element stride/alignment");

        const int64_t tail_begin = static_cast<int64_t>((offset / elem_stride) % static_cast<size_t>(tail_src_elems));
        const int64_t tail_end = tail_begin + tail_dst_elems;
        FRONT_END_CHECK_IMPLEMENTED(tail_begin >= 0 && tail_end <= tail_src_elems,
                                    "VIEW op_case 3 fallback slice out of bounds");

        std::vector<int64_t> flat_shape;
        for (int i = 0; i < slice_dim; ++i) {
            flat_shape.push_back(static_cast<int64_t>(input_llama_shape[i]));
        }
        flat_shape.push_back(tail_src_elems);

        auto flat = std::make_shared<ov::op::v1::Reshape>(
            input,
            ov::op::v0::Constant::create(ov::element::i64, {flat_shape.size()}, flat_shape),
            false);

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
        auto sliced = std::make_shared<ov::op::v8::Slice>(flat, begin, end, stride, axes);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
            sliced,
            ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape),
            false);
        return {reshaped};
    }

    // op_case 4: view offset selects one index from a middle dimension, then output keeps another source dim.
    // Example: src [N,M,K,1] -> dst [N,K,1,1] with offsets 0, nb1, 2*nb1, ...
    if (context.get_op_case() == 4) {
        auto input = context.get_input(0);
        auto src_shape = context.get_input_shape(0).to_shape();
        auto dst_shape = context.get_output_shape().to_shape();
        auto src_stride = context.get_input_stride(0);
        auto dst_stride = context.get_output_stride();

        FRONT_END_CHECK_IMPLEMENTED(src_shape.size() == dst_shape.size() &&
                                        src_shape.size() == src_stride.size() &&
                                        src_shape.size() == dst_stride.size(),
                                    "VIEW op_case 4 shape/stride rank mismatch");

        std::set<size_t> used_dst_strides;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] > 1) {
                used_dst_strides.insert(dst_stride[i]);
            }
        }

        int64_t slice_axis = -1;
        for (size_t i = 0; i < src_shape.size(); ++i) {
            if (src_shape[i] > 1 && used_dst_strides.find(src_stride[i]) == used_dst_strides.end()) {
                slice_axis = static_cast<int64_t>(i);
                break;
            }
        }
        FRONT_END_CHECK_IMPLEMENTED(slice_axis >= 0, "VIEW op_case 4 failed to infer slice axis");

        const size_t offset = context.get_output_op_offset();
        const size_t axis_stride = src_stride[static_cast<size_t>(slice_axis)];
        FRONT_END_CHECK_IMPLEMENTED(axis_stride > 0, "VIEW op_case 4 invalid axis stride");

        const int64_t axis_size = static_cast<int64_t>(src_shape[static_cast<size_t>(slice_axis)]);
        const int64_t slice_index = static_cast<int64_t>((offset / axis_stride) % static_cast<size_t>(axis_size));

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_index});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_index + 1});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_axis});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);

        if (context.get_op_dynamic_dim() != -1) {
            dst_shape[3 - context.get_op_dynamic_dim()] = -1;
        }

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
            sliced,
            ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape),
            false);
        return rename_outputs_with_suffix({reshaped}, context.get_name());
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
