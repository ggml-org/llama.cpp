#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <climits>
#include <cstdint>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather_elements.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/topk.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_argsort(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    const int32_t order = context.get_output_op_params()[0];

    ov::op::v11::TopK::Mode mode;
    switch (order) {
    case GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::TopK::Mode::MIN;
        break;
    case GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::TopK::Mode::MAX;
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    // Stateful models drop the leading size-1 batch dim, so the expert axis is 2 there
    // instead of 3 (same rank-3-vs-rank-4 split as get_rows.cpp / process_view_input).
    const int axis = (context.is_stateful() && input.get_partial_shape().rank() == 3) ? 2 : 3;

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(input.get_node_shared_ptr(), {axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::TopK>(input, k, axis, mode, ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    context.get_output_type(), false);

    return rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

OutputVector translate_top_k(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    const int64_t k = context.get_output_shape()[3].get_length();
    const auto shape = input.get_partial_shape();
    if (ggml_openvino_is_gpu() && shape.rank() == 4 && shape[3].is_static() &&
        shape[3].get_length() > 65535 && shape[3].get_length() <= INT32_MAX && k <= 64) {
        const int64_t width = shape[3].get_length();
        int64_t chunk_size = 4096;
        while (chunk_size >= 1024 && width % chunk_size != 0) {
            --chunk_size;
        }
        if (chunk_size >= 1024) {
            const int64_t chunks = width / chunk_size;
            auto chunked = std::make_shared<ov::op::v1::Reshape>(input,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{-1, 1, chunks, chunk_size}), false);
            // Each global top-k candidate is in its chunk's top-k set.
            // Value sorting selects the parallel GPU kernel; ggml allows any index order.
            auto local = std::make_shared<ov::op::v11::TopK>(chunked,
                ov::op::v0::Constant::create(ov::element::i32, {}, {k}), -1,
                ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i32, false);
            std::vector<int32_t> offsets(chunks);
            for (int64_t i = 0; i < chunks; ++i) {
                offsets[i] = i * chunk_size;
            }
            auto indices = std::make_shared<ov::op::v1::Add>(local->output(1),
                ov::op::v0::Constant::create(ov::element::i32, {1, 1, size_t(chunks), 1}, offsets));
            auto candidate_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, chunks * k});
            auto values = std::make_shared<ov::op::v1::Reshape>(local->output(0), candidate_shape, false);
            auto candidate_indices = std::make_shared<ov::op::v1::Reshape>(indices, candidate_shape, false);
            auto selected = std::make_shared<ov::op::v11::TopK>(values,
                ov::op::v0::Constant::create(ov::element::i32, {}, {k}), -1,
                ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i32, false);
            auto result = std::make_shared<ov::op::v6::GatherElements>(candidate_indices, selected->output(1), 3);
            auto output_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{
                get_dimensions(input.get_node_shared_ptr(), {0, 1, 2}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {k})}, 0);
            return rename_outputs_with_suffix({std::make_shared<ov::op::v1::Reshape>(result, output_shape, false)}, context.get_name());
        }
    }
    auto topk = std::make_shared<ov::op::v11::TopK>(
        input, ov::op::v0::Constant::create(ov::element::i32, {}, {k}), -1,
        ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::NONE, ov::element::i32, false);
    return rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
