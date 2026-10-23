#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <algorithm>
#include <climits>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

// Gathering the rows of a quantized embedding table before dequantizing them is on by default on NPU and can be
// forced either way with GGML_OPENVINO_GATHER_BEFORE_DEQUANT (a positive value enables it, 0 disables it).
bool gather_before_dequant_enabled() {
    if (ggml_openvino_getenv_str("GGML_OPENVINO_GATHER_BEFORE_DEQUANT") != nullptr) {
        return ggml_openvino_getenv_int("GGML_OPENVINO_GATHER_BEFORE_DEQUANT") > 0;
    }
    return ggml_openvino_is_npu();
}

// Gathers the constant `c` along its first axis when it holds one entry per table row. A constant whose dimensions
// are all 1 broadcasts against every row and is returned unchanged. Returns an empty Output for any other shape.
Output<Node> gather_table_rows(const Output<Node> & c, const Output<Node> & indices, int64_t rows) {
    const auto & ps = c.get_partial_shape();
    if (ps.is_dynamic()) {
        return {};
    }
    if (ps.rank().get_length() > 0 && ps[0].get_length() == rows) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        return std::make_shared<ov::op::v8::Gather>(c, indices, axis);
    }
    const auto & shape = ps.to_shape();
    if (std::all_of(shape.begin(), shape.end(), [](size_t dim) { return dim == 1; })) {
        return c;
    }
    return {};
}

// A quantized embedding table reaches GET_ROWS as the dequantization chain
//
//   [Convert(f32)] <- [Reshape] <- Multiply(scale, [Subtract(zero point)] <- Convert(quantized Constant))
//
// built by make_int4_weights()/make_int8_weights(). Gathering after this chain dequantizes the whole table (about
// 1.3 GB in f16 for a 248320 x 2560 token_embd) for every token. When the table is tied to the output projection,
// the NPUW partitioner also hands the dequantized table from one submodel to the next. Gathering the rows of the
// quantized constants and of their scales first dequantizes only the selected rows, and leaves the table to be
// consumed once, in its quantized form, by the output projection.
//
// `data` is the 2D table [rows, cols]; the quantized constant is [rows, groups, group_size] (followed by a Reshape)
// or [rows, cols] (channel-wise, no Reshape). Returns an empty Output when `data` is not such a chain, so that the
// caller falls back to a plain Gather.
Output<Node> gather_before_dequant(const Output<Node> & data, const Output<Node> & indices) {
    using namespace ov::op;

    const auto indices_rank = indices.get_partial_shape().rank();
    if (indices_rank.is_dynamic()) {
        return {};
    }

    Output<Node> chain = data;
    const auto top_convert = ov::as_type_ptr<v0::Convert>(chain.get_node_shared_ptr());
    if (top_convert) {
        chain = top_convert->input_value(0);
    }
    const bool has_reshape = ov::is_type<v1::Reshape>(chain.get_node());
    if (has_reshape) {
        chain = chain.get_node()->input_value(0);
    }

    // Multiply(x, scale): exactly one operand is the scale constant.
    const auto multiply = ov::as_type_ptr<v1::Multiply>(chain.get_node_shared_ptr());
    if (!multiply) {
        return {};
    }
    Output<Node> scale;
    Output<Node> dequant_input;
    for (size_t i = 0; i < 2; ++i) {
        if (ov::is_type<v0::Constant>(multiply->input_value(i).get_node())) {
            scale = multiply->input_value(i);
            dequant_input = multiply->input_value(1 - i);
        }
    }
    if (!scale.get_node()) {
        return {};
    }

    // Optional zero point: Subtract(Convert(q), Constant or Convert(Constant)).
    Output<Node> zero_point;
    if (const auto subtract = ov::as_type_ptr<v1::Subtract>(dequant_input.get_node_shared_ptr())) {
        zero_point = subtract->input_value(1);
        dequant_input = subtract->input_value(0);
    }

    const auto convert_q = ov::as_type_ptr<v0::Convert>(dequant_input.get_node_shared_ptr());
    if (!convert_q || !ov::is_type<v0::Constant>(convert_q->input_value(0).get_node())) {
        return {};
    }
    const Output<Node> quantized = convert_q->input_value(0);

    const auto & quantized_ps = quantized.get_partial_shape();
    if (quantized_ps.is_dynamic() || quantized_ps.rank().get_length() != (has_reshape ? 3 : 2)) {
        return {};
    }
    const int64_t rows = quantized_ps[0].get_length();

    Output<Node> rows_q = gather_table_rows(quantized, indices, rows);
    Output<Node> rows_scale = gather_table_rows(scale, indices, rows);
    if (!rows_q.get_node() || !rows_scale.get_node()) {
        return {};
    }
    Output<Node> result = std::make_shared<v0::Convert>(rows_q, convert_q->get_destination_type());

    if (zero_point.get_node()) {
        std::shared_ptr<v0::Convert> zero_point_convert =
            ov::as_type_ptr<v0::Convert>(zero_point.get_node_shared_ptr());
        const Output<Node> zero_point_const = zero_point_convert ? zero_point_convert->input_value(0) : zero_point;
        if (!ov::is_type<v0::Constant>(zero_point_const.get_node())) {
            return {};
        }
        Output<Node> rows_zero_point = gather_table_rows(zero_point_const, indices, rows);
        if (!rows_zero_point.get_node()) {
            return {};
        }
        if (zero_point_convert) {
            rows_zero_point =
                std::make_shared<v0::Convert>(rows_zero_point, zero_point_convert->get_destination_type());
        }
        result = std::make_shared<v1::Subtract>(result, rows_zero_point);
    }
    result = std::make_shared<v1::Multiply>(result, rows_scale);

    if (has_reshape) {
        // [indices..., groups, group_size] -> [indices..., cols]: keep the index dimensions (0) and infer the rest.
        std::vector<int64_t> pattern(indices_rank.get_length(), 0);
        pattern.push_back(-1);
        result = std::make_shared<v1::Reshape>(
            result, v0::Constant::create(ov::element::i64, ov::Shape{pattern.size()}, pattern), true);
    }
    if (top_convert) {
        result = std::make_shared<v0::Convert>(result, top_convert->get_destination_type());
    }
    return result;
}

}  // namespace

OutputVector translate_get_rows(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    Output<Node> res;
    Output<Node> data;
    int64_t row_offset = 0;
    if (context.get_view_input_size(0) > 0 && context.get_input(0).get_partial_shape().rank() == 2) {
        // A row range view over a 2D weight Constant folds into the gather indices,
        // which keeps the dequantization subgraph intact for the plugins.
        const auto view_shape = context.get_view_input_ggml_shape(0, 0);
        const auto view_stride = context.get_view_input_stride(0, 0);
        const size_t view_offset = context.get_view_input_offset(0, 0);
        const size_t row_bytes = view_stride[2];
        data = context.get_input(0);
        FRONT_END_OP_CONVERSION_CHECK(row_bytes > 0 && view_offset % row_bytes == 0 &&
                                          data.get_partial_shape()[1].compatible(view_shape[3]),
                                      "GET_ROWS: view over a weight must be a row range");
        row_offset = static_cast<int64_t>(view_offset / row_bytes);
    } else {
        data = process_view_input_new(context, 0);
    }

    auto op_case = context.get_op_case();
    if (op_case == 3 || op_case == 4) {
        return {data};
    }

    ov::Output<ov::Node> indices;
    if ((op_case == 1 || op_case == 2) && context.has_input("s_copy_active_slot_len")) {
        // Recurrent state reorder (inp->s_copy): slice the active (op_case 1) or extra (op_case 2)
        // segment from the s_copy index list at runtime, instead of baking the static view offset,
        // so the cached IR works for any number of active sequences.
        auto s_copy = context.get_input(1);
        auto len = context.get_input("s_copy_active_slot_len");
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        if (op_case == 1) {
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, begin, len, step, axis);
        } else {
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
            indices = std::make_shared<ov::op::v8::Slice>(s_copy, len, end, step, axis);
        }
    } else {
        indices = process_view_input_new(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    // Squeeze the leading dims down to [b,x']. Stateful models drop one rank, so a hardcoded
    // {0,1} would also strip the batch dim whenever b == 1 (every decode step).
    const auto indices_rank = indices.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(indices_rank.is_static(), "Expected static rank for GET_ROWS indices");
    std::vector<int64_t> indices_squeeze_axes;
    for (int64_t i = 0; i + 2 < indices_rank.get_length(); ++i) {
        indices_squeeze_axes.push_back(i);
    }
    if (!indices_squeeze_axes.empty()) {
        indices = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {indices_squeeze_axes.size()},
                                                  indices_squeeze_axes));
    }
    if (row_offset != 0) {
        indices = std::make_shared<ov::op::v1::Add>(
            indices, ov::op::v0::Constant::create(indices.get_element_type(), {}, {row_offset}));
    }
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            // data: [batch, rows, ...], indices: [batch, n] - this is a batched gather
            // (batch_dims=1) along the rows axis. The data and indices batch dims are
            // logically equal (both == n_tokens) but reach this node through independent
            // reshapes, so the GPU plugin's gather shape inference cannot prove
            // data.shape[0] == indices.shape[0] and rejects the node. We must tie both
            // batch dims to the SAME value, and crucially that value must stay DYNAMIC.
            const auto data_ps = data.get_partial_shape();
            const auto idx_ps = indices.get_partial_shape();
            const bool data_batch_static = data_ps.rank().is_static() && data_ps[0].is_static();
            const bool idx_batch_dynamic = idx_ps.rank().is_dynamic() || idx_ps[0].is_dynamic();

            if (data_batch_static && idx_batch_dynamic) {
                // MoE per-expert-scale path: `data` is a statically-tiled REPEAT
                // (ggml_repeat_4d(scale, 1, n_expert, n_tokens, 1)) whose batch dim is a
                // compile-time-constant n_tokens, and every batch slice is IDENTICAL (it was
                // tiled from a single [1, n_expert, 1] scale). `indices` (selected_experts)
                // carries the genuinely dynamic token dim. Broadcasting indices up to the
                // static data batch (the naive fix) would freeze the token dim to the
                // captured prefill length, and that static value then flows through the
                // gather into the residual stream, making every following decoder layer
                // static -> triggers the GPU in-place-concat KV-cache corruption (only
                // layer 0 stays dynamic). A static->dynamic Broadcast cannot expand, so
                // instead collapse the redundant data batch to 1 and broadcast 1->dynamic to
                // match the indices batch. Mathematically identical (the slices are equal),
                // and the whole graph stays dynamic.
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto data_b1 = std::make_shared<ov::op::v8::Slice>(data, zero, one, one, axis0);  // [1, rows, ...]

                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_batch = get_dimensions(idx_shape, {0});  // [batch] (dynamic)
                auto data_b1_shape = std::make_shared<ov::op::v3::ShapeOf>(data_b1, ov::element::i64);
                const auto rank = data_ps.rank().get_length();
                std::vector<int> rest_axes;
                for (int a = 1; a < rank; ++a) {
                    rest_axes.push_back(a);
                }
                auto data_rest = get_dimensions(data_b1_shape, rest_axes);  // [rows, ...]
                auto data_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{idx_batch, data_rest}, 0);
                data =
                    std::make_shared<ov::op::v3::Broadcast>(data_b1, data_target, ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
            } else {
                // General case: tie the indices batch to the data batch (the data batch is
                // already dynamic, e.g. the routing-weights gather whose data comes from the
                // activations). Broadcast indices to [data_batch, indices_n].
                auto data_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
                auto data_batch = get_dimensions(data_shape, {0});  // [batch]
                auto idx_shape = std::make_shared<ov::op::v3::ShapeOf>(indices, ov::element::i64);
                auto idx_n = get_dimensions(idx_shape, {1});  // [n]
                auto idx_target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{data_batch, idx_n}, 0);
                indices = std::make_shared<ov::op::v3::Broadcast>(indices, idx_target,
                                                                  ov::op::BroadcastType::BIDIRECTIONAL);
                res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
            }
        }
    } else if (context.is_stateful() && data.get_partial_shape().rank() == 3) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
    } else {
        if (gather_before_dequant_enabled() && data.get_partial_shape().rank() == 2) {
            res = gather_before_dequant(data, indices);
        }
        // For a tied embedding the same weight also feeds the lm_head matmul. Gathering the
        // dequantized table would give that dequantization chain a second consumer, forcing the
        // full table to be materialized; gather the packed rows instead and dequantize those.
        if (!res.get_node()) {
            if (auto compressed = gather_compressed_rows(data, indices)) {
                res = compressed;
            }
        }
        if (!res.get_node()) {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        }
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    if (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
