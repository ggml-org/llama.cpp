#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/type.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_elements_update.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = process_view_input_new(context, 0);
    auto indices = process_view_input_new(context, 1);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    const auto indices_shape = context.get_input_shape(1);
    const bool multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_length() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_length() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_length() > 1));

    // On NPU ScatterUpdate is not in-place: it copies the whole destination to write
    // one KV row, which dominates decode once ctx grows. ScatterElementsUpdate is
    // measurably cheaper for the same single-row write.
    const bool use_elements_update =
        !multidim_indices && !context.is_stateful() && ggml_openvino_getenv_int("GGML_OPENVINO_KV_SCATTER_ELEMENTS");

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    const bool is_cache_write = context.get_op_case() != 2;

    if (is_cache_write && context.is_static() && !context.is_stateful() && !multidim_indices &&
        ggml_openvino_npu_kv_head_major_enabled()) {
        // Head-major cache [1, heads, rows, head_size]: write the new rows along axis 2 without reshaping the cache.
        ov::Output<Node> cache = context.get_input(2);
        while (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(cache.get_node_shared_ptr())) {
            cache = reshape->input_value(0);
        }
        const auto cache_shape = cache.get_partial_shape();
        if (cache_shape.rank().is_static() && cache_shape.rank().get_length() == 4 && cache_shape[1].is_static() &&
            cache_shape[3].is_static() && cache_shape[1].get_length() > 1) {
            const int64_t heads = cache_shape[1].get_length();
            // Transposed V cache is [1, heads, head_size, rows]: rows go along axis 3.
            const bool v_transposed = cache.get_node()->get_friendly_name().rfind("cache_v_l", 0) == 0;
            const int64_t head_size = (v_transposed ? cache_shape[2] : cache_shape[3]).get_length();
            auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
                indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
            auto rows = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, heads, head_size}),
                false);
            auto rows_bhsd = std::make_shared<ov::op::v1::Transpose>(
                rows, ov::op::v0::Constant::create(ov::element::i64, {4},
                                                   v_transposed ? std::vector<int64_t>{0, 2, 3, 1} :
                                                                  std::vector<int64_t>{0, 2, 1, 3}));
            if (v_transposed && context.is_prefill() && ggml_openvino_npu_hfa_prefill_enabled()) {
                // On NPU a chunk-sized scatter along the innermost axis of [1, heads, head_size, rows] is very slow:
                // the HFA prefill outputs the chunk's V^T rows [1, heads, head_size, n] instead and the host copies
                // them into the cache (as GenAI does for its prefill KV).
                return rename_outputs_with_suffix({rows_bhsd}, context.get_name());
            }
            Output<Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(
                cache, ind_squeezed, rows_bhsd,
                v_transposed ? ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3}) : axes);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
    }

    auto dst = process_view_input_new(context, 2);
    Output<Node> res;
    if (is_cache_write && context.is_stateful()) {
        int concat_axis = 1;
        if (context.get_op_case() == 1) {
            // `-fa off` writes V transposed through a [1, n] reshape of the cache. The state keeps V token-major
            // like K, and the new values are already in token order, so append them to the state itself
            dst = dst.get_node()->input_value(0);
        }
        FRONT_END_OP_CONVERSION_CHECK(dst.get_partial_shape()[2].is_static() && dst.get_partial_shape()[3].is_static(),
                                      "Unsupported KV cache layout for stateful execution");
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else if (is_cache_write && is_kvcache_4d_shape(dst.get_partial_shape())) {
        // cache [1, ctx, n_heads_kv, head_size]: write whole tokens on axis 1. This is the pattern of the GPU
        // plugin's StatelessKVFusion (1-D axis, read with VariadicSplit, see permute), which writes the new rows in
        // place instead of copying the whole cache first
        FRONT_END_OP_CONVERSION_CHECK(!multidim_indices, "SET_ROWS into a 4-D KV cache needs one index per token");
        const int64_t n_heads_kv = dst.get_partial_shape()[2].get_length();
        const int64_t head_size = dst.get_partial_shape()[3].get_length();
        auto data_tokens = std::make_shared<ov::op::v1::Reshape>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, n_heads_kv, head_size}),
            false);
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_tokens,
                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
        return rename_outputs_with_suffix({res}, context.get_name());
    } else if (multidim_indices || !is_cache_write) {
        // updates [.., n_rows, row_size] with one index per row; stateful models may drop the leading 1 of the
        // updates but not of the destination
        const auto dst_rank = dst.get_partial_shape().rank();
        const auto data_rank = data.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(dst_rank.is_static() && data_rank.is_static() &&
                                          data_rank.get_length() <= dst_rank.get_length(),
                                      "SET_ROWS with per-row indices: updates ", data.get_partial_shape(),
                                      " do not fit the destination ", dst.get_partial_shape());
        if (data_rank.get_length() < dst_rank.get_length()) {
            std::vector<int64_t> lead_axes(dst_rank.get_length() - data_rank.get_length());
            std::iota(lead_axes.begin(), lead_axes.end(), 0);
            data = std::make_shared<ov::op::v0::Unsqueeze>(
                data, ov::op::v0::Constant::create(ov::element::i64, {lead_axes.size()}, lead_axes));
        }
        auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);

        auto indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        std::vector<int> row_dims(dst_rank.get_length() - 1);
        std::iota(row_dims.begin(), row_dims.end(), 0);
        auto indices_rows_shape =
            std::make_shared<ov::op::v0::Concat>(OutputVector{get_dimensions(updates_shape, row_dims), one}, 0);
        auto indices_rows = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rows_shape, false);
        auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(indices_rows, updates_shape);
        auto rows_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-2});

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, data, rows_axis);
    } else {
        auto row_size = context.get_input_shape(2)[3].get_length();
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        // skip the reshape when the rows already have that shape: it costs a GPU primitive per cache write
        const auto & data_shape = data.get_partial_shape();
        const bool data_is_rows = data_shape.rank().is_static() && data_shape.rank().get_length() == 4 &&
                                  data_shape[0] == 1 && data_shape[1] == 1 && data_shape[3] == row_size;
        ov::Output<ov::Node> data_reshaped = data;
        if (!data_is_rows) {
            data_reshaped = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4},
                                             {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_size}),
                false);
        }
        if (use_elements_update) {
            auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(data_reshaped, ov::element::i64);
            auto ind_rank4 = std::make_shared<ov::op::v1::Reshape>(
                ind_squeezed, ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, -1, 1}), false);
            auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(ind_rank4, updates_shape);
            res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, data_reshaped, axes);
        } else {
            res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
        }
    }

    auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_node_shared_ptr());
    if (!multidim_indices && is_cache_write && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
