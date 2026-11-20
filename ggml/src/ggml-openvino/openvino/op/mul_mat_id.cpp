#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "gather_matmul.hpp"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <openvino/op/add.hpp>
#include <openvino/op/bitwise_and.hpp>
#include <openvino/op/bitwise_right_shift.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cum_sum.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/less.hpp>
#include <openvino/op/logical_and.hpp>
#include <openvino/op/logical_not.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_logical_or.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/topk_base.hpp>
#include <optional>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t> & values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> slice_axis(const ov::Output<ov::Node> & input, int64_t axis, int64_t begin, int64_t end) {
    return std::make_shared<ov::op::v8::Slice>(input, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<int> trailing_dims(const ov::Output<ov::Node> & input, int count) {
    const auto rank = input.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    const int rank_len = static_cast<int>(rank.get_length());
    FRONT_END_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<int> dims;
    for (int i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    return dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(const ov::Output<ov::Node> & input,
                                                  const std::vector<int> & dims) {
    const auto & partial_shape = input.get_partial_shape();
    if (partial_shape.is_static()) {
        std::vector<int64_t> values;
        values.reserve(dims.size());
        for (const int64_t dim : dims) {
            values.push_back(partial_shape[dim].get_length());
        }
        return const_i64(values);
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    return get_dimensions(shape, dims);
}

// TopK indices are never -1. Model graphs take the ids from the router's TopK through views and
// reshapes, so they skip the -1 handling below and keep the GatherMatmul pattern that the GPU MoE
// fusion matches.
bool ids_from_topk(ov::Output<ov::Node> ids) {
    for (;;) {
        const auto node = ids.get_node_shared_ptr();
        if (ov::is_type<ov::op::util::TopKBase>(node)) {
            return ids.get_index() == 1;
        }
        if (!ov::is_type<ov::op::v1::Reshape>(node) && !ov::is_type<ov::op::v0::Unsqueeze>(node) &&
            !ov::is_type<ov::op::v0::Squeeze>(node) && !ov::is_type<ov::op::v0::Convert>(node) &&
            !ov::is_type<ov::op::v1::Transpose>(node) && !ov::is_type<ov::op::v8::Slice>(node) &&
            !ov::is_type<ov::op::v1::StridedSlice>(node)) {
            return false;
        }
        ids = node->input_value(0);
    }
}

// An id of -1 skips the slot and zeroes the matching row of the result. The skipped slot looks up a
// valid expert, and the result is multiplied by this mask: 1 for a kept slot, 0 for a skipped one,
// shaped [ids..., 1] to broadcast over the row.
ov::Output<ov::Node> kept_slots_mask(const ov::Output<ov::Node> & ids, const ov::element::Type & type) {
    auto zero = ov::op::v0::Constant::create(ids.get_element_type(), ov::Shape{}, {0});
    ov::Output<ov::Node> keep = std::make_shared<ov::op::v1::GreaterEqual>(ids, zero);
    keep = std::make_shared<ov::op::v0::Convert>(keep, type);
    return std::make_shared<ov::op::v0::Unsqueeze>(keep, const_i64({-1}));
}

ov::Output<ov::Node> clamp_skipped_ids(const ov::Output<ov::Node> & ids) {
    auto zero = ov::op::v0::Constant::create(ids.get_element_type(), ov::Shape{}, {0});
    return std::make_shared<ov::op::v1::Maximum>(ids, zero);
}

// GatherMatmul computes a wrong row when two slots of a token name the same expert (ggml leaves
// that undefined too, see TAG_MUL_MAT_ID_DUPLICATE_ID), so a skipped slot cannot simply look up
// expert 0 while another slot of the row keeps it. A row with S skipped slots keeps at most
// n_used - S experts, so [0, n_used) holds at least S unused ones: the r-th skipped slot of a row
// looks up the r-th of them. ids is [n_tokens, n_used].
ov::Output<ov::Node> distinct_ids_for_skipped_slots(const ov::Output<ov::Node> & ids) {
    using namespace ov::op;
    const auto & ps = ids.get_partial_shape();
    if (ps.rank().is_dynamic() || ps[ps.rank().get_length() - 1].is_dynamic()) {
        return clamp_skipped_ids(ids);
    }
    const auto type = ids.get_element_type();
    const int64_t n_used = ps[ps.rank().get_length() - 1].get_length();
    std::vector<int64_t> range(n_used);
    std::iota(range.begin(), range.end(), 0);
    auto experts = v0::Constant::create(type, ov::Shape{static_cast<size_t>(n_used)}, range);
    auto zero = v0::Constant::create(type, ov::Shape{}, {0});
    auto one = v0::Constant::create(type, ov::Shape{}, {1});
    auto last_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});

    // free[t, e]: no slot of row t uses expert e, for e in [0, n_used)
    auto uses = std::make_shared<v1::Equal>(std::make_shared<v0::Unsqueeze>(ids, const_i64({-1})), experts);
    auto free = std::make_shared<v1::LogicalNot>(std::make_shared<v1::ReduceLogicalOr>(uses, const_i64({-2}), false));
    // 1-based rank of each free expert in its row, and of each skipped slot in its row
    auto free_rank = std::make_shared<v0::CumSum>(std::make_shared<v0::Convert>(free, type), last_axis);
    auto skip = std::make_shared<v1::Less>(ids, zero);
    auto skip_rank = std::make_shared<v1::Add>(
        std::make_shared<v0::CumSum>(std::make_shared<v0::Convert>(skip, type), last_axis, true), one);

    // pick[t, j, e]: expert e is the skip_rank[t, j]-th free expert of row t
    auto pick = std::make_shared<v1::LogicalAnd>(
        std::make_shared<v1::Equal>(std::make_shared<v0::Unsqueeze>(skip_rank, const_i64({-1})),
                                    std::make_shared<v0::Unsqueeze>(free_rank, const_i64({-2}))),
        std::make_shared<v0::Unsqueeze>(free, const_i64({-2})));
    auto picked = std::make_shared<v1::ReduceSum>(
        std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(pick, type), experts), const_i64({-1}), false);
    return std::make_shared<v1::Select>(skip, picked, ids);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(const NodeContext & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    const int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    const int64_t rows = static_cast<int64_t>(packed_shape[2]);
    const int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    const int64_t qk = 32;
    const int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, false);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);
    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }
    const auto output_type = context.get_output_type();
    // Gather and MatMul handle repeated ids, so clamping a skipped slot to expert 0 is enough here.
    std::optional<ov::Output<ov::Node>> keep;
    if (!ids_from_topk(ids)) {
        keep = kept_slots_mask(ids, output_type);
        ids = clamp_skipped_ids(ids);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    static const std::vector<float> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<float> e8m0_lut(256);
    for (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        memcpy(&e8m0_lut[i], &bits, sizeof(float));
    }
    e8m0_lut[0] = std::numeric_limits<float>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<float>::quiet_NaN();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto low = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    auto high_shift = std::make_shared<ov::op::v15::BitwiseRightShift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{low, high_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, false);

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims, ov::op::BroadcastType::BIDIRECTIONAL);

    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    if (keep) {
        result = std::make_shared<ov::op::v1::Multiply>(result, *keep);
    }
    return result;
}

}  // namespace

OutputVector translate_mul_mat_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = process_view_input_new(context, 0);
    auto activations = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    if (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    if (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    if (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        return rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
                                          context.get_name());
    }

    // General (non-packed) path: dense F32/F16/BF16 weights, or the f16 dequantization chain for
    // quantized MoE experts (see extract_quantized_weights / make_int4_weights / make_int8_weights in
    // ggml-quants.cpp). Routed through ov::op::internal::GatherMatmul instead of a naive
    // Gather+Broadcast+MatMul, so the selected expert's full weight matrix is never materialized per
    // token. The CPU plugin's ConvertGatherMatmulToGatherMatmulCompressed pass (run during
    // compile_model) fuses the dequantization chain feeding GatherMatmul's B input into a
    // GatherMatmulCompressed node automatically, as long as MarkDequantization has marked the chain --
    // see translate_session.cpp's apply_transformations for the MarkDequantization registration.
    //
    // OpenVINO sees GGML tensors in reversed dimension order:
    //   activations: [1, n_tokens, n_used_or_1, k]
    //   ids: [1, 1, n_tokens, n_used]
    // expert_weights is either [1, n_expert, m, k] (4D, e.g. non-quantized weights without a
    // pre-built extra) or already [n_expert, m, k] (3D, weights routed through
    // process_weight_tensor) -- GatherMatmul's B input expects the latter.
    auto expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    if (expert_weights_rank.get_length() == 4) {
        auto expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, false);
    }

    auto activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    auto ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    const auto output_type = context.get_output_type();
    std::optional<ov::Output<ov::Node>> keep;
    if (!ids_from_topk(ids)) {
        keep = kept_slots_mask(ids, output_type);
        ids = distinct_ids_for_skipped_slots(ids);
    }

    const auto activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    if (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    auto activations_transpose_order = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_order);

    ov::Output<ov::Node> result = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    auto result_transpose_order = const_i64({1, 0, 2});
    result = std::make_shared<ov::op::v1::Transpose>(result, result_transpose_order);
    if (!context.is_stateful()) {
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        result = std::make_shared<ov::op::v0::Unsqueeze>(result, unsqueeze_axes);
    }

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    if (keep) {
        result = std::make_shared<ov::op::v1::Multiply>(result, *keep);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
