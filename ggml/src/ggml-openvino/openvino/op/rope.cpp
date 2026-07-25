#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext & context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    const int mode = op_case;

    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE, false);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (context.get_view_input_size(0) > 0) {
        data_node = process_view_input_new(context, 0).get_node_shared_ptr();
        if (context.is_stateful()) {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {3}, std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        } else {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {4},
                std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        }
    }

    auto output_type = context.get_output_type();
    if (data_node->get_element_type() != ov::element::f32) {
        data_node = std::make_shared<ov::op::v0::Convert>(data_node, ov::element::f32);
    }

    // TODO(openvino-gpu-rope-fusion): TEMPORARY WORKAROUND - do NOT revert until the
    // OpenVINO GPU plugin is updated.
    //
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE. The matcher requires rank-4 inputs, which is why
    // the original even/odd Slice translation (kept in the `else if (mode ==
    // TYPE_NORMAL)` branch below for reference) does not get fused.
    //
    // Once the GPU plugin's RoPE fusion is extended to also recognize the original
    // even/odd Slice form, this Flux rewrite should be removed and both modes should
    // be restored to the captured even/odd translation. Until then, keep both paths:
    // the active Flux rewrite here and the previous translation preserved below.
    if (mode == TYPE_NORMAL) {
        // Emit the Flux-style interleaved-RoPE pattern so the GPU plugin's
        // RoPEFusionFlux matcher folds this subgraph into ov::op::internal::RoPE:
        //   x_paired   = Reshape(x, [1, S, n_heads, head_size/2, 2])
        //   x0, x1     = Split(x_paired, axis=-1, num_splits=2)
        //   x1_neg     = x1 * -1
        //   x_rotated  = Reshape(Concat([x1_neg, x0], axis=-1), [1, S, n_heads, head_size])
        //   y          = x * t_cos + x_rotated * t_sin
        // Mathematically equivalent to the even/odd Slice form below.
        //
        // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
        // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
        // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
        // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
        // rank-4 output, so downstream attention is unaffected.
        if (context.is_stateful()) {
            auto r4_shape = ov::op::v0::Constant::create(
                ov::element::i64, {4},
                std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, r4_shape, false);
        }
        const int64_t head_size = static_cast<int64_t>(output_shape[3]);
        const int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        const int64_t half = head_size / 2;

        auto neg_one_f = ov::op::v0::Constant::create(data_node->get_element_type(), ov::Shape{}, {-1.0f});

        auto paired_shape =
            ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(data_node, paired_shape, false);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        Output<Node> x0 = data_split->outputs()[0];
        Output<Node> x1 = data_split->outputs()[1];

        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);

        auto flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, n_heads, head_size});
        auto x_rotated = std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, false);

        // Expand cos/sin from [..., head_size/2] to [..., head_size] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        auto expand_cos_sin = [&](Output<Node> cs) {
            auto cs_unsq =
                std::make_shared<ov::op::v0::Unsqueeze>(cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            auto bcast_target =
                ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast =
                std::make_shared<ov::op::v3::Broadcast>(cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, head_size});
            return std::make_shared<ov::op::v1::Reshape>(bcast, flat, true);
        };
        Output<Node> cos_full = expand_cos_sin(cos_theta_node);
        Output<Node> sin_full = expand_cos_sin(sin_theta_node);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(data_node, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        res = std::make_shared<ov::op::v1::Add>(y1, y2);
    }
    // PRESERVED PREVIOUS TRANSLATION - Re-enable this branch (and remove the Flux branch above) once
    // the GPU plugin's RoPE fusion is updated to recognize the even/odd Slice form;
    // see the TODO(openvino-gpu-rope-fusion) note above. Do not delete.
    //
    // Original even/odd Slice form. In stateless mode it ran on rank-4 data
    // ([1, S, n_heads, head_size]); in stateful mode on rank-3 data
    // ([S, n_heads, head_size]). Either way it does not match RoPEFusionFlux
    // (which needs rank-4 x in the interleaved layout), so the RoPE stays as
    // discrete elementwise ops.
    //
    // } else if (mode == TYPE_NORMAL) {
    //     auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    //     auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    //     auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    //     auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    //     auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
    //     Output<Node> even_slice;
    //     Output<Node> odd_slice;
    //     // stateful data is rank 3 (unsqueeze at axis 3), stateless is rank 4 (axis 4)
    //     int32_t unsqueeze_dim = context.is_stateful() ? 3 : 4;
    //     even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, neg_one);
    //     odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, neg_one);
    //
    //     Output<Node> first_half =
    //         std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
    //                                                std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
    //     Output<Node> second_half =
    //         std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
    //                                           std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));
    //
    //     first_half = std::make_shared<ov::op::v0::Unsqueeze>(first_half,
    //                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
    //     second_half = std::make_shared<ov::op::v0::Unsqueeze>(second_half,
    //                                                           ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
    //     auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, unsqueeze_dim);
    //
    //     auto data_shape = ov::op::v0::Constant::create(
    //         ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
    //     res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
    else if (mode == TYPE_NEOX) {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, -1);
    } else if (mode == TYPE_IMROPE) {
        int64_t n_dims = data_node->get_output_partial_shape(0)[3].get_length();
        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                    std::vector<int64_t>{1, -1, 1, (n_dims >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
    }

    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
