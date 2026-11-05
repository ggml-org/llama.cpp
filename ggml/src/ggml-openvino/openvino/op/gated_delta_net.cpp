#include "gated_delta_net.hpp"

#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
#include "recurrent_cache_update.hpp"
#endif
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/less.hpp>
#include <openvino/op/loop.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

static OutputVector translate_gated_delta_net_ref(const NodeContext & context);

static bool match_gdn_l2_norm(const Output<Node> & normalized, Output<Node> & input, float & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    const auto scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_node_shared_ptr());
    if (!scale) {
        return false;
    }
    const auto factor = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_input_node_shared_ptr(1));
    const auto rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_input_node_shared_ptr(0));
    if (!factor || ov::shape_size(factor->get_shape()) != 1 || !rms) {
        return false;
    }
    const auto x = rms->input_value(0);
    const auto & shape = x.get_partial_shape();
    if (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_length() <= 0 || normalized.get_partial_shape() != shape) {
        return false;
    }
    const float dim = static_cast<float>(shape[3].get_length());
    if (factor->cast_vector<float>()[0] != 1.0f / std::sqrt(dim)) {
        return false;
    }
    const auto reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_input_node_shared_ptr(1));
    if (!reciprocal) {
        return false;
    }
    const auto one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_input_node_shared_ptr(0));
    const auto root = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_input_node_shared_ptr(1));
    if (!one || ov::shape_size(one->get_shape()) != 1 || one->cast_vector<float>()[0] != 1.0f || !root) {
        return false;
    }
    const auto add = ov::as_type_ptr<ov::op::v1::Add>(root->get_input_node_shared_ptr(0));
    if (!add) {
        return false;
    }
    const auto mean = ov::as_type_ptr<ov::op::v1::ReduceMean>(add->get_input_node_shared_ptr(0));
    const auto rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(add->get_input_node_shared_ptr(1));
    if (!mean || !mean->get_keep_dims() || !rms_eps || ov::shape_size(rms_eps->get_shape()) != 1) {
        return false;
    }
    const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(mean->get_input_node_shared_ptr(1));
    const auto square = ov::as_type_ptr<ov::op::v1::Multiply>(mean->get_input_node_shared_ptr(0));
    if (!axes || axes->cast_vector<int64_t>() != std::vector<int64_t>{-1} || !square ||
        square->input_value(0) != x || square->input_value(1) != x) {
        return false;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_vector<float>()[0];
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return false;
    }
    input = x;
    return true;
}

#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
static OutputVector translate_gdn_snapshots(const Output<Node> & q,
                                          const Output<Node> & k,
                                          const Output<Node> & v,
                                          const Output<Node> & state,
                                          const Output<Node> & g,
                                          const Output<Node> & beta,
                                          int64_t features,
                                          bool norm,
                                          float q_eps,
                                          float k_eps,
                                          int64_t snapshot_count) {
    // Standalone/unmatched graphs have no full-model cache controls. Use a temporary
    // bank in GGML [head,value,key] order instead of extending ordinary GDN semantics.
    auto i64 = [](const Shape & shape, const std::vector<int64_t> & values) {
        return ov::op::v0::Constant::create(ov::element::i64, shape, values);
    };
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(q, ov::element::i64);
    auto batch = std::make_shared<ov::op::v8::Gather>(shape, i64({1}, {0}), i64({}, {0}));
    auto tokens = std::make_shared<ov::op::v8::Gather>(shape, i64({1}, {1}), i64({}, {0}));
    auto initial = std::make_shared<ov::op::v1::Reshape>(state, i64({4}, {1, 1, -1, features}), false);
    auto spare_rows = std::make_shared<ov::op::v1::Multiply>(batch, i64({1}, {snapshot_count - 1}));
    auto spare_shape = std::make_shared<ov::op::v0::Concat>(
        OutputVector{i64({2}, {1, 1}), spare_rows, i64({1}, {features})}, 0);
    auto spare = std::make_shared<ov::op::v3::Broadcast>(
        ov::op::v0::Constant::create(state.get_element_type(), {}, {0}), spare_shape);
    auto cache = std::make_shared<ov::op::v0::Concat>(OutputVector{initial, spare}, 2);
    auto rows = std::make_shared<ov::op::v4::Range>(
        i64({}, {0}), std::make_shared<ov::op::v0::Squeeze>(batch), i64({}, {1}), ov::element::i64);
    auto active = std::make_shared<ov::op::v1::Reshape>(rows, i64({4}, {1, 1, 1, -1}), false);
    auto control = std::make_shared<ov::op::v1::Reshape>(
        std::make_shared<ov::op::v0::Concat>(OutputVector{i64({1}, {0}), batch, batch, i64({1}, {0})}, 0),
        i64({4}, {1, 1, 1, 4}), false);
    auto update = std::make_shared<ov::op::internal::RecurrentGatedDeltaNetUpdate>(
        OutputVector{q, k, v, cache, g, beta, active, i64({1, 1, 1, 0}, {}),
                     i64({1, 1, 1, 1}, {0}), i64({1, 1, 1, 1}, {0}), i64({1, 1, 1, 2}, {0, 0}), control},
        norm, q_eps, k_eps, snapshot_count);
    auto valid_rows = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v1::Minimum>(tokens, i64({1}, {snapshot_count})), batch);
    auto snapshots = std::make_shared<ov::op::v8::Slice>(
        update->output(1), i64({1}, {0}), valid_rows, i64({1}, {1}), i64({1}, {2}));
    return {update->output(0), snapshots};
}
#endif

static OutputVector translate_gdn(const NodeContext & context, const OutputVector * cache_inputs) {
    auto v_shape = context.get_input_shape(2).to_shape();  // [B, T, H_v, S_v]
    auto q_shape = context.get_input_shape(0).to_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    const int64_t H_v = v_shape[2];
    const int64_t S_v = v_shape[3];
    const int64_t H_k = q_shape[2];
    const int64_t snapshot_count = context.get_output_op_params()[0];
    // const int64_t S_k = q_shape[3];

    auto q = context.get_input(0);
    auto k = context.get_input(1);
    auto v = process_view_input(context, 2, H_v * S_v, 3);
    auto g = context.get_input(3);
    auto beta = context.get_input(4);
    auto state = context.get_input(5);

    Output<Node> raw_q, raw_k;
    float q_eps = 1e-6f, k_eps = 1e-6f;
    const bool fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        match_gdn_l2_norm(q, raw_q, q_eps) && match_gdn_l2_norm(k, raw_k, k_eps);
    if (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    if (context.get_op_case() == 1 && ggml_openvino_gdn_prescale_enabled()) {
        // the output is linear in q and the state does not depend on q; the next rms_norm removes the scale
        q = std::make_shared<ov::op::v1::Multiply>(
            q, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {ggml_openvino_gdn_q_scale()}));
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    if (H_v != H_k) {
        const int64_t repeat = H_v / H_k;
        auto repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    if (context.get_view_input_size(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto v_shape = context.get_input_shape(2).to_shape();
        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), true);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    auto state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    if (!cache_inputs && snapshot_count == 1) {
        state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);
    }

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    if (context.has_input("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        const auto & g_ps = g.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_length() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        const int64_t n_tokens = g_ps[1].get_length();
        std::vector<int64_t> positions(n_tokens);
        std::iota(positions.begin(), positions.end(), 0);
        auto valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_tokens}, positions),
            context.get_input("chunk_valid_len"));
        auto mask = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, mask);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, mask);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
    if (cache_inputs) {
        OutputVector args{q, k, v, (*cache_inputs)[0], g, beta};
        args.insert(args.end(), cache_inputs->begin() + 1, cache_inputs->end());
        auto fused = std::make_shared<ov::op::internal::RecurrentGatedDeltaNetUpdate>(
            args, fuse_qk_l2norm, q_eps, k_eps, snapshot_count);
        fused->set_friendly_name(context.get_name() + "/RecurrentGatedDeltaNetUpdate");
        // The matched state VIEW is consumed only by the overridden full-cache
        // writeback. Route that cache directly instead of materializing snapshots.
        if (context.get_output_names().size() == 3) {
            return {fused->output(0), fused->output(0), fused->output(1)};
        }
        return {fused->output(0), fused->output(1)};
    }
#endif

    Output<Node> attn_4d, states;
    if (snapshot_count > 1) {
#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
        auto outputs = translate_gdn_snapshots(q, k, v, state, g, beta, H_v * S_v * q_shape[3],
                                               fuse_qk_l2norm, q_eps, k_eps, snapshot_count);
        attn_4d = outputs[0];
        states = outputs[1];
#else
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "GATED_DELTA_NET snapshot/rollback requires a patched OpenVINO package");
#endif
    } else {
        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(
            q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
        attn_4d = gdn->output(0);
        states = std::make_shared<ov::op::v1::Transpose>(gdn->output(1), state_perm);
    }

    if (snapshot_count == 1 && context.get_output_names().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        return rename_outputs_with_suffix({attn_4d, states}, context.get_name());
    }

    auto flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, false);
    auto new_state = std::make_shared<ov::op::v1::Reshape>(states, flat_shape_1d, false);
    auto packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn, new_state}, 0);
    auto out_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    auto res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, false);

    if (context.get_output_names().size() == 3) {
        auto flat_states = std::make_shared<ov::op::v1::Reshape>(
            states,
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, S_v * S_v * H_v}),
            false);
        return rename_outputs_with_suffix({res, attn_4d, flat_states}, context.get_name());
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_gated_delta_net(const NodeContext & context) {
    return translate_gdn(context, nullptr);
}

#ifdef GGML_OPENVINO_HAS_CACHE_UPDATE
OutputVector translate_recurrent_gated_delta_net(const NodeContext & context, const OutputVector & cache_inputs) {
    return translate_gdn(context, &cache_inputs);
}
#endif

static OutputVector translate_gated_delta_net_ref(const NodeContext & context) {
    num_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    auto q = process_view_input_new(context, 0);
    auto k = process_view_input_new(context, 1);
    auto v = process_view_input_new(context, 2);
    auto g = process_view_input_new(context, 3);
    auto beta = process_view_input_new(context, 4);
    auto state = process_view_input_new(context, 5);

    auto v_shape = context.get_input_shape(2).to_shape();  // [B, T, H_v, S_v]
    auto q_shape = context.get_input_shape(0).to_shape();  // [B, T, H_k, S_k]
    auto g_shape = context.get_input_shape(3).to_shape();  // [B, T, H_v, 1 or S_v]

    const int64_t B = v_shape[0];
    const int64_t T = v_shape[1];
    const int64_t H_v = v_shape[2];
    const int64_t S_v = v_shape[3];
    const int64_t H_k = q_shape[2];
    const bool kda = (g_shape[3] == (size_t) S_v);

    const int64_t rq1 = H_v / H_k;  // head repeat factor
    const float scale = 1.0f / std::sqrt((float) S_v);

    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    auto perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    auto k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    auto v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    auto g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    auto beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    if (rq1 > 1) {
        auto q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        auto k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        auto bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        auto q_bcast =
            std::make_shared<ov::op::v3::Broadcast>(q_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto k_bcast =
            std::make_shared<ov::op::v3::Broadcast>(k_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        auto perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        auto q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        auto k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        auto new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, false);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, false);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    auto merge_bh = [&](const ov::Output<ov::Node> & x, int64_t last_dim) {
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, last_dim});
        return std::make_shared<ov::op::v1::Reshape>(x, shape, false);
    };

    auto q_m = merge_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    auto k_m = merge_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    auto v_m = merge_bh(v_t, S_v);            // [B*H_v, T, S_v]
    auto g_m = merge_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    auto beta_m = merge_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    auto state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    auto state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, false);

    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<float>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    auto body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    auto body_cond_out = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});

    // Gather current token from invariant inputs using iteration counter
    auto q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    auto b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    auto q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    auto k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    auto v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    auto g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    auto b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    auto exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    auto exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    auto state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    auto k_col = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    auto sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_col, false, false);  // [B*H_v, S_v, 1]
    auto sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    auto v_minus_sk = std::make_shared<ov::op::v1::Subtract>(v_cur, sk_sq);              // [B*H_v, S_v]
    auto delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    auto delta_col = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    auto k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    auto outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_col, k_row, false, false);  // [B*H_v, S_v, S_v]
    auto state_updated = std::make_shared<ov::op::v1::Add>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    auto q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    auto sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_col, false, false);  // [B*H_v, S_v, 1]
    auto sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    auto attn_out = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    auto attn_out_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_out, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond_out, state_updated, attn_out_unsq},
        ov::ParameterVector{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    auto exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->set_merged_input(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->set_invariant_input(body_q, q_m);
    loop->set_invariant_input(body_k, k_m);
    loop->set_invariant_input(body_v, v_m);
    loop->set_invariant_input(body_g, g_m);
    loop->set_invariant_input(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    auto final_state_out = loop->get_iter_value(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    auto attn_concat_out = loop->get_concatenated_slices(attn_out_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    auto attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    auto attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_out, attn_4d_shape, false);
    auto attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    auto flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    auto attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, false);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    auto state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    auto state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_out, state_4d_shape, false);
    if (context.get_output_names().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        return rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    auto state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, false);

    // Concat [attn | state] and reshape to final output
    auto packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn_1d, state_1d}, 0);
    auto out_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    auto res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
