#include "fuse_moe_compressed.h"

#include <cstring>
#include <limits>
#include <set>
#include <memory>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/pattern/op/optional.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "../op/gather_matmul.hpp"
#include "../op/moe_compressed.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {

struct dequant_inputs {
    ov::Output<ov::Node> weight;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    bool has_zp = false;
    bool ok = false;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(const ov::Output<ov::Node> & b) {
    dequant_inputs res;

    auto node = b.get_node_shared_ptr();
    while (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node)) {
        node = node->get_input_node_shared_ptr(0);
    }

    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(node);
    if (!mul) {
        return res;
    }
    res.scale = mul->input_value(1);

    auto lhs = mul->get_input_node_shared_ptr(0);
    if (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        auto zp_node = sub->get_input_node_shared_ptr(1);
        while (ov::is_type<ov::op::v0::Convert>(zp_node)) {
            zp_node = zp_node->get_input_node_shared_ptr(0);
        }
        res.zp = zp_node->output(0);
        res.has_zp = true;
        lhs = sub->get_input_node_shared_ptr(0);
    }
    while (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_input_node_shared_ptr(0);
    }
    if (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        return res;
    }

    res.weight = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.weight.get_partial_shape().is_static();
    return res;
}

size_t logical_k(const ov::Shape & shape) {
    return shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> slice_experts_dim1(const ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_node_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    const auto & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    const auto & type = constant->get_element_type();

    size_t row_elems = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        row_elems *= shape[i];
    }
    const size_t row_bits = row_elems * type.bitwidth();
    OPENVINO_ASSERT(row_bits % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    const size_t row_bytes = row_bits / 8;

    ov::Shape out_shape = shape;
    out_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> out_data(shape[0] * static_cast<size_t>(end - begin) * row_bytes);
    const auto * src = static_cast<const uint8_t *>(constant->get_data_ptr());
    const size_t src_row_bytes = shape[1] * row_bytes;
    for (size_t e = 0; e < shape[0]; ++e) {
        std::memcpy(out_data.data() + e * static_cast<size_t>(end - begin) * row_bytes,
                   src + e * src_row_bytes + static_cast<size_t>(begin) * row_bytes,
                   static_cast<size_t>(end - begin) * row_bytes);
    }

    return std::make_shared<ov::op::v0::Constant>(type, out_shape, out_data.data());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    using namespace ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    auto hidden_gate_m = any_input();
    auto a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_input() });
    auto a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_input() });
    auto hidden_up_m = any_input();
    auto a_up_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_up_m, any_input() }) }),
          any_input() });

    auto gate_w_m = any_input();
    auto up_w_m = any_input();
    auto down_w_m = any_input();
    auto ids_gate_m = any_input();
    auto ids_up_m = any_input();
    auto ids_down_m = any_input();

    auto bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_input() });
    auto gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_input() }), any_input() }) });

    auto silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    auto bgm_up_m = wrap_type<ov::op::internal::GatherMatmul>({ a_up_m, up_w_m, ids_up_m, any_input() });
    auto up_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_up_m, any_input() }), any_input() }) });
    auto swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, up_u_m });

    auto d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_input() }) }),
          any_input() });
    auto bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_input() });
    auto down_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_down_m, any_input() }), any_input() }) });

    auto routing_m = any_input();
    auto weighted_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, routing_m });
    auto root_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_input() });

    const auto callback = [=](Matcher & m) {
        auto & pm = m.get_pattern_value_map();

        const auto gate = unwrap_dequant(pm.at(gate_w_m));
        const auto up = unwrap_dequant(pm.at(up_w_m));
        const auto down = unwrap_dequant(pm.at(down_w_m));
        if (!gate.ok || !up.ok || !down.ok) {
            return false;
        }

        const auto gate_shape = gate.weight.get_shape();
        const auto up_shape = up.weight.get_shape();
        const auto down_shape = down.weight.get_shape();
        if (gate_shape != up_shape || gate_shape.size() < 3 || down_shape.size() < 3) {
            return false;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        if (gate.has_zp != down.has_zp || gate_shape.size() != down_shape.size()) {
            return false;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        if (gate.has_zp) {
            static const std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            if (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(down.zp.get_element_type()) == 0) {
                return false;
            }
        }

        // Config holds a single group_size for all three projections.
        const auto group_of = [](const dequant_inputs & w) {
            const auto s = w.weight.get_shape();
            return s.size() == 4 ? s[3] : logical_k(s);
        };
        if (group_of(gate) != group_of(up) || group_of(gate) != group_of(down)) {
            return false;
        }

        // all three branches must route the same hidden state through the same experts
        if (pm.at(hidden_gate_m) != pm.at(hidden_up_m)) {
            return false;
        }

        auto ids = pm.at(ids_down_m);
        const auto ids_pshape = ids.get_partial_shape();
        if (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            return false;
        }
        const size_t top_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        auto routing = pm.at(routing_m);
        const auto routing_pshape = routing.get_partial_shape();
        if (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_length() != 4 ||
            routing_pshape[3] != 1) {
            return false;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        if (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        if (routing.get_partial_shape() != ids.get_partial_shape()) {
            return false;
        }

        const size_t down_k = logical_k(down_shape);
        const auto down_scale_shape = down.scale.get_shape();
        const size_t down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_shape);
        config.inter_size = gate_shape[1];
        config.num_expert = gate_shape[0];
        config.num_shared_expert = 0;
        config.top_k = top_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<size_t>::max() : down_k / down_groups;
        config.has_batch_dim = true;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.out_type = ov::element::dynamic;

        auto absent_zp = [] {
            auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            return zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        const auto a_transpose = pm.at(a_gate_m).get_node_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->input_value(1));

        const ov::OutputVector args = {
            hidden, routing, ids,
            gate.weight,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            up.weight,    up.scale,     up.has_zp ? up.zp : absent_zp(),
            down.weight,  down.scale,   down.has_zp ? down.zp : absent_zp(),
        };

        auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> result = moe->output(0);
        const auto root_type = m.get_match_root()->get_output_element_type(0);
        if (result.get_element_type() != root_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, root_type);
        }

        result.get_node_shared_ptr()->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_node_shared_ptr());
        ov::replace_node(m.get_match_root(), result.get_node_shared_ptr());
        register_new_node(moe);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(root_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
    using namespace ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    auto hidden_m = any_input();
    auto a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_input() });
    auto a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_input() });

    auto gate_up_w_m = any_input();
    auto ids_gate_up_m = any_input();
    auto bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_up_w_m, ids_gate_up_m, any_input() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    auto gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_input() });
    auto gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_input() }) });

    auto gate_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_input(), any_input(), any_input(), any_input() });
    auto up_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_input(), any_input(), any_input(), any_input() });
    auto gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_slice_m });
    auto geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, up_slice_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    auto d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_input() }), any_input() }) }),
          any_input() });
    auto down_w_m = any_input();
    auto ids_down_m = any_input();
    auto bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_input() });
    // Same two shapes as the gate/up branch above.
    auto down_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_down_m }), any_input() });
    auto down_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ down_t_m, any_input() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    auto down_scale_m = any_input();
    auto down_scaled_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, down_scale_m });

    auto routing_m = any_input();
    auto weighted_m = wrap_type<ov::op::v1::Multiply>({ down_scaled_m, routing_m });
    auto root_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_input() });

    const auto callback = [=](Matcher & m) {
        auto & pm = m.get_pattern_value_map();

        const auto gate_up = unwrap_dequant(pm.at(gate_up_w_m));
        const auto down = unwrap_dequant(pm.at(down_w_m));
        if (!gate_up.ok || !down.ok) {
            return false;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        if (!gate_up.has_zp || !down.has_zp) {
            return false;
        }

        const auto gate_up_shape = gate_up.weight.get_shape();
        const auto down_shape = down.weight.get_shape();
        if (gate_up_shape.size() < 3 || down_shape.size() < 3 || gate_up_shape.size() != down_shape.size()) {
            return false;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        static const std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        if (int_types.count(gate_up.weight.get_element_type()) == 0 ||
            int_types.count(down.weight.get_element_type()) == 0 ||
            int_types.count(gate_up.zp.get_element_type()) == 0 ||
            int_types.count(down.zp.get_element_type()) == 0) {
            return false;
        }

        const auto group_of = [](const dequant_inputs & w) {
            const auto s = w.weight.get_shape();
            return s.size() == 4 ? s[3] : logical_k(s);
        };
        if (group_of(gate_up) != group_of(down)) {
            return false;
        }

        auto gelu_node = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_node_shared_ptr());
        if (!gelu_node || gelu_node->get_approximation_mode() != ov::op::GeluApproximationMode::ERF) {
            return false;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        auto gate_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(gate_slice_m).get_node_shared_ptr());
        auto up_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(up_slice_m).get_node_shared_ptr());
        auto gate_end_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_slice->input_value(2).get_node_shared_ptr());
        auto up_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(up_slice->input_value(1).get_node_shared_ptr());
        if (!gate_end_c || !up_begin_c) {
            return false;
        }
        const auto gate_end = gate_end_c->cast_vector<int64_t>().at(0);
        const auto up_begin = up_begin_c->cast_vector<int64_t>().at(0);
        if (gate_end != up_begin || gate_end <= 0 || gate_end >= static_cast<int64_t>(gate_up_shape[1])) {
            return false;
        }
        const int64_t mid = gate_end;
        const int64_t full = static_cast<int64_t>(gate_up_shape[1]);

        auto ids = pm.at(ids_down_m);
        const auto ids_pshape = ids.get_partial_shape();
        if (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            return false;
        }
        const size_t top_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        const auto routing_in = pm.at(routing_m);
        const auto routing_pshape = routing_in.get_partial_shape();
        const auto routing_rank = routing_pshape.rank();
        if (routing_rank.is_dynamic() || (routing_rank.get_length() != 4 && routing_rank.get_length() != 3) ||
            routing_pshape[routing_rank.get_length() - 1] != 1) {
            return false;
        }
        const bool batchless = routing_rank.get_length() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        auto down_scale = pm.at(down_scale_m);
        if (down_scale.get_partial_shape() != routing_pshape) {
            return false;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_in, down_scale);
        if (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        if (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        if (routing.get_partial_shape() != ids.get_partial_shape()) {
            return false;
        }

        const size_t down_k = logical_k(down_shape);
        const auto down_scale_shape = down.scale.get_shape();
        const size_t down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        const ov::Output<ov::Node> gate_weight = slice_experts_dim1(gate_up.weight, 0, mid);
        const ov::Output<ov::Node> up_weight = slice_experts_dim1(gate_up.weight, mid, full);
        const ov::Output<ov::Node> gate_scale = slice_experts_dim1(gate_up.scale, 0, mid);
        const ov::Output<ov::Node> up_scale = slice_experts_dim1(gate_up.scale, mid, full);
        const ov::Output<ov::Node> gate_zp = slice_experts_dim1(gate_up.zp, 0, mid);
        const ov::Output<ov::Node> up_zp = slice_experts_dim1(gate_up.zp, mid, full);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_up_shape);
        config.inter_size = static_cast<size_t>(mid);
        config.num_expert = gate_up_shape[0];
        config.num_shared_expert = 0;
        config.top_k = top_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<size_t>::max() : down_k / down_groups;
        config.has_batch_dim = true;
        config.has_zp = true;
        config.out_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        const auto a_transpose = pm.at(a_m).get_node_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->input_value(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        const ov::OutputVector args = {
            hidden,      routing,     ids,
            gate_weight, gate_scale,  gate_zp,
            up_weight,   up_scale,    up_zp,
            down.weight, down.scale,  down.zp,
        };

        auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        ov::Output<ov::Node> result = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        if (batchless) {
            result = std::make_shared<ov::op::v0::Squeeze>(
                result, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        const auto root_type = m.get_match_root()->get_output_element_type(0);
        if (result.get_element_type() != root_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, root_type);
        }
        if (result.get_partial_shape() != m.get_match_root()->get_output_partial_shape(0)) {
            return false;
        }

        result.get_node_shared_ptr()->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_node_shared_ptr());
        ov::replace_node(m.get_match_root(), result.get_node_shared_ptr());
        register_new_node(moe);
        return true;
    };

    register_matcher(
        std::make_shared<Matcher>(root_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
