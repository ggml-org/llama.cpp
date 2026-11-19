#include "fuse_moe_router.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {

bool constant_is(const ov::Output<ov::Node> & output, const std::vector<int64_t> & values) {
    auto node = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return node && node->get_element_type().is_integral_number() && node->cast_vector<int64_t>() == values;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using namespace ov::pass::pattern;
    using namespace ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::TopK>({softmax, any_input()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_input(), any_input(), any_input(), any_input()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_input()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_input()});
    auto data = wrap_type<v0::Squeeze>({probs, any_input()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_input(), any_input()});
    auto ids_count = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_input(), any_input()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_count});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_input()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_input()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_input()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_input()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_input()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    const auto callback = [=](Matcher & m) {
        const auto & pm = m.get_pattern_value_map();
        const auto node = [&](const std::shared_ptr<ov::Node> & p) { return pm.at(p).get_node_shared_ptr(); };
        const auto input_is = [&](const std::shared_ptr<ov::Node> & p, size_t i, const std::vector<int64_t> & v) {
            return constant_is(node(p)->input_value(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(node(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(node(softmax));
        auto tk = ov::as_type_ptr<v11::TopK>(node(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(node(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(node(clamp));
        const auto shape = mm->get_output_partial_shape(0);
        if (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_input_partial_shape(0).rank() != 4 ||
            mm->get_input_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::TopK::Mode::MAX ||
            tk->get_sort_type() != v11::TopK::SortType::SORT_VALUES || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            return false;
        }
        const int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(node(ids)->get_input_node_shared_ptr(2));
        if (!end || !end->get_element_type().is_integral_number() || ov::shape_size(end->get_shape()) != 1) {
            return false;
        }
        const int64_t k = end->cast_vector<int64_t>()[0];
        const auto sorted_shape = tk->get_output_partial_shape(1);
        if (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !input_is(probs, 1, {1, -1, experts, 1}) ||
            !input_is(data, 1, {0}) || !input_is(ids_2d, 1, {0, 1}) ||
            !input_is(weights_4d, 1, {0}) || !input_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(node(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(node(weights))->get_special_zero() ||
            !input_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(node(gather))->get_batch_dims() != 1 ||
            !input_is(data_batch, 1, {0}) || !input_is(data_batch, 2, {0}) ||
            !input_is(ids_count, 1, {1}) || !input_is(ids_count, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(node(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(node(ids_count))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(node(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(node(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!input_is(sum, 1, {-1}) && !input_is(sum, 1, {3})) ||
            !input_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(node(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        if (!(limit->get_min() <= 0.5 * double(k) / experts && limit->get_max() >= 2.0)) {
            return false;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        for (const auto & input : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(input.get_node()->shared_from_this());
            if (!slice || input.get_index() != 0 || slice->get_input_size() != 5 ||
                !constant_is(slice->input_value(1), {0}) || !constant_is(slice->input_value(2), {k}) ||
                !constant_is(slice->input_value(3), {1}) ||
                (!constant_is(slice->input_value(4), {3}) && !constant_is(slice->input_value(4), {-1}))) {
                return false;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->input_value(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->input_value(1), false, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::TopK>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES, tk->get_index_element_type(), false);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, node(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_node(node(norm), weights_out);
        for (const auto & slice : slices) {
            ov::replace_node(slice, ids_out);
        }
        return true;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
