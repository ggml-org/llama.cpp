#include "kv_state_seq_axis.h"

#include <memory>
#include <openvino/core/graph_util.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {

const std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    static const std::vector<int64_t> perm{0, 2, 1, 3};
    return perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
bool can_move_seq_axis(const ov::PartialShape & shape) {
    return shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(const std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_input_node_shared_ptr(0));
    if (!concat || concat->get_input_size() != 2 || concat->get_axis() != 1) {
        return nullptr;
    }
    auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(concat->get_input_node_shared_ptr(0));
    if (!read_value || read_value->get_variable() != assign->get_variable()) {
        return nullptr;
    }
    if (!can_move_seq_axis(read_value->get_output_partial_shape(0))) {
        return nullptr;
    }
    return concat;
}

}  // namespace

bool KVStateSeqAxis::run_on_model(const std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    for (const auto & op : model->get_ops()) {
        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    bool changed = false;
    for (const auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        if (!concat) {
            continue;
        }
        auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(concat->get_input_node_shared_ptr(0));

        auto variable = read_value->get_variable();
        auto info = variable->get_info();
        const auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_value->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->input_value(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        for (auto & reader : readers) {
            auto * node = reader.get_node();
            if (ov::is_type<ov::op::v6::Assign>(node)) {
                continue;
            }
            bool dropped = false;
            if (auto * transpose = ov::as_type<ov::op::v1::Transpose>(node)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
                if (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = true;
                }
            }
            if (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = true;
    }

    return changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
