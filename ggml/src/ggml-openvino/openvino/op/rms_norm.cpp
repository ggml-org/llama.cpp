#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/squeeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rms_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> input_node;
    if (op_case == 3) {
        input_node = context.get_input(0);
    } else if (op_case == 1) {
        input_node = process_view_input_new(context, 0);
    } else if (op_case == 2 || op_case == 3) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        const auto attention_rows_name = "gdn_attention_rows_" + context.get_view_input_src_name(0, 0);
        if (context.has_input(attention_rows_name)) {
            state_end = context.get_input(attention_rows_name);
        } else if (context.has_input("s_copy_active_slot_len")) {
            auto len = context.get_input("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } else {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        ov::Output<ov::Node> gdn_attn_output;
        if (context.has_input(attention_rows_name)) {
            auto indices = std::make_shared<ov::op::v4::Range>(
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                std::make_shared<ov::op::v0::Squeeze>(state_end),
                ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);
            gdn_attn_output = std::make_shared<ov::op::v8::Gather>(context.get_input(0), indices,
                ov::op::v0::Constant::create(ov::element::i64, {}, {2}));
        } else {
            gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
                context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
        }

        auto input_shape = context.get_input_shape(0).to_shape();
        ov::Output<ov::Node> sequences = ov::op::v0::Constant::create(ov::element::i64, {1}, {input_shape[0]});
        if (context.has_input("s_copy_active_slot_len")) {
            sequences = context.get_input("s_copy_active_slot_len");
        }
        auto shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            sequences, ov::op::v0::Constant::create(ov::element::i64, {3},
                std::vector<int64_t>{-1, (int64_t) input_shape[2], (int64_t) input_shape[3]})}, 0);
        input_node = std::make_shared<ov::op::v1::Reshape>(gdn_attn_output, shape, false);

    } else {
        input_node = process_view_input_new(context, 0);
    }

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    if (op_case == 3 && ggml_openvino_gdn_prescale_enabled()) {
        // input is the GDN output pre-scaled by ggml_openvino_gdn_q_scale(): rms_norm(s*x, s*s*eps) == rms_norm(x, eps)
        const float q_scale = ggml_openvino_gdn_q_scale();
        eps *= q_scale * q_scale;
    }

    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
