#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/sqrt.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rms_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto op_case = context.get_op_case();

    ov::Output<ov::Node> input_node;
    if (op_case == 1) {
        input_node = process_view_input_new(context, 0);
    } else if (op_case == 2) {
        auto ssm_state_size = context.get_ssm_state_size();
        // The GDN op packs [attn | new_state] along the row axis; the state occupies the last
        // ssm_state_size * n_seqs rows. Slice it off (scaling by the active sequence count) to keep
        // just the attention output.
        ov::Output<ov::Node> state_end;
        if (context.has_input("s_copy_active_slot_len")) {
            auto len = context.get_input("s_copy_active_slot_len");
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), len);
            state_end = std::make_shared<ov::op::v0::Negative>(state_rows);
        } else {
            state_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size});
        }
        auto gdn_attn_output = std::make_shared<ov::op::v8::Slice>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), state_end,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));

        auto input_shape = context.get_input_shape(0).to_shape();
        input_node = std::make_shared<ov::op::v1::Reshape>(
            gdn_attn_output,
            ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) input_shape[2], (int64_t) input_shape[3]}),
            false);

    } else {
        input_node = process_view_input_new(context, 0);
    }
    auto square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

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
