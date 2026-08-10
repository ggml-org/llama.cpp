#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"

#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/topk.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML_OP_ARGMAX: src0 is a matrix [ne0, ne1]; the result is a 1-D I32 tensor of ne1 entries
// holding, for each row, the index of the maximum along ne0. ggml shapes arrive reversed here, so
// ne0 is the last OV axis and ne1 is axis 2. TopK(k=1, axis=last, MAX) yields exactly that index,
// then reshape to the [1,1,1,ne1] layout the decoder expects for a 1-D ggml output.
//
// Note ggml's argmax returns the FIRST maximum on a tie, while TopK's tie-breaking is not specified
// to match. Ties on real logits are vanishingly rare and none were observed, but the two are not
// proven equivalent.
OutputVector translate_argmax(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    auto k = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto topk = std::make_shared<ov::op::v11::TopK>(input,
                                                   k,
                                                   3,
                                                   ov::op::v11::TopK::Mode::MAX,
                                                   ov::op::v11::TopK::SortType::SORT_VALUES,
                                                   context.get_output_type(),
                                                   false);

    // ne1 is the dynamic row count, so build the target shape at runtime.
    auto leading = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 1, 1});
    auto rows = get_dimensions(input.get_node_shared_ptr(), {2});
    auto target_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{leading, rows}, 0);
    auto res = std::make_shared<ov::op::v1::Reshape>(topk->output(1), target_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
