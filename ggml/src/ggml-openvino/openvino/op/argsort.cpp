#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"

#include <openvino/frontend/exception.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/topk.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_argsort(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = process_view_input_new(context, 0);

    const int32_t order = context.get_output_op_params()[0];

    ov::op::v11::TopK::Mode mode;
    switch (order) {
    case GGML_SORT_ORDER_ASC:
        mode = ov::op::v11::TopK::Mode::MIN;
        break;
    case GGML_SORT_ORDER_DESC:
        mode = ov::op::v11::TopK::Mode::MAX;
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported GGML_OP_ARGSORT order: ", order);
    }

    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(input.get_node_shared_ptr(), {3}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

    auto topk = std::make_shared<ov::op::v11::TopK>(input, k, 3, mode, ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    context.get_output_type(), false);

    return rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
