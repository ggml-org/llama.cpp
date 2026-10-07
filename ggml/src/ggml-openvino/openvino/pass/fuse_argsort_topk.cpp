#include "fuse_argsort_topk.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

FuseArgsortTopK::FuseArgsortTopK() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v11::TopK>();

    const auto callback = [](ov::pass::pattern::Matcher & m) {
        auto topk = ov::as_type_ptr<ov::op::v11::TopK>(m.get_match_root());
        if (!topk->output(0).get_target_inputs().empty() || topk->output(1).get_target_inputs().empty() ||
            topk->get_sort_type() != ov::op::v11::TopK::SortType::SORT_VALUES) {
            return false;
        }

        const auto shape = topk->get_output_partial_shape(1);
        if (shape.rank().is_dynamic()) {
            return false;
        }
        const int64_t rank = shape.rank().get_length();
        const int64_t axis = topk->get_axis();
        if (shape[axis].is_dynamic()) {
            return false;
        }

        int64_t prefix = -1;
        for (const auto & input : topk->output(1).get_target_inputs()) {
            const auto * slice = ov::as_type<ov::op::v8::Slice>(input.get_node());
            if (!slice || input.get_index() != 0 || slice->get_input_size() != 5) {
                return false;
            }
            int64_t values[4];
            for (size_t i = 0; i < 4; ++i) {
                auto value = ov::as_type_ptr<ov::op::v0::Constant>(slice->get_input_node_shared_ptr(i + 1));
                if (!value || ov::shape_size(value->get_shape()) != 1) {
                    return false;
                }
                values[i] = value->cast_vector<int64_t>()[0];
            }
            const int64_t slice_axis = values[3] < 0 ? values[3] + rank : values[3];
            if (values[0] != 0 || values[2] != 1 || slice_axis != axis || values[1] <= 0 ||
                values[1] >= shape[axis].get_length() || (prefix != -1 && prefix != values[1])) {
                return false;
            }
            prefix = values[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->set_argument(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
