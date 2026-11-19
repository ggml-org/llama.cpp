#include "keep_sum_square_fp32.h"

#include <map>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/visibility.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <unordered_set>
#include <vector>

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
// from transformations/rt_info/disable_precision_conversion.hpp, which the OpenVINO runtime package does not ship
TRANSFORMATIONS_API void disable_conversion(const std::shared_ptr<Node> & node, const element::Type & to);
}  // namespace ov

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {
// the only consumer of node's output 0, or nullptr
std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node> & node) {
    const auto targets = node->output(0).get_target_inputs();
    return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
}

bool is_scalar_constant(const ov::Output<ov::Node> & output) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 1;
}

// Marks the island to stay in f32 and puts a no-op Convert on every edge into or out of it.
// MarkSugraphsToKeepInMixedPrecision spreads the mark through elementwise/reshape/slice/concat ops in both
// directions and stops at a Convert. Without the Converts it reaches most of the graph, which then runs in
// f32 with a Reorder at every boundary and makes decode slow.
void keep_island_in_fp32(const std::vector<std::shared_ptr<ov::Node>> & island) {
    std::unordered_set<const ov::Node *> members;
    for (const auto & node : island) {
        members.insert(node.get());
        ov::disable_conversion(node, ov::element::f16);
    }

    std::map<ov::Output<ov::Node>, std::shared_ptr<ov::Node>> converts_in;
    for (const auto & node : island) {
        for (auto input : node->inputs()) {
            const auto source = input.get_source_output();
            const auto * producer = source.get_node();
            if (members.count(producer) || ov::is_type<ov::op::v0::Constant>(producer) ||
                ov::is_type<ov::op::v0::Convert>(producer)) {
                continue;
            }
            auto & convert = converts_in[source];
            if (!convert) {
                convert = std::make_shared<ov::op::v0::Convert>(source, source.get_element_type());
            }
            input.replace_source_output(convert);
        }
    }

    for (const auto & node : island) {
        for (const auto & output : node->outputs()) {
            std::shared_ptr<ov::Node> convert;
            for (auto target : output.get_target_inputs()) {
                const auto * consumer = target.get_node();
                if (members.count(consumer) || ov::is_type<ov::op::v0::Convert>(consumer)) {
                    continue;
                }
                if (!convert) {
                    convert = std::make_shared<ov::op::v0::Convert>(output, output.get_element_type());
                }
                target.replace_source_output(convert);
            }
        }
    }
}
}  // namespace

KeepSumSquareFP32::KeepSumSquareFP32() {
    auto square = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>();
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum>({square, ov::pass::pattern::any_input()});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto sum = m.get_match_root();
        auto square_node = m.get_pattern_value_map().at(square).get_node_shared_ptr();
        if (square_node->input_value(0) != square_node->input_value(1)) {
            return false;
        }
        std::vector<std::shared_ptr<ov::Node>> island{square_node, sum};

        // optional ggml_scale (e.g. 1/(n-1) in a std), then Sqrt brings the result back into f16 range
        auto next = single_consumer(sum);
        if (ov::is_type<ov::op::v1::Multiply>(next) &&
            (is_scalar_constant(next->input_value(0)) || is_scalar_constant(next->input_value(1)))) {
            island.push_back(next);
            next = single_consumer(next);
        }
        if (ov::is_type<ov::op::v0::Sqrt>(next)) {
            island.push_back(next);

            // a * s / ||x|| (Gemma 3n altup magnitude matching): a * s overflows f16 before the Divide
            std::vector<std::shared_ptr<ov::Node>> norms{next};
            for (size_t i = 0; i < norms.size(); ++i) {
                for (const auto & target : norms[i]->output(0).get_target_inputs()) {
                    auto consumer = target.get_node()->shared_from_this();
                    if (ov::is_type<ov::op::v0::Tile>(consumer) || ov::is_type<ov::op::v3::Broadcast>(consumer)) {
                        norms.push_back(consumer);
                    } else if (ov::is_type<ov::op::v1::Divide>(consumer) && target.get_index() == 1) {
                        island.push_back(consumer);
                        auto numerator = consumer->get_input_node_shared_ptr(0);
                        if (ov::is_type<ov::op::v1::Multiply>(numerator)) {
                            island.push_back(numerator);
                        }
                    }
                }
            }
        }

        keep_island_in_fp32(island);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::KeepSumSquareFP32"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
