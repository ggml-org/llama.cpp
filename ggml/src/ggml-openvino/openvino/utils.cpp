#include "utils.hpp"

#include <ctime>
#include <memory>
#include <openvino/op/gather.hpp>
#include <openvino/op/shape_of.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    int dim = nb.size() - 1;
    size_t bytes = nb[dim];
    for (int i = dim; i > 0; i--) {
        bytes *= ne[i];
        if (bytes != nb[i - 1]) {
            return i;
        }
    }
    return 0;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix) {
    for (const auto& output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
    }
    return outputs;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
