#pragma once

#include <openvino/op/shape_of.hpp>

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

void dump_ov_model(const std::shared_ptr<ov::Model> model);

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);

template <typename T>
std::vector<int> argsort_descend(const std::vector<T>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {
        return v[i1] > v[i2];
    });
    return idx;
}

template <typename T>
std::vector<T> sorted_descend(std::vector<T> v) {
    std::sort(v.begin(), v.end(), [](T a, T b) {
        return a > b;
    });
    return v;
}

template <typename T>
bool is_permuted(const std::vector<T>& strides) {
    for (size_t i = 0; i < strides.size() - 1; ++i) {
        if (strides[i] < strides[i + 1]) {
            return true;
        }
    }
    return false;
}

template <typename T>
std::vector<T> permute(const std::vector<T>& x, const std::vector<int>& perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    for (int i : perm) {
        result.push_back(x[i]);
    }
    return result;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape, const std::vector<int>& dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

namespace op {
template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    return {std::make_shared<T>(context.get_input(0), context.get_input(1))};
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
