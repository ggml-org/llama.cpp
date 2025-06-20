#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define GGML_ROPE_TYPE_NEOX 2

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {
float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
                              float dims[2]) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}
}  // namespace

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    ov::Output<Node> res;

    auto data_node = context.get_input(0);
    auto pos_node = context.get_input(1);
    pos_node = std::make_shared<ov::op::v0::Convert>(pos_node, ov::element::f32);

    auto permutation_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
    Output<Node> pos_node_reshaped = std::make_shared<ov::op::v1::Transpose>(pos_node, permutation_node);

    auto output_shape = context.get_output_shape(0);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t* op_params = context.get_output_op_params(0);
    const int n_dims = op_params[1];
    const int mode = op_params[2];
    const int n_ctx_orig = op_params[4];
    memcpy(&freq_base, op_params + 5, sizeof(float));
    memcpy(&freq_scale, op_params + 6, sizeof(float));
    memcpy(&ext_factor, op_params + 7, sizeof(float));
    memcpy(&attn_factor, op_params + 8, sizeof(float));
    memcpy(&beta_fast, op_params + 9, sizeof(float));
    memcpy(&beta_slow, op_params + 10, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    // TODO: corr_dims is not used in the current implementation
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    // TODO: GGML_OP_ROPE_BACK -> false
    bool forward = true;
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int64_t ne0 = output_shape[2].get_length();
    std::vector<float> factor(ne0 / 2);
    factor[0] = freq_scale;
    for (int64_t i = 1; i < ne0 / 2; i++) {
        factor[i] = theta_scale * factor[i - 1];
    }

    Output<Node> factor_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{factor.size()}, factor);
    if (context.get_input_size() == 3) {
        auto freq_factors_node = context.get_input(2);
        factor_node = std::make_shared<ov::op::v1::Divide>(factor_node, freq_factors_node);
    }

    auto half_last_dim = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {output_shape[2].get_length() / 2});
    Output<Node> input_shape_node = std::make_shared<ov::op::v0::Concat>(
        OutputVector{get_dimensions(data_node.get_node_shared_ptr(), {0, 1}), half_last_dim},
        0);
    Output<Node> factor_broadcasted_node = std::make_shared<ov::op::v3::Broadcast>(factor_node, input_shape_node);

    Output<Node> cos_factor_broadcasted_node = std::make_shared<ov::op::v0::Cos>(
        std::make_shared<ov::op::v1::Multiply>(factor_broadcasted_node, pos_node_reshaped));
    Output<Node> sin_factor_broadcasted_node = std::make_shared<ov::op::v0::Sin>(
        std::make_shared<ov::op::v1::Multiply>(factor_broadcasted_node, pos_node_reshaped));

    float mscale = attn_factor;
    Output<Node> mscale_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{mscale});
    Output<Node> mscale_sin_sign_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{mscale * sin_sign});
    Output<Node> cos_theta_node = std::make_shared<ov::op::v1::Multiply>(cos_factor_broadcasted_node, mscale_node);
    Output<Node> sin_theta_node = std::make_shared<ov::op::v1::Multiply>(sin_factor_broadcasted_node, mscale_node);

    if (!is_neox) {
        auto input_shape = context.get_input_shape(0);

        auto begin_even = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {0, 0, 0});
        auto begin_odd = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {0, 0, 1});
        auto end = std::make_shared<ov::op::v0::ShapeOf>(data_node);
        auto stride = ov::op::v0::Constant::create(ov::element::i64, Shape{3}, {1, 1, 2});
        auto even_slice = std::make_shared<ov::op::v8::Slice>(data_node, begin_even, end, stride);
        auto odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, begin_odd, end, stride);

        auto first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        auto second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, 2);
        auto shape_const = ov::op::v0::Constant::create(
            ov::element::i64,
            Shape{3},
            std::vector<int64_t>{-1, input_shape[1].get_length(), input_shape[2].get_length()});
        res = std::make_shared<ov::op::v1::Reshape>(stack, shape_const, false);
    } else {
        auto slice_node =
            std::make_shared<ov::op::v1::Split>(data_node,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}),
                                                2);
        Output<Node> slice_data_node_0 = slice_node->outputs()[0];
        Output<Node> slice_data_node_1 = slice_node->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, 2);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
