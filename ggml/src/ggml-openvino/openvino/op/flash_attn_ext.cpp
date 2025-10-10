#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_flash_attn_ext(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto mask = context.get_input(3);

    float* params = reinterpret_cast<float*>(context.get_output_op_params(0));
    float scale         = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> mask_sliced, res;
    std::string mask_name = "KQ_mask_sliced";
    if (context.get_input_names()[3].find("swa") != std::string::npos) {
        mask_name = "KQ_mask_swa_sliced";
    }
    if (context.has_input(mask_name)) {
        mask_sliced = context.get_input(mask_name);
    } else {
        auto token_len = get_dimensions(q, {2});
        auto kv_len = get_dimensions(k.get_node_shared_ptr(), {2});

        auto zero_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {0,0});
        auto one_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {1,1});
        auto zero_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto two_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {1,2});

        auto stop = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{token_len, kv_len}, 0);
        mask_sliced =
            std::make_shared<ov::op::v8::Slice>(mask, zero_2d, stop, one_2d, axes);
        mask_sliced = std::make_shared<ov::op::v0::Unsqueeze>(mask_sliced, zero_1d);
    }

    if (mask_sliced.get_element_type() != ov::element::f16) {
        mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
    }

    auto tile_kv = [](int64_t q_batch, int64_t kv_batch, ov::Output<Node> kv, bool is_static) {
        int64_t factor = q_batch / kv_batch;
        if (factor > 1) {
            auto q_batch_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{q_batch});
            auto kv_batch_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{kv_batch});
            auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

            ov::Output<ov::Node> kv_broadcast_shape, kv_unsqueezed, new_kv_shape;
            if (is_static) {
                auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {1});
                kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

                auto kv_last_two_dims = get_dimensions(kv.get_node_shared_ptr(), {1, 2});
                kv_broadcast_shape =
                    std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv_batch_node, factor_node, kv_last_two_dims}, 0);
                new_kv_shape =
                    std::make_shared<ov::op::v0::Concat>(ov::OutputVector{q_batch_node, kv_last_two_dims}, 0);
            } else {
                auto one_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
                kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

                auto kv_last_two_dims = get_dimensions(kv.get_node_shared_ptr(), {2, 3});
                kv_broadcast_shape =
                    std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_1d, kv_batch_node, factor_node, kv_last_two_dims}, 0);
                new_kv_shape =
                    std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_1d, q_batch_node, kv_last_two_dims}, 0);
            }

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed, kv_broadcast_shape);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, false);
        }
        return kv;
    };

    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    k = tile_kv(q_shape[0], k_shape[0], k, context.is_static());
    v = tile_kv(q_shape[0], k_shape[0], v, context.is_static());

    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_sliced, scale_node, false);
    auto sdpa_f32 = std::make_shared<ov::op::v0::Convert>(sdpa, ov::element::f32);
    if (context.is_static()) {
        res = std::make_shared<ov::op::v1::Transpose>(sdpa_f32,
                                                      ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));
    } else {
        res = std::make_shared<ov::op::v1::Transpose>(sdpa_f32,
                                                      ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
