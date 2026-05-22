#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_flash_attn_ext(const NodeContext & context) {
    num_inputs_check(context, 4, 4);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto mask = context.get_input(3);

    float * params = reinterpret_cast<float *>(context.get_output_op_params());
    float scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    std::string mask_name = "KQ_mask_sliced";
    if (context.get_input_names()[3].find("swa") != std::string::npos) {
        mask_name = "KQ_mask_swa_sliced";
    }
    if (context.has_input(mask_name)) {
        mask = context.get_input(mask_name);
    }

    if (mask.get_element_type() != ov::element::f16) {
        mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    const int64_t num_heads     = q_shape[1];
    const int64_t num_heads_kv  = k_shape[1];
    const int64_t head_size     = q_shape[3];
    const int64_t factor        = num_heads / num_heads_kv;

    // Optional path: skip the explicit Broadcast that materialises K and V at
    // num_heads. Express attention manually so MatMul's NUMPY-broadcast handles
    // the GQA expansion at kernel level (K and V are read once from DRAM).
    // Opt in with GGML_OPENVINO_MANUAL_GQA_ATTN=1.
    static const bool manual_gqa_enabled = getenv("GGML_OPENVINO_MANUAL_GQA_ATTN") != nullptr;
    const bool use_manual_gqa_attention =
        manual_gqa_enabled && factor > 1 && num_heads_kv > 1 && !context.is_stateful();

    if (use_manual_gqa_attention) {
        // K, V arrive as [1, num_heads_kv, S, head_size]. Reshape to
        //   K_r: [1, num_heads_kv, 1, S, head_size]
        //   Q_r: [1, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V.
        auto k_5d_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5},
            std::vector<int64_t>{1, num_heads_kv, 1, -1, head_size});
        auto v_5d_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5},
            std::vector<int64_t>{1, num_heads_kv, 1, -1, head_size});
        auto q_5d_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5},
            std::vector<int64_t>{1, num_heads_kv, factor, -1, head_size});

        auto k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, false);
        auto v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, false);
        auto q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, false);

        // QK^T → [1, num_heads_kv, factor, S_q, S_k]
        auto qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/false, /*tB=*/true);
        auto qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_node);

        // Mask shape is [B, 1, S_q, S_k] in stateless. We need to broadcast it to
        // [1, num_heads_kv, factor, S_q, S_k]. NUMPY broadcast on Add will handle
        // the trailing dims if we Unsqueeze the mask twice on the leading head
        // dimensions to bring it to rank 5.
        auto mask_unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(
            mask, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        // mask_unsq1: [1, B, 1, S_q, S_k] (rank 5)
        ov::Output<ov::Node> qk_masked = std::make_shared<ov::op::v1::Add>(qk_scaled, mask_unsq1);

        auto softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [1, num_heads_kv, factor, S_q, head_size]
        auto attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [1, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        auto out_4d_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{1, num_heads, -1, head_size});
        auto out_4d = std::make_shared<ov::op::v1::Reshape>(attn, out_4d_shape, false);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            out_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    auto tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        if (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_broadcast_shape, kv_unsqueezed, new_kv_shape;
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_broadcast_shape = ov::op::v0::Constant::create(
                ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed, kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(num_heads, num_heads_kv, head_size, k);
    v = tile_kv(num_heads, num_heads_kv, head_size, v);

    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale_node, false);
    res = std::make_shared<ov::op::v1::Transpose>(sdpa,
                                                  ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
