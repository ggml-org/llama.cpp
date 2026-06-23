#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) softmax over the last dimension
OutputVector translate_soft_max(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&max_bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    ov::Output<ov::Node> logits = context.get_input(0);

    // Apply scale first: logits = src0 * scale
    if (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && context.get_input_size() < 2),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    if (context.get_input_size() > 1) {
        ov::Output<ov::Node> mask = context.get_input(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        if (context.get_input_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        if (context.has_input(mask_name)) {
            mask = context.get_input(mask_name);
        }

        if (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        if (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            const uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(n_head))));
            const float m0 = std::pow(2.0f, -(max_bias) / static_cast<float>(n_head_log2));
            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<float>(n_head_log2));

            std::vector<float> slopes(n_head);
            for (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1)) :
                                              std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            if (slope_node.get_element_type() != mask.get_element_type()) {
                slope_node = std::make_shared<ov::op::v0::Convert>(slope_node, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_node, slope_shape, false);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(logits, -1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
