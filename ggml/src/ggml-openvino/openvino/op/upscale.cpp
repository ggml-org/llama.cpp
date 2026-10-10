#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/interpolate.hpp>
#include <openvino/op/matmul.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_upscale(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    using Interpolate = ov::op::v4::Interpolate;
    auto input = process_view_input_new(context, 0);

    const auto input_shape = context.get_input_shape(0).to_shape();
    const auto output_shape = context.get_output_shape().to_shape();

    if (input_shape == output_shape) {
        return rename_outputs_with_suffix({input}, context.get_name());
    }

    ov::Output<ov::Node> res = input;

    // Resample batch / channel dimensions (ne[3] and ne[2], corresponding to OV axes 0 and 1)
    // using nearest-neighbor index mapping: i0_d = floor(i_d * in_d / out_d).
    for (size_t axis = 0; axis < 2; ++axis) {
        const size_t in_dim = input_shape[axis];
        const size_t out_dim = output_shape[axis];
        if (in_dim != out_dim) {
            std::vector<int64_t> indices(out_dim);
            for (size_t i = 0; i < out_dim; ++i) {
                indices[i] = static_cast<int64_t>((i * in_dim) / out_dim);
            }
            auto indices_node = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
            auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            res = std::make_shared<ov::op::v8::Gather>(res, indices_node, axis_node);
        }
    }

    // Spatial interpolation for ne[1] and ne[0] (OV axes 2 and 3)
    if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
        const int32_t * op_params = context.get_output_op_params();
        const int32_t mode_flags = op_params != nullptr ? op_params[0] : 0;

        const int op_case = context.get_op_case();
        const bool align_corners = (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
        const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS) != 0;

        if (op_case == 2 && antialias) { // GGML_SCALE_MODE_BILINEAR with antialias
            const size_t H_in = input_shape[2];
            const size_t W_in = input_shape[3];
            const size_t H_out = output_shape[2];
            const size_t W_out = output_shape[3];

            const float pixel_offset = 0.5f;

            // Height projection: [H_out, H_in]
            std::vector<float> Wy_data(H_out * H_in, 0.0f);
            const float sf1 = static_cast<float>(H_out) / H_in;
            const float support1 = std::max(1.0f, 1.0f / sf1);
            const float invscale1 = 1.0f / support1;

            for (size_t i1 = 0; i1 < H_out; ++i1) {
                const float y = (static_cast<float>(i1) + pixel_offset) / sf1;
                const auto y_start = static_cast<int64_t>(y - support1 + pixel_offset);
                const size_t y_min = y_start > 0 ? static_cast<size_t>(y_start) : 0;
                const auto y_end = static_cast<int64_t>(y + support1 + pixel_offset);
                const size_t y_max = y_end > 0 ? std::min<size_t>(static_cast<size_t>(y_end), H_in) : 0;

                float total_weight = 0.0f;
                for (size_t sy = y_min; sy < y_max; ++sy) {
                    float diff = std::abs((static_cast<float>(sy) - y + pixel_offset) * invscale1);
                    float weight = std::max(1.0f - diff, 0.0f);
                    Wy_data[i1 * H_in + sy] = weight;
                    total_weight += weight;
                }
                if (total_weight > 0.0f) {
                    for (size_t sy = y_min; sy < y_max; ++sy) {
                        Wy_data[i1 * H_in + sy] /= total_weight;
                    }
                }
            }

            // Width projection: [W_in, W_out]
            std::vector<float> Wx_data(W_in * W_out, 0.0f);
            const float sf0 = static_cast<float>(W_out) / W_in;
            const float support0 = std::max(1.0f, 1.0f / sf0);
            const float invscale0 = 1.0f / support0;

            for (size_t i0 = 0; i0 < W_out; ++i0) {
                const float x = (static_cast<float>(i0) + pixel_offset) / sf0;
                const auto x_start = static_cast<int64_t>(x - support0 + pixel_offset);
                const size_t x_min = x_start > 0 ? static_cast<size_t>(x_start) : 0;
                const auto x_end = static_cast<int64_t>(x + support0 + pixel_offset);
                const size_t x_max = x_end > 0 ? std::min<size_t>(static_cast<size_t>(x_end), W_in) : 0;

                float total_weight = 0.0f;
                for (size_t sx = x_min; sx < x_max; ++sx) {
                    float diff = std::abs((static_cast<float>(sx) - x + pixel_offset) * invscale0);
                    float weight = std::max(1.0f - diff, 0.0f);
                    Wx_data[sx * W_out + i0] = weight;
                    total_weight += weight;
                }
                if (total_weight > 0.0f) {
                    for (size_t sx = x_min; sx < x_max; ++sx) {
                        Wx_data[sx * W_out + i0] /= total_weight;
                    }
                }
            }

            auto Wy_node = ov::op::v0::Constant::create(ov::element::f32, {H_out, H_in}, Wy_data);
            auto Wx_node = ov::op::v0::Constant::create(ov::element::f32, {W_in, W_out}, Wx_data);

            auto res_y = std::make_shared<ov::op::v0::MatMul>(Wy_node, res);
            res = std::make_shared<ov::op::v0::MatMul>(res_y, Wx_node);
        } else {
            Interpolate::InterpolateAttrs attrs;
            attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
            attrs.antialias = antialias;
            attrs.pads_begin = {0, 0, 0, 0};
            attrs.pads_end = {0, 0, 0, 0};

            switch (op_case) {
            case 1:  // GGML_SCALE_MODE_NEAREST
                attrs.mode = Interpolate::InterpolateMode::NEAREST;
                attrs.nearest_mode = Interpolate::NearestMode::FLOOR;
                attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ASYMMETRIC;
                break;
            case 2:  // GGML_SCALE_MODE_BILINEAR
                attrs.mode = Interpolate::InterpolateMode::LINEAR;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                break;
            case 3:  // GGML_SCALE_MODE_BICUBIC
                attrs.mode = Interpolate::InterpolateMode::CUBIC;
                attrs.cube_coeff = -0.75;
                attrs.coordinate_transformation_mode = align_corners
                    ? Interpolate::CoordinateTransformMode::ALIGN_CORNERS
                    : Interpolate::CoordinateTransformMode::HALF_PIXEL;
                break;
            default:
                FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported upscale op_case: ", op_case);
            }

            std::vector<int64_t> target_shape_vec = {
                static_cast<int64_t>(output_shape[2]),
                static_cast<int64_t>(output_shape[3]),
            };
            std::vector<float> scales_vec = {
                static_cast<float>(output_shape[2]) / static_cast<float>(input_shape[2]),
                static_cast<float>(output_shape[3]) / static_cast<float>(input_shape[3]),
            };

            auto target_shape_node = ov::op::v0::Constant::create(ov::element::i64, {2}, target_shape_vec);
            auto scales_node = ov::op::v0::Constant::create(ov::element::f32, {2}, scales_vec);
            auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3});

            res = std::make_shared<Interpolate>(
                res, target_shape_node, scales_node, axes_node, attrs);
        }
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
