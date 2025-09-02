#include "ggml-quants.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/parallel.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "ggml-impl.h"
#include "ggml.h"

void unpack_32_4(const uint8_t* data, uint8_t* dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

// Extracts (weight, scales, biases) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights
    auto data = static_cast<uint8_t*>(tensor->data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block)));
        biases[i] = ov::float16(-8.f * static_cast<float>(scales[i]));
        unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
    });
}

// Extracts (weight, scales, biases) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit bias|32 x 4bit weights|.
void extract_q4_1_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes bias, 32x0.5 byte weights
    auto data = static_cast<uint8_t*>(tensor->data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block)));
        biases[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block + 2)));
        unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
    });
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights
    auto data = static_cast<uint8_t*>(tensor->data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (size_t i = 0; i < scales_arr.get_size(); i++) {
        uint8_t* block_data = data + i * bytes_per_block;
        scales[i] = ov::float16::from_bits(*(uint16_t*)block_data);
        biases[i] = ov::float16(-128.f * static_cast<float>(scales[i]));
        for (size_t j = 0; j < weights_per_block; ++j) {
            uint8_t x = block_data[j + 2];  // j+2 to skip the scale bytes.
            // Original data is in int8_t, so we add a bias of -128 and invert the
            // first bit.
            x ^= 1 << 7;
            weights[i * weights_per_block + j] = x;
        }
    }
}

void unpack_256_4(const uint8_t* data, uint8_t* dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

void extract_q4_k_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    // TODO tensor->nb[3]
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor->data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t* block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        float scale_scales = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data)));
        float scale_biases = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 1)));

        // Extract qs1 and qs2
        uint8_t* qs1 = block_data + 4;
        // uint8_t* qs2 = block_data + 16;

        scales[i * 8] = ov::float16(scale_scales * static_cast<float>((*(qs1) & 0b111111)));
        scales[i * 8 + 1] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111)));
        scales[i * 8 + 2] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111)));
        scales[i * 8 + 3] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111)));
        scales[i * 8 + 4] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4)));
        scales[i * 8 + 5] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4)));
        scales[i * 8 + 6] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4)));
        scales[i * 8 + 7] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4)));

        biases[i * 8] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 4) & 0b111111)));
        biases[i * 8 + 1] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 5) & 0b111111)));
        biases[i * 8 + 2] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 6) & 0b111111)));
        biases[i * 8 + 3] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 7) & 0b111111)));
        biases[i * 8 + 4] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4)));
        biases[i * 8 + 5] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4)));
        biases[i * 8 + 6] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4)));
        biases[i * 8 + 7] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4)));
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

void extract_q6_k_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor->data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    // std::string name(tensor.name, tensor.namelen);
    for (size_t i = 0; i < n_super_block; i++) {
        uint8_t* block_data = data + i * bytes_per_block;

        float scale_factor =
            static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 104)));  // (128+64+16)/2

        for (size_t j = 0; j < 16; j++) {
            scales[j + i * 16] =
                ov::float16(scale_factor * static_cast<float>(*((int8_t*)(block_data + 128 + 64 + j))));
            biases[j + i * 16] = ov::float16(-32.f * static_cast<float>(scales[j + i * 16]));
        }

        // Extract ql and qh
        uint8_t* ql = block_data;
        uint8_t* qh = block_data + 128;

        // Extract weights
        for (int64_t j = 0; j < 32; ++j) {
            weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
            weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
            weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
            weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
            weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
            weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
            weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
            weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
        }
    }
}

// TODO Reorder for make_intX_weights

ov::Output<ov::Node> make_int8_weights(ov::Tensor& weight, ov::Tensor& scales, ov::Tensor& biases, size_t group_size) {
    ov::Shape orig_shape = weight.get_shape();

    // Expand dimensions for scales and biases
    auto scale_shape = scales.get_shape();

    ov::Shape packed_shape = {orig_shape[0], orig_shape[1] / group_size, group_size};

    if (packed_shape[1] == 1) {
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        biases.set_shape(scale_shape);
    }

    // Create graph nodes
    auto weights_node = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, packed_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
    ov::Tensor biases_u8(ov::element::u8, scale_shape);

    // Calculate zero point
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    uint8_t* bias_u8_data = biases_u8.data<uint8_t>();
    for (size_t i = 0; i < biases_u8.get_size(); ++i) {
        bias_u8_data[i] = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i]) / static_cast<float>(scale_data[i]));
    }

    auto zero_point = std::make_shared<ov::op::v0::Constant>(biases_u8);
    float zp_value;
    if (ov::op::util::get_single_value(zero_point, zp_value)) {
        zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
    }

    // Quantization operations
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);

    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY
    );
    ov::Output<ov::Node> w_zp_s =
        std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        w_zp_s = std::make_shared<ov::op::v1::Reshape>(w_zp_s, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(w_zp_s, ov::element::f32);
}

ov::Output<ov::Node> make_int4_weights(ov::Tensor& weight, ov::Tensor& scales, ov::Tensor& biases, size_t group_size) {
    ov::Shape orig_weight_shape = weight.get_shape();

    // Expand dimensions for scales and biases
    ov::Shape scale_bias_shape = scales.get_shape();

    // Create INT4 weight tensor
    ov::Shape packed_shape = {
        orig_weight_shape[0],
        orig_weight_shape[1] / group_size,
        group_size
    };

    // Requantized channel-wise case
    if (packed_shape[1] == 1) {
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_bias_shape.push_back(1);
        scales.set_shape(scale_bias_shape);
        biases.set_shape(scale_bias_shape);
    }

    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

    // Pack zero points: two subsequent values into one
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::Tensor zero_point_tensor(ov::element::u4, scale_bias_shape);
    uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
    for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
        uint8_t bias1 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2]) / static_cast<float>(scale_data[i * 2]));
        uint8_t bias2 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2 + 1]) / static_cast<float>(scale_data[i * 2 + 1]));
        zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
    }

    auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);
    float zp_value;
    if (ov::op::util::get_single_value(zero_points_node, zp_value)) {
        zero_points_node = ov::op::v0::Constant::create(zero_points_node->get_element_type(), {}, {zp_value});
    }
    auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    // Perform dequantization
    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);

    ov::Output<ov::Node> w_zp_s =
        std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{orig_weight_shape.size()}, orig_weight_shape);

        w_zp_s = std::make_shared<ov::op::v1::Reshape>(w_zp_s, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(w_zp_s, ov::element::f32);
}

std::shared_ptr<ov::Node> requantize(const ggml_tensor* tensor, ExtraQuantType requant_type) {
    std::vector<float> weights_f32(tensor->ne[0] * tensor->ne[1]);
    ggml_get_type_traits(tensor->type)->to_float(tensor->data, weights_f32.data(), ggml_nelements(tensor));

    std::shared_ptr<ov::Node> weight_node;
    ov::Shape node_shape = {(uint64_t) (tensor->ne[1]), (uint64_t) (tensor->ne[0])};

    if (requant_type == ExtraQuantType::F16) {
        ov::Tensor weights(ov::element::f16, node_shape);
        ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), ggml_nelements(tensor));
        std::shared_ptr<ov::Node> weight_node = std::make_shared<ov::op::v0::Constant>(weights);
        weight_node->set_friendly_name(tensor->name);
        return weight_node;
    }

    int64_t block_size = node_shape[1];
    if (requant_type == ExtraQuantType::Q4_0_128) {
        block_size = 128;
    }
    auto scales_shape = ov::Shape{node_shape[0], node_shape[1] / block_size};

    ov::Tensor weights;
    ov::Tensor scales(ov::element::f16, scales_shape);
    ov::Tensor bias(ov::element::f16, scales_shape);

    if (requant_type == ExtraQuantType::Q4_0_C) {
        weights = ov::Tensor(ov::element::u4, node_shape);
        quantize_q4_0(weights_f32.data(), weights, scales, bias, weights.get_size(), block_size);
        weight_node = make_int4_weights(weights, scales, bias, block_size).get_node_shared_ptr();
    } else if (requant_type == ExtraQuantType::Q8_1_C) {
        weights = ov::Tensor(ov::element::u8, node_shape);
        quantize_q8_1(weights_f32.data(), weights, scales, bias, weights.get_size(), block_size);
        weight_node = make_int8_weights(weights, scales, bias, block_size).get_node_shared_ptr();
    } else if (requant_type == ExtraQuantType::Q4_0_128) {
        weights = ov::Tensor(ov::element::u4, node_shape);
        quantize_q4_0(weights_f32.data(), weights, scales, bias, weights.get_size(), block_size);
        weight_node = make_int4_weights(weights, scales, bias, block_size).get_node_shared_ptr();
    }

    weight_node->set_friendly_name(tensor->name);
    return weight_node;
}

void quantize_q4_0(const float* x, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& biases_arr, int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto* weights = static_cast<uint8_t*>(weights_arr.data());
    auto* scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto* biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max
        float max = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        biases[i] = ov::float16(-8.f * d);

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 2 * j] * id;
            const float x1 = x[i * qk + 2 * j + 1] * id;
            const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
            weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
        }
    }
}

void quantize_q8_1(const float* x, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& biases_arr, int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto* weights = static_cast<uint8_t*>(weights_arr.data());
    auto* scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto* biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (int i = 0; i < nb; i++) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }

        const float d = (max - min) / ((1 << 8) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        biases[i] = ov::float16(min);

        for (int j = 0; j < qk; ++j) {
            const float x0 = (x[i * qk + j] - min) * id;
            const uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}
