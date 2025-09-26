#pragma once
#include <cstdint>
#include <openvino/op/constant.hpp>
#include <openvino/runtime/tensor.hpp>

#include "ggml.h"

void unpack_32_4(const uint8_t* data, uint8_t* dst);

void extract_q4_0_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

void extract_q4_1_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

void extract_q8_0_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

void unpack_256_4(const uint8_t* data, uint8_t* dst);

void extract_q4_k_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

void extract_q5_k_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

void extract_q6_k_data(const ggml_tensor* tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr);

static constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

ov::Output<ov::Node> make_int8_weights(ov::Tensor& weight,
                                       ov::Tensor& scales,
                                       ov::Tensor& biases,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE);

ov::Output<ov::Node> make_int4_weights(ov::Tensor& weight,
                                       ov::Tensor& scales,
                                       ov::Tensor& biases,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE);

enum class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q8_0_C, Q8_0_32 };

std::shared_ptr<ov::Node> requantize(const ggml_tensor* tensor, ExtraQuantType requant_type);

void quantize_q4_0(const float* x, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& biases_arr, int64_t k,
                   int64_t qk);
void quantize_q8_1(const float* x, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& biases_arr, int64_t k,
                   int64_t qk);
void quantize_q8_0(const float* x, ov::Tensor& weights_arr, ov::Tensor& scales_arr, ov::Tensor& biases_arr, int64_t k,
                   int64_t qk);

namespace ov {
namespace op {
namespace util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
bool get_single_value(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                      float& value,
                      bool check_value_range = true);
}  // namespace util
}  // namespace op
}  // namespace ov
