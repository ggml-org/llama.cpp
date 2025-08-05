#include <cstdint>
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
