#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#include "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

// Check if a type can be directly mapped to ZDNN (without dequantization)
bool ggml_zdnn_type_is_native(ggml_type type);

// Check if a type requires dequantization (is a quantized type we support)
bool ggml_zdnn_type_needs_dequant(ggml_type type);

// Dequantize data from quantized format to F32
void ggml_zdnn_dequantize(const void * src, float * dst, ggml_type type, int64_t nelements);

void ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      const ggml_tensor       * src,
                      const int64_t           * ne,
                      const zdnn_data_layouts   layout);

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer);

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
