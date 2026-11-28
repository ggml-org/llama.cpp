#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#include "common.hpp"

void ggml_zdnn_quantize_row(const float  * row,
                                  int64_t  ne0,
                                  int64_t  group_size,
                            const float  * channel_scales,
                                  int8_t * quants,
                                  int8_t * residuals,
                                  int64_t  quants_stride,
                                  float  * scales,
                                  int64_t  scales_stride);

void ggml_zdnn_dequantize_row(      float  * output,
                              const float  * partial,
                              const float  * partial_res,
                              const float  * weights_scales,
                                    float    inputs_scale,
                                    int64_t  ne0,
                                    bool     accumulate);

void ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         const ggml_tensor * inputs,
                               int64_t       group_size,
                         const float       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               float       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
