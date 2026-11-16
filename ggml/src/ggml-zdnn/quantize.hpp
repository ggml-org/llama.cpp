#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#include "common.hpp"

void ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         const ggml_tensor * inputs,
                               int64_t       group_size,
                         const float       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               float       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
