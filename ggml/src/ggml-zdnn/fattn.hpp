#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#include "common.hpp"

bool ggml_zdnn_supports_flash_attn_ext(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op);

void ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
