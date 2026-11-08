#ifndef GGML_ZDNN_MMID_HPP
#define GGML_ZDNN_MMID_HPP

#include "common.hpp"

bool ggml_zdnn_supports_mul_mat_id(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op);

void ggml_zdnn_mul_mat_id(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_MMID_HPP
