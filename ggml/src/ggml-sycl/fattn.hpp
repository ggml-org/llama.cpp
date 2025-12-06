#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"

// Flash attention operation for SYCL backend
// This implements the Flash Attention algorithm optimized for SYCL devices
void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Check if flash attention is supported for given tensor
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
