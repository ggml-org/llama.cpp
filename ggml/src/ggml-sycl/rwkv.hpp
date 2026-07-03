#ifndef GGML_SYCL_RWKV_HPP
#define GGML_SYCL_RWKV_HPP

#include "common.hpp"

void ggml_sycl_op_rwkv_lerp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_op_rwkv_rk(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_RWKV_HPP
