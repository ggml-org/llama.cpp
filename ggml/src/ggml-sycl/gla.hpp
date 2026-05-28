#ifndef GGML_SRC_GGML_SYCL_GLA_HPP_
#define GGML_SRC_GGML_SYCL_GLA_HPP_

#include "common.hpp"

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SRC_GGML_SYCL_GLA_HPP_
