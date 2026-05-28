#ifndef GGML_SRC_GGML_SYCL_PAD_REFLECT_1D_HPP_
#define GGML_SRC_GGML_SYCL_PAD_REFLECT_1D_HPP_

#include "common.hpp"

#define SYCL_PAD_REFLECT_1D_BLOCK_SIZE 256

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif  // GGML_SRC_GGML_SYCL_PAD_REFLECT_1D_HPP_
