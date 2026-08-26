#ifndef GGML_SYCL_CONV2D_HPP
#define GGML_SYCL_CONV2D_HPP

#include "common.hpp"

#define SYCL_CONV2D_BLOCK_SIZE 256

void ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CONV2D_HPP
