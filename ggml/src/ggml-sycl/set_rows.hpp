#ifndef GGML_SRC_GGML_SYCL_SET_ROWS_HPP_
#define GGML_SRC_GGML_SYCL_SET_ROWS_HPP_

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SRC_GGML_SYCL_SET_ROWS_HPP_
