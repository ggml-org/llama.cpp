#ifndef GGML_SRC_GGML_SYCL_COUNT_EQUAL_HPP_
#define GGML_SRC_GGML_SYCL_COUNT_EQUAL_HPP_
#include "common.hpp"

#define SYCL_COUNT_EQUAL_CHUNK_SIZE 128

void ggml_sycl_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SRC_GGML_SYCL_COUNT_EQUAL_HPP_
