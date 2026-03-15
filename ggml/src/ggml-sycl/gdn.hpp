#ifndef GGML_SYCL_GDN_HPP
#define GGML_SYCL_GDN_HPP

#include "common.hpp"

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_GDN_HPP
