#ifndef GGML_ZDNN_MMQ_HPP
#define GGML_ZDNN_MMQ_HPP

#include "common.hpp"

void ggml_zdnn_mul_mat_q(
          ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst);

#endif  // GGML_ZDNN_MMQ_HPP
