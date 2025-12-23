#ifndef GGML_ZDNN_ELEMENTWISE_HPP
#define GGML_ZDNN_ELEMENTWISE_HPP

#include "common.hpp"

// Element-wise ADD: dst = src0 + src1
void ggml_zdnn_add(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst);

// Element-wise MUL: dst = src0 * src1
void ggml_zdnn_mul(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst);

// Softmax: dst = softmax(src0)
void ggml_zdnn_softmax(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

#endif // GGML_ZDNN_ELEMENTWISE_HPP
