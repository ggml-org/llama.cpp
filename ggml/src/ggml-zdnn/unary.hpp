#ifndef GGML_ZDNN_UNARY_HPP
#define GGML_ZDNN_UNARY_HPP

#include "common.hpp"

// GELU activation: dst = gelu(src0)
void ggml_zdnn_gelu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// ReLU activation: dst = relu(src0)
void ggml_zdnn_relu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Tanh activation: dst = tanh(src0)
void ggml_zdnn_tanh(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Sigmoid activation: dst = sigmoid(src0)
void ggml_zdnn_sigmoid(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Exp: dst = exp(src0)
void ggml_zdnn_exp(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

#endif // GGML_ZDNN_UNARY_HPP
