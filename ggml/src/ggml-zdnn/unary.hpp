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

// Neg: dst = -src0
void ggml_zdnn_neg(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Sqrt: dst = sqrt(src0)
void ggml_zdnn_sqrt(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Log: dst = log(src0)
void ggml_zdnn_log(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// SILU activation: dst = src0 * sigmoid(src0)
void ggml_zdnn_silu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// Leaky ReLU activation: dst = leaky_relu(src0, negative_slope)
void ggml_zdnn_leaky_relu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst,
    float negative_slope);

#endif // GGML_ZDNN_UNARY_HPP
