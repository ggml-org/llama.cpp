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

// Element-wise SUB: dst = src0 - src1
void ggml_zdnn_sub(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst);

// Element-wise DIV: dst = src0 / src1
void ggml_zdnn_div(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst);

// Softmax: dst = softmax(src0)
void ggml_zdnn_softmax(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// RMS Normalization: dst = src0 / sqrt(mean(src0^2) + eps) * src1
void ggml_zdnn_rms_norm(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,  // weight (optional)
          ggml_tensor * dst,
    float eps);

// GET_ROWS: Extract rows from src0 using indices in src1
// Used for embedding lookups
void ggml_zdnn_get_rows(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,  // embedding table
    const ggml_tensor * src1,  // indices (I32)
          ggml_tensor * dst);

// CONT: Make tensor contiguous (copy non-contiguous to contiguous)
void ggml_zdnn_cont(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

// CPY: Copy tensor data (possibly with type conversion)
void ggml_zdnn_cpy(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst);

#endif // GGML_ZDNN_ELEMENTWISE_HPP
