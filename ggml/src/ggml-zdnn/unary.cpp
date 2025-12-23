#include "ggml.h"
#include "unary.hpp"
#include "utils.hpp"

// GELU activation: dst = gelu(src0)
void ggml_zdnn_gelu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_gelu(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// ReLU activation: dst = relu(src0)
void ggml_zdnn_relu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // zdnn_relu takes a clipping value - pass NULL for no clipping
    ZDNN_CHECK(zdnn_relu(&src0_extra->ztensor, NULL, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Tanh activation: dst = tanh(src0)
void ggml_zdnn_tanh(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_tanh(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Sigmoid activation: dst = sigmoid(src0)
void ggml_zdnn_sigmoid(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_sigmoid(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Exp: dst = exp(src0)
void ggml_zdnn_exp(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_exp(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}
