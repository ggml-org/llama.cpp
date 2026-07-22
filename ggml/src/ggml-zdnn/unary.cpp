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

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

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

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

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

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

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

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

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

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    ZDNN_CHECK(zdnn_exp(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Neg: dst = -src0
// Implemented as 0 - src0 using zdnn_sub
void ggml_zdnn_neg(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Create a zeros ztensor with same shape as src0
    zdnn_tensor_desc zeros_pre_tfm_desc, zeros_tfm_desc;
    zdnn_ztensor zeros_ztensor;

    zdnn_init_pre_transformed_desc(
        src0_extra->pre_tfm_desc.layout,
        src0_extra->pre_tfm_desc.type,
        &zeros_pre_tfm_desc,
        src0_extra->pre_tfm_desc.dim4,
        src0_extra->pre_tfm_desc.dim3,
        src0_extra->pre_tfm_desc.dim2,
        src0_extra->pre_tfm_desc.dim1
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&zeros_pre_tfm_desc, &zeros_tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&zeros_pre_tfm_desc, &zeros_tfm_desc, &zeros_ztensor));

    // Initialize zeros buffer and transform
    size_t num_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    float * zeros_data = (float *)calloc(num_elements, sizeof(float));
    GGML_ASSERT(zeros_data != nullptr);
    ZDNN_CHECK(zdnn_transform_ztensor(&zeros_ztensor, zeros_data));

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    // Compute 0 - src0
    ZDNN_CHECK(zdnn_sub(&zeros_ztensor, &src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    // Cleanup
    free(zeros_data);
    zdnn_free_ztensor_buffer(&zeros_ztensor);

    GGML_UNUSED(ctx);
}

// Sqrt: dst = sqrt(src0)
void ggml_zdnn_sqrt(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    ZDNN_CHECK(zdnn_sqrt(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Log: dst = log(src0)
void ggml_zdnn_log(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    ZDNN_CHECK(zdnn_log(&src0_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// SILU activation: dst = src0 * sigmoid(src0)
void ggml_zdnn_silu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Create a temporary ztensor for sigmoid result
    zdnn_tensor_desc temp_pre_tfm_desc, temp_tfm_desc;
    zdnn_ztensor temp_ztensor;

    zdnn_init_pre_transformed_desc(
        src0_extra->pre_tfm_desc.layout,
        src0_extra->pre_tfm_desc.type,
        &temp_pre_tfm_desc,
        src0_extra->pre_tfm_desc.dim4,
        src0_extra->pre_tfm_desc.dim3,
        src0_extra->pre_tfm_desc.dim2,
        src0_extra->pre_tfm_desc.dim1
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&temp_pre_tfm_desc, &temp_tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&temp_pre_tfm_desc, &temp_tfm_desc, &temp_ztensor));

    // Compute sigmoid(src0) into temp
    ZDNN_CHECK(zdnn_sigmoid(&src0_extra->ztensor, &temp_ztensor));

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    // Compute src0 * sigmoid(src0) into dst
    ZDNN_CHECK(zdnn_mul(&src0_extra->ztensor, &temp_ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    // Cleanup
    zdnn_free_ztensor_buffer(&temp_ztensor);

    GGML_UNUSED(ctx);
}

// Leaky ReLU activation: dst = leaky_relu(src0, negative_slope)
void ggml_zdnn_leaky_relu(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst,
    float negative_slope) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Reset destination tensor if already transformed (required by zDNN)
    if (dst_extra->ztensor.is_transformed) {
        zdnn_reset_ztensor(&dst_extra->ztensor);
    }

    // zdnn_leaky_relu takes: input, clipping_value (NULL for no clipping), alpha, output
    ZDNN_CHECK(zdnn_leaky_relu(&src0_extra->ztensor, NULL, negative_slope, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}
