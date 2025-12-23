#include "ggml.h"
#include "elementwise.hpp"
#include "utils.hpp"

// Element-wise ADD: dst = src0 + src1
void ggml_zdnn_add(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_add(&src0_extra->ztensor, &src1_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Element-wise MUL: dst = src0 * src1
void ggml_zdnn_mul(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_mul(&src0_extra->ztensor, &src1_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Element-wise SUB: dst = src0 - src1
void ggml_zdnn_sub(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_sub(&src0_extra->ztensor, &src1_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Element-wise DIV: dst = src0 / src1
void ggml_zdnn_div(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_div(&src0_extra->ztensor, &src1_extra->ztensor, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// Softmax: dst = softmax(src0)
// Note: ZDNN softmax operates on the last dimension
void ggml_zdnn_softmax(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // ZDNN softmax requires a save area
    void * save_area = ggml_aligned_malloc(ZDNN_SOFTMAX_SAVEAREA_SIZE);
    GGML_ASSERT(save_area != nullptr);

    ZDNN_CHECK(zdnn_softmax(&src0_extra->ztensor, save_area, SOFTMAX_ACT_NONE, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    free(save_area);

    GGML_UNUSED(ctx);
}

// RMS Normalization: dst = src0 / sqrt(mean(src0^2) + eps) * src1
// Note: src1 (weight) is optional
void ggml_zdnn_rms_norm(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,  // weight (optional)
          ggml_tensor * dst,
    float eps) {

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    // Weight tensor is optional
    const zdnn_ztensor * weight = nullptr;
    if (src1 != nullptr && src1->extra != nullptr) {
        ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
        weight = &src1_extra->ztensor;
    }

    ZDNN_CHECK(zdnn_rmsnorm(&src0_extra->ztensor, weight, eps, &dst_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}

// GET_ROWS: Extract rows from src0 using indices in src1
// Used for embedding lookups
// Note: Operates on raw float data, not ztensors
void ggml_zdnn_get_rows(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,  // embedding table (F32/F16)
    const ggml_tensor * src1,  // indices (I32)
          ggml_tensor * dst) {

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Get dimensions
    const int64_t ne0 = src0->ne[0];  // embedding dimension
    const int64_t ne1 = src0->ne[1];  // vocabulary size

    // Number of indices to look up
    const int64_t num_indices = ggml_nelements(src1);

    // Get raw data pointers
    const float * src_data = (const float *)src0->data;
    const int32_t * indices = (const int32_t *)src1->data;
    float * dst_data = (float *)dst->data;

    ZDNN_CHECK(zdnn_get_rows(src_data, ne0, ne1, indices, num_indices, dst_data));

    GGML_UNUSED(ctx);
}

// CONT: Make tensor contiguous (copy non-contiguous to contiguous)
// Copies element-by-element respecting source strides
void ggml_zdnn_cont(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;

    // If source is already contiguous, simple memcpy
    if (ggml_is_contiguous(src0)) {
        memcpy(dst_data, src_data, ggml_nbytes(src0));
    } else {
        // Copy element by element respecting strides
        const int64_t ne0 = src0->ne[0];
        const int64_t ne1 = src0->ne[1];
        const int64_t ne2 = src0->ne[2];
        const int64_t ne3 = src0->ne[3];

        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        const size_t src_offset = i0 * src0->nb[0] + i1 * src0->nb[1] +
                                                  i2 * src0->nb[2] + i3 * src0->nb[3];
                        const size_t dst_idx = i0 + i1 * ne0 + i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2;
                        dst_data[dst_idx] = *(const float *)((const char *)src0->data + src_offset);
                    }
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

// CPY: Copy tensor data (F32 to F32 only for now)
void ggml_zdnn_cpy(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Use CONT implementation for the copy
    ggml_zdnn_cont(ctx, src0, dst);
}
