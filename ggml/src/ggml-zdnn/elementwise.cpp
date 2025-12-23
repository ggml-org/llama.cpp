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

    // Get raw data pointers
    const float * src_data = (const float *)src0->data;
    const int32_t * indices = (const int32_t *)src1->data;
    float * dst_data = (float *)dst->data;

    // Check if we have batched indices (higher dimensions in src1)
    const bool batched = (src1->ne[1] > 1 || src1->ne[2] > 1);

    if (batched) {
        // Use batched version for 4D tensors
        ZDNN_CHECK(zdnn_get_rows_batched(
            src_data,
            src0->ne[0],  // embedding dimension
            src0->ne[1],  // vocabulary size
            src0->ne[2],  // batch dim 1 (usually 1)
            src0->ne[3],  // batch dim 2 (usually 1)
            indices,
            src1->ne[0],  // indices per batch
            src1->ne[1],  // batch dim 1
            src1->ne[2],  // batch dim 2
            dst_data));
    } else {
        // Use simple version for 2D case
        const int64_t ne0 = src0->ne[0];  // embedding dimension
        const int64_t ne1 = src0->ne[1];  // vocabulary size
        const int64_t num_indices = ggml_nelements(src1);
        ZDNN_CHECK(zdnn_get_rows(src_data, ne0, ne1, indices, num_indices, dst_data));
    }

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
    GGML_ASSERT(src0->data != nullptr);
    GGML_ASSERT(dst->data != nullptr);

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

// Helper to compute byte offset for a linear index in a tensor
static size_t tensor_offset_for_linear_index(const ggml_tensor * t, int64_t idx) {
    const int64_t i0 = idx % t->ne[0];
    idx /= t->ne[0];
    const int64_t i1 = idx % t->ne[1];
    idx /= t->ne[1];
    const int64_t i2 = idx % t->ne[2];
    const int64_t i3 = idx / t->ne[2];
    return i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

// CPY: Copy tensor data (F32 to F32 only for now)
// Handles both non-contiguous source and non-contiguous destination
// Source and destination may have different shapes (but same element count)
void ggml_zdnn_cpy(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,
          ggml_tensor * dst) {

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements == ggml_nelements(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->data != nullptr);
    GGML_ASSERT(dst->data != nullptr);

    const bool src_contiguous = ggml_is_contiguous(src0);
    const bool dst_contiguous = ggml_is_contiguous(dst);

    // Fast path: both contiguous with same layout
    if (src_contiguous && dst_contiguous) {
        memcpy(dst->data, src0->data, ggml_nbytes(src0));
        GGML_UNUSED(ctx);
        return;
    }

    // General path: iterate by linear element index
    // Each tensor may have a different shape, so we convert linear index
    // to coordinates separately for each tensor
    for (int64_t i = 0; i < n_elements; i++) {
        const size_t src_offset = tensor_offset_for_linear_index(src0, i);
        const size_t dst_offset = tensor_offset_for_linear_index(dst, i);
        *(float *)((char *)dst->data + dst_offset) =
            *(const float *)((const char *)src0->data + src_offset);
    }

    GGML_UNUSED(ctx);
}

// ROPE: Rotary Position Embedding
void ggml_zdnn_rope(
    const ggml_backend_zdnn_context * ctx,
    const ggml_tensor * src0,  // input tensor
    const ggml_tensor * src1,  // positions (I32)
          ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Extract parameters from op_params
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];

    float freq_base, freq_scale;
    memcpy(&freq_base,  (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));

    // Get tensor dimensions
    const int64_t ne0 = src0->ne[0];  // n_embd
    const int64_t ne1 = src0->ne[1];  // n_head
    const int64_t ne2 = src0->ne[2];  // n_seq
    const int64_t ne3 = src0->ne[3];  // batch

    // Get raw data pointers
    const float * input = (const float *)src0->data;
    const int32_t * positions = (const int32_t *)src1->data;
    float * output = (float *)dst->data;

    ZDNN_CHECK(zdnn_rope(input, positions, ne0, ne1, ne2, ne3, n_dims, mode, freq_base, freq_scale, output));

    GGML_UNUSED(ctx);
}
