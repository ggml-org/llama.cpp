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
