/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "aclnn_ops.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_addcdiv.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_avgpool2d.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_elu.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#include <aclnnop/aclnn_group_norm.h>
#include <aclnnop/aclnn_grouped_matmul_v3.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_im2col.h>
#include <aclnnop/aclnn_index_copy.h>
#include <aclnnop/aclnn_index_fill_tensor.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_max_pool.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_reflection_pad1d.h>
#include <aclnnop/aclnn_repeat.h>
#include <aclnnop/aclnn_repeat_interleave.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_roll.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_sum.h>
#include <aclnnop/aclnn_threshold.h>
#include <aclnnop/aclnn_tril.h>
#include <aclnnop/aclnn_triu.h>
#include <aclnnop/aclnn_upsample_nearest_2d.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#include <aclnnop/aclnn_zero.h>
#include <float.h>

#include <cmath>
#include <cstring>
#include <exception>
#include <vector>

#define GGML_COMMON_DECL_C

#include "../ggml-common.h"

void bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst  = ggml_cann_create_tensor(dst);
    }
}

void ggml_cann_op_unary(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

void ggml_cann_op_unary_gated(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_tensor_ptr acl_src0, acl_src1;
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);

        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
    } else {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_src0     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_src1 = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_size(src0));
        if (swapped) {
            std::swap(acl_src0, acl_src1);
        }
    }

    unary_op(ctx, acl_src0.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_src1.get());
}

/**
 * @brief Repeats elements of a tensor along each dimension according to the
 * specified repeat array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be repeated.
 * @param acl_dst The destination tensor after repeating.
 * @param repeat_array The array specifying the number of repetitions along each
 * dimension.
 */
static void aclnn_repeat(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_src,
                         aclTensor *                 acl_dst,
                         int64_t *                   repeat_array) {
    // repeat tensor along each dim with repeat_array
    acl_int_array_ptr repeats = ggml_cann_create_int_array(repeat_array, GGML_MAX_DIMS);

    GGML_CANN_CALL_ACLNN_OP(ctx, Repeat, acl_src, repeats.get(), acl_dst);
}

/**
 * @brief Casts the data type of a source tensor to a destination tensor.
 *
 * This function casts the data type of the source tensor `acl_src` to the
 * specified data type `cast_data_type` and stores the result in the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose data type will be casted.
 * @param acl_dst The destination tensor where the casted result will be stored.
 * @param cast_data_type The target data type to which the source tensor will be
 * casted.
 */
static void aclnn_cast(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       aclDataType                 cast_data_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_data_type, acl_dst);
}

void ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = { dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2], dst->ne[1] / src->ne[1],
                               dst->ne[0] / src->ne[0] };

    aclnn_repeat(ctx, acl_src.get(), acl_dst.get(), repeatsArray);
}

void aclnn_add(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    float          alphaValue = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0, acl_src1, alpha.get(), acl_dst);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_src0, acl_src1, alpha.get());
    }
}

void aclnn_sub(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    float          alphaValue = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha.get(), acl_dst);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha.get());
    }
}

void aclnn_mul(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    if (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
    }
}

void aclnn_div(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    if (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDiv, acl_src, acl_other);
    }
}

/**
 * @brief Multiplies elements of a tensor by a scalar value, optionally
 * in-place.
 *
 * This function multiplies each element of the source tensor `acl_src` by the
 * scalar `scale` and stores the result in the destination tensor `acl_dst`. If
 * `inplace` is true, `acl_dst` will not be used and the operation is performed
 *  in-place on `acl_src`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {scale}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be multiplied.
 * @param scale The scalar value by which each element of `acl_src` will be
 *  multiplied.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
static void aclnn_muls(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       float                       scale,
                       aclTensor *                 acl_dst,
                       bool                        inplace) {
    acl_scalar_ptr acl_scale = ggml_cann_create_scalar(&scale, aclDataType::ACL_FLOAT);
    if (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale.get());
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale.get(), acl_dst);
    }
}

void ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    acl_scalar_ptr acl_negative_slope = ggml_cann_create_scalar(&negative_slope, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src.get(), acl_negative_slope.get(), acl_dst.get());
}

/**
 * @brief Concatenates a list of tensors along a specified dimension and stores
 * the result in a destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param tensorList The list of tensors to be concatenated.
 * @param acl_dst The destination tensor where the concatenated result will be
 * stored.
 * @param concat_dim The dimension along which the tensors will be concatenated.
 */
static void aclnn_concat(ggml_backend_cann_context & ctx,
                         aclTensorList *             tensorList,
                         aclTensor *                 acl_dst,
                         int64_t                     concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

void ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src0     = dst->src[0];
    ggml_tensor *  src1     = dst->src[1];
    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    int32_t acl_dim = 3 - dim;

    acl_tensor_list_ptr tensor_list = ggml_cann_create_tensor_list(acl_src0, acl_src1);
    aclnn_concat(ctx, tensor_list.get(), acl_dst.get(), acl_dim);
}

/**
 * @brief Creates a tensor with values starting from `start`, incremented by
 * `step`, and ending before `stop`.
 *
 * This function performs the operation:
 * \f[
 *    \text {out }_{i+1}=\text {out }_i+\text {step}
 * \f]
 * the range is [start, stop).
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor where the values will be stored.
 * @param start The starting value of the range.
 * @param stop The ending value of the range (exclusive).
 * @param step The step size between consecutive values.
 * @param n_elements The number of elements in the destination tensor.
 */
static void aclnn_arange(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_dst,
                         float                       start,
                         float                       stop,
                         float                       step,
                         int64_t                     n_elements) {
    int64_t steps = (int64_t) std::ceil((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    acl_scalar_ptr acl_start = ggml_cann_create_scalar(&start, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_end   = ggml_cann_create_scalar(&stop, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_step  = ggml_cann_create_scalar(&step, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_start.get(), acl_end.get(), acl_step.get(), acl_dst);
}

void ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    float   start;
    float   stop;
    float   step;
    memcpy(&start, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&step, (float *) dst->op_params + 2, sizeof(float));

    aclnn_arange(ctx, acl_dst.get(), start, stop, step, n_elements);
}

void ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&min, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&max, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src.get(), acl_min.get(), acl_max.get(), acl_dst.get());
}

void ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    acl_scalar_ptr scale   = ggml_cann_create_scalar(&v, aclDataType::ACL_FLOAT);
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src.get(), scale.get(), acl_dst.get());
}

void ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *        src   = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    acl_tensor_ptr       acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr       acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), ggml_nelements(dst) * sizeof(int64_t));
    void *               buffer = temp_buffer_allocator.get();
    acl_tensor_ptr       tmp_tensor =
        ggml_cann_create_tensor(buffer, ACL_INT64, ggml_type_size(dst->type), dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src.get(), -1, (order == GGML_SORT_ORDER_DESC ? true : false),
                            tmp_tensor.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor.get(), ggml_cann_type_mapping(dst->type), acl_dst.get());
}

void ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    std::vector<int64_t> normData = { dst->ne[0] };
    acl_int_array_ptr    norm     = ggml_cann_create_int_array(normData.data(), normData.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src.get(), norm.get(), nullptr, nullptr, eps, acl_dst.get(), nullptr,
                            nullptr);
}

void ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    size_t               type_size = ggml_type_size(src->type);
    int64_t              n_bytes   = src->ne[3] * src->ne[2] * src->ne[1] * type_size;
    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes);
    void *               buffer = temp_buffer_allocator.get();

    int64_t div_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
    size_t  div_nb[GGML_MAX_DIMS];
    div_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        div_nb[i] = div_nb[i - 1] * div_ne[i - 1];
    }
    acl_tensor_ptr acl_div = ggml_cann_create_tensor(buffer, ACL_FLOAT, type_size, div_ne, div_nb, GGML_MAX_DIMS);

    std::vector<int64_t> norm_dims  = { 3 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.data(), norm_dims.size());

    float          p_value  = 2.0f;
    acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_value, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), true, acl_div.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div.get(), acl_dst.get());
}

void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);

    int64_t logits_ne[] = { nc, nr };
    size_t  logits_nb[2];
    logits_nb[0]              = ggml_type_size(src0->type);
    logits_nb[1]              = logits_nb[0] * logits_ne[0];
    acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->data, ACL_FLOAT, sizeof(float), logits_ne, logits_nb, 2);

    size_t               log_softmax_type_size = sizeof(float);
    int64_t              log_softmax_n_bytes   = nr * nc * log_softmax_type_size;
    ggml_cann_pool_alloc log_softmax_allocator(ctx.pool(), log_softmax_n_bytes);
    void *               log_softmax_buffer = log_softmax_allocator.get();

    int64_t log_softmax_ne[] = { nc, nr };
    size_t  log_softmax_nb[2];
    log_softmax_nb[0]              = log_softmax_type_size;
    log_softmax_nb[1]              = log_softmax_nb[0] * log_softmax_ne[0];
    acl_tensor_ptr acl_log_softmax = ggml_cann_create_tensor(log_softmax_buffer, ACL_FLOAT, log_softmax_type_size,
                                                             log_softmax_ne, log_softmax_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, LogSoftmax, acl_logits.get(), 1, acl_log_softmax.get());

    int64_t labels_ne[] = { nc, nr };
    size_t  labels_nb[2];
    labels_nb[0]              = ggml_type_size(src1->type);
    labels_nb[1]              = labels_nb[0] * labels_ne[0];
    acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->data, ACL_FLOAT, sizeof(float), labels_ne, labels_nb, 2);

    size_t               mul_type_size = sizeof(float);
    int64_t              mul_n_bytes   = nr * nc * mul_type_size;
    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_n_bytes);
    void *               mul_buffer = mul_allocator.get();

    int64_t mul_ne[] = { nc, nr };
    size_t  mul_nb[2];
    mul_nb[0]                     = mul_type_size;
    mul_nb[1]                     = mul_nb[0] * mul_ne[0];
    acl_tensor_ptr acl_mul_result = ggml_cann_create_tensor(mul_buffer, ACL_FLOAT, mul_type_size, mul_ne, mul_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_log_softmax.get(), acl_labels.get(), acl_mul_result.get());

    size_t               sum_per_sample_type_size = sizeof(float);
    int64_t              sum_per_sample_n_bytes   = nr * sum_per_sample_type_size;
    ggml_cann_pool_alloc sum_per_sample_allocator(ctx.pool(), sum_per_sample_n_bytes);
    void *               sum_per_sample_buffer = sum_per_sample_allocator.get();

    int64_t sum_per_sample_ne[] = { nr };
    size_t  sum_per_sample_nb[1];
    sum_per_sample_nb[0]              = sum_per_sample_type_size;
    acl_tensor_ptr acl_sum_per_sample = ggml_cann_create_tensor(
        sum_per_sample_buffer, ACL_FLOAT, sum_per_sample_type_size, sum_per_sample_ne, sum_per_sample_nb, 1);

    std::vector<int64_t> sum_dims   = { 1 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(sum_dims.data(), sum_dims.size());
    bool                 keep_dims  = false;

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_mul_result.get(), dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_sum_per_sample.get());

    size_t               total_sum_type_size = sizeof(float);
    int64_t              total_sum_n_bytes   = 1 * total_sum_type_size;
    ggml_cann_pool_alloc total_sum_allocator(ctx.pool(), total_sum_n_bytes);
    void *               total_sum_buffer = total_sum_allocator.get();

    int64_t total_sum_ne[] = { 1 };
    size_t  total_sum_nb[1];
    total_sum_nb[0] = total_sum_type_size;

    acl_tensor_ptr acl_total_sum =
        ggml_cann_create_tensor(total_sum_buffer, ACL_FLOAT, total_sum_type_size, total_sum_ne, total_sum_nb, 1);

    std::vector<int64_t> total_sum_dims    = { 0 };
    acl_int_array_ptr total_sum_dims_array = ggml_cann_create_int_array(total_sum_dims.data(), total_sum_dims.size());

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_sum_per_sample.get(), total_sum_dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_total_sum.get());

    float          value        = -1.0f / static_cast<float>(nr);
    acl_scalar_ptr scale_factor = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    acl_tensor_ptr acl_dst =
        ggml_cann_create_tensor(dst->data, ACL_FLOAT, sizeof(float), total_sum_ne, total_sum_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_total_sum.get(), scale_factor.get(), acl_dst.get());
}

void ggml_cann_group_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int n_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int64_t N   = src->ne[3];
    int64_t C   = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t  type_size = ggml_type_size(src->type);
    int64_t ne[]      = { n_groups, N };
    size_t  nb[]      = { type_size, type_size * n_groups };
    size_t  n_bytes   = N * n_groups;

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes * 2);
    void *               buffer       = temp_buffer_allocator.get();
    acl_tensor_ptr       acl_mean_out = ggml_cann_create_tensor(buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    acl_tensor_ptr       acl_rstd_out =
        ggml_cann_create_tensor((char *) buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, GroupNorm, acl_src.get(), nullptr, nullptr, N, C, HxW, n_groups, eps, acl_dst.get(),
                            acl_mean_out.get(), acl_rstd_out.get());
}

void ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_size(src0), nb1, nb2, nb3 };

    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    acl_scalar_ptr alpha      = nullptr;
    float          alphaValue = 1.0f;
    alpha                     = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);

    if (!inplace) {
        size_t cpy_size = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
        acl_tensor_ptr acl_src0 =
            ggml_cann_create_tensor(src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);

        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0.get(), acl_src1.get(), alpha.get(), acl_dst.get());
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst.get(), acl_src1.get(), alpha.get());
    }
}

/**
 * @brief Performs sum reduction on a given tensor along specified dimensions.
 *
 * This function reduces the input tensor by summing along the specified dimensions.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the reduced result will be stored.
 * @param dim An array of dimension indices.
 * @param dim_size The number of dimensions.
 */
static void aclnn_reduce_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t * dim, size_t dim_size) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor *     src         = dst->src[0];
    acl_tensor_ptr    acl_src     = ggml_cann_create_tensor(src);
    acl_tensor_ptr    acl_dst     = ggml_cann_create_tensor(dst);
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(dim, dim_size);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_src.get(), reduce_dims.get(), true, ggml_cann_type_mapping(dst->type),
                            acl_dst.get());
}

void ggml_cann_sum_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 3 };
    aclnn_reduce_sum(ctx, dst, reduce_dims, 1);
}

void ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 0, 1, 2, 3 };
    aclnn_reduce_sum(ctx, dst, reduce_dims, 4);
}

void ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_size{ dst->ne[1], dst->ne[0] };
    acl_int_array_ptr    output_size_array = ggml_cann_create_int_array(output_size.data(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src.get(), output_size_array.get(), acl_dst.get());
}

/**
 * @brief Pads a tensor with a specified value along each dimension.
 *
 * This function performs padding of the source tensor `acl_src` and stores the
 * result in the destination tensor `acl_dst`. The padding values for each
 * dimension are specified in the `paddings` array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be padded.
 * @param acl_dst The destination tensor where the padded result will be stored.
 * @param paddings An array specifying the padding values for each dimension.
 * The size of the array should be twice the number of dimensions of the tensor.
 * @param value The value to be used for padding. The default value is 0.0.
 */
static void aclnn_pad(ggml_backend_cann_context & ctx,
                      aclTensor *                 acl_src,
                      aclTensor *                 acl_dst,
                      int64_t *                   paddings,
                      float                       value = 0.0f) {
    acl_int_array_ptr acl_pad   = ggml_cann_create_int_array(paddings, GGML_MAX_DIMS * 2);
    acl_scalar_ptr    acl_value = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad.get(), acl_value.get(), acl_dst);
}

void ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    const int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    const int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    const int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    const int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    const int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    const int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    const int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    const int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    int64_t paddings[] = { lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3 };
    aclnn_pad(ctx, acl_src.get(), acl_dst.get(), paddings);
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_avg_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       k0   = opts[1];
    const int       k1   = opts[2];
    const int       s0   = opts[3];
    const int       s1   = opts[4];
    const int       p0   = opts[5];
    const int       p1   = opts[6];

    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    std::vector<int64_t> padding_avg_dims = { p1, p0 };  // (padH, padW)

    acl_int_array_ptr kernel_size  = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr strides      = ggml_cann_create_int_array(stride_dims.data(), 2);
    acl_int_array_ptr paddings_avg = ggml_cann_create_int_array(padding_avg_dims.data(), 2);

    bool    ceil_mode         = false;
    bool    count_include_pad = true;
    int64_t divisor_override  = 0;
    int8_t  cube_math_type    = 0;
#ifdef ASCEND_310P
    cube_math_type = 1;
#endif

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src.get(), kernel_size.get(), strides.get(), paddings_avg.get(),
                            ceil_mode, count_include_pad, divisor_override, cube_math_type, acl_dst.get());
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_max_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       k0   = opts[1];
    const int       k1   = opts[2];
    const int       s0   = opts[3];
    const int       s1   = opts[4];
    const int       p0   = opts[5];
    const int       p1   = opts[6];

    int64_t temp_ne[] = { src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2], src->ne[3] };
    size_t  temp_nb[GGML_MAX_DIMS];

    temp_nb[0] = ggml_element_size(src);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        temp_nb[i] = temp_nb[i - 1] * temp_ne[i - 1];
    }

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    void *               buffer = temp_buffer_allocator.get();
    acl_tensor_ptr tmp_tensor   = ggml_cann_create_tensor(buffer, ACL_FLOAT, ggml_element_size(src), temp_ne, temp_nb,
                                                          GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = { p0, p0, p1, p1, 0, 0, 0, 0 };
    float   value      = -FLT_MAX;
    aclnn_pad(ctx, acl_src.get(), tmp_tensor.get(), paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = { 0, 0, 0, 0 };
    std::vector<int64_t> dilation_size    = { 1, 1 };
    acl_int_array_ptr    kernel_size      = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr    strides          = ggml_cann_create_int_array(stride_dims.data(), 2);
    acl_int_array_ptr    paddings_max     = ggml_cann_create_int_array(padding_max_dims.data(), 4);
    acl_int_array_ptr    dilations        = ggml_cann_create_int_array(dilation_size.data(), 2);

    bool    ceil_mode = false;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor.get(), kernel_size.get(), strides.get(), auto_pads,
                            paddings_max.get(), dilations.get(), ceil_mode, acl_dst.get());
}

void ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    const int32_t *   opts = (const int32_t *) dst->op_params;
    enum ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ABORT("fatal error");
            break;
    }
}

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * This function copies data from the source tensor `acl_src` to the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor from which data will be copied.
 * @param acl_dst The destination tensor where the data will be copied to.
 */
static void cann_copy(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

void ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    if (ggml_are_same_shape(src0, dst)) {
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
        if (dst->type == src0->type) {
            cann_copy(ctx, acl_src.get(), acl_dst.get());
        } else {
            aclnn_cast(ctx, acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    } else {
        void *               src_trans_buffer = src0->data;
        ggml_cann_pool_alloc src_buffer_allocator;
        if (!ggml_is_contiguous(src0)) {
            acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
            src_buffer_allocator.alloc(ctx.pool(), ggml_nelements(src0) * ggml_type_size(src0->type));
            src_trans_buffer = src_buffer_allocator.get();
            size_t src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_size(src0->type);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            acl_tensor_ptr src_trans_tensor =
                ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(src0->type),
                                        ggml_type_size(src0->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src.get(), src_trans_tensor.get());
        }

        size_t src_reshape_nb[GGML_MAX_DIMS];
        src_reshape_nb[0] = ggml_type_size(src0->type);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            src_reshape_nb[i] = src_reshape_nb[i - 1] * dst->ne[i - 1];
        }

        acl_tensor_ptr trans_acl_src =
            ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    dst->ne, src_reshape_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

        if (dst->type == src0->type) {
            cann_copy(ctx, trans_acl_src.get(), acl_dst.get());
        } else {
            aclnn_cast(ctx, trans_acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    }
}

/**
 * @brief Creates an ACL tensor initialized with zeros using a provided buffer.
 *
 * This function initializes a tensor with zeros using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @return A tensor smart pointer initialized with zeros.
 */
static acl_tensor_ptr aclnn_zero(ggml_backend_cann_context & ctx,
                                 void *                      buffer,
                                 size_t                      n_bytes,
                                 int64_t *                   ne,
                                 int64_t                     dims,
                                 aclDataType                 type,
                                 size_t                      type_size) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_size;
    for (int i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    acl_tensor_ptr zero = ggml_cann_create_tensor(buffer, type, type_size, ne, nb, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, zero.get());
    return zero;
    GGML_UNUSED(n_bytes);
}

/**
 * @brief Creates an ACL tensor initialized with value using a provided buffer.
 *
 * This function initializes a tensor with value using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @param value The value to be used for initializing the tensor (default
 * is 1.0).
 * @return A tensor smart pointer initialized with value.
 */
static acl_tensor_ptr aclnn_values(ggml_backend_cann_context & ctx,
                                   void *                      buffer,
                                   size_t                      n_bytes,
                                   int64_t *                   ne,
                                   int64_t                     dims,
                                   aclDataType                 type,
                                   size_t                      type_size,
                                   float                       value = 1.0f) {
    acl_tensor_ptr acl_tensor = aclnn_zero(ctx, buffer, n_bytes, ne, dims, type, type_size);
    float          alpha_host = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_host, aclDataType::ACL_FLOAT);
    acl_scalar_ptr other      = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_tensor.get(), other.get(), alpha.get());
    return acl_tensor;
}

/**
 * @brief Fills a tensor with a scalar value.
 *
 * This function fills the destination tensor `acl_dst` with the scalar value
 * `scalar`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param scalar The scalar value used to fill the tensor.
 * @param acl_dst The destination tensor to be filled with the scalar value.
 */
static void aclnn_fill_scalar(ggml_backend_cann_context & ctx, float scalar, aclTensor * acl_dst) {
    acl_scalar_ptr acl_scalar = ggml_cann_create_scalar(&scalar, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst, acl_scalar.get());
}

/**
 * @brief Get or expand a cached tensor filled with a scalar value.
 *
 * This function manages cached device memory for tensors. If the current
 * cache size is insufficient for the requested tensor shape, the old memory will
 * be released and new memory will be allocated. The allocated buffer is
 * initialized  with the given scalar value using CANN operations.
 * Finally, an aclTensor object is created from the cached memory and returned.
 *
 * @param ctx           The CANN backend context that manages device memory.
 * @param buffer        A pointer to the cached device buffer (will be allocated
 *                      or reallocated if necessary).
 * @param cache_element The current number of cached elements. This will be
 *                      updated when the cache is expanded.
 * @param ne            The tensor shape array (number of elements in each dimension).
 * @param nb            The stride size for each dimension.
 * @param dtype         Data type of cached tensor.
 * @param dims          The number of tensor dimensions.
 * @param value         The scalar value used to fill the tensor (supports zero
 *                      initialization via memset or arbitrary values via fill_scalar).
 * @return              A tensor smart pointer created from the cached buffer.
 */
static acl_tensor_ptr get_cache_acl_tensor(ggml_backend_cann_context & ctx,
                                           void **                     buffer,
                                           int64_t &                   cache_element,
                                           int64_t *                   ne,
                                           size_t *                    nb,
                                           ggml_type                   dtype,
                                           int64_t                     dims,
                                           float                       value) {
    // Calculate total number of elements
    int64_t n_element = 1;
    for (int i = 0; i < dims; i++) {
        n_element *= ne[i];
    }
    size_t size = n_element * ggml_type_size(dtype);

    // Allocate or expand cache if needed
    if (cache_element < n_element) {
        if (*buffer != nullptr) {
            aclrtFree(*buffer);
            *buffer = nullptr;
        }

        ACL_CHECK(aclrtMalloc(buffer, size, ACL_MEM_MALLOC_HUGE_FIRST));
        cache_element = n_element;

        // Initialize cache
        int64_t        pool_ne[1] = { n_element };
        size_t         pool_nb[1] = { ggml_type_size(dtype) };
        acl_tensor_ptr acl_value =
            ggml_cann_create_tensor(*buffer, ggml_cann_type_mapping(dtype), ggml_type_size(dtype), pool_ne, pool_nb, 1);
        aclnn_fill_scalar(ctx, value, acl_value.get());
    }

    return ggml_cann_create_tensor(*buffer, ggml_cann_type_mapping(dtype), ggml_type_size(dtype), ne, nb, dims);
}

void ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // build gamma.
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    // gamma's type is the same with dst.
    acl_gamma_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * src->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma = get_cache_acl_tensor(
        ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, src->ne, acl_gamma_nb, dst->type,
        1,    // dims
        1.0f  // value
    );

    // build rstd.
    int64_t acl_rstd_ne[] = { src->ne[1], src->ne[2], src->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    // rstd will always be F32.
    acl_rstd_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS - 1,
                             0.0f  // value
        );

    GGML_CANN_CALL_ACLNN_OP(ctx, RmsNorm, acl_src.get(), acl_gamma.get(), eps, acl_dst.get(), acl_rstd.get());
}

// TODO: performace is low.
void ggml_cann_diag_mask(ggml_backend_cann_context & ctx, ggml_tensor * dst, float value) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    const int n_past = ((int32_t *) dst->op_params)[0];

    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), ggml_nbytes(src));
    void *               buffer = one_tensor_allocator.get();

    acl_tensor_ptr mask_tensor = ggml_cann_create_tensor(buffer, ggml_cann_type_mapping(src->type),
                                                         ggml_type_size(src->type), src->ne, src->nb, GGML_MAX_DIMS);

    aclnn_fill_scalar(ctx, value, mask_tensor.get());

    float          alphaValue = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceTriu, mask_tensor.get(), n_past + 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), n_past + 1, acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst.get(), mask_tensor.get(), alpha.get());
}

/**
 * @brief Permutes the dimensions of a tensor according to a specified order.
 *
 * This function permutes the dimensions of the source tensor `acl_src`
 * according to the order specified in the `new_dim` array and stores the result
 * in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose dimensions will be permuted.
 * @param acl_dst The destination tensor where the permuted result will be
 * stored.
 * @param new_dim An array specifying the new order of dimensions for the
 * tensor.
 * @param dims The number of dimensions in the tensor.
 */
static void aclnn_permute(ggml_backend_cann_context & ctx,
                          aclTensor *                 acl_src,
                          aclTensor *                 acl_dst,
                          int64_t *                   new_dim,
                          uint64_t                    dims) {
    acl_int_array_ptr acl_dims = ggml_cann_create_int_array(new_dim, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, Permute, acl_src, acl_dims.get(), acl_dst);
}

static void ggml_cann_im2col_2d_post_process(ggml_backend_cann_context & ctx,
                                             ggml_tensor *               dst,
                                             ggml_tensor *               src1,
                                             aclTensor *                 tmp_cast_tensor,
                                             aclTensor *                 tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t        dst_ne[] = { dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3] };
    size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[3] };
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t permute_dim[] = { 0, 2, 1 };
    if (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, acl_dst.get(), permute_dim, 3);
    } else {
        aclnn_permute(ctx, tmp_im2col_tensor, acl_dst.get(), permute_dim, 3);
    }
}

static void ggml_cann_im2col_1d_post_process(ggml_backend_cann_context &  ctx,
                                             ggml_tensor *                dst,
                                             ggml_tensor *                src1,
                                             aclTensor *                  tmp_cast_tensor,
                                             aclTensor *                  tmp_im2col_tensor,
                                             const std::vector<int64_t> & im2col_op_params) {
    // get params
    const int64_t KH             = im2col_op_params[0];
    const int64_t KW             = im2col_op_params[1];
    const int64_t IW             = im2col_op_params[2];
    const int64_t IC             = im2col_op_params[3];
    const int64_t N              = im2col_op_params[4];
    const int64_t OH             = im2col_op_params[5];
    const int64_t OW             = im2col_op_params[6];
    const int64_t s0             = im2col_op_params[7];
    const int64_t p0             = im2col_op_params[8];
    const int64_t d0             = im2col_op_params[9];
    const int64_t n_bytes_factor = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    ggml_cann_pool_alloc tmp_permute_allocator(ctx.pool());
    tmp_permute_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
    void * tmp_permute_buffer = tmp_permute_allocator.get();

    int64_t tmp_permute_ne[] = { IC * KH * KW, OW * OH * n_bytes_factor, N };
    size_t  tmp_permute_nb[GGML_MAX_DIMS - 1];
    tmp_permute_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    acl_tensor_ptr tmp_permute_tensor =
        ggml_cann_create_tensor(tmp_permute_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_permute_ne, tmp_permute_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t permute_dim[] = { 0, 2, 1 };
    if (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, tmp_permute_tensor.get(), permute_dim, 3);
    } else {
        aclnn_permute(ctx, tmp_im2col_tensor, tmp_permute_tensor.get(), permute_dim, 3);
    }

    // number of times the kernel moves in W dimension
    const int n_step_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    size_t    offset;
    void *    cur_dst_buffer = dst->data, *cur_permute_buffer = tmp_permute_buffer;

    // memory copy with offset to restore 1D im2col from 2d
    if (IC > 1) {
        offset          = IC * KH * KW * n_step_w * ggml_type_size(dst->type);
        size_t cpy_size = KH * KW * ggml_type_size(dst->type);

        for (int c = 0; c < IC; c++) {
            cur_permute_buffer = (char *) tmp_permute_buffer + offset + KH * KW * c * ggml_type_size(dst->type);
            cur_dst_buffer     = (char *) dst->data + c * KH * KW * n_step_w * ggml_type_size(dst->type);

            for (int i = 0; i < n_step_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(cur_dst_buffer, cpy_size, cur_permute_buffer, cpy_size,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
                cur_dst_buffer     = (char *) cur_dst_buffer + KH * KW * ggml_type_size(dst->type);
                cur_permute_buffer = (char *) cur_permute_buffer + KH * KW * IC * ggml_type_size(dst->type);
            }
        }
    } else {
        offset = KH * KW * n_step_w * ggml_type_size(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->data, offset, (char *) tmp_permute_buffer + offset, offset,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
    }
}

void ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // kernel
    ggml_tensor * src1 = dst->src[1];  // input

    GGML_TENSOR_BINARY_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    const bool    is_2D = ((const int32_t *) (dst->op_params))[6] == 1;
    const int32_t s0    = ((const int32_t *) (dst->op_params))[0];
    const int32_t s1    = is_2D ? ((const int32_t *) (dst->op_params))[1] : 1;
    const int32_t p0    = ((const int32_t *) (dst->op_params))[2];
    const int32_t p1    = is_2D ? ((const int32_t *) (dst->op_params))[3] : 1;
    const int32_t d0    = ((const int32_t *) (dst->op_params))[4];
    const int32_t d1    = is_2D ? ((const int32_t *) (dst->op_params))[5] : 1;

    const int64_t N  = ne13;
    const int64_t IC = ne12;
    const int64_t KH = ne01;
    const int64_t KW = ne00;
    const int64_t IW = ne10;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    const int64_t n_bytes_factor = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    acl_tensor_ptr acl_src1        = ggml_cann_create_tensor(src1);
    int64_t        tmp_im2col_ne[] = { OW * OH * n_bytes_factor, IC * KH * KW, N };
    size_t         tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_size(src1->type);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_size(src1) * n_bytes_factor);
    void *               tmp_im2col_buffer = im2col_allocator.get();

    acl_tensor_ptr tmp_im2col_tensor =
        ggml_cann_create_tensor(tmp_im2col_buffer, ggml_cann_type_mapping(src1->type), ggml_type_size(src1->type),
                                tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims   = { KH, KW };
    std::vector<int64_t> dilation_size = { d1, d0 };
    std::vector<int64_t> padding_dims  = { p1, p0 };
    std::vector<int64_t> stride_dims   = { s1, s0 };
    acl_int_array_ptr    kernel_size   = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr    dilations     = ggml_cann_create_int_array(dilation_size.data(), 2);
    acl_int_array_ptr    paddings      = ggml_cann_create_int_array(padding_dims.data(), 2);
    acl_int_array_ptr    strides       = ggml_cann_create_int_array(stride_dims.data(), 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Im2col, acl_src1.get(), kernel_size.get(), dilations.get(), paddings.get(),
                            strides.get(), tmp_im2col_tensor.get());

    // Cast if dst is f16.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    void *               tmp_cast_buffer = nullptr;
    if (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
        tmp_cast_buffer = tmp_cast_allocator.get();
        size_t temp_cast_nb[GGML_MAX_DIMS - 1];
        temp_cast_nb[0] = ggml_type_size(dst->type);
        for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
            temp_cast_nb[i] = temp_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                    tmp_im2col_ne, temp_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor.get(), tmp_cast_tensor.get(), ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    if (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get());
    } else {
        std::vector<int64_t> im2col_op_params = { KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_bytes_factor };
        ggml_cann_im2col_1d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get(),
                                         im2col_op_params);
    }
}

/**
 * @brief Applies element-wise exponential function to the elements of a tensor.
 *
 * This function computes the exponential of each element in the source tensor
 * `acl_src` and stores the result back into the same tensor.
 * The operation is defined as:
 * \f[
 *     \text {acl_src }_i=e^{acl\_src_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The tensor on which the exponential function will be applied.
 */
static void aclnn_exp(ggml_backend_cann_context & ctx, aclTensor * acl_src) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceExp, acl_src);
}

void aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    if (acl_dst == nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCos, acl_src);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cos, acl_src, acl_dst);
    }
}

void aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    if (acl_dst == nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSin, acl_src);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sin, acl_src, acl_dst);
    }
}

void ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim        = dst->op_params[0];
    const int max_period = dst->op_params[1];
    int       half       = dim / 2;

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    float   start             = 0;
    float   stop              = half;
    float   step              = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[]   = { half };
    size_t  tmp_arange_nb[]   = { sizeof(dst->type) };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * sizeof(dst->type));
    void *               tmp_arange_buffer = arange_allocator.get();
    acl_tensor_ptr       tmp_arange_tensor =
        ggml_cann_create_tensor(tmp_arange_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_arange_ne, tmp_arange_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor.get(), start, stop, step, n_elements_arange);

    // freq
    float freq_param = -logf(max_period) / half;
    bool  inplace    = true;
    aclnn_muls(ctx, tmp_arange_tensor.get(), freq_param, nullptr, inplace);
    aclnn_exp(ctx, tmp_arange_tensor.get());

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_permute_ne[] = { src->ne[1], src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_permute_nb[GGML_MAX_DIMS];
    tmp_permute_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    ggml_cann_pool_alloc permute_allocator(ctx.pool(), ggml_nbytes(src));
    void *               tmp_permute_buffer = permute_allocator.get();
    acl_tensor_ptr       tmp_permute_tensor =
        ggml_cann_create_tensor(tmp_permute_buffer, ggml_cann_type_mapping(src->type), ggml_type_size(src->type),
                                tmp_permute_ne, tmp_permute_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t permute_dim[] = { 0, 1, 3, 2 };
    int64_t num_dims      = 4;
    aclnn_permute(ctx, acl_src.get(), tmp_permute_tensor.get(), permute_dim, num_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = { src->ne[1] * half, src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    int mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void *               tmp_mul_buffer = mul_allocator.get();
    acl_tensor_ptr       tmp_mul_tensor =
        ggml_cann_create_tensor(tmp_mul_buffer, ggml_cann_type_mapping(src->type), ggml_type_size(src->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_permute_tensor.get(), tmp_arange_tensor.get(), tmp_mul_tensor.get());

    // cos
    ggml_cann_pool_alloc cos_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void *               tmp_cos_buffer = cos_allocator.get();
    acl_tensor_ptr       tmp_cos_tensor =
        ggml_cann_create_tensor(tmp_cos_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor.get(), tmp_cos_tensor.get());

    // sin
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void *               tmp_sin_buffer = sin_allocator.get();
    acl_tensor_ptr       tmp_sin_tensor =
        ggml_cann_create_tensor(tmp_sin_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_sin(ctx, tmp_mul_tensor.get(), tmp_sin_tensor.get());

    // concat
    int64_t             concat_dim  = 3;
    acl_tensor_ptr      acl_dst     = ggml_cann_create_tensor(dst);
    acl_tensor_list_ptr tensor_list = ggml_cann_create_tensor_list(tmp_cos_tensor, tmp_sin_tensor);
    aclnn_concat(ctx, tensor_list.get(), acl_dst.get(), concat_dim);
}

/**
 * @brief Raises each element of a tensor to the power of the corresponding
 * element in another tensor.
 *
 * This function computes the element-wise power of the destination tensor
 * `acl_dst` raised to the power of the exponent tensor `acl_exp`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=acl\_dst_i^{\text {acl_exp }_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor, which also serves as the base tensor.
 * @param acl_exp The exponent tensor, each element of which is used to raise
 * the corresponding element in the destination tensor.
 */
static void aclnn_pow_tensor_tensor(ggml_backend_cann_context & ctx, aclTensor * acl_dst, aclTensor * acl_exp) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplacePowTensorTensor, acl_dst, acl_exp);
}

/**
 * @brief Generate a range of values and apply a scalar base exponentiation.
 *
 * This function creates an evenly spaced sequence from `start` to `stop` (exclusive),
 * with step size `step`, stores it in a temporary buffer, and then computes:
 *
 * @f[
 * slope[i] = m^{\left( start + i \cdot step \right)}, \quad 0 \le i < size
 * @f]
 *
 * The results are written to the provided @p slope_buffer.
 *
 * @param ctx           CANN backend context for memory allocation and operator execution.
 * @param slope_buffer  Pointer to the output buffer (float array) for the computed slope values.
 * @param m             Scalar base for the exponentiation.
 * @param size          Number of elements in the generated sequence.
 * @param start         Starting exponent offset.
 * @param stop          Stopping exponent offset (exclusive).
 * @param step          Step size for the exponent increment.
 * @param dtype         Data type for slope tensor.
 */
static void aclnn_get_slope_inner(ggml_backend_cann_context & ctx,
                                  void *                      slope_buffer,
                                  float                       m,
                                  int64_t                     size,
                                  float                       start,
                                  float                       stop,
                                  float                       step,
                                  ggml_type                   dtype) {
    aclDataType acl_type  = ggml_cann_type_mapping(dtype);
    size_t      type_size = ggml_type_size(dtype);

    int64_t ne[] = { size };
    size_t  nb[] = { type_size };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), size * type_size);
    void *               arange_buffer = arange_allocator.get();

    acl_tensor_ptr arange_tensor = ggml_cann_create_tensor(arange_buffer, acl_type, type_size, ne, nb, 1);
    aclnn_arange(ctx, arange_tensor.get(), start, stop, step, size);

    acl_tensor_ptr slope_tensor = ggml_cann_create_tensor(slope_buffer, acl_type, type_size, ne, nb, 1);

    acl_scalar_ptr sc = ggml_cann_create_scalar(&m, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, sc.get(), arange_tensor.get(), slope_tensor.get());
}

/**
 * @brief Compute slope values for multiple attention heads based on ALiBi bias parameters.
 *
 * This function generates slope values for each attention head according to the ALiBi
 * (Attention with Linear Biases) method. It splits the computation into two ranges depending
 * on whether the head index is less than @p n_head_log2 or not, and uses different base values
 * (`m0` and `m1`) for the exponentiation.
 *
 * @f[
 * slope[h] =
 * \begin{cases}
 * m_0^{(h + 1)}, & h < n\_head\_log2 \\
 * m_1^{\left( 2 \cdot (h - n\_head\_log2) + 1 \right)}, & h \geq n\_head\_log2
 * \end{cases}
 * \quad , \quad \text{if } max\_bias > 0
 * @f]
 *
 * If @p max_bias <= 0, all slope values are set to 1.0.
 *
 * @param ctx           CANN backend context for memory allocation and operator execution.
 * @param n_head        Total number of attention heads.
 * @param slope_buffer  Pointer to the output buffer (float array) for storing slopes.
 * @param max_bias      Maximum bias value for slope computation.
 * @param dtype         Data type for slope tensor.
 *
*/
static void aclnn_get_slope(ggml_backend_cann_context & ctx,
                            int64_t                     n_head,
                            void *                      slope_buffer,
                            float                       max_bias,
                            ggml_type                   dtype) {
    const int n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // const float slope = (max_bias > 0.0f) ?
    //                          h < n_head_log2 ?
    //                              powf(m0, h + 1) :
    //                              powf(m1, 2*(h - n_head_log2) + 1) :
    //                          1.0f;
    // arange1
    float start = 0 + 1;
    float end   = (n_head_log2 - 1) + 1;
    float step  = 1;
    float count = n_head_log2;
    // end needs to be +1 because aclnn uses a left-closed, right-open interval.
    aclnn_get_slope_inner(ctx, slope_buffer, m0, count, start, end + 1, step, dtype);
    if (n_head_log2 < n_head) {
        // arange2
        start = 2 * (n_head_log2 - n_head_log2) + 1;
        end   = 2 * ((n_head - 1) - n_head_log2) + 1;
        step  = 2;
        count = n_head - n_head_log2;
        aclnn_get_slope_inner(ctx, (char *) slope_buffer + n_head_log2 * sizeof(float), m1, count, start, end + 1, step,
                              dtype);
    }
}

/**
 * @brief Add ALiBi (Attention with Linear Biases) positional biases to the attention mask.
 *
 * This function computes the ALiBi slopes for each attention head (if max_bias > 0),
 * multiplies them with the attention mask to produce bias tensors, and adds these biases
 * to the destination tensor (@p dst).
 *
 * The function performs necessary broadcasting of the mask and slope tensors to match
 * the shape of the destination tensor, then applies element-wise multiplication and addition
 * using CANN operators.
 *
 * @param ctx         CANN backend context for memory management and operator execution.
 * @param mask        Input attention mask tensor, assumed to be contiguous.
 * @param dst         Destination tensor to which ALiBi biases will be added.
 * @param dst_ptr     Pointer to the memory of the destination tensor.
 * @param max_bias    Maximum bias value controlling the slope scaling.
 *
 * @note
 * - Write data into dst_ptr using only the shape information of the dst tensor.
 * - `GGML_MAX_DIMS + 2` is used to extend tensor dimensions for broadcasting.
 */
static void aclnn_add_alibi(ggml_backend_cann_context & ctx,
                            ggml_tensor *               mask,
                            ggml_tensor *               dst,
                            void *                      dst_ptr,
                            float                       max_bias) {
    void * slope_buffer = nullptr;
    void * bias_buffer  = nullptr;

    if (max_bias > 0.0f) {
        int64_t              n_heads = dst->ne[2];
        ggml_cann_pool_alloc slope_allocator(ctx.pool(), n_heads * sizeof(float));
        slope_buffer = slope_allocator.get();
        ggml_cann_pool_alloc bias_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_size(dst));
        bias_buffer = bias_allocator.get();
        aclnn_get_slope(ctx, n_heads, slope_buffer, max_bias, GGML_TYPE_F32);
    }

    // broadcast for mask, slop and dst;
    int64_t nr2 = dst->ne[2] / mask->ne[2];
    int64_t nr3 = dst->ne[3] / mask->ne[3];

    // broadcast the mask across rows
    int64_t mask_ne[] = { mask->ne[0], dst->ne[1], mask->ne[2], 1, mask->ne[3], 1 };
    size_t  mask_nb[] = { mask_nb[0] = mask->nb[0], mask_nb[1] = mask->nb[1], mask_nb[2] = mask->nb[2],
                          mask_nb[3] = mask->nb[2], mask_nb[4] = mask->nb[3], mask_nb[5] = mask->nb[3] };

    int64_t dst_ne[] = { dst->ne[0], dst->ne[1], mask->ne[2], nr2, mask->ne[3], nr3 };
    size_t  dst_nb[] = { dst_nb[0] = dst->nb[0], dst_nb[1] = dst->nb[1], dst_nb[2] = dst->nb[2],
                         dst_nb[3] = dst->nb[2], dst_nb[4] = dst->nb[3], dst_nb[5] = dst->nb[3] };

    // slope is a 1 dim tensor, slope.ne2 == dst.ne2
    int64_t slope_ne[] = { 1, 1, mask->ne[2], nr2, 1, 1 };
    size_t  slope_nb[GGML_MAX_DIMS + 2];
    slope_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS + 2; i++) {
        slope_nb[i] = slope_nb[i - 1] * slope_ne[i - 1];
    }

    acl_tensor_ptr acl_slope =
        ggml_cann_create_tensor(slope_buffer, ACL_FLOAT, sizeof(float), slope_ne, slope_nb, GGML_MAX_DIMS + 2);
    acl_tensor_ptr acl_mask = ggml_cann_create_tensor(mask, mask_ne, mask_nb, GGML_MAX_DIMS + 2);

    // write data into dst_ptr using only the shape information of the dst tensor.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_ptr, ggml_cann_type_mapping(dst->type),
                                                     ggml_type_size(dst->type), dst_ne, dst_nb, GGML_MAX_DIMS + 2);

    if (max_bias > 0.0f) {
        int64_t bias_ne[] = { mask->ne[0], dst->ne[1], mask->ne[2], nr2, mask->ne[3], 1 };
        size_t  bias_nb[GGML_MAX_DIMS + 2];
        bias_nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS + 2; i++) {
            bias_nb[i] = bias_nb[i - 1] * bias_ne[i - 1];
        }
        acl_tensor_ptr bias_tensor =
            ggml_cann_create_tensor(bias_buffer, ACL_FLOAT, sizeof(float), bias_ne, bias_nb, GGML_MAX_DIMS + 2);

        aclnn_mul(ctx, acl_slope.get(), acl_mask.get(), bias_tensor.get());
        aclnn_add(ctx, acl_dst.get(), bias_tensor.get());
    } else {
        aclnn_add(ctx, acl_dst.get(), acl_mask.get());
    }
}

void ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_cann_dup(ctx, dst);
}

/**
 * @brief Applies the softmax function to a tensor along a specified dimension.
 *
 * This function computes the softmax of the source tensor `acl_src` along the
 * specified dimension `dim` and stores the result in the destination tensor
 * `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the softmax function will be
 * applied.
 * @param dim The dimension along which the softmax function will be computed.
 * @param acl_dst The destination tensor where the softmax results will be
 * stored.
 */
static void aclnn_softmax(ggml_backend_cann_context & ctx, aclTensor * acl_src, int64_t dim, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, acl_src, dim, acl_dst);
}

void ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];  // mask

    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // input mul scale
    acl_scalar_ptr       acl_scale = ggml_cann_create_scalar(&scale, aclDataType::ACL_FLOAT);
    ggml_cann_pool_alloc src_tensor_allocator(ctx.pool(), ggml_nbytes(src0));
    void *               src_tensor_buffer = src_tensor_allocator.get();
    acl_tensor_ptr       softmax_tensor = ggml_cann_create_tensor(src_tensor_buffer, ggml_cann_type_mapping(src0->type),
                                                                  ggml_element_size(src0), src0->ne, src0->nb, GGML_MAX_DIMS);

    aclnn_muls(ctx, acl_src0.get(), scale, softmax_tensor.get(), false);

    // mask
    if (src1) {
        aclnn_add_alibi(ctx, src1, src0, src_tensor_buffer, max_bias);
    }
    // softmax
    aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
}

/**
 * @brief Performs index select operation on a 4D tensor using the CANN backend.
 *
 * This function applies the `IndexSelect` operation along a specific dimension
 * of the source tensor (`src_buffer`) using the indices from the index tensor (`index`).
 * It iterates over the last two dimensions of the source tensor, creates the corresponding
 * CANN tensors for the source, index, and output slices, and executes the `IndexSelect`
 * operation for each slice.
 *
 * @param ctx The context for CANN backend operations.
 * @param src_buffer The source buffer containing the 4D input tensor data.
 * @param src_ne The dimensions of the source tensor.
 * @param src_nb The strides (byte offsets) of the source tensor.
 * @param dst_buffer The destination buffer where the output tensor data will be written.
 * @param dst_ne The dimensions of the destination tensor.
 * @param dst_nb The strides (byte offsets) of the destination tensor.
 * @param index The index tensor specifying the indices to select from the source tensor.
 * @param type The data type of the source and destination tensors.
 */
static void aclnn_index_select_4d(ggml_backend_cann_context & ctx,
                                  void *                      src_buffer,
                                  int64_t *                   src_ne,
                                  size_t *                    src_nb,
                                  void *                      dst_buffer,
                                  int64_t *                   dst_ne,
                                  size_t *                    dst_nb,
                                  ggml_tensor *               index,
                                  ggml_type                   type) {
    for (int64_t i = 0; i < src_ne[3]; i++) {
        for (int64_t j = 0; j < src_ne[2]; j++) {
            // src
            acl_tensor_ptr acl_src_tensor =
                ggml_cann_create_tensor((char *) src_buffer + i * src_nb[3] + j * src_nb[2],
                                        ggml_cann_type_mapping(type), ggml_type_size(type), src_ne, src_nb, 2);

            // index
            acl_tensor_ptr acl_index = ggml_cann_create_tensor(
                (char *) index->data + (i % index->ne[2]) * index->nb[2] + (j % index->ne[1]) * index->nb[1],
                ggml_cann_type_mapping(index->type), ggml_element_size(index), index->ne, index->nb, 1);

            // out
            acl_tensor_ptr acl_out =
                ggml_cann_create_tensor((char *) dst_buffer + i * dst_nb[3] + j * dst_nb[2],
                                        ggml_cann_type_mapping(type), ggml_type_size(type), dst_ne, dst_nb, 2);
            GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, acl_src_tensor.get(), 0, acl_index.get(), acl_out.get());
        }
    }
}

/**
 * @brief Performs inplace index copy operation on a 4D tensor using the CANN backend.
 *
 * This function applies the `IndexCopy` operation along a specific dimension of the
 * destination tensor (`dst_buffer`) by copying elements from the source tensor (`src_buffer`)
 * to positions specified by the index tensor (`index`).
 * It iterates over the last two dimensions of the tensors, creates the corresponding
 * CANN tensors for source, index, and destination slices, and performs the index copy
 * operation for each slice.
 *
 * @param ctx The context for CANN backend operations.
 * @param src_buffer The source buffer containing the 4D input tensor data to be copied.
 * @param src_ne The dimensions of the source tensor.
 * @param src_nb The strides (byte offsets) of the source tensor.
 * @param dst_buffer The destination buffer where values will be copied to.
 * @param dst_ne The dimensions of the destination tensor.
 * @param dst_nb The strides (byte offsets) of the destination tensor.
 * @param index The index tensor specifying target positions in the destination tensor.
 * @param type The data type of the source and destination tensors.
 */
static void aclnn_index_copy_4d(ggml_backend_cann_context & ctx,
                                void *                      src_buffer,
                                int64_t *                   src_ne,
                                size_t *                    src_nb,
                                void *                      dst_buffer,
                                int64_t *                   dst_ne,
                                size_t *                    dst_nb,
                                ggml_tensor *               index,
                                ggml_type                   type) {
    for (int64_t i = 0; i < src_ne[3]; i++) {
        for (int64_t j = 0; j < src_ne[2]; j++) {
            // src
            acl_tensor_ptr acl_src_tensor =
                ggml_cann_create_tensor((char *) src_buffer + i * src_nb[3] + j * src_nb[2],
                                        ggml_cann_type_mapping(type), ggml_type_size(type), src_ne, src_nb, 2);

            // index
            acl_tensor_ptr acl_index = ggml_cann_create_tensor(
                (char *) index->data + (i % index->ne[2]) * index->nb[2] + (j % index->ne[1]) * index->nb[1],
                ggml_cann_type_mapping(index->type), ggml_element_size(index), index->ne, index->nb, 1);

            // out
            acl_tensor_ptr acl_out =
                ggml_cann_create_tensor((char *) dst_buffer + i * dst_nb[3] + j * dst_nb[2],
                                        ggml_cann_type_mapping(type), ggml_type_size(type), dst_ne, dst_nb, 2);
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexCopy, acl_out.get(), 0, acl_index.get(), acl_src_tensor.get());
        }
    }
}

void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // src
    ggml_tensor * src1 = dst->src[1];  // index

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    switch (src0->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            if (src0->type == dst->type) {
                aclnn_index_select_4d(ctx, src0->data, src0->ne, src0->nb, dst->data, dst->ne, dst->nb, src1,
                                      dst->type);
            } else {
                acl_tensor_ptr       acl_src0 = ggml_cann_create_tensor(src0);
                ggml_cann_pool_alloc src_buffer_allocator(ctx.pool(), ggml_nelements(src0) * ggml_element_size(dst));
                void *               src_trans_buffer = src_buffer_allocator.get();
                size_t               src_trans_nb[GGML_MAX_DIMS];
                src_trans_nb[0] = dst->nb[0];
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr src_trans_tensor =
                    ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(dst->type),
                                            ggml_type_size(dst->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), src_trans_tensor.get(), ggml_cann_type_mapping(dst->type));
                aclnn_index_select_4d(ctx, src_trans_buffer, src0->ne, src_trans_nb, dst->data, dst->ne, dst->nb, src1,
                                      dst->type);
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                // add 1 dim for bcast mul.
                size_t  weight_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                int64_t weight_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
                int64_t scale_offset = 0;
                // [3,4,5,64] -> [3,4,5,2,32]
                weight_ne[0]         = QK8_0;
                weight_ne[1]         = src0->ne[0] / QK8_0;
                weight_nb[0]         = sizeof(int8_t);
                weight_nb[1]         = weight_nb[0] * weight_ne[0];
                for (int i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    weight_ne[i] = src0->ne[i - 1];
                    weight_nb[i] = weight_nb[i - 1] * weight_ne[i - 1];
                }
                // [3,4,5,64] -> [3,4,5,2,1]
                scale_ne[0] = 1;
                scale_ne[1] = src0->ne[0] / QK8_0;
                scale_nb[0] = sizeof(uint16_t);
                scale_nb[1] = scale_nb[0] * scale_ne[0];
                for (int i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    scale_ne[i] = src0->ne[i - 1];
                    scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                }
                // [3,4,5,64] -> [3,4,5,2,32]
                dequant_ne    = weight_ne;
                dequant_nb[0] = ggml_type_size(dst->type);
                for (int i = 1; i < GGML_MAX_DIMS + 1; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                }
                scale_offset = ggml_nelements(src0) * sizeof(int8_t);
                ggml_cann_pool_alloc dequant_buffer_allocator(ctx.pool(),
                                                              ggml_nelements(src0) * ggml_type_size(dst->type));
                acl_tensor_ptr       acl_weight_tensor = ggml_cann_create_tensor(src0->data, ACL_INT8, sizeof(int8_t),
                                                                                 weight_ne, weight_nb, GGML_MAX_DIMS + 1);
                acl_tensor_ptr       acl_scale_tensor =
                    ggml_cann_create_tensor(src0->data, ACL_FLOAT16, sizeof(uint16_t), scale_ne, scale_nb,
                                            GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_offset);
                acl_tensor_ptr dequant_tensor =
                    ggml_cann_create_tensor(dequant_buffer_allocator.get(), ggml_cann_type_mapping(dst->type),
                                            ggml_type_size(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
                aclnn_mul(ctx, acl_weight_tensor.get(), acl_scale_tensor.get(), dequant_tensor.get());
                dequant_nb[0] = ggml_type_size(dst->type);
                dequant_ne    = src0->ne;
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                }
                aclnn_index_select_4d(ctx, dequant_buffer_allocator.get(), dequant_ne, dequant_nb, dst->data, dst->ne,
                                      dst->nb, src1, dst->type);
                break;
            }
        default:
            GGML_ABORT("Unsupported tensor type for GGML_OP_GET_ROWS");
            break;
    }
}

void ggml_cann_set_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // src
    ggml_tensor * src1 = dst->src[1];  // index

    switch (dst->type) {
        case GGML_TYPE_F32:
            {
                aclnn_index_copy_4d(ctx, src0->data, src0->ne, src0->nb, dst->data, dst->ne, dst->nb, src1, dst->type);
                break;
            }
        case GGML_TYPE_F16:
            {
                acl_tensor_ptr       acl_src0 = ggml_cann_create_tensor(src0);
                ggml_cann_pool_alloc src_buffer_allocator(ctx.pool(), ggml_nelements(src0) * sizeof(uint16_t));
                void *               src_trans_buffer = src_buffer_allocator.get();
                size_t               src_trans_nb[GGML_MAX_DIMS];
                src_trans_nb[0] = sizeof(uint16_t);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr src_trans_tensor = ggml_cann_create_tensor(
                    src_trans_buffer, ACL_FLOAT16, ggml_type_size(dst->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), src_trans_tensor.get(), ggml_cann_type_mapping(dst->type));
                aclnn_index_copy_4d(ctx, src_trans_buffer, src0->ne, src_trans_nb, dst->data, dst->ne, dst->nb, src1,
                                    dst->type);
                break;
            }
        default:
            GGML_ABORT("Unsupported tensor type for GGML_OP_SET_ROWS");
            break;
    }
}

/**
 * @brief Repeats elements of a tensor along a specified dimension.
 *
 * This function repeats each element of the source tensor `acl_src` a specified
 * number of times (`repeats`) along the specified dimension `dim` and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be repeated.
 * @param acl_dst The destination tensor where the repeated elements will be
 * stored.
 * @param dim The dimension along which the elements will be repeated.
 * @param repeats The number of times each element will be repeated.
 * @param output_size The size of the output tensor.
 */
static void aclnn_repeat_interleave(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    aclTensor *                 acl_dst,
                                    int64_t                     dim,
                                    int64_t                     repeats,
                                    int64_t                     output_size) {
    GGML_CANN_CALL_ACLNN_OP(ctx, RepeatInterleaveIntWithDim, acl_src, repeats, dim, output_size, acl_dst);
}

/**
 * @brief Performs matrix multiplication with floating-point precision on
 * tensors using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor and the
 * weight tensor, handling broadcasting and transposing as needed, and stores
 * the result in the destination tensor `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_cann_mat_mul_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];  // weight
    ggml_tensor * input  = dst->src[1];  // input

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(input, weight, dst);

    int64_t n_dims = bcast_dims;
    if (bcast_input_ne[3] == bcast_weight_ne[3] && bcast_input_ne[3] == 1) {
        if (bcast_input_ne[2] == 1 && bcast_weight_ne[2] == 1) {
            n_dims = 2;
        } else if (bcast_input_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_input_tensor = ggml_cann_create_tensor(input, bcast_input_ne, bcast_input_nb, n_dims);
    int64_t        transpose_ne[]   = { bcast_weight_ne[1], bcast_weight_ne[0], bcast_weight_ne[2],
                                        bcast_weight_ne[3], bcast_weight_ne[4], bcast_weight_ne[5] };
    size_t         transpose_nb[]   = { bcast_weight_nb[1], bcast_weight_nb[0], bcast_weight_nb[2],
                                        bcast_weight_nb[3], bcast_weight_nb[4], bcast_weight_nb[5] };
    acl_tensor_ptr acl_weight_tensor;

    // Only check env once.
    static bool weight_to_nz = parse_bool(get_env("GGML_CANN_WEIGHT_NZ").value_or("on"));
    if (weight_to_nz && is_matmul_weight(weight)) {
        acl_weight_tensor = ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_FRACTAL_NZ);
    } else {
        acl_weight_tensor = ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
    }
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    switch (n_dims) {
        case 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, Mm, acl_input_tensor.get(), acl_weight_tensor.get(), acl_dst.get(), 2);
            break;
        case 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_input_tensor.get(), acl_weight_tensor.get(), acl_dst.get(),
                                    2);
            break;
        default:
            // ALLOW_FP32_DOWN_PRECISION, when input is
            // fp32, atlas a2 will transpose it to HFLOAT32.
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_input_tensor.get(), acl_weight_tensor.get(), acl_dst.get(), 1);
            break;
    }
}

/**
 * @brief Performs matrix multiplication with quantized weights and
 * floating-point inputs using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor `src1` and
 * the weight tensor `src0`, handling broadcasting, transposing, and
 * quantization as needed, and stores the result in the destination tensor
 * `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_cann_mul_mat_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst, const enum ggml_type type) {
    ggml_tensor * src0 = dst->src[0];  // weight
    ggml_tensor * src1 = dst->src[1];  // input

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    float weight_elem_size;
    if (type == GGML_TYPE_Q4_0) {
        weight_elem_size = float(sizeof(uint8_t)) / 2;
    } else if (type == GGML_TYPE_Q8_0) {
        weight_elem_size = float(sizeof(uint8_t));
    } else {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    float  weight_nb[]   = { src0->ne[0] * weight_elem_size, weight_elem_size };
    size_t weight_stride = src0->ne[1] * src0->ne[0] * weight_elem_size;
    size_t weight_size   = weight_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_size = sizeof(uint16_t);
    size_t scale_nb[]      = { src0->ne[0] / QK8_0 * scale_elem_size, scale_elem_size };
    size_t scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_size;
    char * scale_offset    = (char *) src0->data + weight_size;

    // input
    size_t               input_elem_size = sizeof(uint16_t);
    int64_t              input_ne[]      = { src1->ne[0], src1->ne[1] };
    size_t               input_nb[]      = { input_elem_size, input_ne[0] * input_elem_size };
    size_t               input_stride    = input_ne[0] * input_ne[1] * input_elem_size;
    ggml_cann_pool_alloc input_alloctor(ctx.pool());
    void *               input_buffer = src1->data;

    // case in
    if (src1->type != GGML_TYPE_F16) {
        acl_tensor_ptr acl_src1_tensor = ggml_cann_create_tensor(src1);
        input_buffer                   = input_alloctor.alloc(ggml_nelements(src1) * input_elem_size);

        int64_t * input_cast_ne = src1->ne;
        size_t    input_cast_nb[GGML_MAX_DIMS];
        input_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_cast_nb[i] = input_cast_nb[i - 1] * input_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_input_tensor = ggml_cann_create_tensor(input_buffer, ACL_FLOAT16, input_elem_size,
                                                                  input_cast_ne, input_cast_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src1_tensor.get(), acl_input_tensor.get(), ACL_FLOAT16);
    }

    // output
    size_t               output_elem_size = sizeof(uint16_t);
    size_t               output_nb[]      = { output_elem_size, dst->ne[0] * output_elem_size };
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    void *               output_buffer = output_allocator.alloc(ggml_nelements(dst) * output_elem_size);
    size_t               output_stride = dst->ne[0] * dst->ne[1] * output_elem_size;

    // aclnn
    int64_t              max_elem_size = 65535;
    int64_t              split_size    = (src0->ne[1] / max_elem_size) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    for (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        for (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            acl_tensor_ptr acl_input_tensor = ggml_cann_create_tensor(
                (char *) input_buffer + batch1 * input_stride, ACL_FLOAT16, input_elem_size, input_ne, input_nb, 2);

            // first split
            int64_t weight_ne_offset = 0;
            int64_t weight_ne[2]     = { max_elem_size > src0->ne[1] ? src0->ne[1] : max_elem_size, src0->ne[0] };
            int64_t scale_ne_offset  = 0;
            int64_t scale_ne[2]      = { weight_ne[0], weight_ne[1] / QK8_0 };
            int64_t output_ne_offset = 0;
            int64_t output_ne[2]     = { weight_ne[0], dst->ne[1] };

            acl_tensor_ptr acl_weight_tensor =
                ggml_cann_create_tensor((char *) src0->data + batch0 * weight_stride, ggml_cann_type_mapping(type),
                                        weight_elem_size, weight_ne, weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
            acl_tensor_ptr acl_scale_tensor =
                ggml_cann_create_tensor(scale_offset + batch0 * scale_stride, ACL_FLOAT16, scale_elem_size, scale_ne,
                                        scale_nb, 2, ACL_FORMAT_ND, scale_ne_offset);
            acl_tensor_ptr acl_output_tensor =
                ggml_cann_create_tensor((char *) output_buffer + batch1 * output_stride, ACL_FLOAT16, output_elem_size,
                                        output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_offset);
            int64_t antiquantGroupSize = 0;
            if (src0->ne[0] > QK8_0) {
                antiquantGroupSize = QK8_0;
            }
            GGML_CANN_CALL_ACLNN_OP(ctx, WeightQuantBatchMatmulV2, acl_input_tensor.get(), acl_weight_tensor.get(),
                                    acl_scale_tensor.get(), nullptr, nullptr, nullptr, nullptr, antiquantGroupSize,
                                    acl_output_tensor.get());

            // other splits
            for (int64_t split = 1; split < split_size; split++) {
                weight_ne_offset += weight_elem_size * weight_ne[0] * weight_ne[1];
                weight_ne[0] =
                    max_elem_size * (split + 1) > src0->ne[1] ? src0->ne[1] - (max_elem_size * split) : max_elem_size;
                scale_ne_offset += scale_elem_size * scale_ne[0] * scale_ne[1];
                scale_ne[0] = weight_ne[0];
                output_ne_offset += output_elem_size * output_ne[0] * output_ne[1];
                output_ne[0] = weight_ne[0];

                acl_weight_tensor =
                    ggml_cann_create_tensor((char *) src0->data + batch0 * weight_stride, ggml_cann_type_mapping(type),
                                            weight_elem_size, weight_ne, weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
                acl_scale_tensor =
                    ggml_cann_create_tensor(scale_offset + batch0 * scale_stride, ACL_FLOAT16, scale_elem_size,
                                            scale_ne, scale_nb, 2, ACL_FORMAT_ND, scale_ne_offset);
                acl_output_tensor =
                    ggml_cann_create_tensor((char *) output_buffer + batch1 * output_stride, ACL_FLOAT16,
                                            output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_offset);
                GGML_CANN_CALL_ACLNN_OP(ctx, WeightQuantBatchMatmulV2, acl_input_tensor.get(), acl_weight_tensor.get(),
                                        acl_scale_tensor.get(), nullptr, nullptr, nullptr, nullptr, antiquantGroupSize,
                                        acl_output_tensor.get());
            }
        }
    }

    // cast out
    if (dst->type != GGML_TYPE_F16) {
        int64_t * output_cast_ne = dst->ne;
        size_t    output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_output_tensor = ggml_cann_create_tensor(output_buffer, ACL_FLOAT16, output_elem_size,
                                                                   output_cast_ne, output_cast_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_tensor    = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

void ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_cann_mat_mul_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            break;
        default:
            GGML_ABORT("Unsupported type for mul_mat");
            break;
    }
}

/**
 * @brief Rolls the elements of a tensor along a specified dimension.
 *
 * This function rolls the elements of the source tensor `acl_src` by the
 * specified shifts `shifts` along the specified dimensions `dims`, and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be rolled.
 * @param acl_dst The destination tensor where the rolled elements will be
 * stored.
 * @param shifts An array specifying the number of positions by which elements
 * are shifted.
 * @param dims An array specifying the dimensions along which elements are
 * shifted.
 */
static void aclnn_roll(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       int64_t *                   shifts,
                       int64_t *                   dims) {
    acl_int_array_ptr acl_shifts = ggml_cann_create_int_array(shifts, 1);
    acl_int_array_ptr acl_dims   = ggml_cann_create_int_array(dims, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Roll, acl_src, acl_shifts.get(), acl_dims.get(), acl_dst);
}

/**
 * @brief Fills specified positions of a tensor with a scalar value.
 *
 * This function fills the positions in the source tensor `acl_src` specified by
 * `index` along the dimension `dim` with the scalar value `value`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor where the positions will be filled.
 * @param dim The dimension along which the positions are specified.
 * @param index An array specifying the positions to be filled.
 * @param index_num The number of positions specified in the index array.
 * @param value The scalar value used to fill the specified positions.
 */
static void aclnn_index_fill_tensor(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    int64_t                     dim,
                                    int64_t *                   index,
                                    int64_t                     index_num,
                                    float                       value) {
    acl_int_array_ptr acl_index = ggml_cann_create_int_array(index, index_num);
    acl_scalar_ptr    acl_value = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexFillTensor, acl_src, dim, acl_index.get(), acl_value.get());
}

/**
 * @brief Initializes and caches sine/cosine positional encoding values
 *        (used in RoPE, Rotary Position Embedding) for attention layers.
 *
 * This function computes and caches the sin/cos values of
 * θ = position * theta_scale for RoPE encoding. The cache is shared
 * across attention layers, and only the first attention layer will
 * trigger initialization. The cache includes repeated sin/cos values
 * with different repeat methods depending on the @param is_neox flag.
 *
 * Steps performed by this function:
 *   1. Identify whether the target tensor belongs to Q/K in attention
 *      and restrict computation to the first layer only.
 *   2. Initialize the theta scale array (arange → power → freq scaling).
 *   3. Allocate sin/cos caches if the max prompt length increases.
 *   4. Compute θ = position * theta_scale.
 *   5. Compute sin(θ), cos(θ) and optionally scale by attn_factor.
 *   6. Expand sin/cos values by repeat or repeat_interleave depending
 *      on whether @param is_neox is enabled.
 *
 * @param ctx                The CANN backend context, holding memory pool,
 *                           stream, and persistent buffers for rope init/cache.
 * @param dst                The destination ggml_tensor whose computation
 *                           depends on the RoPE values (usually Qcur/Kcur).
 * @param theta_scale        Scalar exponent base for computing theta scale values.
 * @param freq_scale         Frequency scaling factor, applied to theta scale.
 * @param attn_factor        Attention scaling factor, applied to sin/cos.
 * @param is_neox            Whether to use Neox-style repeat strategy
 *                           (dim expansion vs repeat_interleave).
 */
static void aclnn_cache_init(ggml_backend_cann_context & ctx,
                             ggml_tensor *               dst,
                             float *                     corr_dims,
                             float                       ext_factor,
                             float                       theta_scale,
                             float                       freq_scale,
                             float                       attn_factor,
                             bool                        is_neox) {
    ggml_tensor * src0 = dst->src[0];  // input
    ggml_tensor * src1 = dst->src[1];  // position
    ggml_tensor * src2 = dst->src[2];  // freq_factors

    if (src2 == nullptr && ctx.rope_cache.cached && ctx.rope_cache.ext_factor == ext_factor &&
        ctx.rope_cache.theta_scale == theta_scale && ctx.rope_cache.freq_scale == freq_scale &&
        ctx.rope_cache.attn_factor == attn_factor && ctx.rope_cache.is_neox == is_neox) {
        // use cache.
        return;
    }

    int64_t theta_scale_length = src0->ne[0] / 2;
    int64_t theta_scale_ne[]   = { theta_scale_length, 1, 1, 1 };
    size_t  theta_scale_nb[]   = { sizeof(float), sizeof(float), sizeof(float), theta_scale_length * sizeof(float) };

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_length = src1->ne[0];
    int64_t position_ne[]   = { 1, 1, position_length, 1 };
    size_t  position_nb[]   = { sizeof(int32_t), sizeof(int32_t), sizeof(int32_t), sizeof(int32_t) * position_length };

    int64_t theta_ne[] = { theta_scale_length, 1, position_length, 1 };
    size_t  theta_nb[GGML_MAX_DIMS];
    theta_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        theta_nb[i] = theta_nb[i - 1] * theta_ne[i - 1];
    }

    // theta_scale arange, [0,1,...,ne00/2 - 1]
    acl_tensor_ptr acl_theta_scale_tensor;
    // cache theta scale
    if (ctx.rope_cache.theta_scale_length != theta_scale_length ||
        // theta_scale and freq_scale should not change during the current token inference process,
        // so we can directly use == here instead of comparing the absolute difference.
        ctx.rope_cache.theta_scale != theta_scale || ctx.rope_cache.freq_scale != freq_scale) {
        ctx.rope_cache.theta_scale_length = theta_scale_length;

        if (ctx.rope_cache.theta_scale_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.theta_scale_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.theta_scale_cache, theta_scale_length * sizeof(float),
                              ACL_MEM_MALLOC_HUGE_FIRST));

        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, sizeof(float),
                                                         theta_scale_ne, theta_scale_nb, 1);

        float start      = 0;
        float step       = 1;
        float stop       = theta_scale_length;
        float n_elements = theta_scale_length;
        aclnn_arange(ctx, acl_theta_scale_tensor.get(), start, stop, step, n_elements);

        ggml_cann_pool_alloc yarn_ramp_allocator(ctx.pool());
        acl_tensor_ptr       acl_yarn_ramp_tensor;
        if (ext_factor != 0) {
            // -rope_yarn_ramp
            // const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
            // return MIN(1, MAX(0, y)) - 1;
            yarn_ramp_allocator.alloc(theta_scale_length * sizeof(float));
            void * yarn_ramp_buffer = yarn_ramp_allocator.get();
            acl_yarn_ramp_tensor =
                ggml_cann_create_tensor(yarn_ramp_buffer, ACL_FLOAT, sizeof(float), theta_scale_ne, theta_scale_nb, 1);
            float          zero_value = 0, one_value = 1;
            float          denom_safe_value = MAX(0.001f, corr_dims[1] - corr_dims[0]);
            acl_scalar_ptr low              = ggml_cann_create_scalar(&corr_dims[0], aclDataType::ACL_FLOAT);
            acl_scalar_ptr zero             = ggml_cann_create_scalar(&zero_value, aclDataType::ACL_FLOAT);
            acl_scalar_ptr one              = ggml_cann_create_scalar(&one_value, aclDataType::ACL_FLOAT);
            acl_scalar_ptr denom_safe       = ggml_cann_create_scalar(&denom_safe_value, aclDataType::ACL_FLOAT);
            acl_scalar_ptr ext_factor_sc    = ggml_cann_create_scalar(&ext_factor, aclDataType::ACL_FLOAT);

            GGML_CANN_CALL_ACLNN_OP(ctx, Subs, acl_theta_scale_tensor.get(), low.get(), one.get(),
                                    acl_yarn_ramp_tensor.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDivs, acl_yarn_ramp_tensor.get(), denom_safe.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceThreshold, acl_yarn_ramp_tensor.get(), zero.get(), zero.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceClampMax, acl_yarn_ramp_tensor.get(), one.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), one.get(), one.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), ext_factor_sc.get());

            // theta_interp = freq_scale * theta_extrap;
            // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            // theta = freq_scale * theta_extrap * (1 - ramp_mix) + theta_extrap * ramp_mix;
            // theta = freq_scale * theta_extrap - freq_scale * theta_extrap * ramp_mix + theta_extrap * ramp_mix;
            // theta = theta_extrap * (freq_scale - freq_scale * ramp_mix + ramp_mix);
            //
            // we cache (freq_scale - freq_scale * ramp_mix + ramp_mix), Considering that the rope_yarn_ramp here is the inverse
            // cache freq_scale + (freq_scale - 1) * ramp_mix
            float          freq_scale_1    = freq_scale - 1;
            acl_scalar_ptr freq_scale_sc   = ggml_cann_create_scalar(&freq_scale, aclDataType::ACL_FLOAT);
            acl_scalar_ptr freq_scale_1_sc = ggml_cann_create_scalar(&freq_scale_1, aclDataType::ACL_FLOAT);
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), freq_scale_1_sc.get());
            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_yarn_ramp_tensor.get(), freq_scale_sc.get(), one.get());
        }

        // power
        acl_scalar_ptr acl_theta_scale = ggml_cann_create_scalar(&theta_scale, aclDataType::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, acl_theta_scale.get(), acl_theta_scale_tensor.get(),
                                acl_theta_scale_tensor.get());

        if (ext_factor != 0) {
            aclnn_mul(ctx, acl_theta_scale_tensor.get(), acl_yarn_ramp_tensor.get());
        } else if (freq_scale != 1) {
            aclnn_muls(ctx, acl_theta_scale_tensor.get(), freq_scale, nullptr, true);
        }
    } else {
        // use cache
        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, sizeof(float),
                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
    }

    ggml_cann_pool_alloc freq_fac_res_allocator(ctx.pool());
    // freq_factors
    if (src2) {
        freq_fac_res_allocator.alloc(theta_scale_length * sizeof(float));
        void *         freq_fac_res_ptr = freq_fac_res_allocator.get();
        acl_tensor_ptr acl_freq_factors_tensor =
            ggml_cann_create_tensor(src2->data, ggml_cann_type_mapping(src2->type), ggml_type_size(src2->type),
                                    theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_freq_fac_res_tensor = ggml_cann_create_tensor(freq_fac_res_ptr, ACL_FLOAT, sizeof(float),
                                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        aclnn_div(ctx, acl_theta_scale_tensor.get(), acl_freq_factors_tensor.get(), acl_freq_fac_res_tensor.get());
        std::swap(acl_theta_scale_tensor, acl_freq_fac_res_tensor);
    }

    // init sin_repeat && cos_repeat, only to accelerate first layer on each device
    if (position_length > ctx.rope_cache.position_length) {
        ctx.rope_cache.position_length = position_length;
        if (ctx.rope_cache.sin_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.sin_cache));
        }
        if (ctx.rope_cache.cos_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.cos_cache));
        }
        int64_t repeat_theta_length = theta_scale_length * position_length * 2;
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.sin_cache, repeat_theta_length * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.cos_cache, repeat_theta_length * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // position
    acl_tensor_ptr acl_position_tensor =
        ggml_cann_create_tensor(src1->data, ggml_cann_type_mapping(src1->type), ggml_type_size(src1->type), position_ne,
                                position_nb, GGML_MAX_DIMS);

    // power * position
    int64_t              theta_length = theta_scale_length * position_length;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(), theta_length * sizeof(float));
    void *               theta_buffer = theta_allocator.get();

    acl_tensor_ptr acl_theta_tensor =
        ggml_cann_create_tensor(theta_buffer, ACL_FLOAT, sizeof(float), theta_ne, theta_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor.get(), acl_theta_scale_tensor.get(), acl_theta_tensor.get());

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), theta_length * sizeof(float));
    void *               sin_buffer = sin_allocator.get();
    acl_tensor_ptr       acl_sin_tensor =
        ggml_cann_create_tensor(sin_buffer, ACL_FLOAT, sizeof(float), theta_ne, theta_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_theta_tensor.get(), acl_sin_tensor.get());

    ggml_cann_pool_alloc cos_allocator(ctx.pool(), theta_length * sizeof(float));
    void *               cos_buffer = cos_allocator.get();
    acl_tensor_ptr       acl_cos_tensor =
        ggml_cann_create_tensor(cos_buffer, ACL_FLOAT, sizeof(float), theta_ne, theta_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_theta_tensor.get(), acl_cos_tensor.get());

    if (ext_factor != 0) {
        attn_factor *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // attn_factor
    if (attn_factor != 1) {
        aclnn_muls(ctx, acl_sin_tensor.get(), attn_factor, nullptr, true);
        aclnn_muls(ctx, acl_cos_tensor.get(), attn_factor, nullptr, true);
    }

    int64_t sin_reshape_ne[4] = { src0->ne[0], 1, src0->ne[2], 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, sizeof(float),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, sizeof(float),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    // repeat
    if (is_neox) {
        int64_t repeatsArray[] = { 1, 1, 1, 2 };
        aclnn_repeat(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), repeatsArray);
    } else {
        int64_t num_repeats = 2;
        int64_t dim         = 3;
        int64_t output_size = theta_scale_length * num_repeats;
        aclnn_repeat_interleave(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), dim, num_repeats, output_size);
        aclnn_repeat_interleave(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), dim, num_repeats, output_size);
    }

    // Other layers use cache except first layer.
    ctx.rope_cache.cached      = true;
    ctx.rope_cache.ext_factor  = ext_factor;
    ctx.rope_cache.theta_scale = theta_scale;
    ctx.rope_cache.freq_scale  = freq_scale;
    ctx.rope_cache.attn_factor = attn_factor;
    ctx.rope_cache.is_neox     = is_neox;
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor * x,
                                                         const aclTensor * cos,
                                                         const aclTensor * sin,
                                                         int64_t           mode,
                                                         const aclTensor * yOut,
                                                         uint64_t *        workspaceSize,
                                                         aclOpExecutor **  executor);
aclnnStatus aclnnRotaryPositionEmbedding(void *          workspace,
                                         uint64_t        workspaceSize,
                                         aclOpExecutor * executor,
                                         aclrtStream     stream);
#ifdef __cplusplus
}
#endif

void ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // input

    // param
    float     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    // TODO: n_dims <= ne0
    GGML_ASSERT(n_dims == ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    // init ctx.rope_cos/rope_sin cache
    aclnn_cache_init(ctx, dst, corr_dims, ext_factor, theta_scale, freq_scale, attn_factor, is_neox);

    int64_t sin_reshape_ne[4] = { ne00, 1, ne02, 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, sizeof(float),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, sizeof(float),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

#ifdef ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    void *               input_roll_buffer;
    acl_tensor_ptr       acl_minus_one_tensor;
    void *               minus_one_scale_buffer = nullptr;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(ctx.pool(), sizeof(float) * src0->ne[0]);
    if (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        input_roll_buffer        = roll_allocator.get();
        int64_t input_roll_ne[4] = { 2, src0->ne[1] * (src0->ne[0] / 2), src0->ne[2], src0->ne[3] };
        size_t  input_roll_nb[GGML_MAX_DIMS];
        input_roll_nb[0] = ggml_type_size(src0->type);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_roll_nb[i] = input_roll_nb[i - 1] * input_roll_ne[i - 1];
        }
        acl_tensor_ptr acl_input_roll_tensor =
            ggml_cann_create_tensor(input_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    input_roll_ne, input_roll_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_input_tensor =
            ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    input_roll_ne, input_roll_nb, GGML_MAX_DIMS);

        int64_t shifts[] = { 1 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_input_tensor.get(), acl_input_roll_tensor.get(), shifts, dims);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_buffer = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_values(ctx, minus_one_scale_buffer, sizeof(float) * src0->ne[0], minus_one_ne,
                                            GGML_MAX_DIMS, ACL_FLOAT, sizeof(float), 1);
        int64_t   dim        = 3;
        int64_t * index      = new int64_t[src0->ne[0]];
        for (int i = 0; i < src0->ne[0]; i++) {
            index[i] = i / 2 * 2;
        }
        int64_t index_num = src0->ne[0];
        float   value     = -1;
        aclnn_index_fill_tensor(ctx, acl_minus_one_tensor.get(), dim, index, index_num, value);
    } else {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        input_roll_buffer = roll_allocator.get();
        acl_tensor_ptr acl_input_roll_tensor =
            ggml_cann_create_tensor(input_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    src0->ne, src0->nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_input_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = { src0->ne[0] / 2 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_input_tensor.get(), acl_input_roll_tensor.get(), shifts, dims);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_buffer  = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor     = aclnn_values(ctx, minus_one_scale_buffer, sizeof(float) * src0->ne[0], minus_one_ne,
                                                GGML_MAX_DIMS, ACL_FLOAT, sizeof(float), 1);
        // -1 * first half
        int64_t first_half_ne[4] = { src0->ne[0] / 2, 1, 1, 1 };
        size_t  first_half_nb[GGML_MAX_DIMS];
        first_half_nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            first_half_nb[i] = first_half_nb[i - 1] * first_half_ne[i - 1];
        }
        acl_tensor_ptr acl_first_half_tensor = ggml_cann_create_tensor(minus_one_scale_buffer, ACL_FLOAT, sizeof(float),
                                                                       first_half_ne, first_half_nb, GGML_MAX_DIMS);
        bool           inplace               = true;
        float          scale                 = -1;
        aclnn_muls(ctx, acl_first_half_tensor.get(), scale, nullptr, inplace);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(), ggml_nbytes(src0));
    void *               input_roll_mul_scale_buffer = roll_mul_scale_allocator.get();
    size_t               input_nb[GGML_MAX_DIMS];
    input_nb[0] = ggml_type_size(src0->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        input_nb[i] = input_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_input_roll_mul_scale_tensor =
        ggml_cann_create_tensor(input_roll_mul_scale_buffer, ggml_cann_type_mapping(src0->type),
                                ggml_type_size(src0->type), src0->ne, input_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_input_roll_reshape_tensor =
        ggml_cann_create_tensor(input_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                src0->ne, input_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_input_roll_reshape_tensor.get(), acl_minus_one_tensor.get(),
              acl_input_roll_mul_scale_tensor.get());

    // output
    void * output_fp32_buffer;
    if (src0->type == GGML_TYPE_F32) {
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
        aclnn_mul(ctx, acl_input_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
        aclnn_add(ctx, acl_src.get(), acl_input_roll_mul_scale_tensor.get(), acl_dst.get());
        // TODO: ne0 != n_dims in mode2
    } else if (src0->type == GGML_TYPE_F16) {
        size_t input_fp32_nb[GGML_MAX_DIMS];
        input_fp32_nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_fp32_nb[i] = input_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(ctx.pool(), ggml_nelements(dst) * sizeof(float));
        void *               input_fp32_buffer1 = fp32_allocator1.get();
        acl_tensor_ptr       input_fp32_tensor1 = ggml_cann_create_tensor(input_fp32_buffer1, ACL_FLOAT, sizeof(float),
                                                                          dst->ne, input_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(ctx.pool(), ggml_nelements(dst) * sizeof(float));
        void *               input_fp32_buffer2 = fp32_allocator2.get();
        acl_tensor_ptr       input_fp32_tensor2 = ggml_cann_create_tensor(input_fp32_buffer2, ACL_FLOAT, sizeof(float),
                                                                          dst->ne, input_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(ctx.pool(), ggml_nelements(dst) * sizeof(float));
        output_fp32_buffer                = fp32_allocator.get();
        acl_tensor_ptr output_fp32_tensor = ggml_cann_create_tensor(output_fp32_buffer, ACL_FLOAT, sizeof(float),
                                                                    dst->ne, input_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get(), input_fp32_tensor1.get());
        aclnn_mul(ctx, acl_input_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get(), input_fp32_tensor2.get());
        aclnn_add(ctx, input_fp32_tensor1.get(), input_fp32_tensor2.get(), output_fp32_tensor.get());
        aclnn_cast(ctx, output_fp32_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
    return;
#endif

    // ggml_mode = 0 --> aclnn_model = 1
    int64_t acl_mode = mode == 0 ? 1 : mode;

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src.get(), acl_cos_reshape_tensor.get(),
                                        acl_sin_reshape_tensor.get(), acl_mode, acl_dst.get());
                break;
            }
        case GGML_TYPE_F16:
            {
                ggml_cann_pool_alloc src_trans_allocator(ctx.pool(), ggml_nelements(src0) * sizeof(float));
                void *               src_trans_buffer = src_trans_allocator.get();
                ggml_cann_pool_alloc dst_trans_allocator(ctx.pool(), ggml_nelements(dst) * sizeof(float));
                void *               dst_trans_buffer = dst_trans_allocator.get();

                size_t src_trans_nb[GGML_MAX_DIMS];
                src_trans_nb[0] = sizeof(float);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
                }

                acl_tensor_ptr acl_src_trans_tensor = ggml_cann_create_tensor(
                    src_trans_buffer, ACL_FLOAT, sizeof(float), src0->ne, src_trans_nb, GGML_MAX_DIMS);
                acl_tensor_ptr acl_dst_trans_tensor = ggml_cann_create_tensor(
                    dst_trans_buffer, ACL_FLOAT, sizeof(float), dst->ne, src_trans_nb, GGML_MAX_DIMS);

                aclnn_cast(ctx, acl_src.get(), acl_src_trans_tensor.get(), ACL_FLOAT);

                GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_trans_tensor.get(),
                                        acl_cos_reshape_tensor.get(), acl_sin_reshape_tensor.get(), acl_mode,
                                        acl_dst_trans_tensor.get());

                aclnn_cast(ctx, acl_dst_trans_tensor.get(), acl_dst.get(), ACL_FLOAT16);
                break;
            }
        default:
            GGML_ABORT("Unsupported tensor type for GGML_OP_ROPE");
            break;
    }
}

void ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ArgMax, acl_src.get(), 3, false, acl_dst.get());
}

void ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // stride
    int64_t s0 = ((const int32_t *) (dst->op_params))[0];

    acl_tensor_ptr acl_input  = ggml_cann_create_tensor(src1, src1->ne, src1->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_weight = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_dst    = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3, ACL_FORMAT_NCL);

    int64_t strideVal[1];
    strideVal[0]                    = s0;
    acl_int_array_ptr stride        = ggml_cann_create_int_array(strideVal, 1);
    int64_t           paddingVal[]  = { 0 };
    acl_int_array_ptr padding       = ggml_cann_create_int_array(paddingVal, 1);
    int64_t           dilationVal[] = { 1 };
    acl_int_array_ptr dilation      = ggml_cann_create_int_array(dilationVal, 1);
    int8_t            cubeMathType  = 0;

#ifdef ASCEND_310P
    cubeMathType = 1;
#endif

    GGML_CANN_CALL_ACLNN_OP(ctx, Convolution, acl_input.get(), acl_weight.get(), nullptr, stride.get(), padding.get(),
                            dilation.get(), true, padding.get(), 1, acl_dst.get(), cubeMathType);
}

void ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_input = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst   = ggml_cann_create_tensor(dst);

    float          alphaValue = 1.0f;
    acl_scalar_ptr alpha      = nullptr;
    alpha                     = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Elu, acl_input.get(), alpha.get(), alpha.get(), alpha.get(), acl_dst.get());
}

void ggml_cann_mean(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t           reduceDimValue[] = { 3 };
    acl_int_array_ptr reduceDim        = ggml_cann_create_int_array(reduceDimValue, 1);
    bool              keepDim          = true;

    GGML_CANN_CALL_ACLNN_OP(ctx, Mean, acl_src.get(), reduceDim.get(), keepDim, ACL_FLOAT, acl_dst.get());
}

void ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *     src0             = dst->src[0];
    int32_t *         opts             = (int32_t *) dst->op_params;
    int64_t           paddingsArray[2] = { opts[0], opts[1] };
    acl_int_array_ptr paddings         = ggml_cann_create_int_array(paddingsArray, 2);

    for (int64_t i = 0; i < src0->ne[3]; i++) {
        acl_tensor_ptr acl_src =
            ggml_cann_create_tensor((char *) src0->data + i * src0->ne[3], ggml_cann_type_mapping(src0->type),
                                    ggml_element_size(src0), src0->ne, src0->nb, 3);

        acl_tensor_ptr acl_dst =
            ggml_cann_create_tensor((char *) dst->data + i * src0->ne[3], ggml_cann_type_mapping(dst->type),
                                    ggml_element_size(dst), dst->ne, dst->nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
    }
}

void ggml_cann_count_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceEqTensor, acl_self.get(), acl_other.get());

    ggml_cann_sum(ctx, dst);
}

void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    float          alphaValue = 0.0f;
    acl_scalar_ptr alpha      = nullptr;
    alpha                     = ggml_cann_create_scalar(&alphaValue, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
}

/**
 * @brief Performs expert-specific matrix multiplication (MoE) with
 * floating-point precision using the CANN backend.
 *
 * This function executes a matrix multiplication operation tailored for
 * Mixture of Experts (MoE) models, where the input tensor is multiplied
 * with expert-specific weight matrices. It uses the CANN backend for
 * efficient computation and stores the result in the destination tensor `dst`.
 * The operation may leverage identity-based optimizations or routing masks
 * as part of sparse expert selection.
 *
 * @param ctx The context for executing CANN backend operations.
 * @param dst The destination tensor where the MoE multiplication result
 * will be stored.
 *
 * @note This function assumes floating-point data types and is designed for
 * MoE architectures, possibly involving sparse expert routing.
 */
static void ggml_cann_mul_mat_id_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]  -> [D, M, K, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1 -> [D, 1, K, 1]
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int64_t batch = src1->ne[2];
    GGML_ASSERT(batch == ids->ne[1]);

    ggml_cann_pool_alloc export_allocator(ctx.pool(), src0->ne[0] * src0->ne[1] * ids->ne[0] * ggml_element_size(src0));
    void *               export_ptr = export_allocator.get();
    for (int64_t i = 0; i < batch; i++) {
        acl_tensor_ptr select_index  = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, i * ids->nb[1]);
        acl_tensor_ptr export_weight = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3);

        int64_t select_export_ne[] = { src0->ne[0], src0->ne[1], ids->ne[0] };
        size_t  select_export_nb[3];
        select_export_nb[0] = src0->nb[0];
        for (int k = 1; k < 3; k++) {
            select_export_nb[k] = select_export_nb[k - 1] * select_export_ne[k - 1];
        }

        acl_tensor_ptr select_export =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
                                    select_export_ne, select_export_nb, 3);
        GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, export_weight.get(), 0, select_index.get(), select_export.get());

        int64_t        select_transpose_ne[] = { select_export_ne[1], select_export_ne[0], select_export_ne[2] };
        size_t         select_transpose_nb[] = { select_export_nb[1], select_export_nb[0], select_export_nb[2] };
        acl_tensor_ptr select_export_transpose =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
                                    select_transpose_ne, select_transpose_nb, 3);

        int64_t        active_tensor_ne[] = { src1->ne[0], 1, src1->ne[1] };
        size_t         active_tensor_nb[] = { src1->nb[0], src1->nb[1], src1->nb[1] };
        acl_tensor_ptr active_tensor =
            ggml_cann_create_tensor(src1, active_tensor_ne, active_tensor_nb, 3, ACL_FORMAT_ND, i * src1->nb[2]);

        int64_t        dst_ne[] = { dst->ne[0], 1, dst->ne[1] };
        size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[1] };
        acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, 3, ACL_FORMAT_ND, i * dst->nb[2]);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, active_tensor.get(), select_export_transpose.get(), acl_dst.get(), 2);
    }
}

/**
 * @brief Performs expert-specific matrix multiplication (MoE) with
 * quantized precision using the CANN backend.
 *
 * This function executes a matrix multiplication operation tailored for
 * Mixture of Experts (MoE) models, where the input tensor is multiplied
 * with expert-specific quantized weight matrices. It leverages the CANN
 * backend to perform efficient low-precision computations and stores the
 * quantized result in the destination tensor `dst`.
 *
 * Quantization techniques reduce memory footprint and improve performance
 * by using lower-bit representations (e.g., int8) instead of floating-point.
 * This function is designed to work with such formats and may incorporate
 * optimizations like identity-based fast paths or routing masks for sparse
 * expert selection.
 *
 * @param ctx The context for executing CANN backend operations.
 * @param dst The destination tensor where the quantized MoE multiplication result
 * will be stored.
 *
 * @note This function assumes quantized data types and is designed for
 * MoE architectures with potential sparse expert routing.
 */
static void ggml_cann_mul_mat_id_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    // TODO: Use aclnnGroupedMatMul
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_TENSOR_BINARY_OP_LOCALS

    // copy index from npu to cpu
    int64_t n_as  = ne02;        // A
    int64_t n_ids = ids->ne[0];  // K

    std::vector<char> ids_host(ggml_nbytes(ids));
    ACL_CHECK(aclrtMemcpyAsync(ids_host.data(), ggml_nbytes(ids), ids->data, ggml_nbytes(ids),
                               ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream()));
    ACL_CHECK(aclrtSynchronizeStream(ctx.stream()));

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *) dst->data;

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    const enum ggml_type type = dst->src[0]->type;
    float                weight_elem_size;
    if (type == GGML_TYPE_Q4_0) {
        weight_elem_size = float(sizeof(uint8_t)) / 2;
    } else if (type == GGML_TYPE_Q8_0) {
        weight_elem_size = float(sizeof(uint8_t));
    } else {
        GGML_ABORT("MUL_MAT_ID only support quant type Q4_0 and Q8_0 ");
    }

    // src0_row [D, M, 1, 1] weight without permute
    src0_row.ne[2]       = 1;
    src0_row.ne[3]       = 1;
    src0_row.nb[0]       = weight_elem_size;
    src0_row.nb[1]       = weight_elem_size * ne00;
    src0_row.nb[2]       = weight_elem_size * ne00;
    src0_row.nb[3]       = weight_elem_size * ne00;
    size_t weight_stride = ne00 * ne01 * weight_elem_size;
    size_t weight_size   = weight_stride * ne02 * ne03;

    // scale [D, M, 1, 1] -> scale && permute
    size_t scale_elem_size = sizeof(uint16_t);
    size_t scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_size;

    // src1_row [D, 1, 1, 1] -> input
    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    // dst_row [M, 1, 1, 1] -> out
    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    //create weight for one row
    ggml_cann_pool_alloc weight_allocator(ctx.pool());
    void *               weight_buffer = weight_allocator.alloc(nb02);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            // expert index
            int32_t i02 = *(int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);
            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            // If B = 1 (broadcast), always use 0; otherwise, use id.
            int64_t i11 = (ne11 == 1 ? 0 : id);
            int64_t i12 = iid1;

            int64_t i1 = id;
            int64_t i2 = i12;

            void * src0_tmp_ptr  = src0_original + i02 * weight_stride;
            void * scale_tmp_ptr = src0_original + weight_size + i02 * scale_stride;
            void * src1_tmp_ptr  = src1_original + i11 * nb11 + i12 * nb12;
            void * dst_tmp_ptr   = dst_original + i1 * nb1 + i2 * nb2;

            // mem cpy
            ACL_CHECK(aclrtMemcpyAsync(weight_buffer, weight_stride, src0_tmp_ptr, weight_stride,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
            void * scale_buffer = (char *) weight_buffer + weight_stride;
            ACL_CHECK(aclrtMemcpyAsync(scale_buffer, scale_stride, scale_tmp_ptr, scale_stride,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));

            src0_row.data  = weight_buffer;
            src1_row.data  = src1_tmp_ptr;
            dst_row.data   = dst_tmp_ptr;
            dst_row.src[0] = &src0_row;
            dst_row.src[1] = &src1_row;

            ggml_cann_mul_mat(ctx, &dst_row);
        }
    }
    return;
}

void ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst);
            break;
        default:
            GGML_ABORT("Unsupported type for mul_mat_id");
            break;
    }
}

void ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // q, fp32 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src1 = dst->src[1];  // k, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src2 = dst->src[2];  // v, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src3 = dst->src[3];  // mask, fp16

    // B, N, S, D (uncont) -> B, S, N, D (cont)
    int64_t src0_bsnd_ne[GGML_MAX_DIMS];
    memcpy(src0_bsnd_ne, src0->ne, GGML_MAX_DIMS * sizeof(int64_t));
    size_t src0_bsnd_nb[GGML_MAX_DIMS];
    memcpy(src0_bsnd_nb, src0->nb, GGML_MAX_DIMS * sizeof(size_t));
    int64_t src1_bsnd_ne[GGML_MAX_DIMS];
    memcpy(src1_bsnd_ne, src1->ne, GGML_MAX_DIMS * sizeof(int64_t));
    size_t src1_bsnd_nb[GGML_MAX_DIMS];
    memcpy(src1_bsnd_nb, src1->nb, GGML_MAX_DIMS * sizeof(size_t));
    int64_t src2_bsnd_ne[GGML_MAX_DIMS];
    memcpy(src2_bsnd_ne, src2->ne, GGML_MAX_DIMS * sizeof(int64_t));
    size_t src2_bsnd_nb[GGML_MAX_DIMS];
    memcpy(src2_bsnd_nb, src2->nb, GGML_MAX_DIMS * sizeof(size_t));

    auto transpose12 = [](int64_t * ne, size_t * nb) {
        int64_t ne_tmp = ne[1];
        size_t  nb_tmp = nb[1];
        ne[1]          = ne[2];
        nb[1]          = nb[2];
        ne[2]          = ne_tmp;
        nb[2]          = nb_tmp;
    };

    transpose12(src0_bsnd_ne, src0_bsnd_nb);
    transpose12(src1_bsnd_ne, src1_bsnd_nb);
    transpose12(src2_bsnd_ne, src2_bsnd_nb);

    float maxBias      = 0.0f;
    float scaleValue   = 1.0f;
    float logitSoftcap = 0.0f;
    memcpy(&scaleValue, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&maxBias, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logitSoftcap, (float *) dst->op_params + 2, sizeof(float));

    if (logitSoftcap == 0.0f) {
        size_t faElemSize = sizeof(uint16_t);
        auto   faDataType = ACL_FLOAT16;  //ACL_BF16;

        acl_tensor_ptr acl_q_tensor = nullptr;
        acl_tensor_ptr acl_k_tensor = nullptr;
        acl_tensor_ptr acl_v_tensor = nullptr;

        // Step 1: cast the src0 (Query) to fp16 if needed
        ggml_cann_pool_alloc src0_f16_allocator(ctx.pool());
        void *               src0_f16_buffer = nullptr;

        if (ggml_cann_type_mapping(src0->type) != faDataType) {
            acl_tensor_ptr acl_src0_f32_tensor =
                ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
            src0_f16_buffer = src0_f16_allocator.alloc(ggml_nelements(src0) * faElemSize);

            int64_t * src0_f16_ne = src0_bsnd_ne;
            size_t    src0_f16_nb[GGML_MAX_DIMS];
            src0_f16_nb[0] = sizeof(uint16_t);
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16_nb[i] = src0_f16_nb[i - 1] * src0_f16_ne[i - 1];
            }

            acl_q_tensor = ggml_cann_create_tensor(src0_f16_buffer, faDataType, faElemSize, src0_f16_ne, src0_f16_nb,
                                                   GGML_MAX_DIMS);
            aclnn_cast(ctx, acl_src0_f32_tensor.get(), acl_q_tensor.get(), faDataType);
        } else {
            acl_q_tensor = ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
        }

        // Step 2: create the acl tensors for src1 (Key), src2 (Value),
        //         and the direct output from FusedInferAttention

        acl_k_tensor = ggml_cann_create_tensor(src1, src1_bsnd_ne, src1_bsnd_nb, GGML_MAX_DIMS);
        acl_v_tensor = ggml_cann_create_tensor(src2, src2_bsnd_ne, src2_bsnd_nb, GGML_MAX_DIMS);

        // Step 3: create the PSEShift tensor if needed
        //         this tensor is considered as mask (f16) in the llama.cpp
        acl_tensor_ptr       bcast_pse_tensor;
        ggml_cann_pool_alloc bcast_pse_allocator(ctx.pool());
        if (src3 != nullptr) {
            // Construct the truncated pse tensor (common for prefill/decode)
            int64_t trunc_pse_ne[GGML_MAX_DIMS] = {
                src3->ne[0],  // D
                src0->ne[1],  // S (number of Q tokens)
                src3->ne[2],  // mask N
                src3->ne[3]   // B
            };
            size_t * trunc_pse_nb = src3->nb;

            acl_tensor_ptr acl_mask_f16_trunc_tensor = ggml_cann_create_tensor(
                src3->data, ACL_FLOAT16, sizeof(uint16_t), trunc_pse_ne, trunc_pse_nb, GGML_MAX_DIMS);

            int64_t bcast_pse_ne[GGML_MAX_DIMS];
            size_t  bcast_pse_nb[GGML_MAX_DIMS];
            bcast_pse_ne[0] = src3->ne[0];  // D
            bcast_pse_ne[1] = src0->ne[1];  // S
            bcast_pse_ne[2] = src0->ne[2];  // N (num_heads)
            bcast_pse_ne[3] = src3->ne[3];  // B
            if (maxBias == 0.0f) {
                // When maxBias == 0.0f, use nb = 0 reduce once repeat (Qwen2)
                // Construct the bcast tensor (simulate repeat on the head dimension using stride=0)
                bcast_pse_nb[0] = sizeof(uint16_t);
                bcast_pse_nb[1] = bcast_pse_nb[0] * bcast_pse_ne[0];
                bcast_pse_nb[2] = 0;  // <---- the head dimension shares the same data
                bcast_pse_nb[3] = src3->nb[3];

                bcast_pse_tensor = ggml_cann_create_tensor(src3->data, ACL_FLOAT16, sizeof(uint16_t), bcast_pse_ne,
                                                           bcast_pse_nb, GGML_MAX_DIMS);

            } else {
                bcast_pse_nb[0] = sizeof(uint16_t);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    bcast_pse_nb[i] = bcast_pse_nb[i - 1] * bcast_pse_ne[i - 1];
                }

                void * bcast_pse_buffer =
                    bcast_pse_allocator.alloc(ggml_nelements(src3) * src0->ne[2] * sizeof(uint16_t));

                bcast_pse_tensor = ggml_cann_create_tensor(bcast_pse_buffer, ACL_FLOAT16, sizeof(uint16_t),
                                                           bcast_pse_ne, bcast_pse_nb, GGML_MAX_DIMS);

                int64_t repeats[] = { 1, src0->ne[2], 1, 1 };
                aclnn_repeat(ctx, acl_mask_f16_trunc_tensor.get(), bcast_pse_tensor.get(), repeats);

                // alibi
                // Compute the slope if needed. Derived from ggml_cann_softmax().
                const int64_t        n_heads = src0->ne[2];
                ggml_cann_pool_alloc slope_allocator(ctx.pool(), n_heads * sizeof(uint16_t));
                void *               slope_buffer = slope_allocator.get();
                aclnn_get_slope(ctx, n_heads, slope_buffer, maxBias, GGML_TYPE_F16);

                int64_t slope_ne[] = { 1, 1, n_heads, 1 };
                size_t  slope_nb[GGML_MAX_DIMS];
                slope_nb[0] = sizeof(uint16_t);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    slope_nb[i] = slope_nb[i - 1] * slope_ne[0];
                }

                acl_tensor_ptr slope_tensor = ggml_cann_create_tensor(slope_buffer, ACL_FLOAT16, sizeof(uint16_t),
                                                                      slope_ne, slope_nb, GGML_MAX_DIMS);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, bcast_pse_tensor.get(), slope_tensor.get());
            }
        }

        // Step 4: set the inputs for FusedInferAttention.
        acl_tensor_list_ptr acl_k_tensor_list = ggml_cann_create_tensor_list(acl_k_tensor);
        acl_tensor_list_ptr acl_v_tensor_list = ggml_cann_create_tensor_list(acl_v_tensor);

        int64_t numHeads           = src0->ne[2];  // N
        int64_t numKeyValueHeads   = src1->ne[2];
        // double  scaleValue = 1 / sqrt(src0->ne[0]); // 1/sqrt(d)
        int64_t preTokens          = 65535;
        int64_t nextTokens         = 65535;
        char    layout[5]          = { 'B', 'S', 'N', 'D', 0 };
        int64_t sparseMode         = 0;
        int64_t innerPrecise       = (src0->ne[1] == 1) ? 0 : 2;
        int64_t blockSize          = 0;
        int64_t antiquantMode      = 0;
        bool    softmaxLseFlag     = false;
        int64_t keyAntiquantMode   = 0;
        int64_t valueAntiquantMode = 0;

        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        acl_tensor_ptr       fa_dst_tensor;
        acl_tensor_ptr       acl_dst_tensor;
        ggml_cann_pool_alloc out_f16_allocator(ctx.pool());
        if (dst->type == GGML_TYPE_F32) {
            void * out_f16_buffer = out_f16_allocator.alloc(ggml_nelements(dst) * faElemSize);

            int64_t * out_f16_ne = src0_bsnd_ne;
            size_t    out_f16_nb[GGML_MAX_DIMS];
            out_f16_nb[0] = faElemSize;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                out_f16_nb[i] = out_f16_nb[i - 1] * out_f16_ne[i - 1];
            }

            fa_dst_tensor =
                ggml_cann_create_tensor(out_f16_buffer, faDataType, faElemSize, out_f16_ne, out_f16_nb, GGML_MAX_DIMS);
        } else {
            fa_dst_tensor = ggml_cann_create_tensor(dst);
        }

        GGML_CANN_CALL_ACLNN_OP(ctx, FusedInferAttentionScoreV2, acl_q_tensor.get(), acl_k_tensor_list.get(),
                                acl_v_tensor_list.get(),               // q, k, v
                                bcast_pse_tensor.get(), nullptr,       // pse, mask
                                nullptr, nullptr,                      // actSeqLen, actSeqLenkv
                                nullptr, nullptr,                      // deqScale1, quantScale1
                                nullptr, nullptr, nullptr,             // deqScale2, quantScale2, quantOffset2
                                nullptr, nullptr,                      // antiquantScale, antiquantOffset
                                nullptr,                               // blockTable
                                nullptr, nullptr,                      // qPadSize, kvPadSize
                                nullptr, nullptr,                      // kAntiquantScale, kAntiQuantOffset
                                nullptr, nullptr,                      // vAntiquantScale, vAntiQuantOffset
                                nullptr, nullptr, nullptr,             // kSharedPrefix, vSharedPrefix, actSharedLen
                                numHeads, scaleValue,                  // heads, scaleValue
                                preTokens, nextTokens,                 // preTokens, nextTokens
                                layout,                                // inputLayout
                                numKeyValueHeads,                      // numKVHeads
                                sparseMode, innerPrecise,              // sparseMode, innerPrecise
                                blockSize, antiquantMode,              // blockSize, antiquantMode
                                softmaxLseFlag,                        // softmaxLseFlag
                                keyAntiquantMode, valueAntiquantMode,  // keyAntiqMode, valueAntiqMode
                                fa_dst_tensor.get(),                   // attentionOut
                                nullptr                                // softmaxLse
        );

        if (dst->type == GGML_TYPE_F32) {
            // Step 6: post-processing, permute and cast to f32
            acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
            aclnn_cast(ctx, fa_dst_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
        }
    } else {
        GGML_ABORT("Function is not implemented.");
    }
}
