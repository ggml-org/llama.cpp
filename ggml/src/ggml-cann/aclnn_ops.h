/**
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

#ifndef CANN_ACLNN_OPS
#define CANN_ACLNN_OPS

#include "acl_tensor.h"
#include "common.h"

#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_argsort.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_gelu_v2.h>
#include <aclnnop/aclnn_hardsigmoid.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_leaky_relu.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_logsoftmax.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_norm.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_slice.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_tanh.h>

#include <functional>
#include <unordered_set>

/**
 * @brief   Repeats a ggml tensor along each dimension to match the dimensions
 *          of another tensor.
 *
 * @details This function repeats the elements of a source ggml tensor along
 *          each dimension to create a destination tensor with the specified
 *          dimensions. The operation is performed using the ACL backend and
 *          executed asynchronously on the device.
 *
 * @param   ctx The CANN context used for operations.
 * @param   dst The ggml tensor representing the destination, which op is
 *              GGML_OP_REPEAT and specifies the desired dimensions.
 */
void ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies the Leaky ReLU activation function to a tensor using the CANN
 *          backend.
 *
 * @details This function computes the Leaky ReLU activation for each element of
 *          the input tensor. The Leaky ReLU function allows a small gradient
 *          when the unit is not active (i.e., when the input is negative). The
 *          Leaky ReLU function is defined as:
 *          \f[
 *              \text{dst} = \max(0, src) + \text{negativeSlope} \cdot \min(0,
 *               src)
 *          \f]
 *          `negativeSlope` is in dst->params.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result of the Leaky ReLU
 *            activation is stored, which op is `GGML_OP_LEAKY_RELU`
 */
void ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief    Concatenates multiple tensors along a specified dimension using the
 *           CANN backend.
 *
 * @param ctx        The CANN context used for operations.
 * @param tensorList A pointer to the list of tensors to be concatenated.
 * @param dst        The destination tensor where the result of the
 *                   concatenation is stored. dst->op is `GGML_OP_CONCAT`.
 * @param concat_dim The dimension along which the tensors are concatenated.
 *
 * @attention tensorList length should be 2 and the dimension using for concat
 *            default to 1.
 */
void ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Generates a sequence of evenly spaced values within a specified
 *          interval for a ggml tensor using the CANN backend.
 *
 * @details This function creates a sequence of numbers over a specified i
 *          nterval, starting from `start`, ending before `stop`, and
 *          incrementing by `step`. The sequence is stored in the destination
 *          tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the generated sequence will be stored.
 *            `start`, 'stop' and 'step' are in dst->op_params and dst->op is
 *            `GGML_OP_ARANGE`.
 */
void ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies a clamp operation to the elements of a ggml tensor using the
 *          CANN backend.
 *
 * @details This function clamps the elements of the input tensor `src` to a
 *          specified range defined by `min` and `max` values. The result is
 *          stored in the destination tensor `dst`. The operation is defined as:
 *          \f[
 *              y = \max(\min(x, max\_value), min\_value)
 *           \f]
 *          where `x` is an element of the input tensor, and `y` is the
 *          corresponding element in the output tensor.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the clamped values will be stored.
 *            dst->op is `GGML_OP_CLAMP`, `min` and `max` value is in dst->params.
 */
void ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Scales the elements of a ggml tensor by a constant factor using the
 *          CANN backend.
 *
 * @details This function multiplies each element of the input tensor `src` by
 *          a scaling factor `scale`, storing the result in the destination
 *          tensor `dst`. The operation is defined as:
 *          \f[
 *             dst = src \times scale
 *          \f]
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the scaled values will be stored.
 *            dst->op is `GGML_OP_SCALE` and `scale` value is in dst->params.
 */
void ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Sorts the elements of a ggml tensor and returns the indices that
 *          would sort the tensor using the CANN backend.
 *
 * @details This function performs an argsort operation on the input tensor
 *          `src`. It sorts the elements of `src` in either ascending or
 *          descending order, depending on the `GGML_SORT_ORDER_DESC`,
 *          and returns the indices that would sort the original tensor.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the sorted indices will be stored.
 *            dst->op is `GGML_OP_ARGSORT`.
 */
void ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Layer Normalization for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function applies the Layer Normalization operation on the
 *          input tensor `src` and stores the result in the destination tensor
 *          `dst`. Layer Normalization normalizes the features at each sample in
 *          a mini-batch independently. It is commonly used in neural networks
 *          to normalize the activations of a layer by adjusting and scaling
 *          the outputs.
 *          The operation is defined as:
 *          \f[
 *              \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *          \f]
 *          `Var` defaults dst->ne[0]. `eps` is in dst->params.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 * @attention `Var` defaults to dst->ne[0].
 */
void ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the L2 Normalization for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function applies the L2 Normalization operation on the
 *          input tensor `src` and stores the result in the destination tensor
 *          `dst`. L2 Normalization scales the input tensor such that the
 *          L2 norm along the specified dimension equals 1. This operation
 *          is commonly used in neural networks for feature normalization
 *          and vector scaling.
 *          The operation is defined as:
 *          \f[
 *              \text{out} = \frac{x}{\sqrt{\sum{x^2}}}
 *          \f]
 *          The normalization is performed along the last dimension by default.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 * @attention The normalization is performed along the last dimension of the
 *            input tensor by default.
 */
void ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Cross Entropy Loss for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function computes the cross entropy loss between the predicted
 *          logits and target probability distributions. The operation follows
 *          the same computation pattern as the CPU implementation:
 *          1. Applies log_softmax to the logits along the class dimension
 *          2. Element-wise multiplication with target distributions
 *          3. Summation along the class dimension to get per-sample losses
 *          4. Global summation and scaling by -1/nr to get final loss
 *
 *          The computation can be expressed as:
 *          \f[
 *              \text{loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \cdot \log(\text{softmax}(x_{ij}))
 *          \f]
 *          where \f$N\f$ is the total number of samples, \f$C\f$ is the number
 *          of classes, \f$x\f$ are the logits, and \f$y\f$ are the target
 *          probability distributions.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the computed loss will be stored.
 *            This should be a scalar tensor containing the final loss value.
 *
 * @note This implementation computes cross entropy between probability
 *       distributions, not the typical classification cross entropy that
 *       expects class indices as targets. Both input tensors (src0 and src1)
 *       should have the same shape and represent probability distributions
 *       over the class dimension.
 * @note The function expects two source tensors:
 *       - dst->src[0]: Logits tensor (before softmax)
 *       - dst->src[1]: Target probability distributions tensor
 * @note The computation is performed using CANN backend operators including
 *       LogSoftmax, Mul, ReduceSum, and Muls for the final scaling.
 */
void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief  Computes the Group Normalization for a ggml tensor using the CANN
 *         backend.
 *
 * @brief  This function applies the Group Normalization operation on the input
 *         tensor `src` and stores the result in the destination tensor `dst`.
 *         Group Normalization divides the channels into groups and normalizes
 *         the features within each group across spatial locations.
 *         It is commonly used in convolutional neural networks to improve
 *         training stability and performance.
 *         The operation is defined as:
 *         \f[
 *             \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *         \f]
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            `n_groups` is in dst->params, which split C channel to `n_groups`.
 *            dst->op is `GGML_OP_GROUP_NORM`.
 *
 * @attention eps defaults to 1e-6f.
 */
void ggml_cann_group_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the accumulation of tensors using the CANN backend.
 *
 * @details This function performs an accumulation operation on two tensors.
 *          Depending on the `inplace` flag, it either updates the destination
 *          tensor `dst` in place by adding `alpha * src1` to it, or it creates
 *          a new tensor as the result of `src0 + alpha * src1` and stores it in
 *          `dst`.
 *          The operation is defined as:
 *          \f[
 *               dst = src0 + alpha \times src1
 *          \f]
 *          if `inplace` is `true`, `src0` is equal to 'dst'.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the accumulated values will be stored.
 *            `inplace` is in dst->params, and dst->op is `GGML_OP_ACC`.
 */
void ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the sum of elements along the last dimension of a ggml tensor
 *          using the CANN backend.
 *
 * @details This function performs a reduction sum operation along the last
 *          dimension of the input tensor `src`. The result of the sum is stored
 *          in the destination tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the reduced values will be stored。
 *            dst->op is `GGML_OP_SUM_ROWS`.
 *
 * @attention `reduce_dims` defaults to 3, which means the last dimension.
 */
void ggml_cann_sum_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the sum of elements in a ggml tensor.
 *
 * @details This function performs a reduction sum operation along the last
 *          dimension of the input tensor `src`. The result of the sum is stored
 *          in the destination tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the reduced values will be stored。
 *
 */

void ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Upsamples a ggml tensor using nearest neighbor interpolation using
 *          the CANN backend.
 *
 * @details This function performs upsampling of the input tensor `src` using
 *          nearest neighbor interpolation. The upsampling is applied to the
 *          height and width dimensions (last two dimensions) of the tensor. The
 *          result is stored in the destination tensor `dst`, which must have
 *          the appropriate dimensions for the upsampled output.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the upsampled values will be stored.
 *            dst->op is `GGML_OP_UPSCALE`.
 */
void ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Pads a ggml tensor to match the dimensions of the destination tensor
 *          using the CANN backend.
 *
 * @details This function pads the input tensor `src` so that it matches the
 *          dimensions of the destination tensor `dst`. The amount of padding
 *          is calculated based on the difference in sizes between `src` and
 *          `dst` along each dimension. The padded tensor is stored in `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor, which specifies the target dimensions for
 *            padding. dst->op is `GGML_OP_PAD`.
 */
void ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Executes a 2D pooling operation on a ggml tensor using the CANN
 *          backend.
 *
 * @details This function dispatches the execution of a 2D pooling operation on
 *          the input tensor `dst`. The type of pooling (average or max) is
 *          determined by the `op` parameter, which is read from the operation
 *          parameters of `dst`. The function supports average pooling
 *          (`GGML_OP_POOL_AVG`) and max pooling (`GGML_OP_POOL_MAX`). If an
 *          invalid operation is encountered, the function asserts a failure.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor on which the pooling operation is to be
 *            performed. dst->op is `GGML_OP_POOL_2D`.
 */
void ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Duplicates a ggml tensor using the CANN backend.
 *
 * @details This function duplicates the contents of the source tensor `src` to
 *          the destination tensor `dst`. The function supports various tensor
 *          types and configurations, including handling of extra data, type
 *          conversions, and special cases for contiguous and non-contiguous
 *          tensors.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_DUP`
 *
 * @attention Only support Fp16/FP32. Not support when src and dst have
 *            different shape and dst is no-contiguous.
 * @note:     This func need to simplify.
 */
void ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Root Mean Square (RMS) normalization of a ggml tensor
 *          using the CANN backend.
 *
 * @details This function applies RMS normalization to the input tensor `src`
 *          and stores the result in the destination tensor `dst`. RMS
 *          normalization involves computing the root mean square of the input
 *          tensor along a specified dimension and then dividing each element of
 *          the tensor by this value, adjusted by a small epsilon value to
 *          prevent division by zero.
 *          The operation is defined as:
 *          \f[
 *               \text{RmsNorm}\left(x_i\right)=\frac{x_i}{\text{Rms}(\mathbf{x})} g_i,
 *               \quad \text { where } \text{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+e p s}
 *          \f]
 *          `eps` is in dst->op_params.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            dst->op is `GGML_OP_RMS_NORM`.
 */
void ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies a diagonal mask to the tensor with a specified value.
 *
 * @details This function creates a mask tensor filled with ones, then applies
 *          an upper triangular and lower triangular operation to it based on
 *          the number of past elements specified. Afterward, it adds the masked
 *          tensor to the destination tensor in-place.
 *
 * @param ctx The backend CANN context used for operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_DIAG_MASK`
 * @param value The value to use for masking.
 */
void ggml_cann_diag_mask(ggml_backend_cann_context & ctx, ggml_tensor * dst, float value);

/**
 * @brief   Performs an image-to-column transformation on the input tensor.
 *
 * @details This function takes an input tensor and applies an image-to-column
 *          operation, converting spatial dimensions into column-like
 *          structures suitable for convolutional operations. It supports both
 *          half-precision (F16) and single-precision (F32) floating-point data
 *          types.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor that stores the result of the operation.
 *            dst->op is `GGML_OP_IM2COL`.
 */
void ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes time step embeddings using sine and cosine functions.
 *
 * @details This function calculates time step embeddings by applying sine and
 *          cosine transformations to a given input tensor, which is typically
 *          used in temporal models like diffusion models or transformers to
 *          encode time information effectively.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the result of the embedding operation
 *            will be stored. dst->op is `GGML_OP_TIMESTEP_EMBEDDING`.
 */
void ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst);

// @see ggml_cann_dup.
void ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the softmax activation with optional masking.
 *
 * @details This function computes the softmax activation over the input tensor,
 *          optionally applying a mask and scaling factor. It supports both FP16
 *          and FP32 data types and can handle masking by broadcasting the mask
 *          across rows if necessary.
 *          The function performs the following steps:
 *          1. Multiplies the input tensor by a scale factor.
 *          2. Optionally casts the mask tensor to FP32 if it is in FP16 format.
 *          3. Broadcasts the mask tensor if its dimensions do not match the
 *             input tensor's dimensions.
 *          4. Adds the mask to the scaled input tensor.
 *          5. Applies the softmax activation function along the specified
 *             dimension.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_SOFTMAX`.
 */
void ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Extracts specific rows from a tensor based on indices.
 *
 * @details This function retrieves rows from a source tensor src0 according to
 *          the indices provided in another tensor src1 and stores the result in
 *          a destination tensor (\p dst).
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the extracted rows will be stored.
 */
void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Writes specific rows into a tensor at positions specified by indices.
 *
 * @details This function copies rows from a source tensor into a destination
 *          tensor (\p dst) at the positions indicated by the indices in another
 *          tensor.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the specified rows will be updated.
 */
void ggml_cann_set_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Executes matrix multiplication for the given tensor.
 *
 * @details This function performs matrix multiplication on the source tensors
 *          associated with the destination tensor. It supports matrix
 *          multiplication F32, F16, and Q8_0.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor for storing the result of the matrix
 *            multiplication. dst->op is `GGML_OP_MUL_MAT`.
 */
void ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies Rotary Positional Embedding (RoPE) to the input tensor.
 *
 * @details This function implements the RoPE mechanism, which is a method to
 *          encode positional information into sequence data, particularly
 *          useful in transformer models. It supports both F32 and F16 data
 *          types.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the RoPE-transformed data will be
 *            stored. dst->op is `GGML_OP_ROPE`.
 *
 * @note The function currently does not support cases where the n_dims is less
 *       than the input tensor's first dimension.
 * @note The function currently does not support cases where the freq_factors is
 *       not NULL.
 * @note The function currently does not support cases where the ext_factor is
 *       not equal 0.
 * @note The function currently does not support cases where the freq_scale is
 *       not equal 1.
 */
void ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the index of the maximum value along the specified dimension
 *          of a ggml tensor using the CANN backend.
 *
 * @details This function performs an argmax operation on the input tensor.
 *          It finds the index of the maximum value along the specified axis
 *          and stores these indices in the destination tensor `dst`. The
 *          operation is executed using the CANN backend for optimized performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the indices of the maximum values will
 *            be stored. dst->op is `GGML_OP_ARGMAX`.
 */
void ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Adds two tensors element-wise and stores the result in a destination
 * tensor.
 *
 * This function performs the operation:
 * \f[
 *    dst = acl\_src0 + alpha \times acl\_src1
 * \f]
 * where alpha is a scalar value and defaults to 1.0f.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src0 The first source tensor.
 * @param acl_src1 The second source tensor.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_add(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src0,
               aclTensor *                 acl_src1,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Sub two tensors element-wise and stores the result in a destination
 * tensor.
 *
 * This function performs the operation:
 * \f[
 *    dst = acl\_src0 - alpha \times acl\_src1
 * \f]
 * where alpha is a scalar value and defaults to 1.0f.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src0 The first source tensor.
 * @param acl_src1 The second source tensor.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_sub(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src0,
               aclTensor *                 acl_src1,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Performs element-wise multiplication of two tensors and stores the
 * result in a destination tensor.
 *
 * This function performs element-wise multiplication of the tensors `acl_src`
 * and `acl_other` and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {acl_other }_i
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The first tensor for element-wise multiplication.
 * @param acl_other The second tensor for element-wise multiplication.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_mul(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src,
               aclTensor *                 acl_other,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Matrix division, optionally in-place.
 *
 * This function division each element of the source tensor `acl_src` by the
 * tensor `acl_other` and stores the result in the destination tensor `acl_dst`.
 * If `inplace` is true, `acl_dst` will not be used and the operation is
 * performed in-place on `acl_src`. The operation is defined as: \f[
 *     \text{dst}_i = \frac{\text{acl_src}_i}{\text{acl_other}_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src Numerator tensor..
 * @param acl_other Denominator tensor.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
void aclnn_div(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src,
               aclTensor *                 acl_other,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Applies element-wise cosine function to the elements of a tensor.
 *
 * This function computes the cosine of each element in the source tensor
 * `acl_src` and stores the result in the destination tensor `acl_dst`. The
 * operation is defined as: \f[ \text {acl_dst }_i=\cos \left(\text {acl_src
 * }_i\right) \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the cosine function will be
 * applied.
 * @param acl_dst The destination tensor where the cosine results will be
 * stored.
 */
void aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst);

/**
 * @brief Applies element-wise sine function to the elements of a tensor.
 *
 * This function computes the sine of each element in the source tensor
 `acl_src`
 * and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\sin \left(\text {acl_src }_i\right)
 * \f]

 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the sine function will be applied.
 * @param acl_dst The destination tensor where the sine results will be stored.
 */
void aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst);

/**
 * @brief Prepares broadcast-compatible ACL tensors for two input tensors and one
 * output tensor.
 *
 * This function checks whether broadcasting is needed between `src0` and `src1`.
 * If broadcasting is required, it calculates the proper shapes and creates
 * ACL tensors with broadcast parameters. Otherwise, it directly creates ACL tensors
 * based on the original tensor shapes.
 *
 * @param src0     The first input tensor (reference shape).
 * @param src1     The second input tensor (possibly broadcasted).
 * @param dst      The destination/output tensor.
 * @param acl_src0 Output pointer to the created ACL tensor corresponding to src0.
 * @param acl_src1 Output pointer to the created ACL tensor corresponding to src1.
 * @param acl_dst  Output pointer to the created ACL tensor corresponding to dst.
 */
void bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst);

/**
 * @brief   Computes the 1D transposed convolution (deconvolution) of a ggml
 * tensor using the CANN backend.
 *
 * @details This function performs a 1D transposed convolution (also known as
 * deconvolution) operation on the input tensor. The computed result is stored
 * in the destination tensor `dst`. The operation is optimized using the CANN
 * backend for improved performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the transposed convolution result
 * will be stored. dst->op is `GGML_OP_CONV_TRANSPOSE_1D`.
 */
void ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies the ELU (Exponential Linear Unit) activation to a ggml tensor
 * using the CANN backend.
 *
 * @details This function performs an element-wise ELU activation on the input
 *          tensor.
 *          The result is written to the destination tensor `dst` in-place.
 *          The ELU function is defined as:
 *
 *          \text{ELU}(x) =
 *          \begin{cases}
 *          x, & \text{if } x > 0 \\
 *          \alpha \left( \exp(x) - 1 \right), & \text{if } x \leq 0
 *          \end{cases}
 *
 *          where α (alpha) is a hyperparameter, typically set to 1.0.
 *          This operation is optimized using the CANN backend for high-performance
 *          inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the ELU-activated result will be stored.
 *            dst->op is expected to be `GGML_OP_ELU`.
 */
void ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the mean of a ggml tensor element-wise using the CANN backend.
 *
 * @details This function calculates the element-wise mean of the input tensor.
 *          The result is written to the destination tensor `dst`.
 *          The mean is computed by averaging the values across the entire tensor.
 *
 *          This operation is optimized using the CANN backend for high-performance inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the mean result will be stored.
 *            dst->op is expected to be `GGML_OP_MEAN`.
 */
void ggml_cann_mean(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies 1D reflect padding to a ggml tensor using the CANN backend.
 *
 * @details This function performs 1D reflect padding on the input tensor.
 *          The amount of padding on each side is specified by parameters stored in `dst->op_params`.
 *          The operation reflects the values at the borders of the tensor to generate the padded output.
 *
 *          This operation is optimized using the CANN backend for high-performance inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the padded result will be stored.
 *            dst->op is expected to be `GGML_OP_PAD_REFLECT_1D`.
 */
void ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Counts the number of equal elements in two ggml tensors using the CANN backend.
 *
 * @details This function performs an element-wise comparison between two input tensors,
 *          and counts the number of positions where the elements are equal. The result is
 *          stored in the destination tensor `dst` as a scalar.
 *
 *          The operation is optimized using the CANN backend, making it suitable for
 *          high-performance inference or training scenarios.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_COUNT_EQUAL`.
 */
void ggml_cann_count_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies the Step activation function to a ggml tensor using the CANN backend.
 *
 * @details This function applies a step function element-wise to the input tensor, where
 *          each element is transformed to 1.0 if it is greater than 0, and 0.0 otherwise.
 *          The result is stored in the destination tensor `dst`.
 *
 *          This operation is accelerated using the CANN backend to improve runtime performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_STEP`.
 */
void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Performs the Flash Attention extended operator using the CANN backend.
 *
 * @details This function implements the memory-efficient Flash Attention algorithm
 *          for computing scaled dot-product attention with hardware acceleration.
 *          The result is stored in the destination tensor `dst`.
 *
 *          This operation is accelerated using the CANN backend to improve runtime performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_FLASH_ATTN_EXT`.
 */
void ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Forward Gated Linear Attention on the CANN backend.
 *
 * Expects dst->src[0..4] = {k, v, q, g, s} with shape conventions:
 *   k, v, q, g: [D] with outer dims T x H batched as ne[2]=T, ne[1]=H
 *   s: initial state [B, H, D, D], where B is batch and D=C/H
 * dst holds both outputs (o) and updated state; a scale factor is read from op params.
 *
 * The kernel updates per time step l: S_new = g ⊗ S_old + k ⊗ v, then computes o = (S_new^T q) * scale.
 *
 * @param ctx Backend context providing stream/allocator utilities.
 * @param dst Output tensor; src deps are k, v, q, g, s as above.
 */
void ggml_cann_gated_linear_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Launches an asynchronous task using the memory allocator.
 *
 * This macro submit an asynchronous task on the specified stream.
 * The task uses memory allocated by the allocator. It is guaranteed
 * that the memory will not be accessed by other tasks until this task
 * completes, due to the sequential execution order within the same stream.
 *
 * @param OP_NAME aclnn operator name.
 * @param args Additional arguments required by the task.
 *
 * @note
 * Memory from the allocator will be "freed" immediately and can be
 * reallocated to other pointers. However, it won't be accessed by any
 * other task before this asynchronous task ends, because all tasks in the
 * same stream are executed in queue order.
 */

#    define GGML_CANN_CALL_ACLNN_OP(CTX, OP_NAME, ...)                                           \
        do {                                                                                     \
            uint64_t        workspaceSize = 0;                                                   \
            aclOpExecutor * executor;                                                            \
            void *          workspaceAddr = nullptr;                                             \
            ACL_CHECK(aclnn##OP_NAME##GetWorkspaceSize(__VA_ARGS__, &workspaceSize, &executor)); \
            /* workspace should alloced in main thread to keep malloc order when using vmm. */   \
            if (workspaceSize > 0) {                                                             \
                ggml_cann_pool_alloc workspace_allocator(CTX.pool(), workspaceSize);             \
                workspaceAddr = workspace_allocator.get();                                       \
            }                                                                                    \
            ACL_CHECK(aclnn##OP_NAME(workspaceAddr, workspaceSize, executor, CTX.stream()));     \
        } while (0)

/**
 * @brief   Performs sparse expert-based matrix multiplication using the CANN backend.
 *
 * @details This function implements a MoE-style batched matrix multiplication, where each input token
 *          is routed to one or more experts, and each expert corresponds to a specific [D, M] weight matrix
 *          in the source tensor `src0`. The routing indices are provided via the `ids` tensor.
 *
 *          For each token (from `src1`), the function selects the corresponding expert(s) as specified by `ids`,
 *          performs the matrix multiplication with the selected expert's weight submatrix (from `src0`),
 *          and stores the results in `dst`. This operation is optimized and executed on the CANN backend.
 *
 *          Dimensions:
 *              - src0: [D, M, A, 1], where A is the number of experts
 *              - src1: [D, B, N, 1], where N is batch size and B is the slot count per sample
 *              - ids : [K, N],       where K is the number of experts each token is routed to
 *              - dst : [M, K, N, 1], output tensor storing the result of expert × token multiplication
 *
 *          The function handles two main modes:
 *              - If `ne12 == 1`, a simpler per-token loop is used.
 *              - TODO: If `ne12 > 1`, grouped multiplication and memory copying is used for efficiency.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the expert-weighted token outputs are stored.
 *            Expected to be of shape [M, K, N, 1].
 */
void ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Performs fused ADD + RMS_NORM operation using the CANN backend.
 *
 * This function fuses the ADD and RMS_NORM operations into a single kernel call
 * for better performance. It first adds two input tensors (x1 + x2), then applies
 * RMS normalization to the result.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The ADD operation node, contains the two input tensors to be added.
 * @param rms_norm_tensor The RMS_NORM operation node, contains the gamma weights
 *                        and epsilon parameter.
 */
void ggml_cann_op_add_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               add_node,
                                     ggml_tensor *               rms_norm_node);

/**
 * @brief   Check whether a tensor is a weight tensor for matrix multiplication.
 *
 * @details Checks whether the given tensor serves as weight parameters in matrix multiplication operations,
 *          typically within neural network layers. The function maintains a static set of canonical weight
 *          naming suffixes from Transformer-based architectures. Uses substring matching to identify weight
 *          tensors even with hierarchical naming patterns.
 *
 * @param tensor Pointer to the target ggml_tensor object (const-qualified).
 */
static bool is_matmul_weight(const ggml_tensor * tensor) {
    std::string                                  name = ggml_get_name(tensor);
    static const std::unordered_set<std::string> weight_suffixes{ "output.weight",      "attn_q.weight",
                                                                  "attn_k.weight",      "attn_v.weight",
                                                                  "attn_output.weight", "ffn_gate.weight",
                                                                  "ffn_up.weight",      "ffn_down.weight" };

    for (const auto & suffix : weight_suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Applies a element-wise operation to two input tensors using the CANN
 * backend.
 *
 * This templated function takes a binary operator and applies it to two source
 * tensors
 * associated with the destination tensor. The function handles broadcasting as
 * needed.
 *
 * @tparam binary_op A callable object (e.g., lambda or function pointer) representing
 *         the binary operation to be performed. It must take three arguments:
 *         (ggml_backend_cann_context&, aclTensor*, aclTensor*, aclTensor*).
 *
 * @param ctx The CANN backend context used to manage execution and resources.
 * @param dst The destination tensor.
 */
template <auto binary_op> void ggml_cann_binary_op(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    acl_tensor_ptr acl_src0, acl_src1, acl_dst;

    // Need bcast
    bcast_shape(src0, src1, dst, acl_src0, acl_src1, acl_dst);
    binary_op(ctx, acl_src0.get(), acl_src1.get(), acl_dst.get());
}

/**
 * @brief Applies a unary operation to an input tensor using the CANN backend.
 *
 * This templated function applies a unary operator to the source tensor of `dst`
 * and stores the result in the destination tensor.
 *
 * @tparam unary_op A callable with the signature:
 *         void(ggml_backend_cann_context&, aclTensor *, aclTensor *)
 *         where the first aclTensor is the source and the second is the destination.
 * @param ctx The CANN backend context for managing resources and execution.
 * @param dst The destination tensor. Its src[0] is treated as the input tensor.
 */
template <void unary_op(ggml_backend_cann_context &, aclTensor *, aclTensor *)>
void ggml_cann_op_unary(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

/**
 * @brief Applies a unary operation to a ggml tensor using the CANN backend.
 *
 * @details This function applies a unary operation to the input tensor using
 * a user-provided lambda or callable `unary_op`. The lambda receives the
 * CANN backend context and two ACL tensors: the source and the destination.
 *
 * Internally, this function handles the conversion from GGML tensors to ACL tensors,
 * calls the provided unary op, and manages resource cleanup. The input is assumed
 * to be `dst->src[0]`, and the result is written to `dst`.
 *
 * This utility simplifies writing unary op wrappers by abstracting tensor preparation.
 *
 * @param unary_op A callable that performs the unary operation using CANN ACL APIs.
 * @param ctx The CANN context for operation execution.
 * @param dst The destination ggml_tensor where the result will be stored.
 *            The input tensor is assumed to be `dst->src[0]`.
 *
 * @see GGML_CANN_CALL_OP_UNARY
 */
void ggml_cann_op_unary(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst);

void ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies a gated (GLU-style) unary operation using the CANN backend.
 *
 * @details This function performs a gated activation such as GEGLU or ReGLU.
 * It supports two input modes:
 *
 * 1. **Dual input mode**: `dst->src[0]` and `dst->src[1]` are both valid tensors.
 *    These are used directly as the value and gate tensors.
 *
 * 2. **Packed input mode**: Only `dst->src[0]` is valid, and it is assumed to
 *    contain a concatenation of value and gate along the first dimension. This tensor
 *    will be split into two equal halves to form the value and gate inputs.
 *
 * The function applies a user-provided unary operation (e.g., GELU) to the value tensor,
 * then multiplies the result in-place with the gate tensor:
 *
 * @code
 * dst = unary_op(value) * gate;
 * @endcode
 *
 * The `swapped` parameter (from `dst->op_params[1]`) allows flipping the
 * order of value/gate in the packed input case.
 *
 * @param unary_op A callable that performs the unary operation using CANN ACL APIs.
 *                 It receives (ctx, acl_value_tensor, acl_output_tensor).
 * @param ctx      The CANN context used for execution.
 * @param dst      The destination ggml_tensor. Source tensors are in `dst->src[0]` and optionally `src[1]`.
 *
 * @see GGML_CANN_CALL_OP_UNARY_GATED
 */
void ggml_cann_op_unary_gated(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst);

/**
 * @brief Helper macro to call a unary ACL operator via ggml_cann_op_unary.
 *
 * This macro wraps the specified ACLNN unary operator name into a lambda expression,
 * and passes it to `ggml_cann_op_unary`, which handles the common logic for executing
 * unary ops in the CANN backend.
 *
 * Internally, this macro expands to a lambda like:
 * @code
 * [](ggml_backend_cann_context& ctx, aclTensor* acl_src, aclTensor* acl_dst) {
 *     GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);
 * };
 * @endcode
 *
 * This lambda is then passed to `ggml_cann_op_unary`, which applies the operation.
 *
 * @param OP_NAME The name of the ACL unary operator to invoke via GGML_CANN_CALL_ACLNN_OP.
 *
 * @see ggml_cann_op_unary
 * @see GGML_CANN_CALL_ACLNN_OP
 */
#    define GGML_CANN_CALL_OP_UNARY(OP_NAME)                                                              \
        do {                                                                                              \
            auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) { \
                GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);                                  \
            };                                                                                            \
            ggml_cann_op_unary(lambda, ctx, dst);                                                         \
        } while (0)

/**
 * @brief Helper macro to call a gated unary ACL operator via ggml_cann_op_unary_gated.
 *
 * This macro wraps the specified ACLNN unary operator name into a lambda expression,
 * and passes it to `ggml_cann_op_unary_gated`, which handles the common logic for
 * executing gated unary ops in the CANN backend.
 *
 * Internally, this macro expands to a lambda like:
 * @code
 * [](ggml_backend_cann_context& ctx, aclTensor* acl_src, aclTensor* acl_dst) {
 *     GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);
 * };
 * @endcode
 *
 * This lambda is then passed to `ggml_cann_op_unary_gated`, which applies the operation.
 *
 * @param OP_NAME The name of the ACL unary operator to invoke via GGML_CANN_CALL_ACLNN_OP.
 *
 * @see ggml_cann_op_unary_gated
 * @see GGML_CANN_CALL_ACLNN_OP
 */
#    define GGML_CANN_CALL_OP_UNARY_GATED(OP_NAME)                                                        \
        do {                                                                                              \
            auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) { \
                GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);                                  \
            };                                                                                            \
            ggml_cann_op_unary_gated(lambda, ctx, dst);                                                   \
        } while (0)

#endif  // CANN_ACLNN_OPS

/**
 * @brief Performs outer product operation on two ggml tensors using the CANN backend.
 *
 * @details This function computes the outer product of two input tensors (src0 and src1)
 * and stores the result in the destination tensor. The outer product operation is defined as:
 * dst[i,j,k,l] = sum_m (src0[i,m,k,l] * src1[j,m,k,l])
 *
 * The function supports multiple data types including F32, F16. For floating-point
 * types, it uses batch matrix multiplication for efficient computation.
 *
 * The implementation handles 4D tensor broadcasting and batch processing automatically.
 *
 * @param ctx The CANN backend context for operation execution and memory management.
 * @param dst The destination ggml_tensor where the outer product result will be stored.
 *            The input tensors are assumed to be `dst->src[0]` and `dst->src[1]`.
 *
 * @see GGML_CANN_CALL_ACLNN_OP for CANN operator invocation
 */
void ggml_cann_out_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst);
