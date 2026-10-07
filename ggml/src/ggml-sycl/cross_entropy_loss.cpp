#include "cross_entropy_loss.hpp"

#include <cstdint>
#include <cmath>

template <bool has_shared>
static __dpct_inline__ void cross_entropy_loss_f32_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ labels,
        float * __restrict__ row_loss,
        const int nclasses,
        const int nrows,
        float * __restrict__ smem,
        const sycl::nd_item<3> & item) {

    const int row = item.get_group(2);
    const int tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    float max_logit = -INFINITY;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        if (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    float sum_exp = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    const float log_sum = sycl::log(sum_exp);

    float loss = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (float) nrows;

    if (tid == 0) {
        row_loss[row] = loss;
    }
}

template <bool has_shared>
static __dpct_inline__ void cross_entropy_loss_back_f32_kernel(
        const float * __restrict__ grad,
        const float * __restrict__ logits,
        const float * __restrict__ labels,
        float * __restrict__ dst,
        const int nclasses,
        const int nrows,
        float * __restrict__ smem,
        const sycl::nd_item<3> & item) {

    const int row = item.get_group(2);
    const int tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    float max_logit = -INFINITY;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        if (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    float sum_exp = 0.0f;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        if (has_shared) {
            smem[i] = v;
        } else {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    const float inv_sum = 1.0f / sum_exp;

    const float d_by_nrows = grad[0] / (float) nrows;
    for (int i = tid; i < nclasses; i += WARP_SIZE) {
        const float sm_num = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_num * inv_sum - labels[i]) * d_by_nrows;
    }
}

static void cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        const float * row_loss,
        float * dst,
        const int64_t nrows) {
    if (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, sizeof(float))));
        return;
    }

    ggml_sycl_pool_alloc<float> tmp_alloc(ctx.pool(), nrows);
    float * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * sizeof(float))));

    int64_t cur = nrows;
    while (cur > 1) {
        const int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        const sycl::range<3> block(1, 1, WARP_SIZE);
        const sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(2);
                const int tid = item.get_local_id(2);
                const int64_t i = (int64_t) row * WARP_SIZE + tid;
                float v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                if (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, sizeof(float))));
}

void ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t nclasses = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * logits_d = (const float *) src0->data;
    const float * labels_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    ggml_sycl_pool_alloc<float> row_loss_alloc(ctx.pool(), nrows);
    float * row_loss = row_loss_alloc.get();

    const sycl::range<3> block(1, 1, WARP_SIZE);
    const sycl::range<3> grid(1, 1, nrows);
    const size_t nbytes_shared = (size_t) nclasses * sizeof(float);
    const size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    if (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<true>(
                        logits_d, labels_d, row_loss,
                        (int) nclasses, (int) nrows,
                        get_pointer(smem), item);
                });
        });
    } else {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<false>(
                    logits_d, labels_d, row_loss,
                    (int) nclasses, (int) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

void ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t nclasses = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * logits_d = (const float *) src0f->data;
    const float * labels_d = (const float *) src1f->data;
    float * dst_d = (float *) dst->data;

    const sycl::range<3> block(1, 1, WARP_SIZE);
    const sycl::range<3> grid(1, 1, nrows);
    const size_t nbytes_shared = (size_t) nclasses * sizeof(float);
    const size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    if (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<true>(
                        grad_d, logits_d, labels_d, dst_d,
                        (int) nclasses, (int) nrows,
                        get_pointer(smem), item);
                });
        });
    } else {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<false>(
                    grad_d, logits_d, labels_d, dst_d,
                    (int) nclasses, (int) nrows,
                    nullptr, item);
            });
    }
}
