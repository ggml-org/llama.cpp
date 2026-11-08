#include "solve_tri.hpp"
#include "common.hpp"
#include <oneapi/mkl/blas.hpp>

template <int n_template, int k_template>
static void solve_tri_f32_fast(const float * __restrict__ A,
                               const float * __restrict__ B,
                               float * __restrict__ X,
                               const int64_t ne02, [[maybe_unused]] const int64_t ne03,
                               const int64_t nb02, const int64_t nb03,
                               const int64_t nb12, const int64_t nb13,
                               const int64_t nb2,  const int64_t nb3,
                               const int n_arg, const int k_arg,
                               const sycl::nd_item<2> & item, float * sA) {

    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;

    const int batch_idx = item.get_group(1);
    const int lane      = item.get_local_id(1) % WARP_SIZE;
    const int col_idx   = item.get_local_id(0);

    if (col_idx >= k) {
        return;
    }

    const int64_t i03 = batch_idx / ne02;
    const int64_t i02 = batch_idx % ne02;

    const float * A_batch = (const float *) ((const char *) A + i02 * nb02 + i03 * nb03);
    const float * B_batch = (const float *) ((const char *) B + i02 * nb12 + i03 * nb13);
    float *       X_batch = (float *)       ((char *)       X + i02 * nb2  + i03 * nb3);

    const int offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    for (int i = 0; i < n * n; i += k * WARP_SIZE) {
        const int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    float x_low  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    float x_high = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    const int half      = WARP_SIZE;
    const int nrows_low = (n < half) ? n : half;

#pragma unroll
    for (int row = 0; row < nrows_low; ++row) {
        float sum = 0.0f;
        if (lane < row) {
            sum += sA[row * n + lane] * x_low;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == row) {
            x_low = (x_low - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int row = half; row < n; ++row) {
        float     sum = sA[row * n + lane] * x_low;
        const int j   = half + lane;
        if (j < row) {
            sum += sA[row * n + j] * x_high;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == row - half) {
            x_high = (x_high - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int row = rr * WARP_SIZE + lane;
        if (row < n) {
            const float val            = (row < half) ? x_low : x_high;
            X_batch[row * k + col_idx] = val;
        }
    }
}

static void solve_tri_f32_mkl(dpct::queue_ptr stream,
                               const float * A, float * X,
                               int n, int k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    const float alpha = 1.0f;
    const int64_t total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    const int64_t stride_a = nb02 / sizeof(float);
    const int64_t stride_x = nb2 / sizeof(float);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::right,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        total_batches);
}

inline void ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int n    = src0->ne[0];
    const int k    = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    const float * A_d = static_cast<const float *>(src0->data);
    const float * B_d = static_cast<const float *>(src1->data);
    float * X_d       = static_cast<float *>(dst->data);

    if (X_d != B_d) {
        const int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * sizeof(float));
    }

    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const int64_t total_batches = ne02 * ne03;

    if (n <= 2 * WARP_SIZE && k <= 32) {
        const int smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        const sycl::range<2> grid(1, total_batches);
        const sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } else {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

void ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
