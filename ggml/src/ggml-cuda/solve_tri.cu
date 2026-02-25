#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"

#define MAX_N_FAST 64

static __global__ void get_batch_pointers(const float *  A,
                                          float *        X,
                                          const float ** A_ptrs,
                                          float **       X_ptrs,
                                          int64_t        ne02,
                                          int64_t        total_batches,
                                          size_t         s02,
                                          size_t         s03,
                                          size_t         s2,
                                          size_t         s3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_batches) {
        return;
    }

    const int64_t i3 = idx / ne02;
    const int64_t i2 = idx % ne02;

    A_ptrs[idx] = A + i3 * s03 + i2 * s02;
    X_ptrs[idx] = X + i3 * s3 + i2 * s2;
}

static void solve_tri_f32_cublas(ggml_backend_cuda_context & ctx,
                                 const float *               A,
                                 const float *               B,
                                 float *                     X,
                                 int                         n,
                                 int                         k,
                                 int64_t                     ne02,
                                 int64_t                     ne03,
                                 size_t                      s02,
                                 size_t                      s03,
                                 size_t                      s12,
                                 size_t                      s13,
                                 size_t                      s2,
                                 size_t                      s3,
                                 cudaStream_t                stream) {
    const float   alpha         = 1.0f;
    const int64_t total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    // Bulk copy B -> X (contiguous tensors)
    if (X != B) {
        const int64_t total_elements_BX = n * k * total_batches;
        CUDA_CHECK(cudaMemcpyAsync(X, B, total_elements_BX * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    const int id = ggml_cuda_get_device();

    ggml_cuda_pool_alloc<const float *> A_ptrs_alloc(ctx.pool(id), total_batches);
    ggml_cuda_pool_alloc<float *>       X_ptrs_alloc(ctx.pool(id), total_batches);

    const float ** A_ptrs_dev = A_ptrs_alloc.get();
    float **       X_ptrs_dev = X_ptrs_alloc.get();

    get_batch_pointers<<<(total_batches + 255) / 256, 256, 0, stream>>>(A, X, A_ptrs_dev, X_ptrs_dev, ne02,
                                                                        total_batches, s02, s03, s2, s3);

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

    // Yes, this is necessary, without this we get RMSE errors
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs_dev, n, X_ptrs_dev, k, total_batches));

    // revert to standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_TF32_TENSOR_OP_MATH));

    GGML_UNUSED_VARS(s12, s13);
}

// ======================
// Fast Kernel (n <= 64, k <= max_k_fast) - Warp-based parallel reduction
// ======================
// When ncols_template == 0 the bounds for the loops in this function are not
// known and can't be unrolled. As we want to keep pragma unroll for all other
// cases we supress the clang transformation warning here.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wpass-failed"
#endif  // __clang__
template <int n_template, int ncols_template>
static __global__ void solve_tri_f32_fast(const float * __restrict__ A,
                                          const float * __restrict__ B,
                                          float * __restrict__ X,
                                          const uint3  ne02,
                                          const size_t nb02,
                                          const size_t nb03,
                                          const size_t nb12,
                                          const size_t nb13,
                                          const size_t nb2,
                                          const size_t nb3,
                                          const int    n_arg,
                                          const int    ncols_arg,
                                          const int    ld_arg,
                                          const int    col0_arg) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int n     = n_template == 0 ? n_arg : n_template;
    const int ncols = ncols_template == 0 ? ncols_arg : ncols_template;
    const int ld    = ld_arg;
    const int col0  = col0_arg;

    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int col_idx   = threadIdx.y;

    if (col_idx >= ncols) {
        return;
    }

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
    for (int i = 0; i < n * n; i += ncols * warp_size) {
        const int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    __syncthreads();

    float x_low  = (lane < n) ? B_batch[lane * ld + col0 + col_idx] : 0.0f;
    float x_high = (warp_size + lane < n) ? B_batch[(warp_size + lane) * ld + col0 + col_idx] : 0.0f;

    const int half      = warp_size;
    const int nrows_low = (n < half) ? n : half;

#pragma unroll
    for (int row = 0; row < nrows_low; ++row) {
        float sum = 0.0f;
        if (lane < row) {
            sum += sA[row * n + lane] * x_low;
        }
        sum = warp_reduce_sum<warp_size>(sum);

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
        sum = warp_reduce_sum<warp_size>(sum);

        if (lane == row - half) {
            x_high = (x_high - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int row = rr * warp_size + lane;
        if (row < n) {
            const float val            = (row < half) ? x_low : x_high;
            X_batch[row * ld + col0 + col_idx] = val;
        }
    }
}
#ifdef __clang__
#    pragma clang diagnostic pop
#endif  // __clang__

static void solve_tri_f32_cuda_tile(const float * A,
                                    const float * B,
                                    float *       X,
                                    int           n,
                                    int           ncols,
                                    int           ld,
                                    int           col0,
                                    int64_t       ne02,
                                    int64_t       ne03,
                                    size_t        nb02,
                                    size_t        nb03,
                                    size_t        nb12,
                                    size_t        nb13,
                                    size_t        nb2,
                                    size_t        nb3,
                                    cudaStream_t  stream) {
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const int   warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    dim3        threads(warp_size, ncols);
    dim3        grid(ne02 * ne03);
    if (n == 64) {
        switch (ncols) {
            case 32:
                solve_tri_f32_fast<64, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 16:
                solve_tri_f32_fast<64, 16>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 14:
                solve_tri_f32_fast<64, 14>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 12:
                solve_tri_f32_fast<64, 12>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 10:
                solve_tri_f32_fast<64, 10>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 8:
                solve_tri_f32_fast<64, 8>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 6:
                solve_tri_f32_fast<64, 6>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 4:
                solve_tri_f32_fast<64, 4>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 2:
                solve_tri_f32_fast<64, 2>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            case 1:
                solve_tri_f32_fast<64, 1>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0, ld, col0);
                break;
            default:
                solve_tri_f32_fast<0, 0>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, ncols, ld, col0);
        }
    } else {  // run general case
        solve_tri_f32_fast<0, 0>
            <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, ncols, ld, col0);
    }
}

static void solve_tri_f32_cuda(const float * A,
                               const float * B,
                               float *       X,
                               int           n,
                               int           k,
                               int64_t       ne02,
                               int64_t       ne03,
                               size_t        nb02,
                               size_t        nb03,
                               size_t        nb12,
                               size_t        nb13,
                               size_t        nb2,
                               size_t        nb3,
                               cudaStream_t  stream) {
    const int max_k_fast = 1024 / ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (k <= max_k_fast) {
        solve_tri_f32_cuda_tile(A, B, X, n, k, k, 0, ne02, ne03, nb02, nb03, nb12, nb13, nb2, nb3, stream);
        return;
    }

    for (int col0 = 0; col0 < k; col0 += max_k_fast) {
        const int k_tile = col0 + max_k_fast <= k ? max_k_fast : (k - col0);
        solve_tri_f32_cuda_tile(A, B, X, n, k_tile, k, col0, ne02, ne03, nb02, nb03, nb12, nb13, nb2, nb3, stream);
    }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // A (n×n, lower triangular)
    const ggml_tensor * src1 = dst->src[1];  // B (n×k)

    ggml_is_contiguous(src0);
    ggml_is_contiguous(src1);

    const int64_t n    = src0->ne[0];
    const int64_t k    = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    if (n <= MAX_N_FAST) {
        solve_tri_f32_cuda((const float *) src0->data, (const float *) src1->data, (float *) dst->data, n, k,
                           src0->ne[2], src0->ne[3], src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                           src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                           dst->nb[3] / sizeof(float), ctx.stream());
    } else {
        solve_tri_f32_cublas(ctx, (const float *) src0->data, (const float *) src1->data, (float *) dst->data, n, k,
                             ne02, ne03, src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                             src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                             dst->nb[3] / sizeof(float), ctx.stream());
    }
}
