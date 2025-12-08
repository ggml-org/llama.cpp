#include "common.cuh"
#include "ggml-cuda/vendors/cuda.h"
#include <cublas_api.h>
#include "ggml.h"
#include "solve_tri.cuh"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

static __global__ void get_batch_pointers(const float * A, float * X, const float ** A_ptrs, float ** X_ptrs,
                                          int64_t ne02, int64_t total_batches,
                                          size_t s02, size_t s03, size_t s2, size_t s3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_batches) {
        return;
    }

    const int64_t i3 = idx / ne02;
    const int64_t i2 = idx % ne02;

    A_ptrs[idx] = A + i3 * s03 + i2 * s02;
    X_ptrs[idx] = X + i3 * s3 + i2 * s2;
}

static void solve_tri_f32_cublas(ggml_backend_cuda_context &ctx,
                                 const float * A,
                                 const float * B,
                                 float *       X,
                                 int           n,
                                 int           k,
                                 int64_t       ne02,
                                 int64_t       ne03,
                                 size_t        s02,
                                 size_t        s03,
                                 size_t        s12,
                                 size_t        s13,
                                 size_t        s2,
                                 size_t        s3,
                                 cudaStream_t  stream) {
    const float alpha = 1.0f;
    const int64_t total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    // Bulk copy B -> X (contiguous tensors)
    if (X != B) {
        const int64_t total_elements_BX = n * k * total_batches;
        CUDA_CHECK(cudaMemcpyAsync(X, B, total_elements_BX * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    int id = ggml_cuda_get_device();

    ggml_cuda_pool_alloc<const float *> A_ptrs_alloc(ctx.pool(id), total_batches);
    ggml_cuda_pool_alloc<float *> X_ptrs_alloc(ctx.pool(id), total_batches);

    const float ** A_ptrs_dev = A_ptrs_alloc.get();
          float ** X_ptrs_dev = X_ptrs_alloc.get();

    get_batch_pointers<<<(total_batches + 255) / 256, 256, 0, stream>>>(
        A, X, A_ptrs_dev, X_ptrs_dev, ne02, total_batches, s02, s03, s2, s3);

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

    // Yes, this is necessary, without this we get RMSE errors
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id),
                                    CUBLAS_SIDE_RIGHT,
                                    CUBLAS_FILL_MODE_UPPER,
                                    CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT,
                                    k,
                                    n,
                                    &alpha,
                                    A_ptrs_dev, n,
                                    X_ptrs_dev, k,
                                    total_batches));

    // revert to standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_TF32_TENSOR_OP_MATH));

    GGML_UNUSED_VARS(s12, s13);
}


// ----------------------------------------------------------------------------
// Public entry point
// ----------------------------------------------------------------------------
void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // A (n×n, lower triangular)
    const ggml_tensor * src1 = dst->src[1];   // B (n×k)

    ggml_is_contiguous(src0);
    ggml_is_contiguous(src1);

    const int64_t n = src0->ne[0];
    const int64_t k = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    solve_tri_f32_cublas(ctx,
        (const float *) src0->data,
        (const float *) src1->data,
        (float *) dst->data,
        n, k,
        ne02, ne03,
        src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
        src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
        dst->nb[2]  / sizeof(float), dst->nb[3]  / sizeof(float),
        ctx.stream());
}
