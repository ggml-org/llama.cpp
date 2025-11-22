#include "cumsum.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

#include <cstdint>

__global__ void cumsum_f32_kernel(const float * x, float * dst, int64_t n) {
    // TODO: this is a naive implementation just for getting something working.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = x[0];
        for (int64_t i = 1; i < n; i++) {
            dst[i] = dst[i-1] + x[i];
        }
    }
}

void cumsum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream) {
#ifdef GGML_CUDA_USE_CUB
    size_t tmp_size = 0;

    // Query how much temp storage CUDA UnBound (CUB) needs
    cub::DeviceScan::InclusiveSum(
        nullptr,       // d_temp_storage (null = just query size)
        tmp_size,      // reference to size (will be set by CUB)
        x,             // input pointer
        dst,           // output pointer
        ne,            // number of elements
        stream         // CUDA stream to use
    );

    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);

    // Perform the inclusive scan
    cub::DeviceScan::InclusiveSum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);

#else
    GGML_UNUSED(pool);
    cumsum_f32_kernel<<<1, 1, 0, stream>>>(x, dst, ne);
#endif // GGML_CUDA_USE_CUB
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguously_allocated(src0));

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne0 = src0->ne[0];  // row length (cumsum computed along this dimension)
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];
    const int64_t nrows = ne1 * ne2 * ne3;  // total number of rows

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    for (int64_t i = 0; i < nrows; i++) {
        const float * src_row = src0_d + i * ne0;
        float * dst_row = dst_d + i * ne0;
        cumsum_f32_cuda(pool, src_row, dst_row, ne0, stream);
    }
}
