#include "sum.cuh"
#include "sumrows.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#elif defined(GGML_USE_HIP)
#include <hipcub/hipcub.hpp>
using namespace hipcub;
#endif  // GGML_CUDA_USE_CUB

#include <cstdint>

void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream) {
#if defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP)
    size_t tmp_size = 0;
    CUDA_CHECK(DeviceReduce::Sum(nullptr,       tmp_size, x, dst, ne, stream));
    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    CUDA_CHECK(DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream));
#else
    // Fallback for backends without a device-wide reduction (e.g. MUSA):
    // use the (inefficient) sum_rows implementation.
    sum_rows_f32_cuda(x, dst, ne, 1, stream);
    GGML_UNUSED(pool);
#endif // GGML_CUDA_USE_CUB
}

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguously_allocated(src0));

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne = ggml_nelements(src0);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
}
