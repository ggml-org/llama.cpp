#include "clamp.cuh"

static __device__ __forceinline__ float op_clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

// src and dst may be views: rows are contiguous, dims 1..3 follow the tensor
// strides (in elements).
template <class T>
static __global__ void op_clamp_kernel(const T * x, T * dst, const T min, const T max,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t n,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1,  const int64_t s2,  const int64_t s3) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const int64_t i0 = i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = (i / (ne0*ne1)) % ne2;
    const int64_t i3 = i / (ne0*ne1*ne2);

    dst[i0 + i1*s1 + i2*s2 + i3*s3] = (T)op_clamp((float)x[i0 + i1*s01 + i2*s02 + i3*s03], (float)min, (float)max);
}

template <class T>
static void clamp_cuda(const T * x, T * dst, const T min, const T max, const ggml_tensor * src0, const ggml_tensor * t, cudaStream_t stream) {
    const int64_t n = ggml_nelements(src0);
    const size_t  ts = sizeof(T);
    const int64_t num_blocks = (n + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max,
        src0->ne[0], src0->ne[1], src0->ne[2], n,
        src0->nb[1]/ts, src0->nb[2]/ts, src0->nb[3]/ts,
        t->nb[1]/ts,    t->nb[2]/ts,    t->nb[3]/ts);
}


void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst));

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        clamp_cuda((const half *)src0_d, (half *)dst_d, (half)min, (half)max, src0, dst, stream);
    } else {
        clamp_cuda((const float *)src0_d, (float *)dst_d, (float)min, (float)max, src0, dst, stream);
    }
}
