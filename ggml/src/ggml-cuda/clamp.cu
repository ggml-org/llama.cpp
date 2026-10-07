#include "clamp.cuh"

static __device__ __forceinline__ float op_clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

// src and dst may be views: rows are contiguous, dims 1..3 follow the strides (in elements).
template <class T>
static __global__ void op_clamp_kernel(const T * x, T * dst, const T min, const T max, const uint32_t k,
        const uint3 ne0, const uint3 ne1, const uint3 ne2,
        const uint32_t s01, const uint32_t s02, const uint32_t s03,
        const uint32_t s1,  const uint32_t s2,  const uint32_t s3) {
    const uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const uint2 d0 = fast_div_modulo(i,    ne0); // <i / ne0, i0>
    const uint2 d1 = fast_div_modulo(d0.x, ne1); // <i / (ne0*ne1), i1>
    const uint2 d2 = fast_div_modulo(d1.x, ne2); // <i3, i2>

    const size_t i_src = d0.y + size_t(d1.y)*s01 + size_t(d2.y)*s02 + size_t(d2.x)*s03;
    const size_t i_dst = d0.y + size_t(d1.y)*s1  + size_t(d2.y)*s2  + size_t(d2.x)*s3;

    dst[i_dst] = (T)op_clamp((float)x[i_src], (float)min, (float)max);
}

template <class T>
static void clamp_cuda(const T * x, T * dst, const T min, const T max, const ggml_tensor * src0, const ggml_tensor * t, cudaStream_t stream) {
    const int64_t k  = ggml_nelements(src0);
    const size_t  ts = sizeof(T);
    GGML_ASSERT(k <= std::numeric_limits<uint32_t>::max());

    const uint3 ne0 = init_fastdiv_values(src0->ne[0]);
    const uint3 ne1 = init_fastdiv_values(src0->ne[1]);
    const uint3 ne2 = init_fastdiv_values(src0->ne[2]);

    const int64_t num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, (uint32_t) k, ne0, ne1, ne2,
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
