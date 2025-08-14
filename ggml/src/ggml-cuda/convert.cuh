#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

template<typename src_t, typename dest_t>
 __host__ __device__ inline dest_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<src_t, dest_t>) {
        return x;
    } else {
        return float(x);
    }
}

template<>
__host__ __device__ inline float ggml_cuda_cast<nv_bfloat16, float>(nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<>
__host__ __device__ inline nv_bfloat16 ggml_cuda_cast<float, nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template<>
__host__ __device__ inline half ggml_cuda_cast<nv_bfloat16, half>(nv_bfloat16 x) {
    return half(__bfloat162float(x));
}

template<>
__host__ __device__ inline nv_bfloat16 ggml_cuda_cast<half, nv_bfloat16>(half x) {
    return __float2bfloat16(float(x));
}

template<>
__host__ __device__ inline int ggml_cuda_cast<nv_bfloat16, int>(nv_bfloat16 x) {
    return int(__bfloat162float(x));
}

template<>
__host__ __device__ inline nv_bfloat16 ggml_cuda_cast<int, nv_bfloat16>(int x) {
    return __float2bfloat16(float(x));
}
