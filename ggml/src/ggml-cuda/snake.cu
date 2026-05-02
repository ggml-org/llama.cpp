#include "snake.cuh"

// Fused Snake activation: y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [C], inv_b: [C]
// Supports F32, F16, BF16 data with F32 compute.

template <typename T>
static __device__ __forceinline__ float snake_load(T x);
template <> __device__ __forceinline__ float snake_load(float x)          { return x; }
template <> __device__ __forceinline__ float snake_load(half x)           { return __half2float(x); }
template <> __device__ __forceinline__ float snake_load(nv_bfloat16 x)    { return __bfloat162float(x); }

template <typename T>
static __device__ __forceinline__ T snake_store(float x);
template <> __device__ __forceinline__ float       snake_store<float>(float x)       { return x; }
template <> __device__ __forceinline__ half        snake_store<half>(float x)        { return __float2half(x); }
template <> __device__ __forceinline__ nv_bfloat16 snake_store<nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
static __global__ void kernel_snake(
        const T     * __restrict__ x,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        T           * __restrict__ dst,
        const int T_len,
        const int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_len * C) return;

    const int c = idx / T_len;

    const float xi = snake_load(x[idx]);
    const float s  = sinf(a[c] * xi);
    dst[idx] = snake_store<T>(xi + s * s * inv_b[c]);
}

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * a_d     = (const float *)src1->data;
    const float * inv_b_d = (const float *)src2->data;

    const int T = (int)src0->ne[0];
    const int C = (int)src0->ne[1];
    const int total = T * C;

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32: {
            kernel_snake<<<grid_size, block_size, 0, stream>>>(
                (const float *)src0->data, a_d, inv_b_d, (float *)dst->data, T, C);
        } break;
        case GGML_TYPE_F16: {
            kernel_snake<<<grid_size, block_size, 0, stream>>>(
                (const half *)src0->data, a_d, inv_b_d, (half *)dst->data, T, C);
        } break;
        case GGML_TYPE_BF16: {
            kernel_snake<<<grid_size, block_size, 0, stream>>>(
                (const nv_bfloat16 *)src0->data, a_d, inv_b_d, (nv_bfloat16 *)dst->data, T, C);
        } break;
        default:
            GGML_ABORT("snake: unsupported type");
    }
}
