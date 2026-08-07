#include "snake.cuh"
#include "convert.cuh"

// Fused Snake activation: y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [1, C], inv_b: [1, C]
// Supports F32, F16, BF16 data with F32 compute.

template <typename T>
static __global__ void snake_kernel(
        const T     * __restrict__ x,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        T           * __restrict__ dst,
        const int    total,
        const uint3  T_len_fastdiv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int c = (int) fastdiv((uint32_t) idx, T_len_fastdiv);

    const float xi = ggml_cuda_cast<float>(x[idx]);
    const float s  = sinf(a[c] * xi);
    dst[idx] = ggml_cuda_cast<T>(xi + s * s * inv_b[c]);
}

// Internal launcher with explicit x/a/inv_b/dst tensors.
// Shared by the public op (reads dst->src) and the fusion path (explicit args).
static void launch_snake(ggml_backend_cuda_context & ctx,
                         const ggml_tensor * x,
                         const ggml_tensor * a,
                         const ggml_tensor * inv_b,
                         ggml_tensor *       dst) {
    const float * a_d     = (const float *)a->data;
    const float * inv_b_d = (const float *)inv_b->data;

    const int   T = (int)x->ne[0];
    const int   C = (int)x->ne[1];
    const int   total = T * C;
    const uint3 T_len_fastdiv = init_fastdiv_values((uint64_t) T);

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    cudaStream_t stream = ctx.stream();

    switch (x->type) {
        case GGML_TYPE_F32: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float *)x->data, a_d, inv_b_d, (float *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_F16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half *)x->data, a_d, inv_b_d, (half *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_BF16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const nv_bfloat16 *)x->data, a_d, inv_b_d, (nv_bfloat16 *)dst->data, total, T_len_fastdiv);
        } break;
        default:
            GGML_ABORT("snake: unsupported type");
    }
}

// Fusion entry: caller supplies x/a/inv_b explicitly from the matched
// mul -> sin -> sqr -> mul -> add pattern. The dst is the trailing add output.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor *       dst) {
    launch_snake(ctx, x, a, inv_b, dst);
}
