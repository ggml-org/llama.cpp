#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

template <typename T>
static __global__ void scale_kernel(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = (T)(scale * (float)x[i] + bias);
    }
}

template <typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_kernel<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    const int64_t nelements = ggml_nelements(src0);

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *)src0->data;
            float * dst_d = (float *)dst->data;
            scale_cuda(src0_d, dst_d, scale, bias, nelements, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *)src0->data;
            half * dst_d = (half *)dst->data;
            scale_cuda(src0_d, dst_d, scale, bias, nelements, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
