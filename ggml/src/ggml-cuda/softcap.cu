#include "softcap.cuh"

template <typename T>
static __global__ void softcap_kernel(const T * x, T * dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)(tanhf(scale * (float)x[i]) * softcap);
}

template <typename T>
static void softcap_cuda(const T * x, T * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_kernel<<<num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream>>>(x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    const int nelements = ggml_nelements(src0);

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *)src0->data;
            float * dst_d = (float *)dst->data;
            softcap_cuda(src0_d, dst_d, scale, softcap, nelements, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *)src0->data;
            half * dst_d = (half *)dst->data;
            softcap_cuda(src0_d, dst_d, scale, softcap, nelements, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
