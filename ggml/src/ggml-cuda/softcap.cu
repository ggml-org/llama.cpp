#include "softcap.cuh"

// src rows may be strided (e.g. a half of a fused gate_up output), dst is contiguous
static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k,
        const int ne0, const int ne1, const int ne2, const int64_t s1, const int64_t s2, const int64_t s3) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int i0 = i % ne0;
    int       r  = i / ne0;
    const int i1 = r % ne1;
    r /= ne1;
    const int i2 = r % ne2;
    const int i3 = r / ne2;

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i3*s3 + i2*s2 + i1*s1 + i0]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k,
        const int ne0, const int ne1, const int ne2, const int64_t s1, const int64_t s2, const int64_t s3, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k, ne0, ne1, ne2, s1, s2, s3);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t s1 = src0->nb[1] / sizeof(float);
    const int64_t s2 = src0->nb[2] / sizeof(float);
    const int64_t s3 = src0->nb[3] / sizeof(float);

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), src0->ne[0], src0->ne[1], src0->ne[2], s1, s2, s3, stream);
}
