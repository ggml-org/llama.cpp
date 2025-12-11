#include "common.cuh"
#include "ggml.h"
#include "ggml-cuda/rel-pos.cuh"


template <typename T>
__global__ static void get_rel_pos_kernel(const void * src, void * dst, int C) {
    int kh = gridDim.x;
    int qh = gridDim.y;
    float k_scale = MAX((float)qh / kh, 1.0f);
    float q_scale = MAX((float)kh / qh, 1.0f);
    int ki = blockIdx.x;
    int qi = blockIdx.y;
    int pos = int(qi*q_scale - ki*k_scale + (kh - 1)*k_scale);

    int s0 = C;
    int s1 = C * kh;

    for (int ci = threadIdx.x; ci < C; ci += blockDim.x) {
        ((T *) dst)[qi*s1 + ki*s0 + ci] = ((const T *) src)[pos*C + ci];
    }
}

static unsigned int round_to_pow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
} 

void ggml_cuda_op_get_rel_pos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == dst->type);

    int C  = ne0;
    int kh = ne1;
    int qh = ne2;

    int num_threads = MIN(CUDA_GET_REL_POS_BLOCK_SIZE, MAX(32, round_to_pow2(C)));
    dim3 grid { (unsigned int)kh, (unsigned int)qh, 1 };

    const void * src0_d = (const void *)src0->data;
    void * dst_d = (void *)dst->data;
    cudaStream_t stream = ctx.stream();

    switch (src0->type)
    {
    case GGML_TYPE_F32:
        get_rel_pos_kernel<float><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, C);
        break;
    case GGML_TYPE_F16:
        get_rel_pos_kernel<half><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, C);
        break;
    case GGML_TYPE_BF16:
        get_rel_pos_kernel<nv_bfloat16><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, C);
        break;
    default:
        GGML_ABORT("%s: unsupported type (%s)\n", __func__, ggml_type_name(src0->type));
        break;
    }
}