#include "common.cuh"
#include "ggml.h"
#include "ggml-cuda/rel-pos.cuh"

/*

static void ggml_compute_forward_get_rel_pos_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t kh = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t pos = (kh - i1 - 1) + i2;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}


void ggml_compute_forward_get_rel_pos(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rel_pos_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

struct ggml_tensor * ggml_get_rel_pos(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   qh,
        int                   kh) {
    GGML_ASSERT(qh + kh - 1 <= a->ne[1]);

    const int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, 3, ne);

    result->op     = GGML_OP_GET_REL_POS;
    result->src[0] = a;

    return result;
}

*/

template <typename T>
__global__ static void get_rel_pos_kernel(const void * src, void * dst, int C) {
    int kh = gridDim.x;
    int ki = blockIdx.x;
    int qi = blockIdx.y;
    int pos = (kh - 1) + qi - ki;

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