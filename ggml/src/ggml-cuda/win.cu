#include "common.cuh"
#include "ggml.h"
#include "ggml-cuda/win.cuh"

/*

C++ CPU Implementation:


static void ggml_compute_forward_win_part_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    const int32_t nep0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t nep1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t bs   = ((const int32_t *)(dst->op_params))[2];
    const int32_t w    = ((const int32_t *)(dst->op_params))[3];

    assert(ne00 == ne0);
    assert(ne3  == nep0*nep1*bs);

    // TODO: optimize / multi-thread
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        int px = i3 % nep0;
        int py = (i3 / nep0) % nep1;
        int b  = i3 / (nep0 * nep1); 
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int64_t i03 = b;
                    const int64_t i02 = py*w + i2;
                    const int64_t i01 = px*w + i1;
                    const int64_t i00 = i0;

                    void * sp = ((void *) src0->data) + i03*nb03 + i02*nb02  + i01*nb01 + i00*nb00;
                    void * dp = ((void *) dst->data)  + i3*nb3   + i2*nb2    + i1*nb1   + i0*nb0; 

                    if (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                        *((ggml_fp16_t *) dp) = 0;
                    } else {
                        *((ggml_fp16_t *) dp) = *((ggml_fp16_t *) sp);
                    }
                }
            }
        }
    }
}

void ggml_compute_forward_win_part(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_I32:
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_win_part_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


struct ggml_tensor * ggml_win_part(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   w) {
    // padding
    const int px = (w - a->ne[1]%w)%w;
    const int py = (w - a->ne[2]%w)%w;

    const int bs = a->ne[3];
    const int npx = (px + a->ne[1])/w;
    const int npy = (py + a->ne[2])/w;
    const int np  = npx*npy*bs;

    const int64_t ne[4] = { a->ne[0], w, w, np, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, bs, w };
    ggml_set_op_params(result, params, sizeof(params));

    result->op     = GGML_OP_WIN_PART;
    result->src[0] = a;

    return result;
}


*/

struct win_param {
    int w;
    int C;
    int npx;
    int npy;
    int ne1;
    int ne2;
    size_t nb00;
    size_t nb01;
    size_t nb02;
    size_t nb03;
};

template<typename T>
__global__ static void win_part_kernel(
    const void * src,
    void * dst,
    win_param p)
{
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    int px = i3 % p.npx;
    int py = (i3 / p.npx) % p.npy;
    int b  = i3 / (p.npx * p.npy);

    const int nb0 = sizeof(T);
    const int nb1 = p.C * sizeof(T);
    const int nb2 = p.C * p.w * sizeof(T);
    const int nb3 = p.C * p.w * p.w * sizeof(T);

    if (py*p.w + i2 >= p.ne2 || px*p.w + i1 >= p.ne1) {
        for (int i0 = threadIdx.x; i0 < p.C; i0 += blockDim.x) {
            char * dp = (char *)dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0;
            *((T *) dp) = 0;
        }
        return;
    }

    for (int i0 = threadIdx.x; i0 < p.C; i0 += blockDim.x) {
        int i03 = b;
        int i02 = py*p.w + i2;
        int i01 = px*p.w + i1;
        int i00 = i0;

        const char * sp = (const char *)src + i03*p.nb03 + i02*p.nb02  + i01*p.nb01 + i00*p.nb00;
        char * dp = (char *)dst  + i3*nb3   + i2*nb2    + i1*nb1   + i0*nb0;

        *((T *) dp) = *((const T *) sp);
    }
}


struct win_unpart_param {
    int w;
    int C;
    int npx;
    int npy;
    int w0;
    int h0;
    size_t nb00;
    size_t nb01;
    size_t nb02;
    size_t nb03;
};

template<typename T>
__global__ static void win_unpart_kernel(
    const void * src,
    void * dst,
    win_unpart_param p)
{
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    int ip2 = i2/p.w;
    int ip1 = i1/p.w;

    int i03 = i3*p.npx*p.npy + ip2*p.npx + ip1;
    int i02 = i2%p.w;
    int i01 = i1%p.w;

    const int nb0 = sizeof(T);
    const int nb1 = p.C * sizeof(T);
    const int nb2 = p.C * p.w0 * sizeof(T);
    const int nb3 = p.C * p.w0 * p.h0 * sizeof(T);

    for (int i0 = threadIdx.x; i0 < p.C; i0 += blockDim.x) {
        int i00 = i0;
        const char * sp = (const char *)src + i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
        char * dp = (char *)dst + i3*nb3   + i2*nb2   + i1*nb1   + i0*nb0;

        *((T *) dp) = *((const T *) sp);
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

void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == dst->type);

    int npx = dst->op_params[0];
    int npy = dst->op_params[1];
    int w   = dst->op_params[3];
    int C   = ne0;
    int np  = ne3;

    GGML_ASSERT(ne1 == w && ne2 == w);

    win_param params = {
        w,
        C,
        npx,
        npy,
        (int)ne01,
        (int)ne02,
        src0->nb[0],
        src0->nb[1],
        src0->nb[2],
        src0->nb[3]
    };

    dim3 grid { (unsigned int)w, (unsigned int)w, (unsigned int)np };
    int num_threads = MIN(CUDA_WINPART_BLOCK_SIZE, MAX(32, round_to_pow2(C)));

    const void * src0_d = (const void *)src0->data;
    void * dst_d = (void *)dst->data;
    cudaStream_t stream = ctx.stream();

    switch (src0->type)
    {
    case GGML_TYPE_F32:
        win_part_kernel<float><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    case GGML_TYPE_F16:
        win_part_kernel<half><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    case GGML_TYPE_BF16:
        win_part_kernel<nv_bfloat16><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    default:
        GGML_ABORT("%s: unsupported type (%s)\n", __func__, ggml_type_name(src0->type));
        break;
    }
}


/*

C++ CPU Implementation:

static void ggml_compute_forward_win_unpart_f16(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    const int32_t w = ((const int32_t *)(dst->op_params))[0];

    // padding
    const int px = (w - ne1%w)%w;
    const int py = (w - ne2%w)%w;

    const int npx = (px + ne1)/w;
    const int npy = (py + ne2)/w;

    assert(ne0 == ne00);
    assert(ne03 == npx*npy*ne3);

    // TODO: optimize / multi-thread
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int ip2 = i2/w;
                    const int ip1 = i1/w;
    
                    const int64_t i03 = i3*npx*npy + ip2*npx + ip1;
                    const int64_t i02 = i2%w;
                    const int64_t i01 = i1%w;
                    const int64_t i00 = i0;
    
                    void * sp = ((void *) src0->data) + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00;
                    void * dp = ((void *) dst->data)  + i3*nb3   + i2*nb2   + i1*nb1   + i0*nb0;

                    *((ggml_fp16_t *) dp) = *((ggml_fp16_t *) sp);
                }
            }
        }
    }
}

void ggml_compute_forward_win_unpart(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_I32:
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_win_unpart_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

struct ggml_tensor * ggml_win_unpart(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   w0,
        int                   h0,
        int                   w) {
    return ggml_win_unpart_ext(ctx, a, w0, h0, 1, w);
}

struct ggml_tensor * ggml_win_unpart_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   w0,
        int                   h0,
        int                   b0,
        int                   w) {
    const int64_t ne[4] = { a->ne[0], w0, h0, b0 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    GGML_ASSERT(ggml_is_contiguous(a));

    int32_t params[] = { w };
    ggml_set_op_params(result, params, sizeof(params));

    result->op     = GGML_OP_WIN_UNPART;
    result->src[0] = a;

    return result;
}


*/
void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == dst->type);

    int w  = dst->op_params[0];
    int C  = ne0;
    int w0 = ne1;
    int h0 = ne2;
    int b0 = ne3;

    const int px = (w - ne1%w)%w;
    const int py = (w - ne2%w)%w;

    const int npx = (px + ne1)/w;
    const int npy = (py + ne2)/w;

    assert(ne0 == ne00);
    assert(ne03 == npx*npy*ne3);

    win_unpart_param params = {
        w,
        C,
        npx,
        npy,
        w0,
        h0,
        src0->nb[0],
        src0->nb[1],
        src0->nb[2],
        src0->nb[3]
    };

    dim3 grid { (unsigned int)w0, (unsigned int)h0, (unsigned int)b0 };
    int num_threads = MIN(CUDA_WINPART_BLOCK_SIZE, MAX(32, round_to_pow2(C)));

    const void * src0_d = (const void *)src0->data;
    void * dst_d = (void *)dst->data;
    cudaStream_t stream = ctx.stream();

    switch (src0->type)
    {
    case GGML_TYPE_F32:
        win_unpart_kernel<float><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    case GGML_TYPE_F16:
        win_unpart_kernel<half><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    case GGML_TYPE_BF16:
        win_unpart_kernel<nv_bfloat16><<<grid, num_threads, 0, stream>>>(src0_d, dst_d, params);
        break;
    default:
        GGML_ABORT("%s: unsupported type (%s)\n", __func__, ggml_type_name(src0->type));
        break;
    }

}