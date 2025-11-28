#include "tri.cuh"
#include "ggml.h"

// Triangle type comparison - determines which elements to keep
__device__ static inline bool tri_compare(const int i, const int r, const ggml_tri_type type) {
    switch (type) {
        case GGML_TRI_TYPE_LOWER:      return i < r;
        case GGML_TRI_TYPE_LOWER_DIAG: return i <= r;
        case GGML_TRI_TYPE_UPPER:      return i > r;
        case GGML_TRI_TYPE_UPPER_DIAG: return i >= r;
        default: return false;
    }
}

template<typename T>
static __global__ void tri_kernel(
    const T * src, T * dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
    const ggml_tri_type ttype) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T * src_row = (const T *) ((const char *) src + i1*nb01 + i2*nb02 + i3*nb03);
    T       * dst_row = (T       *) ((      char *) dst + i1*nb1  + i2*nb2  + i3*nb3);

    // Each thread processes elements at stride blockDim.x
    for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        dst_row[i0] = tri_compare(i0, i1, ttype)
            ? src_row[i0] : static_cast<T>(0.f);
    }
}

template<typename T>
static void tri_cuda(
    const T * src, T * dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
    const ggml_tri_type ttype,
    cudaStream_t stream) {

    dim3 block_dims(CUDA_TRI_BLOCK_SIZE, 1, 1);
    dim3 grid_dims(ne01, ne02, ne03);

    tri_kernel<<<grid_dims, block_dims, 0, stream>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        nb00, nb01, nb02, nb03,
        nb0, nb1, nb2, nb3,
        ttype
    );
}

void ggml_cuda_op_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    const ggml_tri_type ttype = static_cast<ggml_tri_type>(ggml_get_op_params_i32(dst, 0));

    GGML_ASSERT(src0->type == dst->type);

    switch(src0->type) {
        case GGML_TYPE_F32:
            {
                tri_cuda(
                    (const float *)src0->data, (float *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        case GGML_TYPE_F16:
            {
                tri_cuda(
                    (const half *)src0->data, (half *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        case GGML_TYPE_BF16:
            {
                tri_cuda(
                    (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        default:
            GGML_ABORT("fatal error");
    }
}
