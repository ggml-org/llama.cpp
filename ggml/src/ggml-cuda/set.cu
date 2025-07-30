#include "ggml-cuda/common.cuh"
#include "set.cuh"

static __global__ void set_f32_cuda_copy(const float * __restrict__ src1,
                                         float * __restrict__ dst,
                                         const size_t ne0,
                                         const size_t ne1,
                                         const size_t ne2,
                                         const size_t ne3,
                                         const int    offset,  // element‐offset
                                         const int    nb1,     // stride in elements along dim1
                                         const int    nb2,     // stride in elements along dim2
                                         const int    nb3      // stride in elements along dim3
) {
    const size_t total = ne0 * ne1 * ne2 * ne3;
    const size_t gid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    // unravel into 4D indices (i0 fastest, then i1, i2, i3):
    size_t       tmp = gid;
    const size_t i0  = tmp % ne0;
    tmp /= ne0;
    const size_t i1 = tmp % ne1;
    tmp /= ne1;
    const size_t i2 = tmp % ne2;
    tmp /= ne2;
    const size_t i3 = tmp;  // < ne3

    // compute flat positions with strides + offset
    const size_t pos = offset + i0 + i1 * (size_t) nb1 + i2 * (size_t) nb2 + i3 * (size_t) nb3;

    dst[pos] = src1[pos];
}

static __global__ void set_f32_cuda(const float * __restrict__ src0,
                                    float * __restrict__ dst,
                                    const size_t ne0,
                                    const size_t ne1,
                                    const size_t ne2,
                                    const size_t ne3,
                                    const int    offset,  // element‐offset into	dst
                                    const int    nb1,     // stride in elements along dim1
                                    const int    nb2,     // stride in elements along dim2
                                    const int    nb3      // stride in elements along dim3
) {
    // src0 is contiguous over ne0*ne1*ne2*ne3 elements
    const size_t total = ne0 * ne1 * ne2 * ne3;
    const size_t gid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    // unravel gid to 4D (same as copy)
    size_t       tmp = gid;
    const size_t i0  = tmp % ne0;
    tmp /= ne0;
    const size_t i1 = tmp % ne1;
    tmp /= ne1;
    const size_t i2 = tmp % ne2;
    tmp /= ne2;
    const size_t i3 = tmp;

    // dst position has the same formula:
    const size_t pos = offset + i0 + i1 * (size_t) nb1 + i2 * (size_t) nb2 + i3 * (size_t) nb3;

    // src0 is contiguous: flat index = gid
    dst[pos] = src0[gid];
}

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t nb1     = dst->op_params[0];
    const int32_t nb2     = dst->op_params[1];
    const int32_t nb3     = dst->op_params[2];
    const int32_t offset  = dst->op_params[3];
    const bool    inplace = dst->op_params[4];

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // dims
    const size_t ne0 = dst->ne[0];
    const size_t ne1 = dst->ne[1];
    const size_t ne2 = dst->ne[2];
    const size_t ne3 = dst->ne[3];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const size_t total   = ne0 * ne1 * ne2 * ne3;
    const int    threads = 256;
    const int    blocks  = (total + threads - 1) / threads;

    if (!inplace) {
        // copy whole src1→dst
        set_f32_cuda_copy<<<blocks, threads, 0, stream>>>(src1_d, dst_d, ne0, ne1, ne2, ne3, offset, nb1, nb2, nb3);
    }

    // then overwrite from src0→dst at same offsets/strides
    set_f32_cuda<<<blocks, threads, 0, stream>>>(src0_d, dst_d, ne0, ne1, ne2, ne3, offset, nb1, nb2, nb3);
}
