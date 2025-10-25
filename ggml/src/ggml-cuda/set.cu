#include "set.cuh"

template<typename T>
static __global__ void k_set(const T * x, const T * y, T * dst, const int64_t ne,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset) {

    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    const int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    const int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    const int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    const int64_t i10 = tmp;

    T val = x[i];
    if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val = y[((i13*ne12 + i12) * ne11 + i11) * ne10 + i10];
    }
    dst[i] = val;
}

template<typename T>
static void set_cuda_impl(const T * x, const T * y, T * dst, const int64_t n_elements,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset, cudaStream_t stream) {

    const int num_blocks = (n_elements + CUDA_SET_BLOCK_SIZE - 1) / CUDA_SET_BLOCK_SIZE;

    k_set<T><<<num_blocks, CUDA_SET_BLOCK_SIZE, 0, stream>>>(x, y, dst, n_elements,
                                                             ne10, ne11, ne12, ne13,
                                                             s1, s2, s3, offset);

}

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {


    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT( dst->type == src0->type);

    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / ggml_element_size(dst);
    const int64_t s2     = dst->op_params[1] / ggml_element_size(dst);
    const int64_t s3     = dst->op_params[2] / ggml_element_size(dst);
    const int64_t offset = dst->op_params[3] / ggml_element_size(dst);
    const bool   inplace = (bool) dst->op_params[4];


    // If not inplace, copy src0 to dst first
    if (!inplace) {

        CUDA_CHECK(cudaMemcpyAsync(dst->data, src0->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
    }

    const int64_t n = ggml_nelements(dst);
  

    switch (src0->type) {
        case GGML_TYPE_F32:
            set_cuda_impl<float>(src0_d, src1_d, dst_d, ggml_nelements(dst),
                                 src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 s1, s2, s3, offset, stream);
            break;
        case GGML_TYPE_I32:
            set_cuda_impl<int32_t>((const int32_t*)src0_d, (const int32_t*)src1_d, (int32_t*)dst_d, ggml_nelements(dst),
                                   src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                   s1, s2, s3, offset, stream);
            break;
        default:
            GGML_ABORT("ggml_cuda_op_set: unsupported src0 type %s", ggml_type_name(src0->type));
            break;
    }
}
