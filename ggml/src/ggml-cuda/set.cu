#include "ggml-cuda/common.cuh"
#include "set.cuh"

static __global__ void set_f32_cuda(const float * __restrict__ src1,
                                    float * __restrict__ dst,
                                    const size_t ne10,
                                    const size_t ne11,
                                    const size_t ne12,
                                    const size_t ne13,
                                    const size_t nb10,
                                    const size_t nb11,
                                    const size_t nb12,
                                    const size_t nb13,
                                    const size_t nb0,
                                    const size_t nb1,
                                    const size_t nb2,
                                    const size_t nb3,
                                    const size_t offset

) {
    const size_t total = ne10 * ne11 * ne12 * ne13;
    const size_t gid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) {
        return;
    }

    size_t tmp = gid;

    const size_t i0 = tmp % ne10;
    tmp /= ne10;
    const size_t i1 = tmp % ne11;
    tmp /= ne11;
    const size_t i2 = tmp % ne12;
    tmp /= ne12;
    const size_t i3 = tmp;

    size_t dst_offset  = offset + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
    size_t src1_offset = i0 * nb10 + i1 * nb11 + i2 * nb12 + i3 * nb13;

    *((float *) ((char *) dst + dst_offset)) = *((const float *) ((const char *) src1 + src1_offset));
}

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // nb0 is implicitly element_size because src0 and dst are contiguous
    const int32_t nb1     = dst->op_params[0];
    const int32_t nb2     = dst->op_params[1];
    const int32_t nb3     = dst->op_params[2];
    const int32_t offset  = dst->op_params[3];
    const bool    inplace = dst->op_params[4];

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    // TODO: support more dtypes.
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS01;
    const int nb0 = ggml_element_size(dst);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (!inplace) {
        // copy whole src0 -> dst.
        CUDA_CHECK(cudaMemcpyAsync(dst_d, src0_d, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
    }

    // set: src1 -> dst

    const size_t total      = ne10 * ne11 * ne12 * ne13;
    const size_t num_blocks = (total + CUDA_SET_BLOCK_SIZE - 1) / CUDA_SET_BLOCK_SIZE;

    set_f32_cuda<<<num_blocks, CUDA_SET_BLOCK_SIZE, 0, stream>>>(
        src1_d, dst_d, ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, offset);
}
