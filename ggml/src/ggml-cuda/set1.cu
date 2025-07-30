#include "ggml-cuda/common.cuh"
#include "set.cuh"

static __global__ void set_f32_cuda_copy( ...) {}



static __global__ void set_f32_cuda( ...) {}





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
    GGML_ASSERT(src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (!inplace) {
        // copy: src1 -> dst.
        set_f32_cuda_copy
    }

    // set: src0 -> dst
    // set_f32_cuda





}
