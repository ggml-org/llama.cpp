#include "scatter.cuh"

static __global__ void scatter_kernel(
        const int32_t * src0, float * dst, const float c,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03
    ) {

    const int64_t total_blocks = ne01 * ne02 * ne03;

    for (int64_t block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {

        const int64_t i1 = block_idx % ne01;
        const int64_t i2 = (block_idx / ne01) % ne02;
        const int64_t i3 = block_idx / (ne01 * ne02);

        float * dst_row = (float *)((char *)dst + i1*nb1 + i2*nb2 + i3*nb3);
        const int * src0_row = (const int *)((const char *)src0 + i1*nb01 + i2*nb02 + i3*nb03);

        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            const int32_t id = src0_row[i0];
            dst_row[id] = c;
        }
    }
}

void ggml_cuda_op_scatter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(int32_t));

    GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(dst));

    float c = ggml_get_op_params_f32(dst, 0);
    bool inplace = ggml_get_op_params_i32(dst, 1);

    // step 1 - copy whole src0 to dst
    if (!inplace) {
        cudaStream_t main_stream = ctx.stream();
        char * dst_ddc = (char *) dst->data;
        char * src0_ddc = (char *) src0->data;

        CUDA_CHECK(cudaMemcpyAsync(dst_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
    }

    // step 2 - set elements in dst indicated by ids to c
    const int32_t * src1_d = (const int32_t *) src1->data;
    float * dst_d = (float *) dst->data;

    int threads = std::min((int) ne10, 512); // ids

    int64_t total_blocks = ne11 * ne12 * ne13;
    int blocks = (int) std::min((int64_t) 65535, total_blocks);

    scatter_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        src1_d, dst_d, c,
        ne10, ne11, ne12, ne13,
        nb1, nb2, nb3,
        nb11, nb12, nb13
    );
}
