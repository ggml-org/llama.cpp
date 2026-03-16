#include "where-id.cuh"

static __global__ void where_id_kernel(
        const float * src0, const int32_t * src1, float * dst,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13
    ) {

    const int64_t total_blocks = ne11 * ne12 * ne13;

    for (int64_t block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {

        const int64_t i1 = block_idx % ne11;
        const int64_t i2 = (block_idx / ne11) % ne12;
        const int64_t i3 = block_idx / (ne11 * ne12);

        float * dst_row = (float *)((char *)dst + i1*nb1 + i2*nb2 + i3*nb3);
        const float * src0_row = (const float *)((const char *)src0 +  i1*nb01 + i2*nb02 + i3*nb03);
        const int * src1_row = (const int *)((const char *)src1 + i1*nb11 + i2*nb12 + i3*nb13);

        for (int64_t i0 = threadIdx.x; i0 < ne10; i0 += blockDim.x) {
            const int32_t id = src1_row[i0];
            dst_row[id] = src0_row[id];
        }
    }
}

void ggml_cuda_op_where_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(src2));

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
    GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(dst));

    // step 1 - copy whole src1 to dst
    cudaStream_t main_stream = ctx.stream();
    char * dst_ddc = (char *) dst->data;
    char * src1_ddc = (char *) src1->data;

    CUDA_CHECK(cudaMemcpyAsync(dst_ddc, src1_ddc, ggml_nbytes(src1), cudaMemcpyDeviceToDevice, main_stream));

    // step 2 - copy elements from src0 indicated by ids to dst
    const float * src0_d = (const float *) src0->data;
    const int32_t * src2_d = (const int32_t *) src2->data;
    float * dst_d = (float *) dst->data;

    int threads = std::min((int) ne20, 768); // ids

    int64_t total_blocks = ne21 * ne22 * ne23;
    int blocks = (int) std::min((int64_t) 65535, total_blocks);

    where_id_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        src0_d, src2_d, dst_d,
        ne20, ne21, ne22, ne23,
        nb1, nb2, nb3,
        nb01, nb02, nb03,
        nb21, nb22, nb23
    );
}
