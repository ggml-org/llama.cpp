#include "mublas.cuh"

static __global__ void k_compute_batched_ptrs(
        const half * src0_as_f16, const half * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

void ggml_cuda_mul_mat_batched_cublas_gemm_batched_ex(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const half * src0_f16, const half * src1_f16, char * dst_t,
    const size_t nbd2, const size_t nbd3,
    const int64_t r2, const int64_t r3,
    const int64_t s11, const int64_t s12, const int64_t s13,
    const void * alpha, const void * beta,
    const cudaDataType_t cu_data_type,
    const cublasComputeType_t cu_compute_type,
    cudaStream_t main_stream
) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // use cublasGemmBatchedEx
    const int64_t ne23 = ne12*ne13;

    // Allocate memory for pointer arrays using cudaMalloc to avoid segmentation faults in muBLAS.
    const void ** ptrs_src;
    void ** ptrs_dst;
    CUDA_CHECK(cudaMalloc((void **)&ptrs_src, sizeof(void *)*2*ne23));
    CUDA_CHECK(cudaMalloc((void **)&ptrs_dst, sizeof(void *)*1*ne23));

    dim3 block_dims(ne13, ne12);
    k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
            src0_f16, src1_f16, dst_t,
            ptrs_src, ptrs_dst,
            ne12, ne13,
            ne23,
            nb02, nb03,
            src1->type == GGML_TYPE_F16 ? nb12 : s12*sizeof(half),
            src1->type == GGML_TYPE_F16 ? nb13 : s13*sizeof(half),
            nbd2, nbd3,
            r2, r3);
    CUDA_CHECK(cudaGetLastError());

    // This operation is essential for musa; without it, generated tokens will
    // be garbled and may eventually cause MUBLAS_STATUS_INTERNAL_ERROR.
    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(
    cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            ne01, ne11, ne10,
            alpha, (const void **) (ptrs_src + 0*ne23), CUDA_R_16F,   nb01/nb00,
                   (const void **) (ptrs_src + 1*ne23), CUDA_R_16F,   s11,
            beta,  (      void **) (ptrs_dst + 0*ne23), cu_data_type, ne0,
            ne23,
            cu_compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaFree(ptrs_src));
    CUDA_CHECK(cudaFree(ptrs_dst));
}