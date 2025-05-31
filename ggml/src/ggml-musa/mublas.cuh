#include "ggml-cuda/common.cuh"

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
);