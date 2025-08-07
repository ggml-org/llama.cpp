#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src0_is_quantized = (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16);
    const bool src1_is_quantized = (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16);

    // if (src0_is_quantized || src1_is_quantized) {
    //     printf("DEBUG: OUT_PROD with quantized tensors - src0_quantized=%d, src1_quantized=%d\n", 
    //            src0_is_quantized, src1_is_quantized);
    //     fflush(stdout);
    // }

    // GGML_ASSERT(src0->type == GGML_TYPE_F32);
    // GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // temp buffers
    float * src0_f32 = nullptr;
    float * src1_f32 = nullptr;
    bool allocated_src0 = false;
    bool allocated_src1 = false;
    cudaStream_t   stream = ctx.stream();

    if (src0_is_quantized) {
        const size_t src0_size = ggml_nelements(src0) * sizeof(float);
        CUDA_CHECK(cudaMallocAsync(&src0_f32, src0_size, stream));
        allocated_src0 = true;

        // Dequantize
        auto dequantize_fn = ggml_get_to_fp32_cuda(src0->type);
        if (dequantize_fn) {
            dequantize_fn(src0->data, src0_f32, ggml_nelements(src0), stream);
        } else {
            CUDA_CHECK(cudaFreeAsync(src0_f32, stream));
            GGML_ABORT("Unsupported quant type for src0");
        }
    } else {
        src0_f32 = (float *) src0->data;
    } 

    if (src1_is_quantized) {
        const size_t src1_size = ggml_nelements(src1) * sizeof(float);
        CUDA_CHECK(cudaMallocAsync(&src1_f32, src1_size, stream));
        allocated_src1 = true;

        auto dequantize_fn = ggml_get_to_fp32_cuda(src1->type);
        if (dequantize_fn) {
            dequantize_fn(src1->data, src1_f32, ggml_nelements(src0), stream);
        } else {
            CUDA_CHECK(cudaFreeAsync(src1_f32, stream));
            GGML_ABORT("Unsupported quant type for src1");
        }
    } else {
        src1_f32 = (float *) src1->data;
    } 
    

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    // const float * src0_d = (const float *) src0->data;
    // const float * src1_d = (const float *) src1->data;

    // Use dequantized data
    const float * src0_d = src0_f32;
    const float * src1_d = src1_f32;
    float       *  dst_d = (float       *)  dst->data;

    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // const int64_t lda = nb01 / sizeof(float);
    const int64_t lda = allocated_src0 ? ne00 : (nb01 / sizeof(float));
    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    // const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    const int64_t           ldb            = allocated_src1 ? 
                                             (src1_T ? ne10 : ne11) :
                                             ((src1_T ?        nb10 :        nb11) /  sizeof(float));
                                
    // GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));
    // Only assert for non dequantized src1
    if (!allocated_src1) {
        GGML_ASSERT((src1_T ? nb11 : nb10) == sizeof(float));
    }

    // data strides in dimensions 2/3
    // const size_t s02 = nb02 / sizeof(float);
    // const size_t s03 = nb03 / sizeof(float);
    // const size_t s12 = nb12 / sizeof(float);
    // const size_t s13 = nb13 / sizeof(float);
    const size_t s02 = allocated_src0 ? (ne00 * ne01) : nb02 / sizeof(float);
    const size_t s03 = allocated_src0 ? (ne00 * ne01 * ne02): nb03 / sizeof(float);
    const size_t s12 = allocated_src1 ? (ne10 * ne11) :  nb12 / sizeof(float);
    const size_t s13 = allocated_src1 ? (ne10 * ne11 * ne12) : nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }

    if (allocated_src0) {
        CUDA_CHECK(cudaFreeAsync(src0_f32, stream));
        // printf("DEBUG: Freed dequantized src0 buffer\n");
    }
    if (allocated_src1) {
        CUDA_CHECK(cudaFreeAsync(src1_f32, stream));
        // // printf("DEBUG: Freed dequantized src1 buffer\n");
    }
    
    // printf("DEBUG: CUDA OUT_PROD completed successfully\n");
    fflush(stdout);
}
