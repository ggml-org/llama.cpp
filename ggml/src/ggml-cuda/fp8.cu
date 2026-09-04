#include "fp8.cuh"

static __global__ void mul_mat_fp8_fallback(
        const char * src0, const char * src1, char * dst, int64_t ne00, int64_t ne01, int64_t ne11,
        int64_t ne12, int64_t ne13, int64_t r2, int64_t r3, int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t nb11, int64_t nb12, int64_t nb13, int64_t nb1, int64_t nb2, int64_t nb3, int64_t ne_dst) {
    for (int64_t id = blockIdx.x; id < ne_dst; id += gridDim.x) {
        int64_t tmp = id / ne01;
        const int64_t i0 = id - tmp*ne01;
        const int64_t i1 = tmp % ne11;
        tmp /= ne11;
        const int64_t i2 = tmp % ne12;
        const int64_t i3 = tmp / ne12;

        const ggml_fp8_e4m3_t * x = (const ggml_fp8_e4m3_t *) (src0 + i0*nb01 + (i2/r2)*nb02 + (i3/r3)*nb03);
        const float * y = (const float *) (src1 + i1*nb11 + i2*nb12 + i3*nb13);
        float sum = 0.0f;
        for (int64_t k = threadIdx.x; k < ne00; k += blockDim.x) {
            sum = fmaf(ggml_cuda_f8_e4m3_to_fp32(x[k].bits), y[k], sum);
        }

        __shared__ float shared[WARP_SIZE];
        sum = block_reduce<block_reduce_method::SUM, 256>(sum, shared);
        if (threadIdx.x == 0) {
            *(float *) (dst + i0*sizeof(float) + i1*nb1 + i2*nb2 + i3*nb3) = sum;
        }
    }
}

void ggml_cuda_mul_mat_fp8_fallback(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(ggml_fp8_e4m3_t));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];
    const int64_t ne_dst = ggml_nelements(dst);
    const int blocks = std::min<int64_t>(ne_dst, 65535);
    mul_mat_fp8_fallback<<<blocks, 256, 0, ctx.stream()>>>(
        (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
        src0->ne[0], src0->ne[1], src1->ne[1], src1->ne[2], src1->ne[3], r2, r3,
        src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3], ne_dst);
    CUDA_CHECK(cudaGetLastError());
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080

struct fp8_abs_src {
    const float * x;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t s1;
    int64_t s2;
    int64_t s3;

    __device__ float operator()(int64_t i) const {
        const int64_t i0 = i % ne0;
        i /= ne0;
        const int64_t i1 = i % ne1;
        i /= ne1;
        const int64_t i2 = i % ne2;
        const int64_t i3 = i / ne2;
        const float value = fabsf(x[i0 + i1*s1 + i2*s2 + i3*s3]);
        return isfinite(value) ? value : 448.0f;
    }
};

static __global__ void fp8_amax_partials(fp8_abs_src src, int64_t ne, float * partials) {
    float amax = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < ne; i += (int64_t) blockDim.x*gridDim.x) {
        amax = fmaxf(amax, src(i));
    }

    __shared__ float shared[WARP_SIZE];
    amax = block_reduce<block_reduce_method::MAX, 256>(amax, shared);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = amax;
    }
}

static __global__ void fp8_amax_final(const float * partials, int n, float * amax) {
    float value = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        value = fmaxf(value, partials[i]);
    }

    __shared__ float shared[WARP_SIZE];
    value = block_reduce<block_reduce_method::MAX, 256>(value, shared);
    if (threadIdx.x == 0) {
        *amax = value;
    }
}

static __global__ void quantize_fp8_e4m3(
        const float * __restrict__ x, uint8_t * __restrict__ y, const float * __restrict__ amax,
        float * __restrict__ scale, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne, int64_t s1, int64_t s2, int64_t s3) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }

    int64_t tmp = i / ne0;
    const int64_t i0 = i - tmp*ne0;
    const int64_t i1 = tmp % ne1;
    tmp /= ne1;
    const int64_t i2 = tmp % ne2;
    const int64_t i3 = tmp / ne2;

    const float d = *amax > 0.0f ? *amax / 448.0f : 1.0f;
    const __nv_fp8_e4m3 q(x[i0 + i1*s1 + i2*s2 + i3*s3] / d);
    y[i] = q.__x;
    if (i == 0) {
        *scale = d;
    }
}

static void fp8_destroy_matmul(
        cublasLtMatmulDesc_t op_desc, cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
        cublasLtMatrixLayout_t d_desc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!fp8_mma_hardware_available(cc) || src0->type != GGML_TYPE_F8_E4M3 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || src0->ne[0] % 16 != 0 || src0->ne[1] % 16 != 0 ||
            src0->nb[0] != sizeof(uint8_t) || src0->nb[1] != (size_t) src0->ne[0] || src1->nb[0] != sizeof(float)) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    cudaStream_t stream = ctx.stream();
    const int64_t ne_src1 = ggml_nelements(src1);
    ggml_cuda_pool_alloc<uint8_t> src1_fp8(ctx.pool(), ne_src1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool(), 1);
    ggml_cuda_pool_alloc<float> src1_amax(ctx.pool(), 1);

    const fp8_abs_src abs_src = {
        (const float *) src1->data, ne10, ne11, ne12,
        (int64_t) (nb11 / sizeof(float)), (int64_t) (nb12 / sizeof(float)), (int64_t) (nb13 / sizeof(float))
    };
    const int reduce_blocks = std::min<int64_t>((ne_src1 + 255)/256, 1024);
    ggml_cuda_pool_alloc<float> reduce_tmp(ctx.pool(), reduce_blocks);
    fp8_amax_partials<<<reduce_blocks, 256, 0, stream>>>(abs_src, ne_src1, reduce_tmp.ptr);
    fp8_amax_final<<<1, 256, 0, stream>>>(reduce_tmp.ptr, reduce_blocks, src1_amax.ptr);

    quantize_fp8_e4m3<<<(ne_src1 + 255)/256, 256, 0, stream>>>(
        (const float *) src1->data, src1_fp8.ptr, src1_amax.ptr, src1_scale.ptr, ne10, ne11, ne12, ne_src1,
        nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t d_desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &src1_scale.ptr, sizeof(src1_scale.ptr)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, ne00, ne01, ne00));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, ne10, ne11, ne10));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, ne0, ne1, ne0));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ctx.cublaslt_handle(), op_desc, a_desc, b_desc, d_desc, d_desc, preference, 1, &heuristic, &returned);
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
        fp8_destroy_matmul(op_desc, a_desc, b_desc, d_desc);
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    for (int64_t i3 = 0; i3 < ne13; ++i3) {
        for (int64_t i2 = 0; i2 < ne12; ++i2) {
            const char * a = (const char *) src0->data + (i2/r2)*nb02 + (i3/r3)*nb03;
            const uint8_t * b = src1_fp8.ptr + (i3*ne12 + i2)*ne11*ne10;
            float * d = (float *) ((char *) dst->data + i2*dst->nb[2] + i3*dst->nb[3]);
            CUBLAS_CHECK(cublasLtMatmul(ctx.cublaslt_handle(), op_desc, &alpha, a, a_desc, b, b_desc,
                &beta, d, d_desc, d, d_desc, &heuristic.algo, nullptr, 0, stream));
        }
    }

    fp8_destroy_matmul(op_desc, a_desc, b_desc, d_desc);
    return true;
}

#else

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif
