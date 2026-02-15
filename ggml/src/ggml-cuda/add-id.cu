#include "add-id.cuh"

static __global__ void add_id_kernel_reference(
        const float * src0, const float * src1, const int32_t * src2, float * dst,
        int64_t ne0, int64_t ne1,
        size_t nb01, size_t nb02,
        size_t nb11,
        size_t nb21
    ) {
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;

    const int i11 = *(const int32_t *) ((const char *) src2 + i1*sizeof(int32_t) + i2*nb21);

    const size_t nb1 = ne0 * sizeof(float);
    const size_t nb2 = ne1 * nb1;

    float * dst_row = (float *)((char *)dst + i1*nb1 + i2*nb2);
    const float * src0_row = (const float *)((const char *)src0 +  i1*nb01 + i2*nb02);
    const float * src1_row = (const float *)((const char *)src1 + i11*nb11);

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

static __global__ void add_id_kernel_vec4(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const int32_t * __restrict__ src2,
        float * __restrict__ dst,
        const int ne0,
        const int ne01,
        const int s0_stride,
        const int s0_stride2,
        const int s1_stride,
        const int s2_stride
    ) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;

    const int i11 = src2[i1 + i2 * s2_stride];

    const int src0_offset = i1 * s0_stride + i2 * s0_stride2;
    const int src1_offset = i11 * s1_stride;
    const int dst_offset = i1 * ne0 + i2 * ne01 * ne0;

    const float4 * __restrict__ src0_vec = reinterpret_cast<const float4 *>(src0 + src0_offset);
    const float4 * __restrict__ src1_vec = reinterpret_cast<const float4 *>(src1 + src1_offset);
    float4 * __restrict__ dst_vec = reinterpret_cast<float4 *>(dst + dst_offset);

    const int ne0_vec = ne0 >> 2;

    for (int i0 = threadIdx.x; i0 < ne0_vec; i0 += blockDim.x) {
        const float4 a = src0_vec[i0];
        const float4 b = src1_vec[i0];
        dst_vec[i0] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

static __global__ void add_id_kernel_contiguous(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const int32_t * __restrict__ src2,
        float * __restrict__ dst,
        const int ne0,
        const int ne01,
        const int s0_stride,
        const int s0_stride2,
        const int s1_stride,
        const int s2_stride
    ) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;

    const int i11 = src2[i1 + i2 * s2_stride];

    const float * __restrict__ src0_row = src0 + i1 * s0_stride + i2 * s0_stride2;
    const float * __restrict__ src1_row = src1 + i11 * s1_stride;
    float * __restrict__ dst_row = dst + i1 * ne0 + i2 * ne01 * ne0;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const int32_t * src2_d = (const int32_t *)src2->data;
    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    const bool is_contiguous = (nb01 == ne00 * sizeof(float)) &&
                               (nb11 == ne10 * sizeof(float));

    const bool is_aligned = ((uintptr_t)src0_d % 16 == 0) &&
                            ((uintptr_t)src1_d % 16 == 0) &&
                            ((uintptr_t)dst_d  % 16 == 0);

    const bool can_vectorize = is_contiguous && is_aligned && (ne00 % 4 == 0) && (ne00 <= INT_MAX);

    const dim3 blocks(ne01, ne02);

    if (can_vectorize) {
        const int threads_vec4 = std::min((int)(ne00 / 4), 768);
        add_id_kernel_vec4<<<blocks, threads_vec4, 0, stream>>>(
            src0_d, src1_d, src2_d, dst_d,
            (int)ne00,
            (int)ne01,
            (int)(nb01 / sizeof(float)),
            (int)(nb02 / sizeof(float)),
            (int)(nb11 / sizeof(float)),
            (int)(nb21 / sizeof(int32_t))
        );
    } else if (is_contiguous && ne00 <= INT_MAX) {
        const int threads = std::min((int)ne00, 768);
        add_id_kernel_contiguous<<<blocks, threads, 0, stream>>>(
            src0_d, src1_d, src2_d, dst_d,
            (int)ne00,
            (int)ne01,
            (int)(nb01 / sizeof(float)),
            (int)(nb02 / sizeof(float)),
            (int)(nb11 / sizeof(float)),
            (int)(nb21 / sizeof(int32_t))
        );
    } else {
        const int threads = std::min((int)ne00, 768);
        add_id_kernel_reference<<<blocks, threads, 0, stream>>>(
            src0_d, src1_d, src2_d, dst_d,
            ne0, ne1,
            nb01, nb02,
            nb11,
            nb21
        );
    }
}
