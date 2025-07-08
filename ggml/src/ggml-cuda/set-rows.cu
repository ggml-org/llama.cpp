#include "set-rows.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

template<typename src_t, typename dst_t>
__device__ void set_rows_1(const src_t * src_f, dst_t * dst_f) {
    GGML_ABORT("unsupport type for set_rows");
}

template<>
__device__ __forceinline__ void set_rows_1<float, half>(const float * src_f, half * dst_h) {
    *dst_h = __float2half(*src_f);
}

template<>
__device__ __forceinline__ void set_rows_1<float, float>(const float * src_f, float * dst_f) {
    *dst_f = *src_f;
}

//TODO: consolidate kernels from cpy.cu, get_rows etc to make this function generic
template<typename src_t, typename dst_t>
static __global__ void k_set_rows(
        const src_t * __restrict__ src0, const int64_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size) {

    const int i03 = blockIdx.z / ne02;
    const int i02 = blockIdx.z % ne02;
    const int i01 = blockDim.x * blockIdx.x +  threadIdx.x;
    const int i00 = blockIdx.y;

    if (i01 >= ne01) {
        return;
    }

    const int i12 = i03 % ne12;
    const int i11 = i02 % ne11;
    const int i10 = i01;

    const int64_t dst_row = *(src1 + i10*nb10 + i11*nb11 + i12*nb12);

    const src_t * src0_row = (const src_t *)src0 + i01*nb01 + i02*nb02 + i03*nb03;
    dst_t * dst_row_ptr    = dst + dst_row*nb1 + i02*nb2 + i03*nb3;

    const src_t* src_elem = src0_row + i00;
    dst_t* dst_elem = dst_row_ptr + i00;
    set_rows_1(src_elem, dst_elem);
}

template<typename src_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const int64_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        cudaStream_t stream) {

    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(
        (ne01 + CUDA_SET_ROWS_BLOCK_SIZE - 1)/CUDA_SET_ROWS_BLOCK_SIZE,
        ne00,
        ne03*ne02
    );

    const int s1 = nb01 / sizeof(src_t);
    const int s2 = nb02 / sizeof(src_t);
    const int s3 = nb03 / sizeof(src_t);

    const int s10 = nb10 / sizeof(int64_t);
    const int s11 = nb11 / sizeof(int64_t);
    const int s12 = nb12 / sizeof(int64_t);

    const int s_dst = nb1 / sizeof(dst_t);
    const int s_dst2 = nb2 / sizeof(dst_t);
    const int s_dst3 = nb3 / sizeof(dst_t);


    if(ne01 > 0 && ne00 > 0) {
        k_set_rows<<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            s1, s2, s3,
            s10, s11, s12,
            s_dst, s_dst2, s_dst3,
            src_type_size, dst_type_size);
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64);

    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d   = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            sizeof(float), sizeof(float),
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            sizeof(float), sizeof(half),
            stream
        );
    } else {
        GGML_ABORT("unsupported type");
    }
}
