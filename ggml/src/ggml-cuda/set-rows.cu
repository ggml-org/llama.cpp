#include "set-rows.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

static __device__ void set_rows_1_f32_f32(const char * src, char * dst) {
    const float * src_f = (const float *) src;
    float * dst_f = (float *) dst;
    *dst_f = *src_f;
}

static __device__ void set_rows_1_f32_f16(const char * src, char * dst) {
    const float * src_f = (const float *) src;
    half * dst_h = (half *) dst;
    *dst_h = __float2half(*src_f);
}

//TODO: consolidate kernels from cpy.cu, get_rows etc to make this function generic
template<set_rows_kernel_t set_rows_1>
static __global__ void k_set_rows(
        const char * __restrict__ src0, const int64_t * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size) {

    const int i03 = blockIdx.z;
    const int i02 = blockIdx.y;
    const int i01 = blockIdx.x * blockDim.y + threadIdx.y;  // Row index

    if (i01 >= ne01) {
        return;
    }

    const int i12 = i03 % ne12;
    const int i11 = i02 % ne11;
    const int i10 = i01;

    const int64_t dst_row = *(int64_t *)((char *)src1 + i10*nb10 + i11*nb11 + i12*nb12);

    const char * src0_row = src0 + i01*nb01 + i02*nb02 + i03*nb03;
    char * dst_row_ptr    = dst + dst_row*nb1 + i02*nb2 + i03*nb3;

    for (int col = threadIdx.x; col < ne00; col += blockDim.x) {
        const char * src_elem = src0_row + col * src_type_size;
        char * dst_elem       = dst_row_ptr + col * dst_type_size;
        set_rows_1(src_elem, dst_elem);
    }
}

template<set_rows_kernel_t set_rows_1>
static void set_rows_cuda(
        const char * src0_d, const int64_t * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        cudaStream_t stream) {

    const int max_threads_per_row = 256;
    const int threads_per_row     = std::min((int)ne00, max_threads_per_row);

    const int max_threads_per_block = 256;
    const int rows_per_block        = std::max(1, max_threads_per_block / threads_per_row);

    const dim3 block_size(threads_per_row, rows_per_block, 1);
    const dim3 grid_size(
        (ne01 + rows_per_block - 1) / rows_per_block, // thread-groups
        ne02,
        ne03
    );

    if (ne01 > 0 && ne00 > 0) {
        k_set_rows<set_rows_1><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            src_type_size, dst_type_size
        );
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
        set_rows_cuda<set_rows_1_f32_f32>(
            (const char *)src0_d, src1_d, (char *)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            sizeof(float), sizeof(float),
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda<set_rows_1_f32_f16>(
            (const char *)src0_d, src1_d, (char *)dst->data,
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
