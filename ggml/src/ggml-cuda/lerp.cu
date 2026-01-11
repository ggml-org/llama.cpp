#include "lerp.cuh"
#include <cstdint>

template <typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static __global__ void k_lerp(
    const src0_t * src0,
    const src1_t * src1,
    const src2_t * src2,
    dst_t * dst,
    const int ne0,
    const int ne1,
    const int ne2,
    const uint3 ne3,
    const uint3 ne03,
    const uint3 ne13,
    const uint3 ne21,
    const uint3 ne22,
    const int s1,
    const int s2,
    const int s3,
    const int s01,
    const int s02,
    const int s03,
    const int s11,
    const int s12,
    const int s13,
    const int s21,
    const int s22,
    const int s23) {

    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= (uint32_t)ne0 || i1 >= (uint32_t)ne1 || i2 >= (uint32_t)ne2 || i3 >= ne3.z) {
        return;
    }

    // src0/src1 broadcast in dim3
    const uint32_t i03 = fastmodulo(i3, ne03);
    const uint32_t i13 = fastmodulo(i3, ne13);

    // src2 broadcast in dim1, dim2
    const uint32_t i21 = fastmodulo(i1, ne21);
    const uint32_t i22 = fastmodulo(i2, ne22);

    const size_t i_src0 = i03*s03 + i2*s02  + i1*s01;
    const size_t i_src1 = i13*s13 + i2*s12  + i1*s11;
    const size_t i_src2 = i3*s23  + i22*s22 + i21*s21;
    const size_t i_dst  = i3*s3   + i2*s2   + i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    const src2_t * src2_row = src2 + i_src2;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const float v0 = (float) src0_row[i0];
        const float v1 = (float) src1_row[i0];
        const float v2 = (float) src2_row[i0];

        dst_row[i0] = (dst_t) (v0 + (v1 - v0) * v2);
    }
}

template <typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static __global__ void k_lerp_unravel(
    const src0_t * src0,
    const src1_t * src1,
    const src2_t * src2,
    dst_t * dst,
    const uint3 ne0,
    const uint3 ne1,
    const uint3 ne2,
    const uint32_t ne3,
    const uint3 prod_012,
    const uint3 prod_01,
    const uint3 ne03,
    const uint3 ne13,
    const uint3 ne21,
    const uint3 ne22,
    const int s1,
    const int s2,
    const int s3,
    const int s01,
    const int s02,
    const int s03,
    const int s11,
    const int s12,
    const int s13,
    const int s21,
    const int s22,
    const int s23) {

    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    // src0/src1 broadcast in dim3
    const int i03 = fastmodulo(i3, ne03);
    const int i13 = fastmodulo(i3, ne13);

    // src2 broadcast in dim1, dim2
    const int i21 = fastmodulo(i1, ne21);
    const int i22 = fastmodulo(i2, ne22);

    const size_t i_src0 = i03*s03 + i2*s02  + i1*s01;
    const size_t i_src1 = i13*s13 + i2*s12  + i1*s11;
    const size_t i_src2 = i3*s23  + i22*s22 + i21*s21;
    const size_t i_dst  = i3*s3   + i2*s2   + i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    const src2_t * src2_row = src2 + i_src2;
    dst_t * dst_row = dst + i_dst;

    const float v0 = (float) src0_row[i0];
    const float v1 = (float) src1_row[i0];
    const float v2 = (float) src2_row[i0];

    // dst = src0 + (src1 - src0) * src2
    dst_row[i0] = (dst_t) (v0 + (v1 - v0) * v2);
}

template <typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void launch_lerp(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * src2,
    ggml_tensor * dst,
    const src0_t * src0_dd,
    const src1_t * src1_dd,
    const src2_t * src2_dd,
    dst_t * dst_dd,
    cudaStream_t stream) {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(int64_t, ne2, src2, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)
    GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)
    GGML_TENSOR_LOCALS(size_t, nb2, src2, nb)
    GGML_TENSOR_LOCALS(size_t, nb,  dst,  nb)

    GGML_ASSERT(ne00 == ne10 && ne00 == ne20 && ne00 == ne0);
    GGML_ASSERT(ne01 % ne21 == 0);
    GGML_ASSERT(ne02 % ne22 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);
    GGML_ASSERT(ne3 % ne13 == 0);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne23);

    size_t s1 = nb1 / sizeof(dst_t);
    size_t s2 = nb2 / sizeof(dst_t);
    size_t s3 = nb3 / sizeof(dst_t);

    size_t s01 = nb01 / sizeof(src0_t);
    size_t s02 = nb02 / sizeof(src0_t);
    size_t s03 = nb03 / sizeof(src0_t);

    size_t s11 = nb11 / sizeof(src1_t);
    size_t s12 = nb12 / sizeof(src1_t);
    size_t s13 = nb13 / sizeof(src1_t);

    size_t s21 = nb21 / sizeof(src2_t);
    size_t s22 = nb22 / sizeof(src2_t);
    size_t s23 = nb23 / sizeof(src2_t);

    GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
    GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
    GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
    GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

    GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
    GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
    GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
    GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

    GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
    GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
    GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
    GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

    GGML_ASSERT(nb20 % sizeof(src2_t) == 0);
    GGML_ASSERT(nb21 % sizeof(src2_t) == 0);
    GGML_ASSERT(nb22 % sizeof(src2_t) == 0);
    GGML_ASSERT(nb23 % sizeof(src2_t) == 0);

    const int block_size = CUDA_LERP_BLOCK_SIZE;

    int64_t hne0 = std::max(ne0 / 2LL, 1LL);

    dim3 block_dims;
    block_dims.x = std::min<unsigned int>(hne0, block_size);
    block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
    block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

    dim3 block_nums(
        (hne0 + block_dims.x - 1) / block_dims.x,
        (ne1 + block_dims.y - 1) / block_dims.y,
        (ne2 * ne3 + block_dims.z - 1) / block_dims.z);

    const uint3 ne03_fastdiv = init_fastdiv_values((uint32_t) ne03);
    const uint3 ne13_fastdiv = init_fastdiv_values((uint32_t) ne13);
    const uint3 ne21_fastdiv = init_fastdiv_values((uint32_t) ne21);
    const uint3 ne22_fastdiv = init_fastdiv_values((uint32_t) ne22);

    if (block_nums.z > 65535 || block_nums.y > 65535) {
        int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
        const uint3 prod_012 = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
        const uint3 prod_01  = init_fastdiv_values((uint32_t) (ne0 * ne1));
        const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
        const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
        const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

        k_lerp_unravel<src0_t, src1_t, src2_t, dst_t>
            <<<block_num, block_size, 0, stream>>>(
                src0_dd, src1_dd, src2_dd, dst_dd,
                ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3,
                prod_012, prod_01,
                ne03_fastdiv, ne13_fastdiv, ne21_fastdiv, ne22_fastdiv,
                s1, s2, s3,
                s01, s02, s03,
                s11, s12, s13,
                s21, s22, s23);
    } else {
        const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);

        k_lerp<src0_t, src1_t, src2_t, dst_t>
            <<<block_nums, block_dims, 0, stream>>>(
                src0_dd, src1_dd, src2_dd, dst_dd,
                ne0, ne1, ne2, ne3_fastdiv,
                ne03_fastdiv, ne13_fastdiv, ne21_fastdiv, ne22_fastdiv,
                s1, s2, s3,
                s01, s02, s03,
                s11, s12, s13,
                s21, s22, s23);
    }
}

void ggml_cuda_op_lerp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    if (src2->type == GGML_TYPE_F32) {
        launch_lerp<float, float, float, float>(
            src0, src1, src2, dst,
            (const float *) src0->data,
            (const float *) src1->data,
            (const float *) src2->data,
            (float *) dst->data,
            stream);
    } else if (src2->type == GGML_TYPE_F16) {
        launch_lerp<float, float, half, float>(
            src0, src1, src2, dst,
            (const float *) src0->data,
            (const float *) src1->data,
            (const half *) src2->data,
            (float *) dst->data,
            stream);
    } else {
        GGML_ABORT("fatal error");
    }
}
