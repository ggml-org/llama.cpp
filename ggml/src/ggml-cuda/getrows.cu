#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

__constant__ static const float tq3_0_centroids_getrows_cuda[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};
__constant__ static const float tq3_0_signs_getrows_cuda[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const int i11 =  z / ne12; // TODO fastdiv
            const int i12 =  z % ne12;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const int i11 = z / ne12; // TODO fastdiv
            const int i12 = z % ne12;

            if (i00 >= ne00) {
                return;
            }

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst, const int64_t ncols, const int64_t nrows_grad) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int dst_row = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i*ncols + col];
    }

    dst[dst_row*ncols + col] = sum;
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename dst_t>
static __global__ void k_get_rows_tq3_0(
        const block_tq3_0 * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int lane = threadIdx.x;
    const int64_t block = blockIdx.y;
    const int64_t z = blockIdx.z;

    if (lane >= QK_TQ3_0 || z >= ne11*ne12) {
        return;
    }

    const int64_t nblocks_per_row = ne00 / QK_TQ3_0;
    if (block >= nblocks_per_row) {
        return;
    }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_tq3_0 * x = (const block_tq3_0 *) src0_row + block;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int g = lane / 8;
    const int r = lane % 8;
    const uint8_t * qp = x->qs + g * 3;
    uint8_t idx;
    switch (r) {
        case 0: idx =  qp[0]       & 7; break;
        case 1: idx = (qp[0] >> 3) & 7; break;
        case 2: idx = ((qp[0] >> 6) | (qp[1] << 2)) & 7; break;
        case 3: idx = (qp[1] >> 1) & 7; break;
        case 4: idx = (qp[1] >> 4) & 7; break;
        case 5: idx = ((qp[1] >> 7) | (qp[2] << 1)) & 7; break;
        case 6: idx = (qp[2] >> 2) & 7; break;
        default: idx = (qp[2] >> 5) & 7; break;
    }

    float val = tq3_0_centroids_getrows_cuda[idx];
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    const float d = __half2float(x->d);
    const float out = val * (d / sqrtf(32.0f)) * tq3_0_signs_getrows_cuda[lane];
    dst_row[block * QK_TQ3_0 + lane] = ggml_cuda_cast<dst_t>(out);
}

template<typename dst_t>
static void get_rows_cuda_tq3_0(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ3_0 == 0);

    const dim3 block_dims(QK_TQ3_0, 1, 1);
    const dim3 block_nums(ne10, ne00 / QK_TQ3_0, MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq3_0<<<block_nums, block_dims, 0, stream>>>(
        (const block_tq3_0 *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        s1, s2, s3,
        nb01, nb02, nb03,
        s10, s11, s12);
}

template<typename dst_t>
static __global__ void k_get_rows_tq3_1s(
        const block_tq3_1s * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int lane = threadIdx.x;
    const int64_t block = blockIdx.y;
    const int64_t z = blockIdx.z;

    if (lane >= QK_TQ3_0 || z >= ne11*ne12) {
        return;
    }

    const int64_t nblocks_per_row = ne00 / QK_TQ3_0;
    if (block >= nblocks_per_row) {
        return;
    }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_tq3_1s * x = (const block_tq3_1s *) src0_row + block;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int g = lane / 8;
    const int r = lane % 8;
    const uint8_t * qp = x->qs + g * 3;
    uint8_t idx;
    switch (r) {
        case 0: idx =  qp[0]       & 7; break;
        case 1: idx = (qp[0] >> 3) & 7; break;
        case 2: idx = ((qp[0] >> 6) | (qp[1] << 2)) & 7; break;
        case 3: idx = (qp[1] >> 1) & 7; break;
        case 4: idx = (qp[1] >> 4) & 7; break;
        case 5: idx = ((qp[1] >> 7) | (qp[2] << 1)) & 7; break;
        case 6: idx = (qp[2] >> 2) & 7; break;
        default: idx = (qp[2] >> 5) & 7; break;
    }

    const float d = lane < 16 ? __half2float(x->d0) : __half2float(x->d1);
    float val = tq3_0_centroids_getrows_cuda[idx] * d;
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    dst_row[block * QK_TQ3_0 + lane] = ggml_cuda_cast<dst_t>(val * (tq3_0_signs_getrows_cuda[lane] / sqrtf(32.0f)));
}

template<typename dst_t>
static void get_rows_cuda_tq3_1s(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ3_0 == 0);

    const dim3 block_dims(QK_TQ3_0, 1, 1);
    const dim3 block_nums(ne10, ne00 / QK_TQ3_0, MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq3_1s<<<block_nums, block_dims, 0, stream>>>(
        (const block_tq3_1s *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        s1, s2, s3,
        nb01, nb02, nb03,
        s10, s11, s12);
}

static __device__ __forceinline__ uint8_t tq3_idx_from_packed_getrows_cuda(const uint8_t * qp, int r) {
    switch (r) {
        case 0: return  qp[0]       & 7;
        case 1: return (qp[0] >> 3) & 7;
        case 2: return ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        case 3: return (qp[1] >> 1) & 7;
        case 4: return (qp[1] >> 4) & 7;
        case 5: return ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        case 6: return (qp[2] >> 2) & 7;
        default: return (qp[2] >> 5) & 7;
    }
}

template<typename dst_t>
static __global__ void k_get_rows_tq3_1s_ap1(
        const block_tq3_1s_ap1 * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int lane = threadIdx.x;
    const int64_t block = blockIdx.y;
    const int64_t z = blockIdx.z;

    if (lane >= QK_TQ3_0 || z >= ne11*ne12) {
        return;
    }

    const int64_t nblocks_per_row = ne00 / QK_TQ3_0;
    if (block >= nblocks_per_row) {
        return;
    }

    const int i10 = blockIdx.x;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const char * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_tq3_1s_ap1 * superblocks = (const block_tq3_1s_ap1 *) src0_row;
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int64_t super_idx = block / 16;
    const int slot = block % 16;
    const block_tq3_1s_ap1 * sb = superblocks + super_idx;
    const uint16_t mask = sb->mask;
    const int promoted_slot = __ffs((int) mask) - 1;
    const bool is_promoted = promoted_slot == slot;
    const uint8_t * base_region = sb->qs;
    const uint8_t * promo_region = sb->qs + 15 * sizeof(block_tq3_1s);
    const int base_slot = slot - (slot > promoted_slot ? 1 : 0);
    const uint8_t * base = base_region + base_slot * sizeof(block_tq3_1s);

    const int g = lane / 8;
    const int r = lane % 8;
    float val;

    if (!is_promoted) {
        const block_tq3_1s * x = (const block_tq3_1s *) base;
        const uint8_t * qp = x->qs + g * 3;
        const uint8_t idx = tq3_idx_from_packed_getrows_cuda(qp, r);
        const float d = lane < 16 ? __half2float(x->d0) : __half2float(x->d1);
        val = tq3_0_centroids_getrows_cuda[idx] * d;
        for (int step = 1; step < 32; step <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
            val = (lane & step) ? (other - val) : (other + val);
        }
        dst_row[block * QK_TQ3_0 + lane] = ggml_cuda_cast<dst_t>(val * (tq3_0_signs_getrows_cuda[lane] / sqrtf(32.0f)));
        return;
    }

    const block_tq3_1s_shift * x = (const block_tq3_1s_shift *) promo_region;
    const uint8_t * qp = x->qs + g * 3;
    const uint8_t idx = tq3_idx_from_packed_getrows_cuda(qp, r);
    const float d = lane < 16 ? __half2float(x->d0) : __half2float(x->d1);
    val = tq3_0_centroids_getrows_cuda[idx] * d + __half2float(x->m);
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }
    dst_row[block * QK_TQ3_0 + lane] = ggml_cuda_cast<dst_t>(val * (tq3_0_signs_getrows_cuda[lane] / sqrtf(32.0f)));
}

template<typename dst_t>
static void get_rows_cuda_tq3_1s_ap1(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ3_1S_AP1 == 0);

    const dim3 block_dims(QK_TQ3_0, 1, 1);
    const dim3 block_nums(ne10, ne00 / QK_TQ3_0, MIN(ne11*ne12, (int64_t) UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq3_1s_ap1<<<block_nums, block_dims, 0, stream>>>(
        (const block_tq3_1s_ap1 *) src0_d, src1_d, dst_d,
        ne00, ne11, ne12,
        s1, s2, s3,
        nb01, nb02, nb03,
        s10, s11, s12);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQ3_0:
            get_rows_cuda_tq3_0(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQ3_1S:
            get_rows_cuda_tq3_1s(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            // TODO: k-quants
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne1, 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10);
}
