#include "common.cuh"

static __device__ __forceinline__ float reduce_row_f32(const float * x, const int ncols) {
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    ggml_cuda_pdl_sync();
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[i];
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // sum up partial sums
    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    return sum;
}

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm>
static __global__ void reduce_rows_f32(const float * x_ptr, float * dst_ptr, const int ncols) {
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int64_t row = blockIdx.x;
    const int col = threadIdx.x;

    const float * GGML_CUDA_RESTRICT x = x_ptr + row*ncols;
    const float sum = reduce_row_f32(x, ncols);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}

template <bool norm>
static __global__ void reduce_rows_f32_strided(const char * x_ptr, float * dst_ptr, const int ncols,
        const int64_t ne1, const int64_t ne2, const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int64_t row = blockIdx.x;
    const int col = threadIdx.x;

    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / (ne1 * ne2);

    const float * GGML_CUDA_RESTRICT x = (const float *) (x_ptr + i1*nb1 + i2*nb2 + i3*nb3);
    const float sum = reduce_row_f32(x, ncols);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}
