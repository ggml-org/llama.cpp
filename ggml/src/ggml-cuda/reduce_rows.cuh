#include "common.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template<bool norm>
static __global__ void reduce_rows_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}