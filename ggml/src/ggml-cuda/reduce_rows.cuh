#include "common.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template<bool norm>
static __global__ void reduce_rows_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    const int num_unroll = 24;
    float temp[num_unroll];
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j){
            if (i < ncols){
                temp[j] = x[row * ncols + i];
            }
            else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j){
            sum += temp[j];
        }
    }

    sum = warp_reduce_sum(sum);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}