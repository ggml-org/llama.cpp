#pragma once

#include "common.cuh"

// Blackwell-targeted NVFP4 MMVQ fast path for the single-column decode regime.
// Returns true when the specialized kernel is launched, false when caller should
// use the generic MMVQ path.
bool ggml_cuda_mul_mat_vec_nvfp4_mma(
        const void * vx, const void * vy, float * dst, float alpha,
        int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int nsamples_x, int nsamples_dst, int stride_sample_x, int stride_sample_y, int stride_sample_dst,
        cudaStream_t stream);
