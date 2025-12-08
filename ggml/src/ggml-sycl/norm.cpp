#include "norm.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include <vector>
#include <cmath>

static void norm_f32(const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, int block_size) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int nthreads = item_ct1.get_local_range(2);
    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto strided_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto packed_offset = calculate_offset<3>({nchannels * nrows * ncols, nrows * ncols, ncols}, {sample, channel, row});

    x += strided_offset;
    dst += packed_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    if  (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        for (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    const float mean = mean_var.x() / ncols;
    const float var = mean_var.y() / ncols - mean * mean;
    const float inv_std = sycl::rsqrt(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

static void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float variance = tmp / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

static void rms_norm_f32(const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto strided_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto packed_offset = calculate_offset<3>({nchannels * nrows * ncols, nrows * ncols, ncols}, {sample, channel, row});

    x   += strided_offset;
    dst += packed_offset;


    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

// Fused ADD + RMS norm kernel with SLM caching
// Combines: add_dst = x + add, dst = RMSNorm(add_dst)
// Writes BOTH the ADD result (for other consumers) AND the RMS_NORM result
// Uses SLM to cache (x + add) to avoid double global memory read
static void add_rms_norm_f32_slm_cached(const float* x, const float* add, float* add_dst, float* dst,
        const int ncols, const int64_t stride_x_row, const int64_t stride_x_channel, const int64_t stride_x_sample,
        const int64_t stride_add_row, const int64_t stride_add_channel, const int64_t stride_add_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t add_dst_stride_row, const int64_t add_dst_stride_channel, const int64_t add_dst_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, float* s_x, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_x_sample, stride_x_channel, stride_x_row}, {sample, channel, row});
    const auto add_dst_offset = calculate_offset<3>({add_dst_stride_sample, add_dst_stride_channel, add_dst_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x       += src_offset;
    add_dst += add_dst_offset;
    dst     += dst_offset;

    // Calculate add pointer offset with broadcasting support
    const int add_row     = row % add_nrows;
    const int add_channel = channel % add_nchannels;
    const int add_sample  = sample % add_nsamples;
    const float* add_row_ptr = add + add_sample * stride_add_sample + add_channel * stride_add_channel + add_row * stride_add_row;

    float tmp = 0.0f; // partial sum for thread in warp

    // First pass: compute (x + add), store in SLM AND add_dst, compute sum of squares
    for (int col = tid; col < ncols; col += block_size) {
        const int add_col = col % add_ncols;
        const float val = x[col] + add_row_ptr[add_col];
        s_x[col] = val;       // Cache summed value in SLM for RMS norm
        add_dst[col] = val;   // Write ADD result for other consumers
        tmp += val * val;
    }

    // Ensure all threads have written to SLM before reduction
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Second pass: read from SLM and write RMS_NORM result
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * s_x[col];
    }
}

// Non-SLM version for large ncols that don't fit in SLM
// Writes BOTH the ADD result (for other consumers) AND the RMS_NORM result
static void add_rms_norm_f32(const float* x, const float* add, float* add_dst, float* dst,
        const int ncols, const int64_t stride_x_row, const int64_t stride_x_channel, const int64_t stride_x_sample,
        const int64_t stride_add_row, const int64_t stride_add_channel, const int64_t stride_add_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t add_dst_stride_row, const int64_t add_dst_stride_channel, const int64_t add_dst_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_x_sample, stride_x_channel, stride_x_row}, {sample, channel, row});
    const auto add_dst_offset = calculate_offset<3>({add_dst_stride_sample, add_dst_stride_channel, add_dst_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x       += src_offset;
    add_dst += add_dst_offset;
    dst     += dst_offset;

    // Calculate add pointer offset with broadcasting support
    const int add_row     = row % add_nrows;
    const int add_channel = channel % add_nchannels;
    const int add_sample  = sample % add_nsamples;
    const float* add_row_ptr = add + add_sample * stride_add_sample + add_channel * stride_add_channel + add_row * stride_add_row;

    float tmp = 0.0f; // partial sum for thread in warp

    // First pass: compute (x + add), write to add_dst, compute sum of squares
    for (int col = tid; col < ncols; col += block_size) {
        const int add_col = col % add_ncols;
        const float val = x[col] + add_row_ptr[add_col];
        add_dst[col] = val;   // Write ADD result for other consumers
        tmp += val * val;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Second pass: read from add_dst and apply scale
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * add_dst[col];
    }
}

// Fused RMS norm + element-wise multiply kernel
// Combines: dst = scale * x * mul where scale = rsqrt(mean(x^2) + eps)
static void rms_norm_mul_f32(const float* x, const float* mul, float* dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    // Calculate mul pointer offset with broadcasting support
    const int mul_row     = row % mul_nrows;
    const int mul_channel = channel % mul_nchannels;
    const int mul_sample  = sample % mul_nsamples;
    const float* mul_row_ptr = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Fused output: scale * x * mul
    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = col % mul_ncols;
        dst[col] = scale * x[col] * mul_row_ptr[mul_col];
    }
}

// SLM-cached fused RMS norm + element-wise multiply kernel
// Caches input row in shared local memory to avoid double global memory read
// Combines: dst = scale * x * mul where scale = rsqrt(mean(x^2) + eps)
static void rms_norm_mul_f32_slm_cached(const float* x, const float* mul, float* dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, float* s_x, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    // Calculate mul pointer offset with broadcasting support
    const int mul_row     = row % mul_nrows;
    const int mul_channel = channel % mul_nchannels;
    const int mul_sample  = sample % mul_nsamples;
    const float* mul_row_ptr = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    // First pass: load x into SLM and compute sum of squares
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        s_x[col] = xi;  // Cache in SLM
        tmp += xi * xi;
    }

    // Ensure all threads have written to SLM before reduction
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Second pass: read from SLM instead of global memory
    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = col % mul_ncols;
        dst[col] = scale * s_x[col] * mul_row_ptr[mul_col];
    }
}

// Fused RMS norm + element-wise multiply + add kernel
// Combines: dst = scale * x * mul + add where scale = rsqrt(mean(x^2) + eps)
static void rms_norm_mul_add_f32(const float* x, const float* mul, const float* add, float* dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    // Calculate mul pointer offset with broadcasting support
    const int mul_row     = row % mul_nrows;
    const int mul_channel = channel % mul_nchannels;
    const int mul_sample  = sample % mul_nsamples;
    const float* mul_row_ptr = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    // Calculate add pointer offset with broadcasting support
    const int add_row     = row % add_nrows;
    const int add_channel = channel % add_nchannels;
    const int add_sample  = sample % add_nsamples;
    const float* add_row_ptr = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Fused output: scale * x * mul + add
    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = col % mul_ncols;
        const int add_col = col % add_ncols;
        dst[col] = scale * x[col] * mul_row_ptr[mul_col] + add_row_ptr[add_col];
    }
}

// SLM-cached fused RMS norm + element-wise multiply + add kernel
// Caches input row in shared local memory to avoid double global memory read
// Combines: dst = scale * x * mul + add where scale = rsqrt(mean(x^2) + eps)
static void rms_norm_mul_add_f32_slm_cached(const float* x, const float* mul, const float* add, float* dst,
        const int ncols, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, float* s_x, int block_size) {

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    // Calculate mul pointer offset with broadcasting support
    const int mul_row     = row % mul_nrows;
    const int mul_channel = channel % mul_nchannels;
    const int mul_sample  = sample % mul_nsamples;
    const float* mul_row_ptr = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    // Calculate add pointer offset with broadcasting support
    const int add_row     = row % add_nrows;
    const int add_channel = channel % add_nchannels;
    const int add_sample  = sample % add_nsamples;
    const float* add_row_ptr = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    // First pass: load x into SLM and compute sum of squares
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        s_x[col] = xi;  // Cache in SLM
        tmp += xi * xi;
    }

    // Ensure all threads have written to SLM before reduction
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Second pass: read from SLM instead of global memory
    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = col % mul_ncols;
        const int add_col = col % add_ncols;
        dst[col] = scale * s_x[col] * mul_row_ptr[mul_col] + add_row_ptr[add_col];
    }
}

static void l2_norm_f32(const float* x, float* dst, const int ncols, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    const int tid = item_ct1.get_local_id(2);
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:3: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        size_t nreduce = nwarps / WARP_SIZE;
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float scale = sycl::rsqrt(sycl::max(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}

static void norm_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                            sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void group_norm_f32_sycl(const float* x, float* dst,
    const int num_groups, const float eps, const int group_size,
    const int ne_elements, queue_ptr stream, int device) {
    if (group_size < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            const float eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(x, dst, group_size, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void rms_norm_f32_sycl(const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, queue_ptr stream, int device) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Maximum ncols for SLM caching (32KB limit to leave room in 64KB SLM)
constexpr int SLM_CACHE_MAX_NCOLS = 8192;

// Fused RMS norm + multiply SYCL host function
static void rms_norm_mul_f32_sycl(const float* x, const float* mul, float* dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, queue_ptr stream, int device) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_f32(x, mul, dst, ncols, stride_row, stride_channel, stride_sample,
                                     mul_stride_row, mul_stride_channel, mul_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                     eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else if (ncols <= SLM_CACHE_MAX_NCOLS) {
        // Use SLM-cached version: cache input row in shared local memory to avoid double global memory read
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            sycl::local_accessor<float, 1> s_x_acc(sycl::range<1>(ncols), cgh);  // Cache for input row
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_f32_slm_cached(x, mul, dst, ncols, stride_row, stride_channel, stride_sample,
                                     mul_stride_row, mul_stride_channel, mul_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                     eps, item_ct1, get_pointer(s_sum_acc), get_pointer(s_x_acc), work_group_size);
                });
            });
    }
    else {
        // Fall back to non-cached version for very large ncols
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_f32(x, mul, dst, ncols, stride_row, stride_channel, stride_sample,
                                     mul_stride_row, mul_stride_channel, mul_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                     eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Fused RMS norm + multiply + add SYCL host function
static void rms_norm_mul_add_f32_sycl(const float* x, const float* mul, const float* add, float* dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, queue_ptr stream, int device) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_add_f32(x, mul, add, dst, ncols, stride_row, stride_channel, stride_sample,
                                         mul_stride_row, mul_stride_channel, mul_stride_sample,
                                         mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                         add_stride_row, add_stride_channel, add_stride_sample,
                                         add_ncols, add_nrows, add_nchannels, add_nsamples,
                                         dst_stride_row, dst_stride_channel, dst_stride_sample,
                                         eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else if (ncols <= SLM_CACHE_MAX_NCOLS) {
        // Use SLM-cached version: cache input row in shared local memory to avoid double global memory read
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            sycl::local_accessor<float, 1> s_x_acc(sycl::range<1>(ncols), cgh);  // Cache for input row
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_add_f32_slm_cached(x, mul, add, dst, ncols, stride_row, stride_channel, stride_sample,
                                         mul_stride_row, mul_stride_channel, mul_stride_sample,
                                         mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                         add_stride_row, add_stride_channel, add_stride_sample,
                                         add_ncols, add_nrows, add_nchannels, add_nsamples,
                                         dst_stride_row, dst_stride_channel, dst_stride_sample,
                                         eps, item_ct1, get_pointer(s_sum_acc), get_pointer(s_x_acc), work_group_size);
                });
            });
    }
    else {
        // Fall back to non-cached version for very large ncols
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_mul_add_f32(x, mul, add, dst, ncols, stride_row, stride_channel, stride_sample,
                                         mul_stride_row, mul_stride_channel, mul_stride_sample,
                                         mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                         add_stride_row, add_stride_channel, add_stride_sample,
                                         add_ncols, add_nrows, add_nchannels, add_nsamples,
                                         dst_stride_row, dst_stride_channel, dst_stride_sample,
                                         eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Fused ADD + RMS norm SYCL host function
// Computes: add_dst = x + add, dst = RMSNorm(add_dst)
// Writes BOTH outputs so other consumers of the ADD result get correct data
static void add_rms_norm_f32_sycl(const float* x, const float* add, float* add_dst, float* dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_x_row, const int64_t stride_x_channel, const int64_t stride_x_sample,
        const int64_t stride_add_row, const int64_t stride_add_channel, const int64_t stride_add_sample,
        const int add_ncols, const int add_nrows, const int add_nchannels, const int add_nsamples,
        const int64_t add_dst_stride_row, const int64_t add_dst_stride_channel, const int64_t add_dst_stride_sample,
        const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, queue_ptr stream, int device) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    add_rms_norm_f32(x, add, add_dst, dst, ncols,
                                     stride_x_row, stride_x_channel, stride_x_sample,
                                     stride_add_row, stride_add_channel, stride_add_sample,
                                     add_ncols, add_nrows, add_nchannels, add_nsamples,
                                     add_dst_stride_row, add_dst_stride_channel, add_dst_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else if (ncols <= SLM_CACHE_MAX_NCOLS) {
        // Use SLM-cached version: cache (x + add) in shared local memory to avoid double computation
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            sycl::local_accessor<float, 1> s_x_acc(sycl::range<1>(ncols), cgh);  // Cache for (x + add)
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    add_rms_norm_f32_slm_cached(x, add, add_dst, dst, ncols,
                                     stride_x_row, stride_x_channel, stride_x_sample,
                                     stride_add_row, stride_add_channel, stride_add_sample,
                                     add_ncols, add_nrows, add_nchannels, add_nsamples,
                                     add_dst_stride_row, add_dst_stride_channel, add_dst_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     eps, item_ct1, get_pointer(s_sum_acc), get_pointer(s_x_acc), work_group_size);
                });
            });
    }
    else {
        // Fall back to non-cached version for very large ncols
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    add_rms_norm_f32(x, add, add_dst, dst, ncols,
                                     stride_x_row, stride_x_channel, stride_x_sample,
                                     stride_add_row, stride_add_channel, stride_add_sample,
                                     add_ncols, add_nrows, add_nchannels, add_nsamples,
                                     add_dst_stride_row, add_dst_stride_channel, add_dst_stride_sample,
                                     dst_stride_row, dst_stride_channel, dst_stride_sample,
                                     eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void l2_norm_f32_sycl(const float* x, float* dst, const int ncols,
    const int nrows, const float eps,
    queue_ptr stream, int device) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    l2_norm_f32(x, dst, ncols, eps, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    l2_norm_f32(x, dst, ncols, eps, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    // Use device-specific data pointers for TP support
    const float * src0_dd = static_cast<const float *>(ggml_sycl_get_data_ptr(dst->src[0], ctx.device));
    float *       dst_dd  = static_cast<float *>(ggml_sycl_get_data_ptr(dst, ctx.device));

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03, s01, s02, s03, eps, main_stream, ctx.device);
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // Use device-specific data pointers for TP support
    const float * src0_dd = static_cast<const float *>(ggml_sycl_get_data_ptr(dst->src[0], ctx.device));
    float *       dst_dd  = static_cast<float *>(ggml_sycl_get_data_ptr(dst, ctx.device));

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // Debug: verify we're using the TP queue
    GGML_SYCL_DEBUG("[RMS_NORM] device=%d, stream=%p, stream_device=%s\n",
                    ctx.device, (void*)main_stream,
                    main_stream->get_device().get_info<sycl::info::device::name>().c_str());

    // Use device-specific data pointers for TP support
    const float * src0_dd = static_cast<const float *>(ggml_sycl_get_data_ptr(dst->src[0], ctx.device));
    float *       dst_dd  = static_cast<float *>(ggml_sycl_get_data_ptr(dst, ctx.device));

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS
    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    // DEBUG: Check hidden state at result_norm (final layer before lm_head)
    bool is_result_norm = dst->name && strstr(dst->name, "result_norm");
    static int result_norm_dbg = 0;
    if (is_result_norm && result_norm_dbg++ < 3) {
        std::vector<float> input_host(8);
        main_stream->memcpy(input_host.data(), src0_dd, 8*sizeof(float)).wait();
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < 8; i++) {
            sum += input_host[i];
            sum_sq += input_host[i] * input_host[i];
        }
        bool has_nan = std::isnan(sum) || std::isnan(sum_sq);
        bool all_zero = (sum == 0 && sum_sq == 0);
        fprintf(stderr, "TP DEBUG result_norm INPUT[0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d zero=%d ne01=%lld\n",
                input_host[0], input_host[1], input_host[2], input_host[3],
                input_host[4], input_host[5], input_host[6], input_host[7],
                sum, has_nan, all_zero, (long long)ne01);
    }

    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03, s01, s02, s03, eps, main_stream, ctx.device);

    // DEBUG: Check output after result_norm
    if (is_result_norm && result_norm_dbg <= 3) {
        main_stream->wait();
        std::vector<float> output_host(8);
        main_stream->memcpy(output_host.data(), dst_dd, 8*sizeof(float)).wait();
        float sum = 0;
        bool has_nan = false;
        for (int i = 0; i < 8; i++) {
            sum += output_host[i];
            if (std::isnan(output_host[i])) has_nan = true;
        }
        fprintf(stderr, "TP DEBUG result_norm OUTPUT[0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d\n",
                output_host[0], output_host[1], output_host[2], output_host[3],
                output_host[4], output_host[5], output_host[6], output_host[7],
                sum, has_nan);
    }
}

// Fused RMS norm + element-wise multiply dispatch function
// This is called from the graph compute loop when fusion is detected
void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    // dst is the RMS_NORM output tensor
    // mul_tensor is the MUL output tensor (which is the final fused output)
    const ggml_tensor * rms_norm_src = dst->src[0];

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(rms_norm_src->data);
    float * mul_dst_dd = static_cast<float *>(mul_tensor->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // Get dimensions from RMS_NORM source
    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    // Get the mul weights tensor (src[0] or src[1] of mul_tensor that isn't dst)
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else {
        mul_src = mul_tensor->src[0];
    }
    GGML_ASSERT(mul_src != nullptr);

    const float * mul_dd = static_cast<const float *>(mul_src->data);

    // Get mul weights dimensions and strides
    const int mul_ncols = mul_src->ne[0];
    const int mul_nrows = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples = mul_src->ne[3];

    const size_t mul_ts = ggml_type_size(mul_src->type);
    const int64_t mul_s01 = mul_src->nb[1] / mul_ts;
    const int64_t mul_s02 = mul_src->nb[2] / mul_ts;
    const int64_t mul_s03 = mul_src->nb[3] / mul_ts;

    // Get destination (mul_tensor) strides
    const size_t dst_ts = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == dst_ts);
    const int64_t dst_s01 = mul_tensor->nb[1] / dst_ts;
    const int64_t dst_s02 = mul_tensor->nb[2] / dst_ts;
    const int64_t dst_s03 = mul_tensor->nb[3] / dst_ts;

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, mul_dst_dd,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          dst_s01, dst_s02, dst_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          eps, main_stream, ctx.device);
}

// Fused RMS norm + element-wise multiply + add dispatch function
// This is called from the graph compute loop when 3-way fusion is detected
void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    // dst is the RMS_NORM output tensor
    // mul_tensor is the MUL output tensor
    // add_tensor is the ADD output tensor (which is the final fused output)
    const ggml_tensor * rms_norm_src = dst->src[0];

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(rms_norm_src->data);
    float * add_dst_dd = static_cast<float *>(add_tensor->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // Get dimensions from RMS_NORM source
    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    // Get the mul weights tensor (src[0] or src[1] of mul_tensor that isn't dst)
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else {
        mul_src = mul_tensor->src[0];
    }
    GGML_ASSERT(mul_src != nullptr);

    const float * mul_dd = static_cast<const float *>(mul_src->data);

    // Get mul weights dimensions and strides
    const int mul_ncols = mul_src->ne[0];
    const int mul_nrows = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples = mul_src->ne[3];

    const size_t mul_ts = ggml_type_size(mul_src->type);
    const int64_t mul_s01 = mul_src->nb[1] / mul_ts;
    const int64_t mul_s02 = mul_src->nb[2] / mul_ts;
    const int64_t mul_s03 = mul_src->nb[3] / mul_ts;

    // Get the add tensor (src[0] or src[1] of add_tensor that isn't mul_tensor)
    const ggml_tensor * add_src = nullptr;
    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else {
        add_src = add_tensor->src[0];
    }
    GGML_ASSERT(add_src != nullptr);

    const float * add_dd = static_cast<const float *>(add_src->data);

    // Get add tensor dimensions and strides
    const int add_ncols = add_src->ne[0];
    const int add_nrows = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples = add_src->ne[3];

    const size_t add_ts = ggml_type_size(add_src->type);
    const int64_t add_s01 = add_src->nb[1] / add_ts;
    const int64_t add_s02 = add_src->nb[2] / add_ts;
    const int64_t add_s03 = add_src->nb[3] / add_ts;

    // Get destination (add_tensor) strides
    const size_t dst_ts = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == dst_ts);
    const int64_t dst_s01 = add_tensor->nb[1] / dst_ts;
    const int64_t dst_s02 = add_tensor->nb[2] / dst_ts;
    const int64_t dst_s03 = add_tensor->nb[3] / dst_ts;

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, add_dst_dd,
                              ne00, ne01, ne02, ne03,
                              s01, s02, s03,
                              mul_s01, mul_s02, mul_s03,
                              mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                              add_s01, add_s02, add_s03,
                              add_ncols, add_nrows, add_nchannels, add_nsamples,
                              dst_s01, dst_s02, dst_s03,
                              eps, main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    float eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    if (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    const float * g_base  = static_cast<const float *>(dst->src[0]->data); // dz
    const float * x_base  = static_cast<const float *>(dst->src[1]->data); // x
          float * dx_base = static_cast<      float *>(dst->data);

    const int64_t D  = dst->ne[0];
    const int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (void) n3;
    const int64_t N  = ggml_nrows(dst);
    if (D == 0 || N == 0) return;

    const ggml_tensor *G = dst->src[0];
    const ggml_tensor *X = dst->src[1];
    const int ts = (int) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    const int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    const int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    const int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    const int device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](int v, int m) { return ((v + m - 1) / m) * m; };
    int wg_cap = 256;
    if (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    int WG = std::max(WARP_SIZE, std::min(roundup((int)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        const int nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item_ct1.get_group(2);
                const int tid = item_ct1.get_local_id(2);

                const int64_t i1 = row % n1;
                const int64_t i2 = (row / n1) % n2;
                const int64_t i3 = row / (n1 * n2);

                const float *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                const float *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                float *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                float sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                float c_xx = 0.f, c_xg = 0.f;
#endif
                for (int64_t col = tid; col < D; col += WG) {
                    const float xv = x_row[col];
                    const float gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#else
                    float y1 = xv * xv - c_xx;
                    float t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    float y2 = xv * gv - c_xg;
                    float t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#else
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#else
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                const auto sub_group = item_ct1.get_sub_group();
                const auto sg_id     = sub_group.get_group_linear_id();
                const auto wi_in_sg  = sub_group.get_local_linear_id();
                const int nthreads   = item_ct1.get_local_range(2);
                const int nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                if (nwarps > 1) {
                    if (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        const unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } else {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                float inv_r = 0.f;
                float coeff = 0.f;
                if (tid == 0) {
                    const float sum_xx_f  = xx_total.x() + xx_total.y();
                    const float sum_xdz_f = xg_total.x() + xg_total.y();
                    const float mean_eps  = sum_xx_f / (float) D + eps;
                    const float sum_eps   = sum_xx_f + eps * (float) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                for (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float * dst_dd = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    l2_norm_f32_sycl(src0_dd, dst_dd, ne00, nrows, eps, main_stream, ctx.device);

}

// Fused ADD + RMS_NORM dispatch function
// Pattern: add_tensor = src0 + src1, rms_norm_tensor = RMS_NORM(add_tensor)
// Writes BOTH outputs so other consumers of add_tensor get correct data
void ggml_sycl_op_add_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_tensor, ggml_tensor * rms_norm_tensor) {
    // add_tensor is the ADD output tensor (intermediate, needed by other consumers)
    // rms_norm_tensor is the RMS_NORM output tensor (final fused output)

    GGML_ASSERT(add_tensor->op == GGML_OP_ADD);
    GGML_ASSERT(rms_norm_tensor->op == GGML_OP_RMS_NORM);
    GGML_ASSERT(rms_norm_tensor->src[0] == add_tensor);

    // Get the two ADD inputs
    const ggml_tensor * add_src0 = add_tensor->src[0];
    const ggml_tensor * add_src1 = add_tensor->src[1];

    GGML_ASSERT(add_src0->type == GGML_TYPE_F32);
    GGML_ASSERT(add_src1->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * x_dd = static_cast<const float *>(add_src0->data);  // First ADD input (x)
    const float * add_dd = static_cast<const float *>(add_src1->data);  // Second ADD input (add)
    float * add_dst_dd = static_cast<float *>(add_tensor->data);  // ADD output (for other consumers)
    float * rms_dst_dd = static_cast<float *>(rms_norm_tensor->data);  // RMS_NORM output

    float eps;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    // Get dimensions from first ADD source (x) - this is the primary tensor
    const int64_t ne00 = add_src0->ne[0];
    const int64_t ne01 = add_src0->ne[1];
    const int64_t ne02 = add_src0->ne[2];
    const int64_t ne03 = add_src0->ne[3];

    const size_t ts0 = ggml_type_size(add_src0->type);
    GGML_ASSERT(add_src0->nb[0] == ts0);
    const int64_t s01 = add_src0->nb[1] / ts0;
    const int64_t s02 = add_src0->nb[2] / ts0;
    const int64_t s03 = add_src0->nb[3] / ts0;

    // Get add tensor (second input) dimensions and strides for broadcasting
    const int add_ncols = add_src1->ne[0];
    const int add_nrows = add_src1->ne[1];
    const int add_nchannels = add_src1->ne[2];
    const int add_nsamples = add_src1->ne[3];

    const size_t add_ts = ggml_type_size(add_src1->type);
    const int64_t add_s01 = add_src1->nb[1] / add_ts;
    const int64_t add_s02 = add_src1->nb[2] / add_ts;
    const int64_t add_s03 = add_src1->nb[3] / add_ts;

    // Get ADD output (add_tensor) strides - needed for other consumers
    const size_t add_dst_ts = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == add_dst_ts);
    const int64_t add_dst_s01 = add_tensor->nb[1] / add_dst_ts;
    const int64_t add_dst_s02 = add_tensor->nb[2] / add_dst_ts;
    const int64_t add_dst_s03 = add_tensor->nb[3] / add_dst_ts;

    // Get RMS_NORM output (rms_norm_tensor) strides
    const size_t dst_ts = ggml_type_size(rms_norm_tensor->type);
    GGML_ASSERT(rms_norm_tensor->nb[0] == dst_ts);
    const int64_t dst_s01 = rms_norm_tensor->nb[1] / dst_ts;
    const int64_t dst_s02 = rms_norm_tensor->nb[2] / dst_ts;
    const int64_t dst_s03 = rms_norm_tensor->nb[3] / dst_ts;

    add_rms_norm_f32_sycl(x_dd, add_dd, add_dst_dd, rms_dst_dd,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          add_dst_s01, add_dst_s02, add_dst_s03,
                          dst_s01, dst_s02, dst_s03,
                          eps, main_stream, ctx.device);
}
