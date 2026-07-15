#include "conv2d-dw.hpp"

struct conv2d_dw_params {
    int in_w, in_h;
    int out_w, out_h;
    int kernel_w, kernel_h;
    int stride_x, stride_y;
    int padding_x, padding_y;
    int dilation_x, dilation_y;
    int channels, batches;
};

struct conv2d_dw_kernel_bounds {
    int y_min, y_max;
    int x_min, x_max;
};

static inline conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(int out_x, int out_y,
                                                                  const conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    return bounds;
}

static inline int dw_calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
struct dw_whcn_layout {
    static int input_index(int n, int c, int y, int x, const conv2d_dw_params & p) {
        return n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    static int kernel_index(int c, int ky, int kx, const conv2d_dw_params & p) {
        return c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    static int output_index(int n, int c, int y, int x, const conv2d_dw_params & p) {
        return n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    static void unpack_indices(int global_idx, const conv2d_dw_params & p,
                               int & n, int & c, int & out_y, int & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
struct dw_cwhn_layout {
    static int input_index(int n, int c, int y, int x, const conv2d_dw_params & p) {
        return n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    static int kernel_index(int c, int ky, int kx, const conv2d_dw_params & p) {
        return (ky * p.kernel_w + kx) * p.channels + c;
    }
    static int output_index(int n, int c, int y, int x, const conv2d_dw_params & p) {
        return n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    static void unpack_indices(int global_idx, const conv2d_dw_params & p,
                               int & n, int & c, int & out_y, int & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename Layout>
static void conv2d_dw_kernel(const float * input, const float * kernel, float * output,
                             const conv2d_dw_params p, const sycl::nd_item<3> & item_ct1) {
    const int global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int total_elements = p.batches * p.channels * p.out_h * p.out_w;

    if (global_idx >= total_elements) {
        return;
    }

    int n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    float acc = 0.0f;
    const conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    for (int ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        const int in_y = dw_calculate_input_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        for (int kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            const int in_x = dw_calculate_input_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += input[Layout::input_index(n, c, in_y, in_x, p)] *
                   kernel[Layout::kernel_index(c, ky, kx, p)];
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename Layout>
static void conv2d_dw_sycl(const float * x_d, const float * w_d, float * y_d,
                            const conv2d_dw_params p, const queue_ptr & stream) {
    const int total      = p.batches * p.channels * p.out_h * p.out_w;
    const int num_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

void ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * w_d = (const float *) kernel->data;
    const float * x_d = (const float *) input->data;
    float *       y_d = (float *) dst->data;

    const int32_t * p          = (const int32_t *) dst->op_params;
    const int       stride_x   = p[0];
    const int       stride_y   = p[1];
    const int       padding_x  = p[2];
    const int       padding_y  = p[3];
    const int       dilation_x = p[4];
    const int       dilation_y = p[5];

    const int in_w     = input->ne[0];
    const int in_h     = input->ne[1];
    const int kernel_w = kernel->ne[0];
    const int kernel_h = kernel->ne[1];
    const int out_w    = dst->ne[0];
    const int out_h    = dst->ne[1];
    const int channels = dst->ne[2];
    const int batches  = dst->ne[3];

    const conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    const queue_ptr stream = ctx.stream();

    if (ggml_is_contiguous(input)) {
        conv2d_dw_sycl<dw_whcn_layout>(x_d, w_d, y_d, params, stream);
    } else if (ggml_is_contiguous_channels(input)) {
        conv2d_dw_sycl<dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
    } else {
        GGML_ABORT("Unsupported memory layout for conv2d_dw");
    }
}
