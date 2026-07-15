#include "conv2d.hpp"
#include "convert.hpp"

struct conv2d_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

static inline int64_t conv2d_max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

static inline int64_t conv2d_min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

static inline conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

static inline int calculate_input_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
static inline int64_t whcn_input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_params & P) {
    return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

static inline int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv2d_params & P) {
    return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

static inline int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_params & P) {
    return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

template <typename T>
static void conv2d_kernel(const float * input, const T * kernel, float * output,
                          const conv2d_params P, const sycl::nd_item<3> & item_ct1) {
    const int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    if (global_idx >= P.TOTAL) {
        return;
    }

    const int64_t out_x  = global_idx % P.OW;
    const int64_t out_y  = (global_idx / P.OW) % P.OH;
    const int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    const int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    float acc = 0.0f;

    const conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                const float input_val  = input[whcn_input_index(n, c_in, in_y, in_x, P)];
                const T     kernel_val = kernel[whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += input_val * ggml_sycl_cast<float>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_sycl(const float * X_D, const T * K_D, float * Y_D,
                        const conv2d_params P, const queue_ptr & stream) {
    const int num_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T>(X_D, K_D, Y_D, P, item_ct1);
        });
}

void ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const float *       K_D    = (const float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    const queue_ptr stream = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];
    const int       ST_Y = p[1];
    const int       PD_X = p[2];
    const int       PD_Y = p[3];
    const int       DL_X = p[4];
    const int       DL_Y = p[5];

    // no cwhn layout support
    GGML_ASSERT(p[6] == 0);

    const int IW = input->ne[0];
    const int IH = input->ne[1];
    const int OW = dst->ne[0];
    const int OH = dst->ne[1];
    const int KW = kernel->ne[0];
    const int KH = kernel->ne[1];
    const int IC = input->ne[2];
    const int OC = kernel->ne[3];
    const int B  = input->ne[3];

    const int64_t     total  = (int64_t) B * OC * OH * OW;
    const conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_sycl<sycl::half>(X_D, (const sycl::half *) K_D, Y_D, params, stream);
    } else {
        conv2d_sycl<float>(X_D, K_D, Y_D, params, stream);
    }
}
