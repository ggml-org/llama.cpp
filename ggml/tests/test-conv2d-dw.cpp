#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

std::vector<float> f32_range(int n, float start, float end) {
    std::vector<float> values(n);
    float step = (end - start) / n;
    for (int i = 0; i < n; i++) {
        values[i] = start + i * step;
    }
    return values;
}

// Most straightforward implementation without any optimizations
std::vector<float> conv_2d_dw_reference(
        int src_w, int src_h, const float * src_data,
        int knl_w, int knl_h, const float * knl_data,
        int channels, int batch, int stride, int pad, int dilation) {

    int dst_w = (src_w + 2 * pad - dilation * (knl_w - 1) - 1) / stride + 1;
    int dst_h = (src_h + 2 * pad - dilation * (knl_h - 1) - 1) / stride + 1;
    std::vector<float> dst_data(dst_w * dst_h * channels * batch);

    for (int b = 0; b < batch; b++) {
        const float * src_base = src_data + b * src_w * src_h * channels;
        float * dst_base = dst_data.data() + b * dst_w * dst_h * channels;
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < dst_h; y++) {
                for (int x = 0; x < dst_w; x++) {
                    float sum = 0;
                    for (int knl_y = 0; knl_y < knl_h; knl_y++) {
                        for (int knl_x = 0; knl_x < knl_w; knl_x++) {
                            int src_x = x * stride + knl_x * dilation - pad;
                            int src_y = y * stride + knl_y * dilation - pad;
                            if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
                                sum += src_base[c * src_w * src_h + src_y * src_w + src_x] *
                                       knl_data[c * knl_w * knl_h + knl_y * knl_w + knl_x];
                            }
                        }
                    }
                    dst_base[c * dst_w * dst_h + y * dst_w + x] = sum;
                }
            }
        }
    }
    return dst_data;
}

bool check_equal(const std::vector<float> & result, const std::vector<float> & expected) {
    if (result.size() != expected.size()) {
        printf("result.size() = %d, expected.size() = %d\n", (int)result.size(), (int)expected.size());
        return false;
    }
    for (int i = 0; i < result.size(); i++) {
        if(std::abs(result[i] - expected[i]) > 1e-5) {
            printf("result[%d] %f != %f expected[%d]\n", i, result[i], expected[i], i);
            return false;
        }
    }
    return true;
}

bool test_conv_2d_dw(
        int channels,
        int kernel_size,
        int stride,
        int pad,
        int dilation,
        bool contiguous_channels) {
    ggml_time_init();

    const int batch = 2;
    const int src_w = 8;
    const int src_h = 6;
    const int knl_w = kernel_size;
    const int knl_h = kernel_size;

    ggml_init_params params {
        /*.mem_size   =*/ 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Build graph
    ggml_tensor * src_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src_w, src_h, channels, batch);
    ggml_tensor * knl_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, knl_w, knl_h, 1, channels);
    ggml_tensor * src = src_input;
    ggml_tensor * knl = knl_input;
    if (contiguous_channels) {
        // Convert tensor to [C, W, H, N] layout in memory, then permute strides back to [W, H, C, N]
        src = ggml_cont(ctx, ggml_permute(ctx, src, 1, 2, 0, 3));
        src = ggml_permute(ctx, src, 2, 0, 1, 3);
        knl = ggml_cont(ctx, ggml_permute(ctx, knl, 2, 3, 1, 0));
        knl = ggml_permute(ctx, knl, 3, 2, 0, 1);
    }
    ggml_tensor * res = ggml_conv_2d_dw_direct(
        ctx, knl, src, stride, stride, pad, pad, dilation, dilation);
    if (contiguous_channels) {
        res = ggml_cont(ctx, res);
    }
    ggml_build_forward_expand(gf, res);

    // Create backend & allocate buffers
    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};

    std::vector<float> src_values = f32_range(ggml_nelements(src), -1.f, 1.f);
    std::vector<float> knl_values = f32_range(ggml_nelements(knl), -1.f, 1.f);
    ggml_backend_tensor_set(src_input, src_values.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_set(knl_input, knl_values.data(), 0, ggml_nbytes(knl));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res_values(ggml_nelements(res));
    ggml_backend_tensor_get(res, res_values.data(), 0, ggml_nbytes(res));

    std::vector<float> expected = conv_2d_dw_reference(
        src_w, src_h, src_values.data(),
        knl_w, knl_h, knl_values.data(),
        channels, batch, stride, pad, dilation);

    bool passed = check_equal(res_values, expected);

    printf("ggml_conv_2d_dw(channels=%d, kernel=%dx%d, stride=%d, pad=%d, dilation=%d, layout=%s): %s\n",
        channels, kernel_size, kernel_size, stride, pad, dilation, contiguous_channels ? "CWHN" : "WHCN",
        passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed;
}

int main(int argc, char ** argv) {
    bool passed = true;
    passed = test_conv_2d_dw(3, 1, 1, 0, 1, false) && passed;
    passed = test_conv_2d_dw(3, 1, 1, 0, 1, true) && passed;
    passed = test_conv_2d_dw(42, 3, 2, 1, 1, false) && passed;
    passed = test_conv_2d_dw(42, 3, 2, 1, 1, true) && passed;
    passed = test_conv_2d_dw(8, 5, 1, 2, 2, false) && passed;
    passed = test_conv_2d_dw(8, 5, 1, 2, 2, true) && passed;
    return passed ? 0 : 1;
}
