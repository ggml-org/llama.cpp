#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool nearly_equal(float actual, float expected, float tolerance = 1e-5f) {
    return std::fabs(actual - expected) <= tolerance * std::max(1.0f, std::fabs(expected));
}

static float regular_hadamard_reference(const std::vector<float> & input, int output) {
    static constexpr int h4[4][4] = {
        { 1,  1,  1, -1},
        { 1,  1, -1,  1},
        { 1, -1,  1,  1},
        {-1,  1,  1,  1},
    };

    float sum = 0.0f;
    for (int column = 0; column < (int) input.size(); ++column) {
        int row_digit    = output;
        int column_digit = column;
        int sign         = 1;
        while (row_digit != 0 || column_digit != 0) {
            sign *= h4[row_digit % 4][column_digit % 4];
            row_digit /= 4;
            column_digit /= 4;
        }
        sum += sign * input[column];
    }
    return sum / std::sqrt((float) input.size());
}

int main(int argc, char ** argv) {
    const bool use_cuda   = argc == 2 && std::strcmp(argv[1], "--cuda") == 0;
    const bool use_vulkan = argc == 2 && std::strcmp(argv[1], "--vulkan") == 0;
    if (argc > 2 || (argc == 2 && !use_cuda && !use_vulkan)) {
        std::fprintf(stderr, "usage: %s [--cuda|--vulkan]\n", argv[0]);
        return 1;
    }

    ggml_init_params params = {
        1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 4, 3);
    ggml_tensor * rot  = ggml_regular_hadamard(ctx, x, 4);
    ggml_tensor * out  = ggml_mul_mat(ctx, w, rot);
    ggml_tensor * x256 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * r256 = ggml_regular_hadamard(ctx, x256, 256);
    ggml_tensor * w256 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 256, 4);
    ggml_tensor * ws256 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * b256 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * q256 = ggml_quantize_i8_convrot(ctx, x256, 256);
    ggml_tensor * fused256 = ggml_mul_mat_i8_tensorwise(ctx, w256, q256, ws256, b256, 256);
    const int large_k    = 512;
    const int large_n    = 128;
    const int large_rows = 65;
    ggml_tensor * x_large  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, large_k, large_rows);
    ggml_tensor * w_large  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, large_k, large_n);
    ggml_tensor * ws_large = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, large_n);
    ggml_tensor * b_large  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, large_n);
    ggml_tensor * q_large  = ggml_quantize_i8_convrot(ctx, x_large, 256);
    ggml_tensor * fused_large = ggml_mul_mat_i8_tensorwise(ctx, w_large, q_large, ws_large, b_large, 256);
    ggml_tensor * fused_large_no_bias = ggml_mul_mat_i8_tensorwise(ctx, w_large, q_large, ws_large, nullptr, 256);
    ggml_tensor * fused_large_direct = ggml_mul_mat_i8_tensorwise(ctx, w_large, x_large, ws_large, b_large, 256);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (!use_vulkan) {
        ggml_build_forward_expand(graph, out);
        ggml_build_forward_expand(graph, r256);
        ggml_build_forward_expand(graph, fused_large_direct);
    }
    ggml_build_forward_expand(graph, fused256);
    ggml_build_forward_expand(graph, fused_large);
    ggml_build_forward_expand(graph, fused_large_no_bias);

    ggml_backend_t backend = nullptr;
    if (use_cuda || use_vulkan) {
        const char * backend_name = use_cuda ? "CUDA" : "Vulkan";
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU && std::strstr(ggml_backend_dev_name(device), backend_name) != nullptr) {
                backend = ggml_backend_dev_init(device, nullptr);
                break;
            }
        }
        if (backend == nullptr) {
            std::fprintf(stderr, "%s backend not found\n", backend_name);
            return 1;
        }
    } else {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 2);
    }
    if (!ggml_backend_supports_op(backend, q_large) ||
        !ggml_backend_supports_op(backend, fused_large) ||
        !ggml_backend_supports_op(backend, fused_large_no_bias) ||
        (!use_vulkan && !ggml_backend_supports_op(backend, fused_large_direct))) {
        std::fprintf(stderr, "backend does not report packed INT8 convrot matmul support\n");
        return 1;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    const std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, -3.0f,
    };
    const std::vector<int8_t> w_data = {
        1, 2, 3, 4,
        -2, 1, 0, 2,
        127, -127, 64, -64,
    };
    std::vector<float> x256_data(256);
    std::vector<int8_t> w256_data(256 * 4);
    for (int i = 0; i < 256; ++i) {
        x256_data[i] = (float) ((i * 17) % 29 - 14) / 7.0f;
        for (int output = 0; output < 4; ++output) {
            w256_data[output * 256 + i] = (int8_t) ((i * (output + 3) + output * 11) % 31 - 15);
        }
    }
    const std::vector<float> ws256_data = { 0.125f, 0.25f, 0.5f, 1.5f };
    const std::vector<float> b256_data  = { -1.0f, 0.5f, 2.0f, -3.0f };
    std::vector<float> x_large_data(large_k * large_rows);
    std::vector<int8_t> w_large_data(large_k * large_n);
    std::vector<float> ws_large_data(large_n);
    std::vector<float> b_large_data(large_n);
    for (int row = 0; row < large_rows; ++row) {
        for (int i = 0; i < large_k; ++i) {
            x_large_data[row * large_k + i] = (float) ((row * 13 + i * 17) % 43 - 21) / 9.0f;
        }
    }
    for (int output = 0; output < large_n; ++output) {
        const int pattern = output % 7;
        for (int i = 0; i < large_k; ++i) {
            w_large_data[output * large_k + i] = (int8_t) ((i * (pattern + 3) + pattern * 11) % 61 - 30);
        }
        ws_large_data[output] = (float) (output % 11 + 1) / 100.0f;
        b_large_data[output]  = (float) (output % 9 - 4) / 7.0f;
    }
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, w_data.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x256, x256_data.data(), 0, ggml_nbytes(x256));
    ggml_backend_tensor_set(w256, w256_data.data(), 0, ggml_nbytes(w256));
    ggml_backend_tensor_set(ws256, ws256_data.data(), 0, ggml_nbytes(ws256));
    ggml_backend_tensor_set(b256, b256_data.data(), 0, ggml_nbytes(b256));
    ggml_backend_tensor_set(x_large, x_large_data.data(), 0, ggml_nbytes(x_large));
    ggml_backend_tensor_set(w_large, w_large_data.data(), 0, ggml_nbytes(w_large));
    ggml_backend_tensor_set(ws_large, ws_large_data.data(), 0, ggml_nbytes(ws_large));
    ggml_backend_tensor_set(b_large, b_large_data.data(), 0, ggml_nbytes(b_large));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> rot_data(ggml_nelements(rot));
    std::vector<float> out_data(ggml_nelements(out));
    if (!use_vulkan) {
        ggml_backend_tensor_get(rot, rot_data.data(), 0, ggml_nbytes(rot));
        ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));
    }

    const std::vector<float> expected_rot = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.25f, -2.75f, -1.25f, 0.25f,
    };
    if (!use_vulkan) {
        for (size_t i = 0; i < expected_rot.size(); ++i) {
            if (!nearly_equal(rot_data[i], expected_rot[i])) {
                std::fprintf(stderr, "regular Hadamard mismatch at %zu: %.8f != %.8f\n", i, rot_data[i], expected_rot[i]);
                return 1;
            }
        }
    }

    std::vector<float> r256_data(ggml_nelements(r256));
    if (!use_vulkan) {
        ggml_backend_tensor_get(r256, r256_data.data(), 0, ggml_nbytes(r256));
    }
    for (int i = 0; i < 256; ++i) {
        const float expected = regular_hadamard_reference(x256_data, i);
        if (!use_vulkan && !nearly_equal(r256_data[i], expected)) {
            std::fprintf(stderr, "regular Hadamard-256 mismatch at %d: %.8f != %.8f\n", i, r256_data[i], expected);
            return 1;
        }
        r256_data[i] = expected;
    }

    float fused_amax = 0.0f;
    for (float value : r256_data) {
        fused_amax = std::max(fused_amax, std::fabs(value));
    }
    const float fused_scale = fused_amax / 127.0f;
    std::vector<int8_t> fused_quantized(256);
    for (int i = 0; i < 256; ++i) {
        int value = (int) std::lrint(r256_data[i] / fused_scale);
        value = std::max(-127, std::min(127, value));
        fused_quantized[i] = (int8_t) value;
    }
    std::vector<float> expected_fused(4);
    for (int output = 0; output < 4; ++output) {
        int32_t sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += (int32_t) w256_data[output * 256 + i] * (int32_t) fused_quantized[i];
        }
        expected_fused[output] = (float) sum * fused_scale * ws256_data[output] + b256_data[output];
    }
    std::vector<float> fused_data(4);
    ggml_backend_tensor_get(fused256, fused_data.data(), 0, ggml_nbytes(fused256));
    for (size_t i = 0; i < expected_fused.size(); ++i) {
        if (!nearly_equal(fused_data[i], expected_fused[i])) {
            std::fprintf(stderr, "fused INT8 convrot mismatch at %zu: %.8f != %.8f\n", i, fused_data[i], expected_fused[i]);
            return 1;
        }
    }

    std::vector<int8_t> q_large_data(ggml_nbytes(q_large));
    std::vector<float> fused_large_data(ggml_nelements(fused_large));
    std::vector<float> fused_large_no_bias_data(ggml_nelements(fused_large_no_bias));
    std::vector<float> fused_large_direct_data(ggml_nelements(fused_large_direct));
    ggml_backend_tensor_get(q_large, q_large_data.data(), 0, ggml_nbytes(q_large));
    ggml_backend_tensor_get(fused_large, fused_large_data.data(), 0, ggml_nbytes(fused_large));
    ggml_backend_tensor_get(fused_large_no_bias, fused_large_no_bias_data.data(), 0, ggml_nbytes(fused_large_no_bias));
    if (!use_vulkan) {
        ggml_backend_tensor_get(fused_large_direct, fused_large_direct_data.data(), 0, ggml_nbytes(fused_large_direct));
    }
    const int large_rows_padded = (large_rows + 3) & ~3;
    std::vector<float> large_activation_scales(large_rows);
    std::memcpy(large_activation_scales.data(), q_large_data.data() + large_k * large_rows_padded, large_rows * sizeof(float));
    for (int row = 0; row < large_rows; ++row) {
        int32_t sums[7] = {};
        for (int pattern = 0; pattern < 7; ++pattern) {
            for (int i = 0; i < large_k; ++i) {
                sums[pattern] += (int32_t) q_large_data[row * large_k + i] * (int32_t) w_large_data[pattern * large_k + i];
            }
        }
        for (int output = 0; output < large_n; ++output) {
            const int32_t sum = sums[output % 7];
            const float expected_no_bias = (float) sum * large_activation_scales[row] * ws_large_data[output];
            const float expected = expected_no_bias + b_large_data[output];
            const size_t index = (size_t) row * large_n + output;
            if (!nearly_equal(fused_large_data[index], expected)) {
                std::fprintf(stderr, "large fused INT8 convrot mismatch at %zu: %.8f != %.8f\n", index, fused_large_data[index], expected);
                return 1;
            }
            if (!nearly_equal(fused_large_no_bias_data[index], expected_no_bias)) {
                std::fprintf(stderr, "large no-bias fused INT8 convrot mismatch at %zu: %.8f != %.8f\n", index, fused_large_no_bias_data[index], expected_no_bias);
                return 1;
            }
            if (!use_vulkan && !nearly_equal(fused_large_direct_data[index], expected)) {
                std::fprintf(stderr, "large direct fused INT8 convrot mismatch at %zu: %.8f != %.8f\n", index, fused_large_direct_data[index], expected);
                return 1;
            }
        }
    }

    std::vector<float> expected_out(6);
    for (int row = 0; row < 2; ++row) {
        const float * activation = expected_rot.data() + row * 4;
        float amax = 0.0f;
        for (int i = 0; i < 4; ++i) {
            amax = std::max(amax, std::fabs(activation[i]));
        }
        const float scale = amax / 127.0f;
        int8_t quantized[4];
        for (int i = 0; i < 4; ++i) {
            int value = (int) std::lrint(activation[i] / scale);
            value = std::max(-127, std::min(127, value));
            quantized[i] = (int8_t) value;
        }
        for (int output = 0; output < 3; ++output) {
            int32_t sum = 0;
            for (int i = 0; i < 4; ++i) {
                sum += (int32_t) w_data[output * 4 + i] * (int32_t) quantized[i];
            }
            expected_out[row * 3 + output] = (float) sum * scale;
        }
    }

    if (!use_vulkan) {
        for (size_t i = 0; i < expected_out.size(); ++i) {
            if (!nearly_equal(out_data[i], expected_out[i])) {
                std::fprintf(stderr, "INT8 matmul mismatch at %zu: %.8f != %.8f\n", i, out_data[i], expected_out[i]);
                return 1;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("INT8 convrot %s test passed\n", use_cuda ? "CUDA" : (use_vulkan ? "Vulkan" : "CPU"));
    return 0;
}
