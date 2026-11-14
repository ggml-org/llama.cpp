#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <random>
#include <vector>

static bool read_floats(const std::string & path, std::vector<float> & values) {
    std::ifstream file(path, std::ios::binary);
    return bool(file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float))) &&
           file.peek() == EOF;
}

int main(int argc, char ** argv) {
    // /conv_14/Conv: 384 channels, 7 taps, stride 1.
    constexpr int channels = 384, taps = 7, width = channels * taps;
    ggml_backend_load_all();
    const auto weight_type = std::getenv("M5_CONV_F16") ? GGML_TYPE_F16 : GGML_TYPE_F32;
    std::mt19937 random(1234);
    std::uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    const bool real = argc == 3 || argc == 4;
    if (argc != 1 && !real) return 2;
    char * end = nullptr;
    const long tile = argc == 4 ? std::strtol(argv[3], &end, 10) : 64;
    if ((end && *end) || tile <= 0 || tile > 1280) return 2;
    const int rows = static_cast<int>(tile);
    std::vector<float> weights(width * channels), patches(width * rows), reference(channels * rows);
    std::vector<float> activation, bias(channels), expected;
    size_t positions = rows;
    if (real) {
        const std::string dir = argv[1];
        const std::string input = dir + "/input-" + argv[2] + ".f32";
        std::ifstream file(input, std::ios::binary | std::ios::ate);
        const auto size = file.tellg();
        if (size <= 0 || size % (channels * sizeof(float)) != 0) return 2;
        activation.resize(size / sizeof(float));
        const auto stride = activation.size() / channels;
        if (stride <= taps || (stride - taps + 1) % rows != 0) return 2;
        positions = stride - taps + 1;
        expected.resize(positions * channels);
        if (!read_floats(input, activation) || !read_floats(dir + "/weight.f32", weights) ||
            !read_floats(dir + "/bias.f32", bias) ||
            !read_floats(dir + "/output-" + argv[2] + ".f32", expected)) return 2;
    }
    for (auto & value : weights) if (!real) value = distribution(random);
    for (auto & value : patches) value = distribution(random);

    const bool direct = real && std::getenv("M5_CONV_DIRECT");
    for (const char * name : {"CPU", "HTP0"}) {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) return 1;
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 4);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true}), ggml_free);
        if (!context) return 1;
        auto * w = direct ? ggml_new_tensor_4d(context.get(), weight_type, taps, 1, channels, channels)
                          : ggml_new_tensor_2d(context.get(), weight_type, width, channels);
        auto * x = direct ? ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, positions + taps - 1, 1, channels, 1)
                          : ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, rows);
        auto * y = direct ? ggml_conv_2d_direct(context.get(), w, x, 1, 1, 0, 0, 1, 1)
                          : ggml_mul_mat(context.get(), w, x);
        if (!ggml_backend_supports_op(backend.get(), y)) {
            std::fprintf(stderr, "%s does not support this operation\n", name);
            return 1;
        }
        auto * graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, y);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) return 1;
        if (weight_type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(weights.size());
            ggml_fp32_to_fp16_row(weights.data(), half.data(), weights.size());
            ggml_backend_tensor_set(w, half.data(), 0, ggml_nbytes(w));
        } else {
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
        }
        if (!real) ggml_backend_tensor_set(x, patches.data(), 0, ggml_nbytes(x));
        std::vector<float> output(ggml_nelements(y));
        std::vector<double> times;
        float max_error = 0;
        float max_matmul = 0;
        for (int iteration = 0; iteration < 9; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            float iteration_error = 0;
            for (size_t position = 0; position < positions; position += direct ? positions : rows) {
                if (direct) {
                    ggml_backend_tensor_set(x, activation.data(), 0, ggml_nbytes(x));
                } else if (real) {
                    for (int row = 0; row < rows; ++row) {
                        for (int c = 0; c < channels; ++c) {
                            std::copy_n(activation.data() + c * (positions + taps - 1) + position + row,
                                        taps, patches.data() + row * width + c * taps);
                        }
                    }
                    ggml_backend_tensor_set(x, patches.data(), 0, ggml_nbytes(x));
                }
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
                ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
                if (real) {
                    for (size_t i = 0; i < output.size(); ++i) {
                        if (iteration == 0) max_matmul = std::max(max_matmul, std::abs(output[i]));
                        const size_t channel = direct ? i / positions : i % channels;
                        const size_t index = direct ? (i % positions) * channels + channel : position * channels + i;
                        output[i] += bias[channel];
                        if (!std::isfinite(output[i])) return 1;
                        if (iteration == 0) {
                            iteration_error = std::max(iteration_error,
                                std::abs(output[i] - expected[index]));
                        }
                    }
                }
            }
            if (iteration == 0) max_error = iteration_error;
            if (iteration > 0) {
                times.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            }
        }
        if (!real && ggml_backend_is_cpu(backend.get())) {
            reference = output;
        } else if (!real) {
            for (size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(output[i])) return 1;
                max_error = std::max(max_error, std::abs(output[i] - reference[i]));
            }
        }
        std::sort(times.begin(), times.end());
        std::printf("%s median_ms=%.3f max_abs_error=%.8g\n", name, (times[3] + times[4]) / 2, max_error);
        if (real) std::printf("%s max_abs_matmul=%.8g\n", name, max_matmul);
        if (!real && max_error > 0.001f) return 1;
    }
}
