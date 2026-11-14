#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Sweep the weight/activation types of a Conv1D expressed as MUL_MAT.
// The im2col expansion is prepared once on the host and reused, so the timed
// section isolates backend compute plus the optional activation upload.

static bool read_floats(const std::string & path, std::vector<float> & values) {
    std::ifstream file(path, std::ios::binary);
    return bool(file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float))) &&
           file.peek() == EOF;
}

struct result_t {
    double device_ms;
    double full_ms;
    float error;
    size_t weight_bytes;
};

static void convert_weights(enum ggml_type type, const std::vector<float> & src,
                            std::vector<uint8_t> & dst, int rows, int cols) {
    if (type == GGML_TYPE_F32) {
        dst.resize(src.size() * sizeof(float));
        std::memcpy(dst.data(), src.data(), dst.size());
        return;
    }
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        dst.resize(src.size() * sizeof(ggml_fp16_t));
        if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(dst.data()), src.size());
        } else {
            ggml_fp32_to_bf16_row(src.data(), reinterpret_cast<ggml_bf16_t *>(dst.data()), src.size());
        }
        return;
    }
    if (cols % ggml_blck_size(type) != 0) {
        throw std::runtime_error("row length is not a multiple of the quant block size");
    }
    dst.resize(ggml_row_size(type, cols) * rows);
    const size_t written = ggml_quantize_chunk(type, src.data(), dst.data(), 0, rows, cols, nullptr);
    if (written != dst.size()) {
        throw std::runtime_error("unexpected quantized weight size");
    }
}

int main(int argc, char ** argv) {
    // /conv_14/Conv: 384 input channels, 384 output channels, 7 taps.
    constexpr int channels = 384, taps = 7, width = channels * taps;
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s CAPTURE_DIR OFFSET ROWS\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int rows = std::atoi(argv[3]);
    if (rows <= 0 || rows > 1280) return 2;

    std::vector<float> weights(width * channels), bias(channels), activation, expected;
    {
        std::ifstream file(dir + "/input-" + argv[2] + ".f32", std::ios::binary | std::ios::ate);
        const auto size = file.tellg();
        if (size <= 0 || size % (channels * sizeof(float)) != 0) return 2;
        activation.resize(size / sizeof(float));
        const size_t stride = activation.size() / channels;
        const size_t positions = stride - taps + 1;
        if (static_cast<size_t>(rows) > positions || positions % rows != 0) return 2;
        expected.resize(positions * channels);
        if (!read_floats(dir + "/input-" + argv[2] + ".f32", activation) ||
            !read_floats(dir + "/weight.f32", weights) ||
            !read_floats(dir + "/bias.f32", bias) ||
            !read_floats(dir + "/output-" + argv[2] + ".f32", expected)) return 2;
    }

    ggml_backend_load_all();
    const enum ggml_type weight_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL,
    };
    const enum ggml_type activation_types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };

    for (const char * backend_name : {"CPU", "HTP0"}) {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(backend_name, nullptr), ggml_backend_free);
        if (!backend) return 1;
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 4);

        for (auto wtype : weight_types) {
            for (auto xtype : activation_types) {
                std::vector<uint8_t> wbytes;
                try {
                    convert_weights(wtype, weights, wbytes, channels, width);
                } catch (const std::exception &) {
                    continue;
                }

                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                    ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true}), ggml_free);
                auto * w = ggml_new_tensor_2d(ctx.get(), wtype, width, channels);
                auto * x = ggml_new_tensor_2d(ctx.get(), xtype, width, rows);
                auto * y = ggml_mul_mat(ctx.get(), w, x);
                if (!ggml_backend_supports_op(backend.get(), y)) {
                    std::printf("%s %-8s %-5s SKIP (unsupported)\n", backend_name,
                                ggml_type_name(wtype), ggml_type_name(xtype));
                    continue;
                }
                auto * graph = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, y);
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
                if (!buffer) return 1;

                // Host im2col: [rows][channels][taps] laid out as GGML expects.
                const size_t positions = activation.size() / channels - taps + 1;
                std::vector<float> patches(width * rows);
                for (int row = 0; row < rows; ++row) {
                    for (int c = 0; c < channels; ++c) {
                        std::copy_n(activation.data() + c * (positions + taps - 1) + row,
                                    taps, patches.data() + row * width + c * taps);
                    }
                }
                std::vector<uint8_t> xbytes;
                if (xtype == GGML_TYPE_F32) {
                    xbytes.assign(reinterpret_cast<const uint8_t *>(patches.data()),
                                  reinterpret_cast<const uint8_t *>(patches.data() + patches.size()));
                } else {
                    xbytes.resize(patches.size() * sizeof(ggml_fp16_t));
                    ggml_fp32_to_fp16_row(patches.data(), reinterpret_cast<ggml_fp16_t *>(xbytes.data()),
                                          patches.size());
                }

                ggml_backend_tensor_set(w, wbytes.data(), 0, wbytes.size());
                ggml_backend_tensor_set(x, xbytes.data(), 0, xbytes.size());
                std::vector<float> output(ggml_nelements(y));
                std::vector<double> device_times, full_times;
                float error = 0;
                constexpr int iterations = 13;
                for (int i = 0; i < iterations; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    if (i > 2) {
                        ggml_backend_tensor_set(x, xbytes.data(), 0, xbytes.size());
                    }
                    const auto after_upload = std::chrono::steady_clock::now();
                    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
                    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
                    const auto end = std::chrono::steady_clock::now();
                    if (i == 0) {
                        for (size_t j = 0; j < output.size(); ++j) {
                            if (!std::isfinite(output[j])) return 1;
                            const float value = output[j] + bias[j % channels];
                            error = std::max(error, std::abs(value - expected[j]));
                        }
                    }
                    if (i > 2) {
                        device_times.push_back(std::chrono::duration<double, std::milli>(end - after_upload).count());
                        full_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                    }
                }
                std::sort(device_times.begin(), device_times.end());
                std::sort(full_times.begin(), full_times.end());
                const int mid = device_times.size() / 2;
                std::printf("%s %-8s %-5s device_ms=%8.3f full_ms=%8.3f max_abs_error=%.7g weight_kib=%zu\n",
                            backend_name, ggml_type_name(wtype), ggml_type_name(xtype),
                            device_times[mid], full_times[mid], error, wbytes.size() / 1024);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
