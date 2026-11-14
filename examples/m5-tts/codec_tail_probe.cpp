#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// Frozen production chunk: conv_32 (dilation 9), Snake, conv_33.
// The input already includes its causal cache. The output stops before the residual add.
int main(int argc, char ** argv) {
    constexpr int channels = 96, rows = 7680, taps = 7, dilation = 9;
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s capture-dir CPU|HTP0 direct|matmul|split|phase-direct|phase-matmul|hvx\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[3];
    const bool phase = mode == "phase-direct" || mode == "phase-matmul";
    const bool split = mode == "split" || mode == "hvx" || phase;
    constexpr int phase_rows = (rows + dilation - 1) / dilation;
    constexpr int phase_input = phase_rows + taps - 1;
    if (!split && mode != "direct" && mode != "matmul" && mode != "split") return 2;
    ggml_backend_load_all();
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_ptr backend(ggml_backend_init_by_name(argv[2], nullptr), ggml_backend_free);
    if (!backend) return 1;
    if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 3);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({48 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}), ggml_free);
    if (!ctx) return 1;

    auto * g = ctx.get();
    auto * x = ggml_new_tensor_4d(g, GGML_TYPE_F32, phase ? phase_input : rows + (taps - 1) * dilation, phase ? dilation : 1, channels, 1);
    auto * w32 = ggml_new_tensor_4d(g, GGML_TYPE_F32, taps, 1, channels, channels);
    auto * w33 = ggml_new_tensor_4d(g, GGML_TYPE_F32, 1, 1, channels, channels);
    auto * b32 = ggml_new_tensor_3d(g, GGML_TYPE_F32, 1, 1, channels);
    auto * b33 = ggml_new_tensor_3d(g, GGML_TYPE_F32, 1, 1, channels);
    auto * alpha = ggml_new_tensor_3d(g, GGML_TYPE_F32, 1, 1, channels);
    auto * beta = ggml_new_tensor_3d(g, GGML_TYPE_F32, 1, 1, channels);
    const auto conv = [&](ggml_tensor * w, ggml_tensor * input, int kernel, int dil) {
        if (mode == "direct" || mode == "phase-direct" || mode == "hvx") return ggml_conv_2d_direct(g, w, input, 1, 1, 0, 0, dil, 1);
        auto * columns = ggml_im2col(g, w, input, 1, 1, 0, 0, dil, 1, true, GGML_TYPE_F32);
        auto * flat = ggml_reshape_2d(g, columns, kernel * channels, phase ? phase_rows * dilation : rows);
        auto * matrix = ggml_reshape_2d(g, w, kernel * channels, channels);
        auto * product = ggml_mul_mat(g, flat, matrix);
        return ggml_reshape_4d(g, product, phase ? phase_rows : rows, phase ? dilation : 1, channels, 1);
    };
    auto * middle = ggml_add(g, conv(w32, x, taps, phase ? 1 : dilation), b32);
    auto * sin = ggml_sin(g, ggml_mul(g, middle, alpha));
    auto * activation = ggml_add(g, middle, ggml_mul(g, ggml_mul(g, sin, sin), beta));
    auto * out = ggml_add(g, conv(w33, activation, 1, 1), b33);
    auto * graph = ggml_new_graph(g);
    ggml_build_forward_expand(graph, split ? middle : out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend.get(), node)) {
            std::fprintf(stderr, "%s does not support %s for this shape\n", argv[2], ggml_op_desc(node));
            return 3;
        }
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(g, backend.get()), ggml_backend_buffer_free);
    if (!buffer) return 1;
    const auto read = [&](const char * name, size_t count) {
        std::vector<float> values(count);
        std::ifstream file(std::string(argv[1]) + "/" + name, std::ios::binary);
        if (!file.read(reinterpret_cast<char *>(values.data()), count * sizeof(float)) ||
            file.peek() != EOF) values.clear();
        return values;
    };
    const std::pair<ggml_tensor *, const char *> constants[] = {
        {w32, "w32"}, {w33, "w33"}, {b32, "b32"}, {b33, "b33"}, {alpha, "alpha"}, {beta, "beta"},
    };
    for (const auto & [tensor, name] : constants) {
        auto values = read(name, ggml_nelements(tensor));
        if (values.empty()) return 2;
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    constexpr int input_rows = rows + (taps - 1) * dilation;
    auto input = read("input", input_rows * channels);
    auto * result = split ? middle : out;
    auto reference = read(split ? "middle" : "reference", rows * channels);
    if (input.empty() || reference.empty()) return 2;
    std::vector<float> phase_input_buffer(phase ? ggml_nelements(x) : 0);
    std::vector<float> values(ggml_nelements(result));
    std::vector<float> interleaved(phase ? reference.size() : 0);
    std::vector<double> times;
    for (int i = 0; i < 17; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (phase) {
            for (int c = 0; c < channels; ++c) {
                for (int t = 0; t < input_rows; ++t) {
                    phase_input_buffer[(c * dilation + t % dilation) * phase_input + t / dilation] =
                        input[c * input_rows + t];
                }
            }
        }
        ggml_backend_tensor_set(x, phase ? phase_input_buffer.data() : input.data(), 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
        ggml_backend_tensor_get(result, values.data(), 0, ggml_nbytes(result));
        if (phase) {
            for (int c = 0; c < channels; ++c) {
                for (int t = 0; t < rows; ++t) {
                    interleaved[c * rows + t] = values[(c * dilation + t % dilation) * phase_rows + t / dilation];
                }
            }
        }
        if (i >= 3) times.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    if (phase) values.swap(interleaved);
    float error = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) return 1;
        error = std::max(error, std::abs(values[i] - reference[i]));
    }
    std::sort(times.begin(), times.end());
    std::printf("%s %s ms=%.3f max_abs_error=%.8g\n", argv[2], argv[3], times[times.size() / 2], error);
}
