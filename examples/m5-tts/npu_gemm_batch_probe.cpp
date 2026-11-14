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
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

// One M5-TTS depth frame invokes 692 MatMuls (5 layers x 7 projections x
// ~16 codebook steps plus heads). This probe builds that many small GEMVs in a
// single GGML graph to measure whether HTP op batching beats the CPU ORT path.

struct shape_t { int k, m; };

static std::vector<uint8_t> make_weights(enum ggml_type type, const std::vector<float> & src,
                                         int m, int k) {
    std::vector<uint8_t> out;
    if (type == GGML_TYPE_F32) {
        out.resize(src.size() * sizeof(float));
        std::memcpy(out.data(), src.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        out.resize(src.size() * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(src.data(), reinterpret_cast<ggml_fp16_t *>(out.data()), src.size());
    } else {
        if (k % ggml_blck_size(type) != 0) {
            throw std::runtime_error("k is not a multiple of the quant block size");
        }
        out.resize(ggml_row_size(type, k) * m);
        if (ggml_quantize_chunk(type, src.data(), out.data(), 0, m, k, nullptr) != out.size()) {
            throw std::runtime_error("unexpected quantized weight size");
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    const int n_ops = argc > 1 ? std::atoi(argv[1]) : 692;
    if (n_ops <= 0 || n_ops > 2048) return 2;

    const shape_t shapes[] = {
        {512, 512}, {512, 128}, {512, 128}, {512, 512},
        {512, 2048}, {512, 2048}, {2048, 512},
    };
    constexpr int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    ggml_backend_load_all();
    for (const char * name : {"CPU", "HTP0"}) {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) return 1;
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 4);

        for (auto wtype : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            const size_t graph_size = 3 * n_ops + 64;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
                           nullptr, true}), ggml_free);
            if (!ctx) return 1;
            auto * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);

            std::vector<std::vector<uint8_t>> wbytes(n_shapes);
            std::vector<ggml_tensor *> w(n_shapes);
            for (int s = 0; s < n_shapes; ++s) {
                std::vector<float> data(size_t(shapes[s].k) * shapes[s].m);
                for (auto & v : data) v = dist(rng);
                try {
                    wbytes[s] = make_weights(wtype, data, shapes[s].m, shapes[s].k);
                } catch (const std::exception &) {
                    return 2;
                }
                w[s] = ggml_new_tensor_2d(ctx.get(), wtype, shapes[s].k, shapes[s].m);
            }

            std::vector<std::vector<float>> xdata(n_ops);
            std::vector<ggml_tensor *> xs(n_ops);
            ggml_tensor * last = nullptr;
            for (int i = 0; i < n_ops; ++i) {
                const int s = i % n_shapes;
                xdata[i].resize(shapes[s].k);
                for (auto & v : xdata[i]) v = dist(rng);
                auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, shapes[s].k, 1);
                xs[i] = x;
                last = ggml_mul_mat(ctx.get(), w[s], x);
                if (!ggml_backend_supports_op(backend.get(), last)) {
                    std::printf("%s %-6s SKIP (unsupported op)\n", name, ggml_type_name(wtype));
                    return 0;
                }
                ggml_build_forward_expand(graph, last);
            }
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer) return 1;

            for (int s = 0; s < n_shapes; ++s) {
                ggml_backend_tensor_set(w[s], wbytes[s].data(), 0, wbytes[s].size());
            }
            for (int i = 0; i < n_ops; ++i) {
                ggml_backend_tensor_set(xs[i], xdata[i].data(), 0, xdata[i].size() * sizeof(float));
            }

            std::vector<float> result(ggml_nelements(last));
            std::vector<double> times;
            for (int i = 0; i < 9; ++i) {
                const auto start = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
                ggml_backend_tensor_get(last, result.data(), 0, result.size() * sizeof(float));
                const auto end = std::chrono::steady_clock::now();
                if (i >= 2) times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
            std::sort(times.begin(), times.end());
            const double median = times[times.size() / 2];
            std::printf("%s %-6s ops=%d total_ms=%9.3f us_per_op=%8.2f finite=%d\n",
                        name, ggml_type_name(wtype), n_ops, median, median * 1000.0 / n_ops,
                        std::isfinite(result[0]) ? 1 : 0);
            std::fflush(stdout);
        }
    }
    return 0;
}
