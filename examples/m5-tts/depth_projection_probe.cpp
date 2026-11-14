#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "m5_depth_heads.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

static int probe_head(const char * weights_path, const char * hidden_path) {
    using namespace m5::heads;
    std::vector<float> weights(kWidth * kVocab), hidden(kWidth), logits(kVocab);
    std::ifstream weights_file(weights_path, std::ios::binary);
    std::ifstream hidden_file(hidden_path, std::ios::binary);
    if (!weights_file.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(float)) ||
        !hidden_file.read(reinterpret_cast<char *>(hidden.data()), hidden.size() * sizeof(float))) return 2;

    std::vector<int8_t> int8_weights(weights.size());
    std::vector<float> scales(kVocab);
    build_int8(weights.data(), int8_weights.data(), scales.data(), kVocab);
    std::vector<float> reference(kVocab);

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_init_by_name("HTP0", nullptr), ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead() * 20 + ggml_graph_overhead(), nullptr, true}), ggml_free);
    if (!backend || !ctx) return 1;
    auto * graph = ggml_new_graph(ctx.get());
    auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, kWidth, kVocab);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kWidth, 1);
    auto * y = ggml_mul_mat(ctx.get(), w, x);
    auto * bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kVocab);
    auto * log_noise = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kVocab);
    auto * scaled = ggml_add(ctx.get(), ggml_scale(ctx.get(), y, 1.0f / 0.9f), bias);
    auto * picks = ggml_top_k(ctx.get(), scaled, 10);
    // The softmax normalization cancels in the argmax over the top 10.
    auto * scores = ggml_sub(ctx.get(), scaled, log_noise);
    auto * masked = ggml_top_k_mask(ctx.get(), scores, picks, nullptr, nullptr, nullptr);
    auto * selected = ggml_argmax(ctx.get(), masked);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> graph_ctx(
        ggml_init({ggml_graph_overhead() + ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (!graph_ctx) return 1;
    auto * fused = ggml_new_graph(graph_ctx.get());
    ggml_build_forward_expand(fused, selected);
    for (int i = 0; i < ggml_graph_n_nodes(fused); ++i) {
        const auto * node = ggml_graph_node(fused, i);
        if (!ggml_backend_supports_op(backend.get(), node)) {
            std::fprintf(stderr, "HTP does not support %s\n", ggml_op_name(node->op));
            return 1;
        }
    }
    if (!ggml_backend_supports_op(backend.get(), y)) return 1;
    ggml_build_forward_expand(graph, y);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) return 1;
    std::vector<uint8_t> packed(ggml_nbytes(w));
    if (ggml_quantize_chunk(GGML_TYPE_Q8_0, weights.data(), packed.data(), 0, kVocab, kWidth, nullptr) != packed.size()) return 1;
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(x, hidden.data(), 0, ggml_nbytes(x));
    std::vector<float> bias_data(kVocab);
    bias_data[0] = -1e5f;
    ggml_backend_tensor_set(bias, bias_data.data(), 0, ggml_nbytes(bias));

    std::mt19937 rng(1234);
    std::exponential_distribution<float> distribution;
    std::vector<float> noise(kVocab);
    std::vector<double> project_us, read_us, sample_us, cpu_us, fused_us;
    int matches = 0, fused_matches = 0;
    std::vector<float> log_noise_data(kVocab);
    constexpr int trials = 128;
    for (int i = -3; i < trials; ++i) {
        for (auto & n : noise) n = std::max(distribution(rng), 1e-38f);
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
        const auto computed = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(y, logits.data(), 0, ggml_nbytes(y));
        const auto read = std::chrono::steady_clock::now();
        const long code = sample(logits.data(), noise.data(), 0, false);
        const auto sampled = std::chrono::steady_clock::now();
        project8(int8_weights.data(), scales.data(), hidden.data(), reference.data(), kVocab);
        const long expected = sample(reference.data(), noise.data(), 0, false);
        const auto cpu_done = std::chrono::steady_clock::now();
        for (int j = 0; j < kVocab; ++j) log_noise_data[j] = std::log(noise[j]);
        const auto fused_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(log_noise, log_noise_data.data(), 0, ggml_nbytes(log_noise));
        if (ggml_backend_graph_compute(backend.get(), fused) != GGML_STATUS_SUCCESS) return 1;
        int32_t fused_code = -1;
        ggml_backend_tensor_get(selected, &fused_code, 0, sizeof(fused_code));
        const auto fused_done = std::chrono::steady_clock::now();
        if (i < 0) continue;
        matches += code == expected;
        fused_matches += fused_code == code;
        project_us.emplace_back(std::chrono::duration<double, std::micro>(computed - start).count());
        read_us.emplace_back(std::chrono::duration<double, std::micro>(read - computed).count());
        sample_us.emplace_back(std::chrono::duration<double, std::micro>(sampled - read).count());
        cpu_us.emplace_back(std::chrono::duration<double, std::micro>(cpu_done - sampled).count());
        fused_us.emplace_back(std::chrono::duration<double, std::micro>(fused_done - fused_start).count());
    }
    auto median = [](std::vector<double> & values) {
        std::sort(values.begin(), values.end());
        return (values[trials / 2 - 1] + values[trials / 2]) / 2;
    };
    std::printf("HTP head: compute_us=%.1f read_us=%.1f cpu_sample_us=%.1f cpu_head_us=%.1f matches=%d/%d\n",
                median(project_us), median(read_us), median(sample_us), median(cpu_us), matches, trials);
    std::printf("HTP head with on-device sampling: wall_us=%.1f matches_vs_cpu_sampling=%d/%d\n",
                median(fused_us), fused_matches, trials);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 4 && std::string(argv[1]) == "--head") {
        ggml_backend_load_all();
        return probe_head(argv[2], argv[3]);
    }
    constexpr int width = 512, projected = 2048;
    if (argc != 2) return 2;
    std::vector<float> weights(width * projected * 2), input(width);
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(float)) ||
        !file.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float)) ||
        file.peek() != EOF) return 2;

    ggml_backend_load_all();
    std::vector<float> reference;
    for (const char * name : {"CPU", "HTP0"}) {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) return 1;
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 4);

        for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
            for (int groups : {2, 1}) {
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                    ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true}), ggml_free);
                if (!ctx) return 1;
                auto * graph = ggml_new_graph(ctx.get());
                auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
                std::vector<ggml_tensor *> w, y;
                const int outputs = projected * 2 / groups;
                for (int i = 0; i < groups; ++i) {
                    w.push_back(ggml_new_tensor_2d(ctx.get(), type, width, outputs));
                    y.push_back(ggml_mul_mat(ctx.get(), w.back(), x));
                    if (!ggml_backend_supports_op(backend.get(), y.back())) return 1;
                    ggml_build_forward_expand(graph, y.back());
                }
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
                if (!buffer) return 1;
                for (int i = 0; i < groups; ++i) {
                    const float * source = weights.data() + i * width * outputs;
                    if (type == GGML_TYPE_F32) {
                        ggml_backend_tensor_set(w[i], source, 0, ggml_nbytes(w[i]));
                    } else {
                        std::vector<uint8_t> bytes(ggml_nbytes(w[i]));
                        if (ggml_quantize_chunk(type, source, bytes.data(), 0, outputs, width, nullptr) != bytes.size()) return 1;
                        ggml_backend_tensor_set(w[i], bytes.data(), 0, bytes.size());
                    }
                }
                std::vector<float> actual(projected * 2);
                std::vector<double> elapsed;
                for (int i = 0; i < 35; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
                    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
                    for (int j = 0; j < groups; ++j) {
                        ggml_backend_tensor_get(y[j], actual.data() + j * outputs, 0, ggml_nbytes(y[j]));
                    }
                    if (i >= 3) elapsed.push_back(std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start).count());
                }
                if (reference.empty()) reference = actual;
                float error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i])) return 1;
                    error = std::max(error, std::abs(actual[i] - reference[i]));
                }
                std::sort(elapsed.begin(), elapsed.end());
                std::printf("%s %s groups=%d median_us=%.1f max_error=%.7g\n",
                    name, ggml_type_name(type), groups, (elapsed[15] + elapsed[16]) / 2, error);
            }
        }
    }
}
