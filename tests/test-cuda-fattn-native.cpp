#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

// Fixed integer inputs make the stock and port binaries consume identical bytes.
static void fill_tensor(ggml_tensor * tensor, uint32_t seed) {
    const size_t count = ggml_nelements(tensor);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = float(int((i*17+seed*13)%257)-128)/256.0f;
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
    } else {
        std::vector<uint8_t> encoded(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, values.data(), encoded.data(), 0,
                ggml_nrows(tensor), tensor->ne[0], nullptr);
        ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    }
}

// Exercise stock allocation and dispatch, including conversion extras beside the output.
static std::vector<float> attention_case(ggml_backend_t backend, ggml_type key, ggml_type value,
        int queries, int tokens, bool strided, int window = 0) {
    constexpr int dim = 256, heads = 24, kv_heads = 4;
    ggml_context_ptr context(ggml_init({2*1024*1024, nullptr, true}));
    if (!context) throw std::runtime_error("failed to create fixture context");
    auto * q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, dim, queries, heads);
    auto * k_base = ggml_new_tensor_3d(context.get(), key, dim, tokens*(strided ? 2 : 1), kv_heads);
    auto * v_base = ggml_new_tensor_3d(context.get(), value, dim, tokens*(strided ? 2 : 1), kv_heads);
    auto * k = strided ? ggml_view_3d(context.get(), k_base, dim, tokens, kv_heads, k_base->nb[1], k_base->nb[2], 0) : k_base;
    auto * v = strided ? ggml_view_3d(context.get(), v_base, dim, tokens, kv_heads, v_base->nb[1], v_base->nb[2], 0) : v_base;
    auto * mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, tokens, queries);
    auto * output = ggml_flash_attn_ext(context.get(), q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(output, window);
    ggml_set_name(output, "stock_attention_output");
    ggml_set_output(output);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("required attention fixture is unsupported");
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) throw std::runtime_error("failed to allocate fixture tensors");
    ggml_backend_buffer_clear(buffer.get(), 0);
    fill_tensor(q, 1);
    fill_tensor(k_base, 2);
    fill_tensor(v_base, 3);
    std::vector<ggml_fp16_t> masks(size_t(tokens)*queries);
    for (int query = 0; query < queries; ++query) {
        for (int row = 0; row < tokens; ++row) {
            const int end = tokens-queries+query;
            masks[size_t(query)*tokens+row] = ggml_fp32_to_fp16(
                row <= end && (!window || row > end-window) ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
    std::vector<float> result(ggml_nelements(output));
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("fixture compute failed");
        ggml_backend_synchronize(backend);
        std::vector<float> actual(result.size());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        if (!std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("nonfinite attention output");
        }
        if (repeat && std::memcmp(result.data(), actual.data(), result.size()*sizeof(float))) {
            throw std::runtime_error("repeated identical execution changed attention bits");
        }
        result = std::move(actual);
    }
    return result;
}

// Snapshot files are local test artifacts; reject incompatible or truncated records.
static void snapshot_case(std::fstream & file, bool writing, const uint32_t (&shape)[7], const std::vector<float> & output) {
    const size_t bytes = output.size()*sizeof(float);
    if (writing) {
        file.write(reinterpret_cast<const char *>(shape), sizeof(shape));
        file.write(reinterpret_cast<const char *>(output.data()), bytes);
        if (!file) throw std::runtime_error("failed to write fixture snapshot");
        return;
    }
    uint32_t expected_shape[7];
    file.read(reinterpret_cast<char *>(expected_shape), sizeof(expected_shape));
    if (!file || std::memcmp(shape, expected_shape, sizeof(shape))) throw std::runtime_error("snapshot shape mismatch or truncation");
    std::vector<float> expected(output.size());
    file.read(reinterpret_cast<char *>(expected.data()), bytes);
    if (!file) throw std::runtime_error("truncated snapshot values");
    size_t mismatches = 0;
    double max_abs = 0.0, squared_error = 0.0, squared_reference = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(expected[i])) throw std::runtime_error("nonfinite snapshot value");
        const double error = double(output[i])-expected[i];
        max_abs = std::max(max_abs, std::abs(error));
        squared_error += error*error;
        squared_reference += double(expected[i])*expected[i];
        mismatches += std::memcmp(&output[i], &expected[i], sizeof(float)) != 0;
    }
    if (mismatches) {
        std::fprintf(stderr, "snapshot drift: mismatches=%zu max_abs=%.9g relative_l2=%.9g\n",
                mismatches, max_abs, std::sqrt(squared_error/std::max(squared_reference, 1e-30)));
        throw std::runtime_error("stock attention output bits changed");
    }
}

int main(int argc, char ** argv) {
    const char * name = "CUDA0";
    const char * path = nullptr;
    bool writing = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--backend") && i+1 < argc) name = argv[++i];
        else if ((!std::strcmp(argv[i], "--write") || !std::strcmp(argv[i], "--compare")) && i+1 < argc && !path) {
            writing = !std::strcmp(argv[i], "--write");
            path = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--backend NAME] [--write FILE | --compare FILE]\n", argv[0]);
            return 1;
        }
    }
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name(name, nullptr));
    if (!backend) { std::fprintf(stderr, "backend %s is unavailable\n", name); return 77; }
    std::fstream snapshot;
    if (path) {
        snapshot.open(path, std::ios::binary | (writing ? std::ios::out | std::ios::trunc : std::ios::in));
        if (!snapshot) { std::fprintf(stderr, "cannot open snapshot %s\n", path); return 1; }
    }
    size_t cases = 0;
    try {
        for (bool quantized : {false, true}) for (bool strided : {false, true}) {
            for (int queries : {1, 2, 3, 4}) for (int tokens : {256, 512, 1024, 8192}) {
                const auto key = quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
                const auto value = quantized ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
                const uint32_t shape[7] = {256, uint32_t(key), uint32_t(value), uint32_t(queries), uint32_t(tokens), uint32_t(strided), 0};
                const auto output = attention_case(backend.get(), key, value, queries, tokens, strided);
                if (path) snapshot_case(snapshot, writing, shape, output);
                ++cases;
                std::printf("native attention %zu: TG%d KV=%d K=%s V=%s strided=%d PASS\n",
                        cases, queries, tokens, ggml_type_name(key), ggml_type_name(value), int(strided));
            }
        }
        for (bool strided : {false, true}) for (int queries : {1, 4, 64}) {
            const uint32_t shape[7] = {256, GGML_TYPE_F16, GGML_TYPE_F16, uint32_t(queries), 4096, uint32_t(strided), 512};
            const auto output = attention_case(backend.get(), GGML_TYPE_F16, GGML_TYPE_F16, queries, 4096, strided, 512);
            if (path) snapshot_case(snapshot, writing, shape, output);
            ++cases;
            std::printf("native attention %zu: TG%d KV=4096 sparse_window=512 strided=%d PASS\n", cases, queries, int(strided));
        }
        if (path && !writing && snapshot.peek() != std::char_traits<char>::eof()) throw std::runtime_error("extra snapshot records");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native attention failed after %zu cases: %s\n", cases, error.what());
        return 1;
    }
    std::printf("native attention control: %zu cases passed\n", cases);
}
