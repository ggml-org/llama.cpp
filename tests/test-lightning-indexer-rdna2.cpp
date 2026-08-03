// Deterministic HIP backend differential fixture for the DeepSeek-V4
// lightning indexer. Run once with the reference dispatcher to dump output,
// then with GGML_HIP_RDNA2_LID_SUBWAVE=4 to require bitwise equality.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

enum class expected_path : uint32_t { unspecified = 0, reference = 1, subwave4 = 2, fallback = 3 };

struct params {
    int64_t kv = 256;
    int64_t batch = 256;
    int warmup = 3;
    int iterations = 10;
    const char * device_name = nullptr;
    const char * dump_output = nullptr;
    const char * compare_output = nullptr;
    bool disable_graphs = false;
    expected_path path = expected_path::unspecified;
};

struct output_file_header {
    uint64_t magic = 0x4c494452444e4132ULL; // "LIDRDNA2"
    uint32_t version = 1;
    uint32_t producer_path = 0;
    uint64_t kv = 0;
    uint64_t batch = 0;
    uint64_t elements = 0;
};
static_assert(sizeof(output_file_header) == 40, "unexpected output header layout");

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(1);
}

int64_t parse_i64(const char * name, const char * value, int64_t minimum, int64_t maximum) {
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "error: %s must be in [%lld, %lld]\n", name,
                static_cast<long long>(minimum), static_cast<long long>(maximum));
        std::exit(1);
    }
    return parsed;
}

params parse_args(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const auto value = [&](const char * option) {
            if (++i >= argc) {
                std::fprintf(stderr, "error: missing value for %s\n", option);
                std::exit(1);
            }
            return argv[i];
        };
        if (std::strcmp(arg, "--kv") == 0) {
            p.kv = parse_i64(arg, value(arg), 1, 1LL << 20);
        } else if (std::strcmp(arg, "--batch") == 0) {
            p.batch = parse_i64(arg, value(arg), 1, 1LL << 20);
        } else if (std::strcmp(arg, "--warmup") == 0) {
            p.warmup = static_cast<int>(parse_i64(arg, value(arg), 0, 1000000));
        } else if (std::strcmp(arg, "--iterations") == 0) {
            p.iterations = static_cast<int>(parse_i64(arg, value(arg), 1, 1000000));
        } else if (std::strcmp(arg, "--device") == 0) {
            p.device_name = value(arg);
        } else if (std::strcmp(arg, "--dump-output") == 0) {
            p.dump_output = value(arg);
        } else if (std::strcmp(arg, "--compare-output") == 0) {
            p.compare_output = value(arg);
        } else if (std::strcmp(arg, "--expect-path") == 0) {
            const char * path = value(arg);
            if (std::strcmp(path, "reference") == 0) p.path = expected_path::reference;
            else if (std::strcmp(path, "subwave4") == 0) p.path = expected_path::subwave4;
            else if (std::strcmp(path, "fallback") == 0) p.path = expected_path::fallback;
            else fail("--expect-path must be reference, subwave4, or fallback");
        } else if (std::strcmp(arg, "--disable-graphs") == 0) {
            p.disable_graphs = true;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            std::printf("usage: %s [--kv N] [--batch N] [--warmup N] [--iterations N] "
                        "[--dump-output FILE] [--compare-output FILE] [--expect-path reference|subwave4|fallback] [--disable-graphs]\n", argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown option: %s\n", arg);
            std::exit(1);
        }
    }
    return p;
}

uint32_t hash_u32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

float deterministic_value(size_t index, uint32_t seed, float scale) {
    const uint32_t bits = hash_u32(static_cast<uint32_t>(index) ^ seed);
    const int32_t centered = static_cast<int32_t>(bits & 0xffffU) - 32768;
    return static_cast<float>(centered) * (scale / 32768.0f);
}

void fill_f32(std::vector<float> & values, uint32_t seed, float scale) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = deterministic_value(i, seed, scale);
    }
}

std::vector<ggml_fp16_t> make_f16(size_t elements, uint32_t seed, float scale) {
    std::vector<float> f32(elements);
    fill_f32(f32, seed, scale);
    std::vector<ggml_fp16_t> f16(elements);
    ggml_fp32_to_fp16_row(f32.data(), f16.data(), static_cast<int64_t>(elements));
    return f16;
}

uint64_t fnv1a(const std::vector<float> & values) {
    uint64_t hash = 1469598103934665603ULL;
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

output_file_header make_header(const params & p, size_t elements, expected_path producer) {
    output_file_header h;
    h.producer_path = static_cast<uint32_t>(producer);
    h.kv = static_cast<uint64_t>(p.kv);
    h.batch = static_cast<uint64_t>(p.batch);
    h.elements = static_cast<uint64_t>(elements);
    return h;
}

void dump_output(const char * path, const params & p, const std::vector<float> & output) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const output_file_header header = make_header(p, output.size(), p.path);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) fail("unable to write output dump");
}

std::vector<float> load_output(const char * path, const params & p, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamsize expected = static_cast<std::streamsize>(sizeof(output_file_header) + elements * sizeof(float));
    if (!file || file.tellg() != expected) fail("baseline output size mismatch");
    file.seekg(0);
    output_file_header header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    const output_file_header wanted = make_header(p, elements, expected_path::reference);
    if (!file || std::memcmp(&header, &wanted, sizeof(header)) != 0) fail("baseline output header mismatch");
    std::vector<float> output(elements);
    file.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) fail("unable to read baseline output");
    return output;
}

ggml_backend_dev_t select_rocm_device(const params & p) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "ROCm") != 0) continue;
        if (p.device_name && std::strcmp(p.device_name, ggml_backend_dev_name(device)) != 0) continue;
        return device;
    }
    fail("no matching ROCm device");
}

void run_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) fail("graph execution failed");
}

} // namespace

int main(int argc, char ** argv) {
    const params p = parse_args(argc, argv);
    const char * override_value = std::getenv("GGML_HIP_RDNA2_LID_SUBWAVE");
    const bool reference_env = !override_value || !override_value[0] || std::strcmp(override_value, "0") == 0;
    const bool subwave4_env = override_value && std::strcmp(override_value, "4") == 0;
    if ((p.dump_output || p.compare_output) && p.path == expected_path::unspecified) {
        fail("--dump-output/--compare-output requires --expect-path");
    }
    if (p.path == expected_path::reference && !reference_env) fail("reference expectation requires override unset or 0");
    if (p.path == expected_path::subwave4 && (!subwave4_env || p.batch != 256 || p.kv < 1 || p.kv > 4096)) {
        fail("subwave4 expectation requires override 4, batch 256, and KV in [1,4096]");
    }
    if (p.path == expected_path::fallback && (!subwave4_env || (p.batch == 256 && p.kv >= 1 && p.kv <= 4096))) {
        fail("fallback expectation requires override 4 and a shape outside the candidate guard");
    }
    if (p.dump_output && p.path != expected_path::reference) fail("baseline dumps must come from the reference path");
    if (p.compare_output && p.path != expected_path::subwave4 && p.path != expected_path::reference && p.path != expected_path::fallback) fail("invalid comparison path");
    if (p.disable_graphs) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);

    ggml_backend_load_all();
    ggml_backend_dev_t device = select_rocm_device(p);
    if (p.path == expected_path::subwave4 && std::strcmp(ggml_backend_dev_description(device), "AMD Radeon Pro V620") != 0) {
        fail("subwave4 expectation requires the validated gfx1030 Radeon Pro V620 target");
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) fail("backend initialization failed");
    std::printf("backend: %s (%s) expected_path=%u\n", ggml_backend_name(backend.get()),
            ggml_backend_dev_description(device), static_cast<unsigned>(p.path));

    constexpr int64_t n_embd = 128;
    constexpr int64_t n_head = 64;
    const size_t q_elements = static_cast<size_t>(n_embd * n_head * p.batch);
    const size_t k_elements = static_cast<size_t>(n_embd * p.kv);
    const size_t w_elements = static_cast<size_t>(n_head * p.batch);
    const size_t m_elements = static_cast<size_t>(p.kv * p.batch);
    const size_t output_elements = m_elements;
    std::vector<float> q(q_elements), w(w_elements);
    fill_f32(q, 0x13579bdfU, 0.125f);
    fill_f32(w, 0x2468ace0U, 0.25f);
    const std::vector<ggml_fp16_t> k = make_f16(k_elements, 0x10293847U, 0.125f);
    const std::vector<ggml_fp16_t> m = make_f16(m_elements, 0xabcdef01U, 0.5f);

    const ggml_init_params init_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 12 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(init_params));
    if (!ctx) fail("GGML context initialization failed");
    ggml_tensor * q_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_embd, n_head, p.batch, 1);
    ggml_tensor * k_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_embd, 1, p.kv, 1);
    ggml_tensor * w_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_head, p.batch, 1, 1);
    ggml_tensor * m_t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, p.kv, p.batch, 1, 1);
    ggml_tensor * output = ggml_lightning_indexer(ctx.get(), q_t, k_t, w_t, m_t);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend.get(), output)) fail("backend does not support LID graph");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) fail("backend tensor allocation failed");
    ggml_backend_tensor_set(q_t, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(k_t, k.data(), 0, k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(w_t, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(m_t, m.data(), 0, m.size() * sizeof(ggml_fp16_t));

    for (int i = 0; i < p.warmup; ++i) run_graph(backend.get(), graph);
    ggml_backend_synchronize(backend.get());
    const int64_t start_us = ggml_time_us();
    for (int i = 0; i < p.iterations; ++i) run_graph(backend.get(), graph);
    ggml_backend_synchronize(backend.get());
    const int64_t elapsed_us = ggml_time_us() - start_us;

    std::vector<float> result(output_elements);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    size_t nonfinite = 0;
    for (float value : result) nonfinite += !std::isfinite(value);
    if (nonfinite) fail("nonfinite output");
    std::printf("LID case: kv=%lld batch=%lld warmup=%d iterations=%d override=%s\n",
            static_cast<long long>(p.kv), static_cast<long long>(p.batch), p.warmup, p.iterations,
            std::getenv("GGML_HIP_RDNA2_LID_SUBWAVE") ? std::getenv("GGML_HIP_RDNA2_LID_SUBWAVE") : "unset");
    std::printf("time: total=%lld us avg=%.2f us\n", static_cast<long long>(elapsed_us),
            static_cast<double>(elapsed_us) / p.iterations);
    std::printf("output: elements=%zu fnv1a=%016llx\n", result.size(), static_cast<unsigned long long>(fnv1a(result)));

    if (p.compare_output) {
        const std::vector<float> baseline = load_output(p.compare_output, p, result.size());
        size_t mismatches = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < result.size(); ++i) {
            mismatches += std::memcmp(&result[i], &baseline[i], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::fabs(result[i] - baseline[i]));
        }
        std::printf("A/B bitwise: mismatches=%zu/%zu max_abs=%g\n", mismatches, result.size(), max_abs);
        if (mismatches) return 2;
    }
    if (p.dump_output) {
        dump_output(p.dump_output, p, result);
        std::printf("wrote output: %s\n", p.dump_output);
    }
    return 0;
}