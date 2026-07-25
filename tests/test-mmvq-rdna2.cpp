// Synthetic HIP backend benchmark for quantized MMVQ and MMQ paths.
//
// This intentionally builds a normal GGML_OP_MUL_MAT graph rather than calling
// an internal kernel. It therefore includes the production dispatcher and the
// F32 -> Q8_1 activation staging used by either selected kernel.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct params {
    ggml_type type = GGML_TYPE_Q4_0;
    int64_t k = 4096;
    int64_t n = 4096;
    int64_t batch = 1;
    int warmup = 10;
    int iterations = 50;
    const char * device_name = nullptr;
    const char * dump_output = nullptr;
    const char * compare_output = nullptr;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    bool disable_graphs = false;
};

struct output_file_header {
    uint64_t magic = 0x4D4D565152444E32ULL; // "MMVQRDN2"
    uint32_t version = 1;
    uint32_t type = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t batch = 0;
    uint64_t elements = 0;
};

static_assert(sizeof(output_file_header) == 48, "unexpected output file header layout");

struct output_stats {
    uint64_t checksum = 1469598103934665603ULL;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double squared_error = 0.0;
    size_t mismatches = 0;
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(1);
}

void usage(const char * program) {
    std::printf("Synthetic quantized MMVQ benchmark using the normal HIP GGML backend graph.\n\n");
    std::printf("usage: %s [options]\n\n", program);
    std::printf("  --type TYPE             q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k (q4_0)\n");
    std::printf("  --k N                   input width; must satisfy the quant block size (4096)\n");
    std::printf("  --n N                   output rows (4096)\n");
    std::printf("  --batch N               activation columns, 1..256; 1..8 normally use MMVQ (1)\n");
    std::printf("  --warmup N              warmup graph executions (10)\n");
    std::printf("  --iterations N          timed graph executions (50)\n");
    std::printf("  --device NAME           exact GGML GPU device name; defaults to the first GPU\n");
    std::printf("  --disable-graphs         set GGML_CUDA_DISABLE_GRAPHS before backend initialization\n");
    std::printf("  --dump-output FILE      write deterministic F32 output for a later A/B comparison\n");
    std::printf("  --compare-output FILE   compare output against a baseline written by --dump-output\n");
    std::printf("  --atol X                absolute A/B tolerance (1e-3)\n");
    std::printf("  --rtol X                relative A/B tolerance (1e-4)\n");
    std::printf("  -h, --help              show this help\n");
}

bool parse_type(const char * value, ggml_type & type) {
    struct type_name {
        const char * name;
        ggml_type type;
    };

    static constexpr type_name supported[] = {
        { "q4_0", GGML_TYPE_Q4_0 },
        { "q4_1", GGML_TYPE_Q4_1 },
        { "q5_0", GGML_TYPE_Q5_0 },
        { "q5_1", GGML_TYPE_Q5_1 },
        { "q8_0", GGML_TYPE_Q8_0 },
        { "q2_k", GGML_TYPE_Q2_K },
        { "q3_k", GGML_TYPE_Q3_K },
        { "q4_k", GGML_TYPE_Q4_K },
        { "q5_k", GGML_TYPE_Q5_K },
        { "q6_k", GGML_TYPE_Q6_K },
    };

    for (const type_name & candidate : supported) {
        if (std::strcmp(value, candidate.name) == 0) {
            type = candidate.type;
            return true;
        }
    }
    return false;
}

int64_t parse_positive_i64(const char * name, const char * value, int64_t minimum, int64_t maximum) {
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "error: %s must be in [%lld, %lld]\n", name,
                     static_cast<long long>(minimum), static_cast<long long>(maximum));
        std::exit(1);
    }
    return parsed;
}

float parse_positive_float(const char * name, const char * value) {
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !(parsed >= 0.0f) || !std::isfinite(parsed)) {
        std::fprintf(stderr, "error: %s must be a finite non-negative number\n", name);
        std::exit(1);
    }
    return parsed;
}

params parse_args(int argc, char ** argv) {
    params result;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const auto require_value = [&](const char * option) -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "error: missing value for %s\n", option);
                std::exit(1);
            }
            return argv[i];
        };

        if (std::strcmp(arg, "--type") == 0) {
            const char * value = require_value(arg);
            if (!parse_type(value, result.type)) {
                std::fprintf(stderr, "error: unsupported type: %s\n", value);
                std::exit(1);
            }
        } else if (std::strcmp(arg, "--k") == 0) {
            result.k = parse_positive_i64(arg, require_value(arg), 32, 1LL << 30);
        } else if (std::strcmp(arg, "--n") == 0) {
            result.n = parse_positive_i64(arg, require_value(arg), 1, 1LL << 30);
        } else if (std::strcmp(arg, "--batch") == 0) {
            result.batch = parse_positive_i64(arg, require_value(arg), 1, 256);
        } else if (std::strcmp(arg, "--warmup") == 0) {
            result.warmup = static_cast<int>(parse_positive_i64(arg, require_value(arg), 0, 1000000));
        } else if (std::strcmp(arg, "--iterations") == 0) {
            result.iterations = static_cast<int>(parse_positive_i64(arg, require_value(arg), 1, 1000000));
        } else if (std::strcmp(arg, "--device") == 0) {
            result.device_name = require_value(arg);
        } else if (std::strcmp(arg, "--dump-output") == 0) {
            result.dump_output = require_value(arg);
        } else if (std::strcmp(arg, "--compare-output") == 0) {
            result.compare_output = require_value(arg);
        } else if (std::strcmp(arg, "--atol") == 0) {
            result.atol = parse_positive_float(arg, require_value(arg));
        } else if (std::strcmp(arg, "--rtol") == 0) {
            result.rtol = parse_positive_float(arg, require_value(arg));
        } else if (std::strcmp(arg, "--disable-graphs") == 0) {
            result.disable_graphs = true;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown option: %s\n", arg);
            usage(argv[0]);
            std::exit(1);
        }
    }

    if (result.k % ggml_blck_size(result.type) != 0) {
        std::fprintf(stderr, "error: k=%lld is not divisible by %lld for %s\n",
                     static_cast<long long>(result.k), static_cast<long long>(ggml_blck_size(result.type)), ggml_type_name(result.type));
        std::exit(1);
    }

    return result;
}

float deterministic_value(size_t index, float phase) {
    return 0.70f * std::sin(0.013f * static_cast<float>(index) + phase)
         + 0.30f * std::cos(0.007f * static_cast<float>(index) - phase);
}

void fill_weights(std::vector<float> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = deterministic_value(i, 0.25f);
    }
}

void fill_activations(std::vector<float> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = deterministic_value(i, 1.50f);
    }
}

ggml_backend_dev_t select_rocm_device(const params & p) {
    constexpr const char * expected_registry = "ROCm";

    ggml_backend_dev_t selected = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        if (std::strcmp(ggml_backend_reg_name(registry), expected_registry) != 0) {
            continue;
        }
        if (p.device_name && std::strcmp(p.device_name, ggml_backend_dev_name(device)) != 0) {
            continue;
        }
        selected = device;
        break;
    }

    if (!selected) {
        if (p.device_name) {
            std::fprintf(stderr, "error: no ROCm GPU backend device named '%s'\n", p.device_name);
        } else {
            std::fprintf(stderr, "error: no ROCm GPU backend device found; ensure the HIP backend is discoverable\n");
        }
        std::exit(1);
    }
    return selected;
}

void run_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "error: graph execution failed: %s\n", ggml_status_to_string(status));
        std::exit(1);
    }
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

output_file_header make_output_header(const params & p, size_t elements) {
    output_file_header header;
    header.type = static_cast<uint32_t>(p.type);
    header.k = static_cast<uint64_t>(p.k);
    header.n = static_cast<uint64_t>(p.n);
    header.batch = static_cast<uint64_t>(p.batch);
    header.elements = static_cast<uint64_t>(elements);
    return header;
}

void dump_output(const char * path, const params & p, const std::vector<float> & output) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::fprintf(stderr, "error: cannot open '%s' for output\n", path);
        std::exit(1);
    }

    const output_file_header header = make_output_header(p, output.size());
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) {
        std::fprintf(stderr, "error: unable to write baseline output '%s'\n", path);
        std::exit(1);
    }
}

std::vector<float> load_output(const char * path, const params & p, size_t expected_elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "error: cannot open baseline output '%s'\n", path);
        std::exit(1);
    }

    const std::streamsize bytes = file.tellg();
    const std::streamsize expected_bytes = static_cast<std::streamsize>(sizeof(output_file_header) + expected_elements * sizeof(float));
    if (bytes != expected_bytes) {
        std::fprintf(stderr, "error: baseline '%s' is %lld bytes; expected %lld bytes\n", path,
                     static_cast<long long>(bytes), static_cast<long long>(expected_bytes));
        std::exit(1);
    }

    file.seekg(0);
    output_file_header header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    const output_file_header expected = make_output_header(p, expected_elements);
    if (!file || header.magic != expected.magic || header.version != expected.version || header.type != expected.type ||
        header.k != expected.k || header.n != expected.n || header.batch != expected.batch || header.elements != expected.elements) {
        std::fprintf(stderr, "error: baseline '%s' was generated for a different MMVQ case\n", path);
        std::exit(1);
    }

    std::vector<float> output(expected_elements);
    file.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) {
        std::fprintf(stderr, "error: unable to read baseline output '%s'\n", path);
        std::exit(1);
    }
    return output;
}

output_stats compare_output(const std::vector<float> & output, const std::vector<float> * baseline, float atol, float rtol) {
    output_stats result;
    result.checksum = fnv1a(output);

    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr, "error: non-finite output at element %zu\n", i);
            std::exit(1);
        }
        if (!baseline) {
            continue;
        }

        const float abs_error = std::fabs(output[i] - (*baseline)[i]);
        const float relative_error = abs_error / std::max(std::fabs((*baseline)[i]), 1e-12f);
        result.max_abs = std::max(result.max_abs, abs_error);
        result.max_rel = std::max(result.max_rel, relative_error);
        result.squared_error += static_cast<double>(abs_error) * abs_error;
        if (abs_error > atol + rtol * std::fabs((*baseline)[i])) {
            ++result.mismatches;
        }
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    const params p = parse_args(argc, argv);

    if (p.disable_graphs) {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = select_rocm_device(p);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) {
        fail("failed to initialize the selected GPU backend");
    }

    std::printf("backend: %s (%s)\n", ggml_backend_name(backend.get()), ggml_backend_dev_description(device));
    std::printf("quantized matmul case: type=%s K=%lld N=%lld batch=%lld warmup=%d iterations=%d\n",
                ggml_type_name(p.type), static_cast<long long>(p.k), static_cast<long long>(p.n),
                static_cast<long long>(p.batch), p.warmup, p.iterations);

    const size_t weight_elements = static_cast<size_t>(p.k * p.n);
    const size_t activation_elements = static_cast<size_t>(p.k * p.batch);
    const size_t output_elements = static_cast<size_t>(p.n * p.batch);
    const size_t packed_weight_bytes = ggml_row_size(p.type, p.k) * static_cast<size_t>(p.n);

    std::vector<float> weight_f32(weight_elements);
    std::vector<float> activation_f32(activation_elements);
    std::vector<uint8_t> weight_packed(packed_weight_bytes);
    fill_weights(weight_f32);
    fill_activations(activation_f32);
    ggml_quantize_chunk(p.type, weight_f32.data(), weight_packed.data(), 0, p.n, p.k, nullptr);
    weight_f32.clear();
    weight_f32.shrink_to_fit();

    const ggml_init_params init_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_weights(ggml_init(init_params));
    ggml_context_ptr ctx(ggml_init(init_params));
    if (!ctx_weights || !ctx) {
        fail("failed to create GGML contexts");
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx_weights.get(), p.type, p.k, p.n);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, p.k, p.batch);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, activation);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    if (!ggml_backend_supports_op(backend.get(), output)) {
        fail("selected backend does not support this MUL_MAT case");
    }

    ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(ctx_weights.get(), backend.get()));
    ggml_backend_buffer_ptr compute_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!weight_buffer || !compute_buffer) {
        fail("failed to allocate backend tensors");
    }
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_set(weights, weight_packed.data(), 0, weight_packed.size());
    ggml_backend_tensor_set(activation, activation_f32.data(), 0, activation_f32.size() * sizeof(float));

    for (int i = 0; i < p.warmup; ++i) {
        run_graph(backend.get(), graph);
    }
    ggml_backend_synchronize(backend.get());

    const int64_t start_us = ggml_time_us();
    for (int i = 0; i < p.iterations; ++i) {
        run_graph(backend.get(), graph);
    }
    ggml_backend_synchronize(backend.get());
    const int64_t elapsed_us = ggml_time_us() - start_us;

    std::vector<float> output_f32(output_elements);
    ggml_backend_tensor_get(output, output_f32.data(), 0, output_f32.size() * sizeof(float));

    const std::vector<float> baseline = p.compare_output
        ? load_output(p.compare_output, p, output_elements)
        : std::vector<float>();
    const output_stats stats = compare_output(output_f32, p.compare_output ? &baseline : nullptr, p.atol, p.rtol);

    if (p.dump_output) {
        dump_output(p.dump_output, p, output_f32);
        std::printf("wrote baseline output: %s\n", p.dump_output);
    }

    const double seconds = static_cast<double>(elapsed_us) / 1e6;
    const double average_us = static_cast<double>(elapsed_us) / p.iterations;
    const double gmacs = (static_cast<double>(p.k) * p.n * p.batch * p.iterations) / (seconds * 1e9);
    const double logical_weight_gbs = (static_cast<double>(packed_weight_bytes) * p.iterations) / (seconds * 1024.0 * 1024.0 * 1024.0);

    std::printf("time: total=%lld us avg=%.2f us logical=%0.2f GMAC/s packed-weight=%0.2f GiB/s\n",
                static_cast<long long>(elapsed_us), average_us, gmacs, logical_weight_gbs);
    std::printf("output: fnv1a=%016llx\n", static_cast<unsigned long long>(stats.checksum));

    if (p.compare_output) {
        const double rmse = std::sqrt(stats.squared_error / output_elements);
        std::printf("A/B: baseline=%s mismatches=%zu/%zu max_abs=%g max_rel=%g rmse=%g (atol=%g rtol=%g)\n",
                    p.compare_output, stats.mismatches, output_elements, stats.max_abs, stats.max_rel, rmse, p.atol, p.rtol);
        if (stats.mismatches != 0) {
            return 2;
        }
    }

    ggml_quantize_free();
    return 0;
}
