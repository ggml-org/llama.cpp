// Synthetic HIP backend benchmark for the DeepSeek-V4 hidden-channel mixer.
//
// This builds a normal GGML_OP_MUL_MAT graph, so it exercises the production
// dispatcher. The target shape is A[K,M]^T * B[K,N] with F32
// K=16384, M=24, N=256.

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
#include <string>
#include <vector>

namespace {

struct params {
    int64_t k = 16384;
    int64_t m = 24;
    int64_t n = 256;
    int warmup = 5;
    int iterations = 25;
    const char * device_name = nullptr;
    const char * dump_output = nullptr;
    const char * compare_output = nullptr;
    float atol = 5e-3f;
    float rtol = 5e-4f;
    bool disable_graphs = false;
    bool skip_cpu_reference = false;
};

struct output_file_header {
    uint64_t magic = 0x48434D4958524432ULL; // "HCMIXRD2"
    uint32_t version = 1;
    uint32_t reserved = 0;
    uint64_t k = 0;
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t elements = 0;
};

static_assert(sizeof(output_file_header) == 48, "unexpected output file header layout");

struct output_stats {
    uint64_t checksum = 1469598103934665603ULL;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    size_t mismatches = 0;
    size_t nonfinite = 0;
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(1);
}

void usage(const char * program) {
    std::printf("Synthetic DSV4 hc_mixes F32 benchmark using the normal HIP GGML backend graph.\n\n");
    std::printf("usage: %s [options]\n\n", program);
    std::printf("  --k N                   reduction width (16384)\n");
    std::printf("  --m N                   output rows (24)\n");
    std::printf("  --n N                   activation/output columns (256)\n");
    std::printf("  --warmup N              warmup graph executions (5)\n");
    std::printf("  --iterations N          timed graph executions (25)\n");
    std::printf("  --device NAME           exact GGML GPU device name; defaults to first ROCm GPU\n");
    std::printf("  --disable-graphs         disable HIP graph capture before backend initialization\n");
    std::printf("  --skip-cpu-reference     skip double-accumulation host reference\n");
    std::printf("  --dump-output FILE       write deterministic F32 output for later A/B comparison\n");
    std::printf("  --compare-output FILE    compare output against --dump-output data\n");
    std::printf("  --atol X                 absolute tolerance (5e-3)\n");
    std::printf("  --rtol X                 relative tolerance (5e-4)\n");
    std::printf("  -h, --help               show this help\n");
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

float parse_tolerance(const char * name, const char * value) {
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

        if (std::strcmp(arg, "--k") == 0) {
            result.k = parse_i64(arg, require_value(arg), 1, 1LL << 30);
        } else if (std::strcmp(arg, "--m") == 0) {
            result.m = parse_i64(arg, require_value(arg), 1, 1LL << 20);
        } else if (std::strcmp(arg, "--n") == 0) {
            result.n = parse_i64(arg, require_value(arg), 1, 1LL << 20);
        } else if (std::strcmp(arg, "--warmup") == 0) {
            result.warmup = static_cast<int>(parse_i64(arg, require_value(arg), 0, 1000000));
        } else if (std::strcmp(arg, "--iterations") == 0) {
            result.iterations = static_cast<int>(parse_i64(arg, require_value(arg), 1, 1000000));
        } else if (std::strcmp(arg, "--device") == 0) {
            result.device_name = require_value(arg);
        } else if (std::strcmp(arg, "--dump-output") == 0) {
            result.dump_output = require_value(arg);
        } else if (std::strcmp(arg, "--compare-output") == 0) {
            result.compare_output = require_value(arg);
        } else if (std::strcmp(arg, "--atol") == 0) {
            result.atol = parse_tolerance(arg, require_value(arg));
        } else if (std::strcmp(arg, "--rtol") == 0) {
            result.rtol = parse_tolerance(arg, require_value(arg));
        } else if (std::strcmp(arg, "--disable-graphs") == 0) {
            result.disable_graphs = true;
        } else if (std::strcmp(arg, "--skip-cpu-reference") == 0) {
            result.skip_cpu_reference = true;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown option: %s\n", arg);
            usage(argv[0]);
            std::exit(1);
        }
    }
    return result;
}

uint32_t hash_u32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

float deterministic_value(size_t index, uint32_t seed) {
    const uint32_t bits = hash_u32(static_cast<uint32_t>(index) ^ seed);
    const int32_t centered = static_cast<int32_t>(bits & 0xffffU) - 32768;
    return static_cast<float>(centered) * (0.125f / 32768.0f);
}

void fill_values(std::vector<float> & values, uint32_t seed) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = deterministic_value(i, seed);
    }
}

ggml_backend_dev_t select_rocm_device(const params & p) {
    ggml_backend_dev_t selected = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        if (std::strcmp(ggml_backend_reg_name(registry), "ROCm") != 0) {
            continue;
        }
        if (p.device_name && std::strcmp(p.device_name, ggml_backend_dev_name(device)) != 0) {
            continue;
        }
        selected = device;
        break;
    }
    if (!selected) {
        fail("no matching ROCm GPU backend device found");
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

output_file_header make_header(const params & p, size_t elements) {
    output_file_header header;
    header.k = static_cast<uint64_t>(p.k);
    header.m = static_cast<uint64_t>(p.m);
    header.n = static_cast<uint64_t>(p.n);
    header.elements = static_cast<uint64_t>(elements);
    return header;
}

void dump_output(const char * path, const params & p, const std::vector<float> & output) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        fail("cannot open output dump");
    }
    const output_file_header header = make_header(p, output.size());
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) {
        fail("unable to write output dump");
    }
}

std::vector<float> load_output(const char * path, const params & p, size_t expected_elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("cannot open baseline output");
    }
    const std::streamsize expected_bytes = static_cast<std::streamsize>(sizeof(output_file_header) + expected_elements * sizeof(float));
    if (file.tellg() != expected_bytes) {
        fail("baseline output size does not match this case");
    }
    file.seekg(0);
    output_file_header header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    const output_file_header expected = make_header(p, expected_elements);
    if (!file || std::memcmp(&header, &expected, sizeof(header)) != 0) {
        fail("baseline output header does not match this case");
    }
    std::vector<float> output(expected_elements);
    file.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) {
        fail("unable to read baseline output");
    }
    return output;
}

std::vector<float> cpu_reference(
        const params & p,
        const std::vector<float> & weights,
        const std::vector<float> & activations) {
    std::vector<float> result(static_cast<size_t>(p.m * p.n));
    for (int64_t col = 0; col < p.n; ++col) {
        for (int64_t row = 0; row < p.m; ++row) {
            double sum = 0.0;
            for (int64_t ik = 0; ik < p.k; ++ik) {
                sum += static_cast<double>(weights[static_cast<size_t>(row * p.k + ik)]) *
                       static_cast<double>(activations[static_cast<size_t>(col * p.k + ik)]);
            }
            result[static_cast<size_t>(col * p.m + row)] = static_cast<float>(sum);
        }
    }
    return result;
}

output_stats compare_output(
        const std::vector<float> & output,
        const std::vector<float> & reference,
        float atol,
        float rtol) {
    output_stats result;
    result.checksum = fnv1a(output);
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || !std::isfinite(reference[i])) {
            ++result.nonfinite;
            continue;
        }
        const float abs_error = std::fabs(output[i] - reference[i]);
        const float relative_error = abs_error / std::max(std::fabs(reference[i]), 1e-12f);
        result.max_abs = std::max(result.max_abs, abs_error);
        result.max_rel = std::max(result.max_rel, relative_error);
        result.squared_error += static_cast<double>(abs_error) * abs_error;
        result.squared_reference += static_cast<double>(reference[i]) * reference[i];
        if (abs_error > atol + rtol * std::fabs(reference[i])) {
            ++result.mismatches;
        }
    }
    return result;
}

void print_comparison(const char * label, const output_stats & stats, size_t elements, float atol, float rtol) {
    const double rmse = std::sqrt(stats.squared_error / elements);
    const double nmse = stats.squared_reference > 0.0 ? stats.squared_error / stats.squared_reference : 0.0;
    std::printf("%s: mismatches=%zu/%zu nonfinite=%zu max_abs=%g max_rel=%g rmse=%g nmse=%g (atol=%g rtol=%g)\n",
            label, stats.mismatches, elements, stats.nonfinite, stats.max_abs, stats.max_rel, rmse, nmse, atol, rtol);
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
        fail("failed to initialize selected GPU backend");
    }

    std::printf("backend: %s (%s)\n", ggml_backend_name(backend.get()), ggml_backend_dev_description(device));
    std::printf("hc_mixes case: K=%lld M=%lld N=%lld warmup=%d iterations=%d graphs=%s override=%s\n",
            static_cast<long long>(p.k), static_cast<long long>(p.m), static_cast<long long>(p.n),
            p.warmup, p.iterations, p.disable_graphs ? "off" : "on",
            std::getenv("GGML_HIP_RDNA2_HC_MIXES") ? std::getenv("GGML_HIP_RDNA2_HC_MIXES") : "unset");

    const size_t weight_elements = static_cast<size_t>(p.k * p.m);
    const size_t activation_elements = static_cast<size_t>(p.k * p.n);
    const size_t output_elements = static_cast<size_t>(p.m * p.n);
    std::vector<float> weights_f32(weight_elements);
    std::vector<float> activations_f32(activation_elements);
    fill_values(weights_f32, 0x13579bdfU);
    fill_values(activations_f32, 0x2468ace0U);

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

    ggml_tensor * weights = ggml_new_tensor_2d(ctx_weights.get(), GGML_TYPE_F32, p.k, p.m);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, p.k, p.n);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, activation);
    ggml_set_name(output, "hc_mixes_test");
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
    ggml_backend_tensor_set(weights, weights_f32.data(), 0, weights_f32.size() * sizeof(float));
    ggml_backend_tensor_set(activation, activations_f32.data(), 0, activations_f32.size() * sizeof(float));

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
    const uint64_t checksum = fnv1a(output_f32);
    const double seconds = static_cast<double>(elapsed_us) / 1e6;
    const double average_us = static_cast<double>(elapsed_us) / p.iterations;
    const double tflops = 2.0 * static_cast<double>(p.k) * p.m * p.n * p.iterations / seconds / 1e12;
    std::printf("time: total=%lld us avg=%.2f us throughput=%.3f TFLOP/s\n",
            static_cast<long long>(elapsed_us), average_us, tflops);
    std::printf("output: fnv1a=%016llx\n", static_cast<unsigned long long>(checksum));

    bool failed = false;
    if (!p.skip_cpu_reference) {
        const int64_t cpu_start_us = ggml_time_us();
        const std::vector<float> reference = cpu_reference(p, weights_f32, activations_f32);
        const output_stats stats = compare_output(output_f32, reference, p.atol, p.rtol);
        print_comparison("CPU reference", stats, output_elements, p.atol, p.rtol);
        std::printf("CPU reference time: %lld us\n", static_cast<long long>(ggml_time_us() - cpu_start_us));
        failed = failed || stats.mismatches != 0 || stats.nonfinite != 0;
    }

    if (p.compare_output) {
        const std::vector<float> baseline = load_output(p.compare_output, p, output_elements);
        const output_stats stats = compare_output(output_f32, baseline, p.atol, p.rtol);
        print_comparison("A/B baseline", stats, output_elements, p.atol, p.rtol);
        failed = failed || stats.mismatches != 0 || stats.nonfinite != 0;
    }

    if (p.dump_output) {
        dump_output(p.dump_output, p, output_f32);
        std::printf("wrote output: %s\n", p.dump_output);
    }

    return failed ? 2 : 0;
}