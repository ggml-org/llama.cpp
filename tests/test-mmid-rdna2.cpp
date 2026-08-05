// Synthetic HIP backend benchmark for the routed quantized MUL_MAT_ID path.
//
// It uses a broadcast activation tensor, expert IDs, and packed quantized
// expert weights so the production HIP backend exercises MMID routing plus the
// selected MMVQ/MMQ implementation.

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

enum class routing_mode : uint32_t {
    uniform = 0,
    hot = 1,
};

enum class weight_fixture : uint32_t {
    prototypes = 0,
    unique = 1,
};

struct params {
    ggml_type type = GGML_TYPE_Q4_K;
    int64_t k = 4096;
    int64_t n = 512;
    int64_t batch = 32;
    int64_t experts = 64;
    int64_t top_k = 10;
    int warmup = 10;
    int iterations = 50;
    routing_mode routing = routing_mode::uniform;
    weight_fixture fixture = weight_fixture::prototypes;
    const char * device_name = nullptr;
    const char * dump_output = nullptr;
    const char * compare_output = nullptr;
    float atol = 1e-3f;
    float rtol = 1e-4f;
    bool disable_graphs = false;
};

struct output_file_header {
    uint64_t magic = 0x4D4D494452444E32ULL; // "MMIDRDN2"
    uint32_t version = 1;
    uint32_t type = 0;
    uint32_t routing = 0;
    uint32_t reserved = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t batch = 0;
    uint64_t experts = 0;
    uint64_t top_k = 0;
    uint64_t elements = 0;
};

static_assert(sizeof(output_file_header) == 72, "unexpected MMID output header layout");

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
    std::printf("Synthetic routed quantized MUL_MAT_ID benchmark using the HIP GGML backend graph.\n\n");
    std::printf("usage: %s [options]\n\n", program);
    std::printf("  --type TYPE             q4_k, q5_k, q6_k, iq2_xxs, iq2_s, iq3_xxs, or iq3_s (q4_k)\n");
    std::printf("  --k N                   input width, divisible by the quant block size (4096)\n");
    std::printf("  --n N                   expert output rows (512)\n");
    std::printf("  --batch N               routed tokens, 1..256 (32)\n");
    std::printf("  --experts N             number of expert matrices (64)\n");
    std::printf("  --top-k N               unique experts selected per token (10)\n");
    std::printf("  --routing MODE          uniform or hot (uniform)\n");
    std::printf("  --fixture MODE          prototypes or unique (prototypes)\n");
    std::printf("  --warmup N              warmup graph executions (10)\n");
    std::printf("  --iterations N          timed graph executions (50)\n");
    std::printf("  --device NAME           exact ROCm GGML device name; defaults to the first GPU\n");
    std::printf("  --disable-graphs        set GGML_CUDA_DISABLE_GRAPHS before backend initialization\n");
    std::printf("  --dump-output FILE      write deterministic F32 output for later A/B comparison\n");
    std::printf("  --compare-output FILE   compare output against a matching baseline\n");
    std::printf("  --atol X                absolute A/B tolerance (1e-3)\n");
    std::printf("  --rtol X                relative A/B tolerance (1e-4)\n");
    std::printf("  -h, --help              show this help\n");
}

bool parse_type(const char * value, ggml_type & type) {
    if (std::strcmp(value, "q4_k") == 0) {
        type = GGML_TYPE_Q4_K;
        return true;
    }
    if (std::strcmp(value, "q5_k") == 0) {
        type = GGML_TYPE_Q5_K;
        return true;
    }
    if (std::strcmp(value, "q6_k") == 0) {
        type = GGML_TYPE_Q6_K;
        return true;
    }
    if (std::strcmp(value, "iq2_xxs") == 0) {
        type = GGML_TYPE_IQ2_XXS;
        return true;
    }
    if (std::strcmp(value, "iq2_s") == 0) {
        type = GGML_TYPE_IQ2_S;
        return true;
    }
    if (std::strcmp(value, "iq3_xxs") == 0) {
        type = GGML_TYPE_IQ3_XXS;
        return true;
    }
    if (std::strcmp(value, "iq3_s") == 0) {
        type = GGML_TYPE_IQ3_S;
        return true;
    }
    return false;
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

float parse_nonnegative_float(const char * name, const char * value) {
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0f) {
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
            result.k = parse_i64(arg, require_value(arg), 32, 1LL << 20);
        } else if (std::strcmp(arg, "--n") == 0) {
            result.n = parse_i64(arg, require_value(arg), 1, 1LL << 20);
        } else if (std::strcmp(arg, "--batch") == 0) {
            result.batch = parse_i64(arg, require_value(arg), 1, 256);
        } else if (std::strcmp(arg, "--experts") == 0) {
            result.experts = parse_i64(arg, require_value(arg), 1, 1024);
        } else if (std::strcmp(arg, "--top-k") == 0) {
            result.top_k = parse_i64(arg, require_value(arg), 1, 1024);
        } else if (std::strcmp(arg, "--routing") == 0) {
            const char * value = require_value(arg);
            if (std::strcmp(value, "uniform") == 0) {
                result.routing = routing_mode::uniform;
            } else if (std::strcmp(value, "hot") == 0) {
                result.routing = routing_mode::hot;
            } else {
                std::fprintf(stderr, "error: --routing must be uniform or hot\n");
                std::exit(1);
            }
        } else if (std::strcmp(arg, "--fixture") == 0) {
            const char * value = require_value(arg);
            if (std::strcmp(value, "prototypes") == 0) {
                result.fixture = weight_fixture::prototypes;
            } else if (std::strcmp(value, "unique") == 0) {
                result.fixture = weight_fixture::unique;
            } else {
                std::fprintf(stderr, "error: --fixture must be prototypes or unique\n");
                std::exit(1);
            }
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
            result.atol = parse_nonnegative_float(arg, require_value(arg));
        } else if (std::strcmp(arg, "--rtol") == 0) {
            result.rtol = parse_nonnegative_float(arg, require_value(arg));
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

    if (result.top_k > result.experts) {
        fail("--top-k cannot exceed --experts");
    }
    if (result.k % ggml_blck_size(result.type) != 0) {
        std::fprintf(stderr, "error: k=%lld is not divisible by %lld for %s\n",
                     static_cast<long long>(result.k), static_cast<long long>(ggml_blck_size(result.type)),
                     ggml_type_name(result.type));
        std::exit(1);
    }
    return result;
}

float deterministic_value(size_t index, float phase) {
    return 0.70f * std::sin(0.013f * static_cast<float>(index) + phase)
         + 0.30f * std::cos(0.007f * static_cast<float>(index) - phase);
}

void quantize_expert_weights(const params & p, std::vector<uint8_t> & packed) {
    const size_t row_bytes = ggml_row_size(p.type, p.k);
    std::vector<float> row(static_cast<size_t>(p.k));
    std::vector<float> importance(static_cast<size_t>(p.k), 1.0f);

    if (p.fixture == weight_fixture::unique) {
        // Slow correctness fixture: every expert/output row has independent
        // packed weights so coupled expert/row addressing errors cannot alias.
        for (int64_t expert = 0; expert < p.experts; ++expert) {
            for (int64_t out = 0; out < p.n; ++out) {
                const size_t row_index = static_cast<size_t>(expert * p.n + out);
                for (int64_t col = 0; col < p.k; ++col) {
                    row[static_cast<size_t>(col)] = deterministic_value(
                        row_index * static_cast<size_t>(p.k) + static_cast<size_t>(col),
                        0.13f * static_cast<float>(expert + 1));
                }
                ggml_quantize_chunk(p.type, row.data(), packed.data() + row_index * row_bytes,
                                    0, 1, p.k, importance.data());
            }
        }
        return;
    }

    // Fast performance fixture: reuse a bounded deterministic row set in an
    // expert-dependent pattern. It intentionally aliases coupled expert/row
    // pairs; use --fixture unique for correctness validation.
    constexpr size_t nprototypes = 1024;
    std::vector<uint8_t> prototypes(nprototypes * row_bytes);
    for (size_t prototype = 0; prototype < nprototypes; ++prototype) {
        for (int64_t col = 0; col < p.k; ++col) {
            row[static_cast<size_t>(col)] = deterministic_value(
                prototype * static_cast<size_t>(p.k) + static_cast<size_t>(col), 0.13f * static_cast<float>(prototype + 1));
        }
        ggml_quantize_chunk(p.type, row.data(), prototypes.data() + prototype * row_bytes,
                            0, 1, p.k, importance.data());
    }

    for (int64_t expert = 0; expert < p.experts; ++expert) {
        for (int64_t out = 0; out < p.n; ++out) {
            const size_t row_index = static_cast<size_t>(expert * p.n + out);
            const size_t prototype = static_cast<size_t>(out + 257 * expert) % nprototypes;
            std::memcpy(packed.data() + row_index * row_bytes,
                        prototypes.data() + prototype * row_bytes, row_bytes);
        }
    }
}

void fill_activations(std::vector<float> & values) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = deterministic_value(i, 1.5f);
    }
}

void fill_ids(const params & p, std::vector<int32_t> & ids) {
    std::fill(ids.begin(), ids.end(), 0);
    const int64_t hot_experts = std::max<int64_t>(p.top_k, std::min<int64_t>(16, p.experts));

    for (int64_t token = 0; token < p.batch; ++token) {
        const int64_t base = p.routing == routing_mode::uniform
            ? (token * p.top_k) % p.experts
            : (token * 3) % hot_experts;
        for (int64_t slot = 0; slot < p.top_k; ++slot) {
            ids[static_cast<size_t>(slot + token * p.experts)] = static_cast<int32_t>((base + slot) %
                (p.routing == routing_mode::uniform ? p.experts : hot_experts));
        }
    }
}

ggml_backend_dev_t select_rocm_device(const params & p) {
    ggml_backend_dev_t selected = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
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
        fail(p.device_name ? "requested ROCm backend device was not found" : "no ROCm backend device was found");
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
    header.routing = static_cast<uint32_t>(p.routing);
    header.reserved = static_cast<uint32_t>(p.fixture);
    header.k = static_cast<uint64_t>(p.k);
    header.n = static_cast<uint64_t>(p.n);
    header.batch = static_cast<uint64_t>(p.batch);
    header.experts = static_cast<uint64_t>(p.experts);
    header.top_k = static_cast<uint64_t>(p.top_k);
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
        std::fprintf(stderr, "error: unable to write '%s'\n", path);
        std::exit(1);
    }
}

std::vector<float> load_output(const char * path, const params & p, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "error: cannot open baseline '%s'\n", path);
        std::exit(1);
    }
    const std::streamsize expected_bytes = static_cast<std::streamsize>(sizeof(output_file_header) + elements * sizeof(float));
    if (file.tellg() != expected_bytes) {
        std::fprintf(stderr, "error: baseline '%s' has the wrong size\n", path);
        std::exit(1);
    }
    file.seekg(0);
    output_file_header header;
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    const output_file_header expected = make_output_header(p, elements);
    if (!file || std::memcmp(&header, &expected, sizeof(header)) != 0) {
        std::fprintf(stderr, "error: baseline '%s' belongs to a different routed MMID case\n", path);
        std::exit(1);
    }
    std::vector<float> output(elements);
    file.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) {
        std::fprintf(stderr, "error: unable to read baseline '%s'\n", path);
        std::exit(1);
    }
    return output;
}

output_stats compare_output(const std::vector<float> & output, const std::vector<float> * baseline, float atol, float rtol) {
    output_stats stats;
    stats.checksum = fnv1a(output);
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr, "error: non-finite output at element %zu\n", i);
            std::exit(1);
        }
        if (!baseline) {
            continue;
        }
        const float abs_error = std::fabs(output[i] - (*baseline)[i]);
        const float rel_error = abs_error / std::max(std::fabs((*baseline)[i]), 1e-12f);
        stats.max_abs = std::max(stats.max_abs, abs_error);
        stats.max_rel = std::max(stats.max_rel, rel_error);
        stats.squared_error += static_cast<double>(abs_error) * abs_error;
        if (abs_error > atol + rtol * std::fabs((*baseline)[i])) {
            ++stats.mismatches;
        }
    }
    return stats;
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
        fail("failed to initialize the selected ROCm backend");
    }

    std::printf("backend: %s (%s)\n", ggml_backend_name(backend.get()), ggml_backend_dev_description(device));
    std::printf("routed MMID case: type=%s K=%lld N=%lld batch=%lld experts=%lld top_k=%lld routing=%s fixture=%s\n",
                ggml_type_name(p.type), static_cast<long long>(p.k), static_cast<long long>(p.n),
                static_cast<long long>(p.batch), static_cast<long long>(p.experts), static_cast<long long>(p.top_k),
                p.routing == routing_mode::uniform ? "uniform" : "hot",
                p.fixture == weight_fixture::prototypes ? "prototypes" : "unique");

    const size_t row_bytes = ggml_row_size(p.type, p.k);
    const size_t weight_bytes = row_bytes * static_cast<size_t>(p.n) * static_cast<size_t>(p.experts);
    std::vector<uint8_t> weights_packed(weight_bytes);
    std::vector<float> activations(static_cast<size_t>(p.k * p.batch));
    std::vector<int32_t> ids_host(static_cast<size_t>(p.experts * p.batch));
    quantize_expert_weights(p, weights_packed);
    fill_activations(activations);
    fill_ids(p, ids_host);

    const ggml_init_params init_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 12 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_weights(ggml_init(init_params));
    ggml_context_ptr ctx(ggml_init(init_params));
    if (!ctx_weights || !ctx) {
        fail("failed to create GGML contexts");
    }

    ggml_tensor * expert_weights = ggml_new_tensor_3d(ctx_weights.get(), p.type, p.k, p.n, p.experts);
    ggml_tensor * ids_full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, p.experts, p.batch);
    ggml_tensor * ids = ggml_view_2d(ctx.get(), ids_full, p.top_k, p.batch, ids_full->nb[1], 0);
    ggml_tensor * activation = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p.k, 1, p.batch);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), expert_weights, activation, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    if (!ggml_backend_supports_op(backend.get(), output)) {
        fail("selected backend does not support this MUL_MAT_ID case");
    }

    ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(ctx_weights.get(), backend.get()));
    ggml_backend_buffer_ptr compute_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!weight_buffer || !compute_buffer) {
        fail("failed to allocate backend tensors");
    }
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(expert_weights, weights_packed.data(), 0, weights_packed.size());
    ggml_backend_tensor_set(ids_full, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(activation, activations.data(), 0, activations.size() * sizeof(float));

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

    const size_t output_elements = static_cast<size_t>(ggml_nelements(output));
    std::vector<float> output_host(output_elements);
    ggml_backend_tensor_get(output, output_host.data(), 0, output_host.size() * sizeof(float));

    const std::vector<float> baseline = p.compare_output
        ? load_output(p.compare_output, p, output_elements)
        : std::vector<float>();
    const output_stats stats = compare_output(output_host, p.compare_output ? &baseline : nullptr, p.atol, p.rtol);

    if (p.dump_output) {
        dump_output(p.dump_output, p, output_host);
        std::printf("wrote baseline output: %s\n", p.dump_output);
    }

    const double seconds = static_cast<double>(elapsed_us) / 1e6;
    const double average_us = static_cast<double>(elapsed_us) / p.iterations;
    const double gmacs = (static_cast<double>(p.k) * p.n * p.batch * p.top_k * p.iterations) / (seconds * 1e9);
    std::printf("time: total=%lld us avg=%.2f us logical=%0.2f GMAC/s\n",
                static_cast<long long>(elapsed_us), average_us, gmacs);
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