#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct rpc_device_spec {
    std::string endpoint;
    uint32_t device = 0;
};

struct bench_options {
    rpc_device_spec src;
    rpc_device_spec dst;
    std::vector<size_t> bytes = {4096, 16384, 73728, 262144, 1048576};
    int repetitions = 9;
    int warmup = 2;
    bool verify = false;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --src ENDPOINT/DEVICE --dst ENDPOINT/DEVICE [options]\n"
        "\n"
        "Measure RPC tensor copy latency between two RPC buffer types.\n"
        "\n"
        "required:\n"
        "  --src ENDPOINT/DEVICE     source RPC endpoint and device index\n"
        "  --dst ENDPOINT/DEVICE     destination RPC endpoint and device index\n"
        "\n"
        "options:\n"
        "  --bytes LIST              comma-separated byte sizes (default: 4096,16384,73728,262144,1048576)\n"
        "  --repetitions N           measured copies per byte size (default: 9)\n"
        "  --warmup N                warmup copies per byte size (default: 2)\n"
        "  --verify                  read back and verify destination contents after each byte size\n"
        "  -h, --help                show this help\n"
        "\n"
        "Output is one JSON object per byte size on stdout. Set GGML_RPC_TRACE=1\n"
        "to confirm whether the measured path used COPY_TENSOR or coordinator\n"
        "fallback GET_TENSOR/SET_TENSOR traffic.\n",
        prog);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static int parse_non_negative_int(const char * value, const char * name) {
    char * end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return (int) parsed;
}

static uint32_t parse_device_index(const std::string & value) {
    char * end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid device index: " + value);
    }
    return (uint32_t) parsed;
}

static rpc_device_spec parse_rpc_device_spec(const std::string & value) {
    const size_t slash = value.rfind('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= value.size()) {
        throw std::runtime_error("RPC device must be ENDPOINT/DEVICE: " + value);
    }

    rpc_device_spec spec;
    spec.endpoint = value.substr(0, slash);
    spec.device = parse_device_index(value.substr(slash + 1));
    return spec;
}

static std::vector<size_t> parse_byte_list(const std::string & value) {
    std::vector<size_t> sizes;
    size_t pos = 0;
    while (pos <= value.size()) {
        size_t comma = value.find(',', pos);
        std::string item = value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (item.empty()) {
            throw std::runtime_error("empty byte size in list: " + value);
        }

        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0' || parsed == 0) {
            throw std::runtime_error("invalid byte size: " + item);
        }
        if (parsed % sizeof(float) != 0) {
            throw std::runtime_error("byte size must be a multiple of sizeof(float): " + item);
        }
        sizes.push_back((size_t) parsed);

        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }

    if (sizes.empty()) {
        throw std::runtime_error("at least one byte size is required");
    }
    return sizes;
}

static bench_options parse_args(int argc, char ** argv) {
    bench_options opts;
    bool have_src = false;
    bool have_dst = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--src") {
            opts.src = parse_rpc_device_spec(require_value("--src"));
            have_src = true;
        } else if (arg == "--dst") {
            opts.dst = parse_rpc_device_spec(require_value("--dst"));
            have_dst = true;
        } else if (arg == "--bytes") {
            opts.bytes = parse_byte_list(require_value("--bytes"));
        } else if (arg == "--repetitions") {
            opts.repetitions = parse_non_negative_int(require_value("--repetitions"), "--repetitions");
        } else if (arg == "--warmup") {
            opts.warmup = parse_non_negative_int(require_value("--warmup"), "--warmup");
        } else if (arg == "--verify") {
            opts.verify = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (!have_src) {
        throw std::runtime_error("--src is required");
    }
    if (!have_dst) {
        throw std::runtime_error("--dst is required");
    }
    if (opts.repetitions <= 0) {
        throw std::runtime_error("--repetitions must be greater than zero");
    }
    return opts;
}

static std::string spec_to_string(const rpc_device_spec & spec) {
    return spec.endpoint + "/" + std::to_string(spec.device);
}

static std::string path_hint(const rpc_device_spec & src, const rpc_device_spec & dst) {
    return src.endpoint == dst.endpoint ? "same-endpoint" : "cross-endpoint-fallback";
}

static std::vector<float> make_pattern(size_t n) {
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (float) ((i * 1315423911ULL + 17ULL) % 1000003ULL) / 1000003.0f;
    }
    return data;
}

static bool verify_tensor(const ggml_tensor * tensor, const std::vector<float> & expected) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    return actual == expected;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    if (n % 2 == 1) {
        return samples[n / 2];
    }
    return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

static double stdev(const std::vector<double> & samples, double avg) {
    if (samples.size() < 2) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (double v : samples) {
        const double d = v - avg;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / (double) (samples.size() - 1));
}

struct bench_result {
    size_t bytes = 0;
    std::vector<double> samples_ms;
    bool verified = false;
};

static bench_result run_size(const bench_options & opts, size_t bytes) {
    const size_t n_floats = bytes / sizeof(float);

    ggml_init_params src_params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_init_params dst_params = src_params;

    ggml_context * src_ctx = ggml_init(src_params);
    ggml_context * dst_ctx = ggml_init(dst_params);
    if (src_ctx == nullptr || dst_ctx == nullptr) {
        throw std::runtime_error("failed to create ggml contexts");
    }

    ggml_tensor * src = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, (int64_t) n_floats);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, (int64_t) n_floats);
    ggml_set_name(src, "rpc_copy_bench_src");
    ggml_set_name(dst, "rpc_copy_bench_dst");

    ggml_backend_buffer_type_t src_buft = ggml_backend_rpc_buffer_type(opts.src.endpoint.c_str(), opts.src.device);
    ggml_backend_buffer_type_t dst_buft = ggml_backend_rpc_buffer_type(opts.dst.endpoint.c_str(), opts.dst.device);
    if (src_buft == nullptr || dst_buft == nullptr) {
        ggml_free(src_ctx);
        ggml_free(dst_ctx);
        throw std::runtime_error("failed to create RPC buffer type");
    }

    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(src_ctx, src_buft);
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(dst_ctx, dst_buft);
    if (src_buf == nullptr || dst_buf == nullptr) {
        if (src_buf != nullptr) {
            ggml_backend_buffer_free(src_buf);
        }
        if (dst_buf != nullptr) {
            ggml_backend_buffer_free(dst_buf);
        }
        ggml_free(src_ctx);
        ggml_free(dst_ctx);
        throw std::runtime_error("failed to allocate RPC tensor buffers");
    }

    const std::vector<float> data = make_pattern(n_floats);
    ggml_backend_tensor_set(src, data.data(), 0, bytes);

    for (int i = 0; i < opts.warmup; ++i) {
        ggml_backend_tensor_copy(src, dst);
    }

    bench_result result;
    result.bytes = bytes;
    result.samples_ms.reserve((size_t) opts.repetitions);
    for (int i = 0; i < opts.repetitions; ++i) {
        const double start_ms = now_ms();
        ggml_backend_tensor_copy(src, dst);
        const double end_ms = now_ms();
        result.samples_ms.push_back(end_ms - start_ms);
    }

    result.verified = !opts.verify || verify_tensor(dst, data);

    ggml_backend_buffer_free(dst_buf);
    ggml_backend_buffer_free(src_buf);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);

    return result;
}

static void print_result_json(
        const bench_options & opts,
        const bench_result & result) {
    const double avg_ms = std::accumulate(result.samples_ms.begin(), result.samples_ms.end(), 0.0) /
        (double) result.samples_ms.size();
    const double median_ms = median(result.samples_ms);
    const double min_ms = *std::min_element(result.samples_ms.begin(), result.samples_ms.end());
    const double max_ms = *std::max_element(result.samples_ms.begin(), result.samples_ms.end());
    const double stdev_ms = stdev(result.samples_ms, avg_ms);
    const double mib_per_s_avg = ((double) result.bytes / (1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::printf("{");
    std::printf("\"src\":\"%s\",", json_escape(spec_to_string(opts.src)).c_str());
    std::printf("\"dst\":\"%s\",", json_escape(spec_to_string(opts.dst)).c_str());
    std::printf("\"path_hint\":\"%s\",", path_hint(opts.src, opts.dst).c_str());
    std::printf("\"type\":\"f32\",");
    std::printf("\"bytes\":%zu,", result.bytes);
    std::printf("\"elements\":%zu,", result.bytes / sizeof(float));
    std::printf("\"repetitions\":%d,", opts.repetitions);
    std::printf("\"warmup\":%d,", opts.warmup);
    std::printf("\"avg_ms\":%.6f,", avg_ms);
    std::printf("\"median_ms\":%.6f,", median_ms);
    std::printf("\"min_ms\":%.6f,", min_ms);
    std::printf("\"max_ms\":%.6f,", max_ms);
    std::printf("\"stdev_ms\":%.6f,", stdev_ms);
    std::printf("\"mib_per_s_avg\":%.6f,", mib_per_s_avg);
    std::printf("\"verified\":%s,", result.verified ? "true" : "false");
    std::printf("\"samples_ms\":[");
    for (size_t i = 0; i < result.samples_ms.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("%.6f", result.samples_ms[i]);
    }
    std::printf("]}\n");
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    try {
        const bench_options opts = parse_args(argc, argv);

        ggml_backend_t src_backend = ggml_backend_rpc_init(opts.src.endpoint.c_str(), opts.src.device);
        ggml_backend_t dst_backend = ggml_backend_rpc_init(opts.dst.endpoint.c_str(), opts.dst.device);
        if (src_backend == nullptr || dst_backend == nullptr) {
            if (src_backend != nullptr) {
                ggml_backend_free(src_backend);
            }
            if (dst_backend != nullptr) {
                ggml_backend_free(dst_backend);
            }
            throw std::runtime_error("failed to initialize RPC backend");
        }

        int rc = 0;
        for (size_t bytes : opts.bytes) {
            bench_result result = run_size(opts, bytes);
            print_result_json(opts, result);
            if (!result.verified) {
                std::fprintf(stderr, "verification failed for %zu bytes\n", bytes);
                rc = 2;
                break;
            }
        }

        ggml_backend_free(dst_backend);
        ggml_backend_free(src_backend);
        return rc;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }
}
