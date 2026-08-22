#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "common.h"

#include <vector>
#include <string>
#include <cstdio>
#include <chrono>
#include <map>
#include <cmath>
#include <algorithm>
#include <thread>
#include <memory>
#include <cstring>

// Smart pointers for RAII cleanup
struct ggml_context_deleter {
    void operator()(ggml_context * ctx) { ggml_free(ctx); }
};
using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buf) { ggml_backend_buffer_free(buf); }
};
using ggml_backend_buffer_ptr = std::unique_ptr<struct ggml_backend_buffer, ggml_backend_buffer_deleter>;

struct ggml_backend_deleter {
    void operator()(ggml_backend_t backend) { ggml_backend_free(backend); }
};
using ggml_backend_ptr = std::unique_ptr<struct ggml_backend, ggml_backend_deleter>;

// Utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

struct BenchmarkParams {
    int64_t m = 4096;
    int64_t k = 14336;
    int64_t n_prefill = 512;
    int64_t n_decode = 1;
    int reps = 5;
    bool verbose = false;
    std::string device_arg = "auto";
};

static void print_usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  -v, --verbose         verbose output\n");
    printf("  -d, --device <dev>    device ID (int) or name (str) to use (default: auto)\n");
    printf("\n");
}

static void run_benchmark(ggml_backend_t backend, const BenchmarkParams & params, ggml_type type_a, const std::string & phase_name, int64_t n) {
    if (params.verbose) {
        printf("Benchmarking %s %s: m=%ld n=%ld k=%ld\n", phase_name.c_str(), ggml_type_name(type_a), params.m, n, params.k);
    }

    // Init context
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params init_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_base   =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(init_params));

    // Create tensors
    // A: Weight matrix (Quantized) [k, m]
    // B: Input matrix [k, n]
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), type_a, params.k, params.m);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.k, n);
    
    // Check support
    if (!ggml_backend_supports_op(backend, a) || !ggml_backend_supports_op(backend, b)) {
        if (params.verbose) printf("Backend does not support input tensors for %s\n", ggml_type_name(type_a));
        return;
    }

    // Build graph: C = A * B
    struct ggml_tensor * c = ggml_mul_mat(ctx.get(), a, b);
    
    if (!ggml_backend_supports_op(backend, c)) {
        if (params.verbose) printf("Backend does not support MUL_MAT for %s\n", ggml_type_name(type_a));
        return;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);

    // Allocate memory
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        printf("Failed to allocate memory\n");
        return;
    }

    // Warmup
    ggml_backend_graph_compute(backend, gf);

    // Run benchmark
    uint64_t t_start = get_time_ns();
    for (int i = 0; i < params.reps; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    uint64_t t_end = get_time_ns();
    
    double t_ns = (double)(t_end - t_start) / params.reps;
    double t_us = t_ns / 1000.0;

    // Stats
    // TOPS: 2*m*n*k
    double ops = 2.0 * params.m * n * params.k;
    double tops = (ops / t_ns) * 1e9 / 1e12; // TOPS

    // Print Row
    if (n > 1) {
        // Prompt Processing: Bandwidth is less relevant, compute bound
        printf("| %-10s | %10.2f | %10.2f |\n", 
               ggml_type_name(type_a), t_us, tops);
    } else {
        // Token Generation: Bandwidth is critical
        // Bandwidth: Size(A) + Size(B) + Size(C)
        size_t size_a = ggml_nbytes(a);
        size_t size_b = ggml_nbytes(b);
        size_t size_c = ggml_nbytes(c);
        size_t total_bytes = size_a + size_b + size_c;
        double gb_s = (double)total_bytes / t_ns; // GB/s
        
        printf("| %-10s | %10.2f | %10.2f | %10.2f |\n", 
               ggml_type_name(type_a), t_us, tops, gb_s);
    }
}

int main(int argc, char ** argv) {
    BenchmarkParams params;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", arg.c_str());
                return 1;
            }
            params.device_arg = argv[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    // Pick backend
    ggml_backend_ptr backend_ptr;
    
    if (params.device_arg != "auto") {
        // Try to parse as integer index
        try {
            int id = std::stoi(params.device_arg);
            if (id >= 0 && id < (int)ggml_backend_dev_count()) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(id);
                printf("Using device %d: %s\n", id, ggml_backend_dev_name(dev));
                backend_ptr.reset(ggml_backend_dev_init(dev, NULL));
            }
        } catch (...) {
            // Not a number, try name lookup
        }

        if (!backend_ptr) {
            // Try by name
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(params.device_arg.c_str());
            if (dev) {
                printf("Using device: %s\n", ggml_backend_dev_name(dev));
                backend_ptr.reset(ggml_backend_dev_init(dev, NULL));
            } else {
                fprintf(stderr, "error: device '%s' not found\n", params.device_arg.c_str());
                fprintf(stderr, "Available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    fprintf(stderr, "  %zu: %s\n", i, ggml_backend_dev_name(d));
                }
                return 1;
            }
        }
    } else {
        // Auto-detect: Prioritize GPU
        if (ggml_backend_dev_count() > 0) {
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    printf("Using auto-detected device %zu: %s\n", i, ggml_backend_dev_name(dev));
                    backend_ptr.reset(ggml_backend_dev_init(dev, NULL));
                    break;
                }
            }
        }
    }

    // Fallback to CPU
    if (!backend_ptr) {
        backend_ptr.reset(ggml_backend_init_by_name("CPU", NULL));
        if (!backend_ptr) {
             // Try fetching CPU backend by index if name fails (fallback)
             for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    backend_ptr.reset(ggml_backend_dev_init(dev, NULL));
                    break;
                }
             }
        }
        printf("Using backend: CPU\n");
    }

    if (!backend_ptr) {
        fprintf(stderr, "error: failed to initialize backend\n");
        return 1;
    }

    // Quant types to test
    std::vector<ggml_type> quants = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, 
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_0,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
        GGML_TYPE_MXFP4
    };

    printf("\n=== Prompt Processing (Prefill) Phase (Batch Size = %ld) ===\n", params.n_prefill);
    printf("| %-10s | %-10s | %-10s |\n", "Quant", "Time (us)", "TOPS");
    printf("|-%-10s-|-%-10s-|-%-10s-|\n", "----------", "----------", "----------");
    
    for (auto type : quants) {
        run_benchmark(backend_ptr.get(), params, type, "Prefill", params.n_prefill);
    }

    printf("\n=== Token Generation (Decoding) Phase (Batch Size = %ld) ===\n", params.n_decode);
    printf("| %-10s | %-10s | %-10s | %-10s |\n", "Quant", "Time (us)", "TOPS", "Eff. BW (GB/s)");
    printf("|-%-10s-|-%-10s-|-%-10s-|-%-14s-|\n", "----------", "----------", "----------", "--------------");

    for (auto type : quants) {
        run_benchmark(backend_ptr.get(), params, type, "Decoding", params.n_decode);
    }

    return 0;
}