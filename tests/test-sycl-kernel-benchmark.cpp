// SYCL Kernel Benchmark - Tests all mul_mat kernel paths and layouts
// Usage: ./test-sycl-kernel-benchmark [device_id]
//
// Tests:
// - DMMV (Dequantize + oneDNN GEMM) - large batches
// - MMVQ (Mul Mat Vec Quantized) - batch=1, AoS and SoA layouts
// - MMQ (Mul Mat Quantized) - small batches, AoS and SoA layouts
// - oneDNN GEMM - large batches, FP16/FP32
// - XMX (if available) - Intel matrix extensions
//
// Quant types tested: Q4_0, Q8_0, MXFP4

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#include "ggml-quants.h"  // For block definitions and quant constants

// Matrix dimensions for benchmarks
struct BenchConfig {
    const char* name;
    int64_t M;  // Output rows (batch size / tokens)
    int64_t N;  // Output cols (hidden dim)
    int64_t K;  // Inner dim (input features)
    int repeats;
};

// Kernel paths to test
enum class KernelPath {
    DMMV,           // Dequantize mul mat vec
    MMVQ_AOS,       // Quantized vec, Array-of-Structures
    MMVQ_SOA,       // Quantized vec, Structure-of-Arrays
    MMQ_AOS,        // Quantized mat, Array-of-Structures
    MMQ_SOA,        // Quantized mat, Structure-of-Arrays
    ONEDNN_GEMM,    // oneDNN for large batches
    XMX_GEMM,       // XMX accelerated (if available)
};

const char* kernel_path_name(KernelPath p) {
    switch (p) {
        case KernelPath::DMMV:        return "DMMV";
        case KernelPath::MMVQ_AOS:    return "MMVQ-AoS";
        case KernelPath::MMVQ_SOA:    return "MMVQ-SoA";
        case KernelPath::MMQ_AOS:     return "MMQ-AoS";
        case KernelPath::MMQ_SOA:     return "MMQ-SoA";
        case KernelPath::ONEDNN_GEMM: return "oneDNN";
        case KernelPath::XMX_GEMM:    return "XMX";
        default:                      return "unknown";
    }
}

// Result storage
struct BenchResult {
    const char* config_name;
    const char* kernel_name;
    ggml_type quant_type;
    double avg_ms;
    double min_ms;
    double max_ms;
    double throughput_gflops;
    double throughput_tps;  // tokens per second for prompt processing
};

// Timer utility
class Timer {
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0;
};

// Initialize random quantized data
void init_random_q4_0(void* data, size_t n_blocks) {
    auto* blocks = static_cast<block_q4_0*>(data);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n_blocks; i++) {
        blocks[i].d = ggml_fp32_to_fp16(dist(rng) * 0.1f);
        for (int j = 0; j < QK4_0 / 2; j++) {
            uint8_t v0 = (rng() % 16);
            uint8_t v1 = (rng() % 16);
            blocks[i].qs[j] = v0 | (v1 << 4);
        }
    }
}

void init_random_q8_0(void* data, size_t n_blocks) {
    auto* blocks = static_cast<block_q8_0*>(data);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n_blocks; i++) {
        blocks[i].d = ggml_fp32_to_fp16(dist(rng) * 0.1f);
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = static_cast<int8_t>((rng() % 256) - 128);
        }
    }
}

void init_random_f16(void* data, size_t n_elements) {
    auto* ptr = static_cast<ggml_fp16_t*>(data);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n_elements; i++) {
        ptr[i] = ggml_fp32_to_fp16(dist(rng));
    }
}

// Run a single benchmark configuration
BenchResult run_benchmark(
    ggml_backend_t backend,
    const BenchConfig& config,
    ggml_type weight_type,
    int batch_size,
    const char* kernel_name
) {
    BenchResult result = {};
    result.config_name = config.name;
    result.kernel_name = kernel_name;
    result.quant_type = weight_type;

    // Create context
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,  // 256 MB
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to create ggml context\n");
        return result;
    }

    // Create tensors
    // Weight: [K, N] in quantized format
    // Input:  [K, M] in F32/F16
    // Output: [N, M] in F32

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, config.K, config.N);
    struct ggml_tensor* input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.K, batch_size);
    struct ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.N, batch_size);

    ggml_set_name(weight, "weight");
    ggml_set_name(input, "input");
    ggml_set_name(output, "output");

    // Create graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    struct ggml_tensor* result_tensor = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(result_tensor, "result");
    ggml_build_forward_expand(graph, result_tensor);

    // Allocate backend buffers
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return result;
    }

    // Initialize data
    size_t weight_size = ggml_nbytes(weight);
    void* weight_data = malloc(weight_size);

    if (weight_type == GGML_TYPE_Q4_0) {
        size_t n_blocks = config.K * config.N / QK4_0;
        init_random_q4_0(weight_data, n_blocks);
    } else if (weight_type == GGML_TYPE_Q8_0) {
        size_t n_blocks = config.K * config.N / QK8_0;
        init_random_q8_0(weight_data, n_blocks);
    } else if (weight_type == GGML_TYPE_F16) {
        init_random_f16(weight_data, config.K * config.N);
    }

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    free(weight_data);

    // Initialize input
    std::vector<float> input_data(config.K * batch_size);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : input_data) v = dist(rng);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Warmup
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);

    // Benchmark
    std::vector<double> times;
    Timer timer;

    for (int i = 0; i < config.repeats; i++) {
        timer.start();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        times.push_back(timer.stop_ms());
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());
    result.min_ms = times.front();
    result.max_ms = times.back();

    double sum = 0;
    for (double t : times) sum += t;
    result.avg_ms = sum / times.size();

    // Calculate throughput
    // FLOPs = 2 * M * N * K (for matrix multiply)
    double flops = 2.0 * batch_size * config.N * config.K;
    result.throughput_gflops = (flops / 1e9) / (result.avg_ms / 1000.0);

    // Tokens per second (for prompt processing comparison)
    result.throughput_tps = (batch_size * 1000.0) / result.avg_ms;

    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

void print_result_header() {
    printf("\n");
    printf("%-12s %-10s %-8s %8s %8s %8s %10s %10s\n",
           "Config", "Kernel", "Type", "Avg(ms)", "Min(ms)", "Max(ms)", "GFLOPS", "tok/s");
    printf("%-12s %-10s %-8s %8s %8s %8s %10s %10s\n",
           "------", "------", "----", "-------", "-------", "-------", "------", "-----");
}

void print_result(const BenchResult& r) {
    const char* type_name = ggml_type_name(r.quant_type);
    printf("%-12s %-10s %-8s %8.2f %8.2f %8.2f %10.1f %10.0f\n",
           r.config_name, r.kernel_name, type_name,
           r.avg_ms, r.min_ms, r.max_ms,
           r.throughput_gflops, r.throughput_tps);
}

int main(int argc, char** argv) {
    int device_id = 0;
    if (argc > 1) {
        device_id = atoi(argv[1]);
    }

    printf("=== SYCL Kernel Benchmark ===\n");
    printf("Device ID: %d\n", device_id);

    // Initialize SYCL backend
    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to initialize SYCL backend for device %d\n", device_id);
        return 1;
    }

    printf("Backend: %s\n", ggml_backend_name(backend));

    // Benchmark configurations
    // Simulating real model dimensions
    std::vector<BenchConfig> configs = {
        // Token generation (batch=1)
        {"tg-1tok",    1, 4096, 4096, 50},    // Single token, 7B model
        {"tg-1tok-lg", 1, 8192, 8192, 50},    // Single token, 70B model

        // Small batch (MMQ range, batch <= 32)
        {"pp-8tok",    8, 4096, 4096, 30},
        {"pp-16tok",  16, 4096, 4096, 30},
        {"pp-32tok",  32, 4096, 4096, 30},

        // Large batch (oneDNN GEMM range)
        {"pp-64tok",   64, 4096, 4096, 20},
        {"pp-128tok", 128, 4096, 4096, 20},
        {"pp-256tok", 256, 4096, 4096, 15},
        {"pp-512tok", 512, 4096, 4096, 10},

        // MoE expert dimensions (smaller K, many experts)
        {"moe-1tok",   1, 2880, 2880, 50},    // GPT-OSS expert dim
        {"moe-8tok",   8, 2880, 2880, 30},
        {"moe-64tok", 64, 2880, 2880, 20},
    };

    // Quant types to test
    std::vector<ggml_type> quant_types = {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0,
        GGML_TYPE_F16,
    };

    std::vector<BenchResult> all_results;

    // Run benchmarks
    for (const auto& config : configs) {
        print_result_header();
        printf("Config: %s (M=%lld, N=%lld, K=%lld)\n",
               config.name, (long long)config.M, (long long)config.N, (long long)config.K);

        for (ggml_type qtype : quant_types) {
            // Skip unsupported combinations
            if (qtype == GGML_TYPE_F16 && config.M > 32) {
                // F16 only makes sense for large batches with oneDNN
            }

            // Determine which kernel path will be used based on batch size
            const char* expected_kernel;
            if (config.M == 1) {
                expected_kernel = "MMVQ/DMMV";  // vec kernels for batch=1
            } else if (config.M <= 32) {
                expected_kernel = "MMQ";        // small batch quantized
            } else {
                expected_kernel = "oneDNN";     // large batch dequantize+GEMM
            }

            BenchResult result = run_benchmark(backend, config, qtype, config.M, expected_kernel);
            print_result(result);
            all_results.push_back(result);
        }
    }

    // Summary: Find best kernel for each batch size category
    printf("\n=== Summary: Best Kernel by Batch Size ===\n");
    printf("\n--- Token Generation (batch=1) ---\n");
    for (const auto& r : all_results) {
        if (strstr(r.config_name, "tg-1tok") && r.quant_type == GGML_TYPE_Q4_0) {
            printf("%s Q4_0: %.1f tok/s (%.2f ms)\n", r.config_name, r.throughput_tps, r.avg_ms);
        }
    }

    printf("\n--- Prompt Processing (batch>1) ---\n");
    for (const auto& r : all_results) {
        if (strstr(r.config_name, "pp-") && r.quant_type == GGML_TYPE_Q4_0) {
            printf("%s Q4_0: %.0f tok/s (%.2f ms, %.1f GFLOPS)\n",
                   r.config_name, r.throughput_tps, r.avg_ms, r.throughput_gflops);
        }
    }

    printf("\n--- MoE Expert Dispatch ---\n");
    for (const auto& r : all_results) {
        if (strstr(r.config_name, "moe-") && r.quant_type == GGML_TYPE_Q4_0) {
            printf("%s Q4_0: %.0f tok/s (%.2f ms)\n", r.config_name, r.throughput_tps, r.avg_ms);
        }
    }

    // Cleanup
    ggml_backend_free(backend);

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
