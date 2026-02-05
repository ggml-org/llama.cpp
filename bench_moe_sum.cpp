// Simple benchmark for GGML_OP_MOE_SUM
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

static double get_time_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

struct BenchResult {
    double moe_sum_ms;
    double add_loop_ms;
    double speedup;
};

// Benchmark 1: Using moe_sum operator
static double benchmark_moe_sum(
    ggml_backend_t backend,
    int64_t hidden_dim,
    int64_t n_expert_used,
    int64_t n_tokens,
    int iterations) {

    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .no_alloc = true,
    };
    ggml_context * ctx = ggml_init(params);

    // Input: [hidden_dim, n_expert_used, n_tokens]
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, n_expert_used, n_tokens);
    ggml_tensor * output = ggml_moe_sum(ctx, input, n_expert_used);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    // Initialize input data
    std::vector<float> input_data(hidden_dim * n_expert_used * n_tokens);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = (float)(i % 100) / 100.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Warmup
    ggml_backend_graph_compute(backend, gf);

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    double end = get_time_ms();

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return end - start;
}

// Benchmark 2: Using traditional ADD loop (equivalent to CPU implementation)
static double benchmark_add_loop(
    ggml_backend_t backend,
    int64_t hidden_dim,
    int64_t n_expert_used,
    int64_t n_tokens,
    int iterations) {

    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .no_alloc = true,
    };
    ggml_context * ctx = ggml_init(params);

    // Input: [hidden_dim, n_expert_used, n_tokens]
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, n_expert_used, n_tokens);

    // Build graph: simulate moe_sum by creating views and adding them
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_tensor * zero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_tensor * cur = ggml_mul(ctx, result, zero);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Add each expert's contribution
    for (int64_t k = 0; k < n_expert_used; k++) {
        ggml_tensor * expert_view = ggml_view_3d(ctx, input,
            hidden_dim, n_tokens, 1,
            input->nb[0], input->nb[2], k * input->nb[1]);
        cur = ggml_add(ctx, cur, expert_view);
    }

    ggml_build_forward_expand(gf, cur);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    // Initialize input data
    std::vector<float> input_data(hidden_dim * n_expert_used * n_tokens);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = (float)(i % 100) / 100.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Warmup
    ggml_backend_graph_compute(backend, gf);

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    double end = get_time_ms();

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return end - start;
}

static BenchResult run_benchmark(
    ggml_backend_t backend,
    const char * backend_name,
    int64_t hidden_dim,
    int64_t n_expert_used,
    int64_t n_tokens,
    int iterations) {

    printf("\n=================================================\n");
    printf("Testing %s backend:\n", backend_name);
    printf("=================================================\n");
    printf("  Hidden dimension: %ld\n", hidden_dim);
    printf("  Number of experts: %ld\n", n_expert_used);
    printf("  Number of tokens: %ld\n", n_tokens);
    printf("  Iterations: %d\n", iterations);
    printf("=================================================\n");

    double time_moe_sum = benchmark_moe_sum(backend, hidden_dim, n_expert_used, n_tokens, iterations);
    double time_add_loop = benchmark_add_loop(backend, hidden_dim, n_expert_used, n_tokens, iterations);

    printf("\nResults (averaged over %d iterations):\n", iterations);
    if (time_moe_sum >= 0) {
        printf("  moe_sum:      %8.2f ms  (%8.2f us/iter)\n", time_moe_sum, time_moe_sum * 1000.0 / iterations);
    } else {
        printf("  moe_sum:      NOT SUPPORTED\n");
    }

    if (time_add_loop >= 0) {
        printf("  add_loop:     %8.2f ms  (%8.2f us/iter)\n", time_add_loop, time_add_loop * 1000.0 / iterations);
    }

    double speedup = 0.0;
    if (time_moe_sum >= 0 && time_add_loop >= 0) {
        speedup = time_add_loop / time_moe_sum;
        printf("\n  Speedup:      %.2fx\n", speedup);

        // Calculate effective bandwidth
        size_t bytes_read = hidden_dim * n_expert_used * n_tokens * sizeof(float);
        size_t bytes_written = hidden_dim * n_tokens * sizeof(float);
        size_t total_bytes = (bytes_read + bytes_written) * iterations;
        double gb_per_sec = (total_bytes / 1e9) / (time_moe_sum / 1000.0);
        printf("  moe_sum bandwidth: %.2f GB/s\n", gb_per_sec);
    }

    printf("=================================================\n");

    return {time_moe_sum, time_add_loop, speedup};
}

int main(int argc, char ** argv) {
    int64_t hidden_dim = 4096;
    int64_t n_expert_used = 4;
    int64_t n_tokens = 256;
    int iterations = 100;
    bool test_gpu = true;
    bool test_cpu = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--hidden") == 0) {
            if (i + 1 < argc) hidden_dim = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--experts") == 0) {
            if (i + 1 < argc) n_expert_used = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokens") == 0) {
            if (i + 1 < argc) n_tokens = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            test_gpu = false;
        } else if (strcmp(argv[i], "--gpu-only") == 0) {
            test_cpu = false;
        }
    }

    printf("=================================================\n");
    printf("GGML_OP_MOE_SUM Performance Benchmark\n");
    printf("=================================================\n");
    printf("Configuration:\n");
    printf("  Hidden dimension: %ld\n", hidden_dim);
    printf("  Number of experts: %ld\n", n_expert_used);
    printf("  Number of tokens: %ld\n", n_tokens);
    printf("  Iterations: %d\n", iterations);
    printf("=================================================\n\n");

    // Initialize backend - load all available backends
    ggml_backend_load_all();

    std::vector<BenchResult> results;

    // Test CPU backend
    if (test_cpu) {
        ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
        if (cpu_reg) {
            ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
            ggml_backend_t backend = ggml_backend_dev_init(cpu_dev, NULL);
            if (backend) {
                results.push_back(run_benchmark(backend, "CPU", hidden_dim, n_expert_used, n_tokens, iterations));
                ggml_backend_free(backend);
            }
        }
    }

    // Test GPU backend
    if (test_gpu) {
        ggml_backend_reg_t gpu_reg = ggml_backend_reg_by_name("CUDA");
        if (!gpu_reg) {
            gpu_reg = ggml_backend_reg_by_name("GPU");
        }
        if (gpu_reg) {
            ggml_backend_dev_t gpu_dev = ggml_backend_reg_dev_get(gpu_reg, 0);
            ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, NULL);
            if (backend) {
                results.push_back(run_benchmark(backend, "GPU", hidden_dim, n_expert_used, n_tokens, iterations));
                ggml_backend_free(backend);
            }
        }
    }

    // Summary
    if (results.size() >= 2) {
        printf("\n=================================================\n");
        printf("Performance Summary:\n");
        printf("=================================================\n");
        for (const auto& r : results) {
            printf("%s: %.2fx speedup\n", r.speedup > 0 ? "" : "GPU", r.speedup);
        }
        printf("=================================================\n");
    }

    return 0;
}
