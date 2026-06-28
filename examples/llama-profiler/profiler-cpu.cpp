#define CPU_WARMUP_ITERS  2
#define CPU_TIMED_ITERS   2

#include "profiler-common.h"

#include "common.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <atomic>
#include <cassert>
#include <functional>
#include <thread>

#if defined(_MSC_VER)
#include <malloc.h>
#include <intrin.h>
#endif

static std::vector<char> g_flush_buffer;

static void flush_caches() {
    if (g_flush_buffer.empty()) return;

    volatile char sum = 0;
    for (size_t i = 0; i < g_flush_buffer.size(); i += 64) {
        sum += g_flush_buffer[i];
        g_flush_buffer[i] = (char)(i & 0xFF);
    }
    g_flush_buffer[0] = sum;

#if defined(_MSC_VER)
    _mm_mfence();
#elif defined(__GNUC__) || defined(__clang__)
    __sync_synchronize();
#endif
}

static void init_flush_buffer() {
    const size_t flush_size = 256 * 1024 * 1024;
    g_flush_buffer.resize(flush_size);
    for (size_t i = 0; i < flush_size; i += 4096) {
        g_flush_buffer[i] = (char)(i & 0xFF);
    }
}

static double benchmark_cpu_dram_bandwidth(int threads) {
    const size_t pool_bytes = 1024ULL * 1024 * 1024;
    const size_t chunk_per_thread = pool_bytes / threads;
    const int iterations = 10;

    std::vector<uint8_t> pool(pool_bytes);
    for (size_t i = 0; i < pool.size(); i += 4096) {
        pool[i] = (uint8_t)(i & 0xFF);
    }

    std::vector<std::thread> workers;
    std::vector<double> thread_bytes(threads, 0.0);

    bench_timer t;
    t.start();

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&pool, &thread_bytes, tid, chunk_per_thread, iterations, pool_bytes]() {
            const size_t start = tid * chunk_per_thread;
            const size_t end_pos = (tid == (int)(pool_bytes / chunk_per_thread) - 1) ? pool_bytes : (start + chunk_per_thread);
            volatile uint64_t local_sink = 0;
            double local_bytes = 0.0;
            for (int iter = 0; iter < iterations; ++iter) {
                const size_t limit = end_pos - sizeof(uint64_t);
                for (size_t offset = start; offset + 64 <= limit; offset += 64) {
                    local_sink += *(const uint64_t *)(pool.data() + offset);
                    local_bytes += 64.0;
                }
            }
            thread_bytes[tid] = local_bytes;
            (void)local_sink;
        });
    }
    for (auto & w : workers) w.join();

    double elapsed = t.stop();
    double total_bytes = 0.0;
    for (int i = 0; i < threads; i++) total_bytes += thread_bytes[i];
    return total_bytes / elapsed / 1e9;
}

struct pcie_stress_ctx {
    std::atomic<bool> active{false};
    std::atomic<bool> stop{false};

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_buffer_t host_buf = nullptr;
    ggml_backend_buffer_t dev_buf  = nullptr;
    ggml_tensor * h_tensor = nullptr;
    ggml_tensor * d_tensor = nullptr;
    ggml_context * ctx = nullptr;
    size_t transfer_size = 256 * 1024 * 1024;
    double calibrated_bw_gb_s = 0.0;
};

static void pcie_stress_loop(pcie_stress_ctx * pcie) {
    pcie->active.store(true, std::memory_order_release);
    while (!pcie->stop.load(std::memory_order_acquire)) {
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
    }
    pcie->active.store(false, std::memory_order_release);
}

static void calibrate_pcie(pcie_stress_ctx * pcie) {
    printf("Calibrating standalone PCIe bandwidth...\n");
    bench_timer t;
    t.start();
    const int cal_iterations = 20;
    for (int i = 0; i < cal_iterations; ++i) {
        ggml_backend_tensor_set_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
        ggml_backend_tensor_get_async(pcie->gpu_backend, pcie->d_tensor,
            pcie->h_tensor->data, 0, pcie->transfer_size);
        ggml_backend_synchronize(pcie->gpu_backend);
    }
    double elapsed = t.stop();
    double bytes_moved = (double)cal_iterations * pcie->transfer_size * 2.0;
    pcie->calibrated_bw_gb_s = bytes_moved / elapsed / 1e9;
    printf("  Standalone PCIe BW: %.1f GB/s\n\n", pcie->calibrated_bw_gb_s);
}

struct bench_result_cpu : bench_result {
    int threads = 0;
    float standalone_gflops = 0.0f;
    float concurrent_gflops = 0.0f;
    float concurrent_efficiency_pct = 0.0f;
    float pcie_standalone_bw_gb_s = 0.0f;

    void print(double pcie_bw_ref = 0.0) const {
        printf("%-20s quant=%-6s threads=%d AI=%.3f FLOP/byte BW=%.2f GB/s Perf=%.2f GFLOP/s",
            op_name.c_str(), quant_type.c_str(), threads,
            arithmetic_intensity, effective_bw_gb_s, effective_gflops);
        if (concurrent_gflops > 0) {
            printf(" | Concur=%.2f (%.1f%%)", concurrent_gflops, concurrent_efficiency_pct);
            if (pcie_bw_ref > 0) {
                double est_pcie_bw = pcie_bw_ref * (concurrent_efficiency_pct / 100.0) * 0.9;
                printf(" PCIe~%.1f GB/s (%.1f%%)", est_pcie_bw, 100.0 * est_pcie_bw / pcie_bw_ref);
            }
        }
        print_dims();
        printf("\n");
    }
};

static double benchmark_mul_mat_raw(
        ggml_backend_t be, int N, int K, int batch_size,
        ggml_type quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 4096ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * A = ggml_new_tensor_2d(ctx, quant, K, N);
    ggml_tensor * B_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, batch_size);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B_tensor);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: MUL_MAT N=%d K=%d B=%d %s (alloc failed)\n", N, K, batch_size, ggml_type_name(quant));
        ggml_free(ctx); return 0.0;
    }

    std::vector<uint8_t> A_data = create_quantized_data(quant, (int64_t)K * N);
    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));
    ggml_backend_tensor_set(B_tensor, B_data.data(), 0, ggml_nbytes(B_tensor));

    for (int i = 0; i < CPU_WARMUP_ITERS; ++i) { flush_caches(); ggml_backend_graph_compute_async(be, gf); }

    double total_time = 0.0;
    bench_timer t;
    for (int i = 0; i < CPU_TIMED_ITERS; ++i) {
        flush_caches();
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }

    double time_per_iter = total_time / CPU_TIMED_ITERS;
    double ops_total = 2.0 * N * K * batch_size;
    double bytes_total = (double)(ggml_nbytes(A) + ggml_nbytes(B_tensor) + ggml_nbytes(C));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_mul_mat_id_raw(
        ggml_backend_t be, int N, int K, int n_experts, int n_experts_used,
        int batch_size, ggml_type quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * A   = ggml_new_tensor_3d(ctx, quant, K, N, n_experts);
    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, batch_size);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts_used, batch_size);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_mul_mat_id(ctx, A, B, ids));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: MUL_MAT_ID N=%d K=%d B=%d (alloc failed)\n", N, K, batch_size);
        ggml_free(ctx); return 0.0;
    }

    std::vector<uint8_t> A_data = create_quantized_data(quant, (int64_t)K * N * n_experts);
    ggml_backend_tensor_set(A, A_data.data(), 0, ggml_nbytes(A));
    std::vector<float> B_data(K * batch_size, 1.0f);
    ggml_backend_tensor_set(B, B_data.data(), 0, ggml_nbytes(B));
    std::vector<int32_t> ids_data(n_experts_used * batch_size);
    for (int i = 0; i < n_experts_used * batch_size; i++) ids_data[i] = i % n_experts;
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

    for (int i = 0; i < CPU_WARMUP_ITERS; ++i) { flush_caches(); ggml_backend_graph_compute_async(be, gf); }

    double total_time = 0.0;
    bench_timer t;
    for (int i = 0; i < CPU_TIMED_ITERS; ++i) {
        flush_caches();
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }

    double time_per_iter = total_time / CPU_TIMED_ITERS;
    double ops_total = 2.0 * N * K * batch_size * n_experts_used;
    double bytes_total = (double)(ggml_nbytes(A) * n_experts_used / n_experts) + ggml_nbytes(B) + (double)(N * batch_size * n_experts_used * 4);

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static double benchmark_flash_attn_raw(
        ggml_backend_t be, int n_tokens, int ctx_len,
        int n_q_heads, int n_kv_heads, int head_dim,
        ggml_type kv_quant, int threads,
        double * out_time_s, double * out_ops, double * out_bytes) {

    ggml_init_params params = { 8192ULL * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_cpu_set_n_threads(be, threads);

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_q_heads, 1);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, kv_quant, head_dim, ctx_len, n_kv_heads, 1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buffer) {
        printf("SKIPPED: FLASH_ATTN (alloc failed)\n");
        ggml_free(ctx); return 0.0;
    }

    std::vector<float> Q_data(ggml_nelements(Q), 1.0f);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    std::vector<uint8_t> KV_data = create_quantized_data(kv_quant, ggml_nelements(K));
    ggml_backend_tensor_set(K, KV_data.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, KV_data.data(), 0, ggml_nbytes(V));

    for (int i = 0; i < CPU_WARMUP_ITERS; ++i) { flush_caches(); ggml_backend_graph_compute_async(be, gf); }

    double total_time = 0.0;
    bench_timer t;
    for (int i = 0; i < CPU_TIMED_ITERS; ++i) {
        flush_caches();
        t.start();
        ggml_backend_graph_compute_async(be, gf);
        total_time += t.stop();
    }

    double time_per_iter = total_time / CPU_TIMED_ITERS;
    double ops_total = 2.0 * n_tokens * head_dim * ctx_len * n_q_heads * 2;
    double bytes_total = (double)(ggml_nbytes(Q) + ggml_nbytes(K) + ggml_nbytes(V) + ggml_nbytes(out));

    if (out_time_s) *out_time_s = time_per_iter;
    if (out_ops) *out_ops = ops_total;
    if (out_bytes) *out_bytes = bytes_total;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ops_total / time_per_iter / 1e9;
}

static bench_result_cpu run_concurrent(
        std::function<double(double *, double *, double *)> bench_fn,
        const std::string & op_name, const char * quant_name, int threads,
        pcie_stress_ctx * pcie) {

    bench_result_cpu result;
    result.op_name = op_name;
    result.quant_type = quant_name;
    result.threads = threads;
    result.pcie_standalone_bw_gb_s = pcie ? (float)pcie->calibrated_bw_gb_s : 0.0f;

    double standalone = bench_fn(&result.time_s, &result.ops, &result.bytes);
    result.calculate_derived();
    result.standalone_gflops = result.effective_gflops;

    if (pcie && pcie->gpu_backend) {
        pcie->stop.store(false, std::memory_order_release);
        std::thread pcie_thread(pcie_stress_loop, pcie);
        while (!pcie->active.load(std::memory_order_acquire)) std::this_thread::yield();

        result.concurrent_gflops = (float)bench_fn(nullptr, nullptr, nullptr);

        pcie->stop.store(true, std::memory_order_release);
        pcie_thread.join();
    } else {
        result.concurrent_gflops = result.standalone_gflops;
    }

    result.concurrent_efficiency_pct = (standalone > 0.0)
        ? (float)(std::min)(100.0, 100.0 * result.concurrent_gflops / result.standalone_gflops)
        : 100.0f;

    return result;
}

static void run_matmul_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto sizes  = get_matmul_sizes(fast);
    auto quants = get_matmul_quants(fast);

    printf("=== MUL_MAT Operations ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            printf("  [Batch=%d]\n", bs);
            for (const auto & sz : sizes) {
                if (qt == GGML_TYPE_Q2_K && (sz.K % 256 != 0)) continue;

                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_mul_mat_raw(be, sz.N, sz.K, bs, qt, threads, t, o, b);
                    }, "MUL_MAT", ggml_type_name(qt), threads, pcie);
                res.N = sz.N; res.K = sz.K; res.B = bs;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void run_moe_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto configs = get_moe_configs(fast);
    auto quants  = get_matmul_quants(fast);

    printf("=== MUL_MAT_ID Operations (MoE) ===\n\n");
    for (ggml_type qt : quants) {
        printf("--- MoE Quantization: %s ---\n", ggml_type_name(qt));
        for (int32_t bs : batch_sizes) {
            printf("  [Batch=%d]\n", bs);
            for (const auto & cfg : configs) {
                if (qt == GGML_TYPE_Q2_K && (cfg.K % 256 != 0)) continue;

                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_mul_mat_id_raw(be, cfg.N, cfg.K, cfg.n_experts, cfg.n_experts_used, bs, qt, threads, t, o, b);
                    }, "MUL_MAT_ID", ggml_type_name(qt), threads, pcie);
                res.N = cfg.N; res.K = cfg.K; res.B = bs;
                res.n_tokens = cfg.n_experts_used; res.ctx_len = cfg.n_experts;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void run_attention_benchmarks(
        ggml_backend_t be, int threads, const std::vector<int32_t> & batch_sizes,
        bool fast, pcie_stress_ctx * pcie,
        std::vector<bench_result_cpu> & results) {

    auto configs  = get_attn_configs(fast);
    auto ctx_lens = get_attn_ctx_lens(fast);

    printf("=== FLASH_ATTN Operations ===\n\n");
    for (const auto & cfg : configs) {
        printf("--- %s (n_q=%d, n_kv=%d, head_dim=%d) ---\n", cfg.name, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim);
        for (int32_t n_tok : batch_sizes) {
            printf("  [n_tokens=%d]\n", n_tok);
            for (int32_t cl : ctx_lens) {
                auto res = run_concurrent(
                    [&](double * t, double * o, double * b) {
                        return benchmark_flash_attn_raw(be, n_tok, cl, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, GGML_TYPE_F16, threads, t, o, b);
                    }, std::string("FLASH_ATTN_") + cfg.name, ggml_type_name(GGML_TYPE_F16), threads, pcie);
                res.n_tokens = n_tok; res.ctx_len = cl; res.n_heads = cfg.n_kv_heads; res.head_dim = cfg.head_dim;
                res.print(pcie ? pcie->calibrated_bw_gb_s : 0.0);
                results.push_back(res);
            }
        }
        printf("\n");
    }
}

static void save_results_cpu(
        const char * path,
        const std::vector<bench_result_cpu> & results,
        const std::vector<int32_t> & batch_sizes,
        int threads, double dram_bw, double pcie_standalone_bw, double pcie_concurrent_bw, double cpu_eff,
        bool has_gpu) {

    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); return; }

    fprintf(f, "# Concurrent Profiling (threads=%d, batch_sizes=[", threads);
    for (size_t i = 0; i < batch_sizes.size(); i++)
        fprintf(f, "%d%s", batch_sizes[i], i + 1 < batch_sizes.size() ? "," : "");
    fprintf(f, "])\n");

    fprintf(f, "# Measured Bandwidths Per Thread Count:\n");
    if (has_gpu) {
        fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s, PCIe_Standalone=%.1f GB/s, PCIe_Concurrent=%.1f GB/s (CPU_Eff=%.1f%%)\n",
            threads, dram_bw, pcie_standalone_bw, pcie_concurrent_bw, cpu_eff);
    } else {
        fprintf(f, "#   Threads=%d: DRAM_BW=%.1f GB/s\n", threads, dram_bw);
    }

    fprintf(f, "# op_name quant threads AI(FLOP/byte) BW(GB/s) GFLOP/s Ridge(FLOP/byte) Concurrent_GFLOP/s PCIe_Concurrent_BW N K B n_tokens ctx_len n_heads head_dim n_elements\n");

    auto ridges = compute_ridge_points(results, dram_bw);
    std::map<std::string, double> ridge_map;
    for (const auto & rr : ridges) ridge_map[rr.key] = rr.ridge;

    for (const auto & r : results) {
        std::string key = r.op_name + "_" + r.quant_type;
        double ridge = ridge_map.count(key) ? ridge_map[key] : 0.0;
        double est_pcie = pcie_standalone_bw * (r.standalone_gflops > 0 ? r.concurrent_gflops / r.standalone_gflops : 1.0) * 0.9;

        fprintf(f, "%s %s %d %.4f %.2f %.2f %.4f %.2f %.2f %d %d %d %d %d %d %d %lld\n",
            r.op_name.c_str(), r.quant_type.c_str(), r.threads,
            r.arithmetic_intensity, r.effective_bw_gb_s, r.effective_gflops, ridge,
            r.concurrent_gflops, est_pcie,
            r.N, r.K, r.B, r.n_tokens, r.ctx_len, r.n_heads, r.head_dim, (long long)r.n_elements);
    }

    fclose(f);
    printf("Results saved to %s (%zu benchmarks)\n", path, results.size());
}

int main(int argc, char ** argv) {
    int32_t fixed_threads = -1;
    bool    fast_mode     = true;
    const char * output_path = "cpu_profile.txt";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], fixed_threads) || fixed_threads <= 0) {
                fprintf(stderr, "Invalid --threads value: %s\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--fast")) {
            fast_mode = true;
        } else if (!strcmp(argv[i], "--full")) {
            fast_mode = false;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [options]\n", argv[0]);
            printf("\n");
            printf("options:\n");
            printf("  -h, --help\n");
            printf("  --fast              fast mode with fewer configs (default)\n");
            printf("  --full              full mode with all configs\n");
            printf("  --threads <n>       number of CPU threads (default: auto)\n");
            printf("  --output <path>     output file (default: cpu_profile.txt)\n");
            return 0;
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    int32_t default_threads = cpu_get_num_math();
    int threads = (fixed_threads > 0) ? fixed_threads : default_threads;
    std::vector<int32_t> batch_sizes = { 1, 64, 512 };

    printf("=== CPU Profiler (cold-cache) ===\n");
    printf("Threads: %d%s\n", threads, fixed_threads > 0 ? " (user)" : " (auto)");
    printf("Mode:    %s\n\n", fast_mode ? "FAST" : "FULL");

    init_flush_buffer();

    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_be) { fprintf(stderr, "Failed to initialize CPU backend\n"); return 1; }

    ggml_quantize_init(GGML_TYPE_Q2_K);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    ggml_quantize_init(GGML_TYPE_Q4_1);
    ggml_quantize_init(GGML_TYPE_Q5_0);
    ggml_quantize_init(GGML_TYPE_Q8_0);
    ggml_quantize_init(GGML_TYPE_MXFP4);

    pcie_stress_ctx pcie;
    pcie.gpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    bool has_gpu = (pcie.gpu_backend != nullptr);

    if (has_gpu) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(pcie.gpu_backend));
        if (host_buft) {
            pcie.host_buf = ggml_backend_buft_alloc_buffer(host_buft, pcie.transfer_size);
            pcie.dev_buf  = ggml_backend_alloc_buffer(pcie.gpu_backend, pcie.transfer_size);
            ggml_init_params p = { pcie.transfer_size + 8 * 1024 * 1024, NULL, true };
            pcie.ctx = ggml_init(p);
            pcie.h_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
            pcie.d_tensor = ggml_new_tensor_1d(pcie.ctx, GGML_TYPE_F32, pcie.transfer_size / 4);
            ggml_backend_tensor_alloc(pcie.host_buf, pcie.h_tensor, ggml_backend_buffer_get_base(pcie.host_buf));
            ggml_backend_tensor_alloc(pcie.dev_buf,  pcie.d_tensor, ggml_backend_buffer_get_base(pcie.dev_buf));
            std::vector<float> init_data(pcie.transfer_size / 4, 1.0f);
            ggml_backend_tensor_set(pcie.h_tensor, init_data.data(), 0, pcie.transfer_size);
            printf("GPU: %s\n", ggml_backend_name(pcie.gpu_backend));
        } else {
            has_gpu = false;
        }
    }
    if (!has_gpu) printf("No GPU — standalone mode\n\n");

    printf("Measuring DRAM bandwidth...\n");
    double dram_bw = benchmark_cpu_dram_bandwidth(threads);
    printf("  DRAM BW: %.1f GB/s\n", dram_bw);
    if (has_gpu) calibrate_pcie(&pcie);

    std::vector<bench_result_cpu> all_results;
    bench_timer overall;
    overall.start();

    run_matmul_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);
    run_moe_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);
    run_attention_benchmarks(cpu_be, threads, batch_sizes, fast_mode, has_gpu ? &pcie : nullptr, all_results);

    double pcie_concurrent_bw = 0.0, cpu_eff = 100.0;
    if (has_gpu && !all_results.empty()) {
        double sum_s = 0.0, sum_c = 0.0;
        for (const auto & r : all_results) { sum_s += r.standalone_gflops; sum_c += r.concurrent_gflops; }
        cpu_eff = (sum_s > 0) ? 100.0 * sum_c / sum_s : 100.0;
        pcie_concurrent_bw = pcie.calibrated_bw_gb_s * (cpu_eff / 100.0) * 0.9;
    }

    printf("\nTotal time: %.1f s, %zu benchmarks\n", overall.stop(), all_results.size());

    save_results_cpu(output_path, all_results, batch_sizes, threads, dram_bw,
        pcie.calibrated_bw_gb_s, pcie_concurrent_bw, cpu_eff, has_gpu);

    if (has_gpu) {
        if (pcie.ctx) ggml_free(pcie.ctx);
        if (pcie.host_buf) ggml_backend_buffer_free(pcie.host_buf);
        if (pcie.dev_buf) ggml_backend_buffer_free(pcie.dev_buf);
        ggml_backend_free(pcie.gpu_backend);
    }
    ggml_backend_free(cpu_be);
    ggml_quantize_free();
    return 0;
}
