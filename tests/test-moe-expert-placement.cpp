// test-moe-expert-placement.cpp
// Micro-benchmark: MoE expert placement strategy
//
// Question: Should MoE experts be statically placed on CPU (KTransformers-style)
// or dynamically shuffled between host and VRAM?
//
// Tests:
//   1. GPU compute on VRAM-resident expert (best case — expert already in VRAM)
//   2. CPU compute on host-resident expert (static CPU placement)
//   3. DMA H2D + GPU compute (dynamic: shuttle expert from host to VRAM)
//   4. Pipelined: DMA expert N+1 while GPU computes expert N
//   5. Break-even analysis: how many reuses justify DMA cost?
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -pthread -o test-moe-expert-placement tests/test-moe-expert-placement.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-moe-expert-placement
//
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

// ─── Timer ────────────────────────────────────────────────
struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ─── Simulated expert compute ─────────────────────────────
// Each expert weight is a [N, K] Q4_0 tensor.
// We simulate a mat-vec by reading all bytes and producing a reduction.
// This measures memory-bandwidth-bound compute (which is what TG is).

// GPU kernel: read expert weight bytes, produce checksum
static double bench_gpu_compute(sycl::queue & q, const uint8_t * buf, size_t nbytes, int iters) {
    const int wg_size   = 256;
    const int n_wgs     = std::min(512, (int)((nbytes + wg_size - 1) / wg_size));
    const int n_threads = n_wgs * wg_size;

    auto * d_out = sycl::malloc_device<uint32_t>(n_wgs, q);
    q.memset(d_out, 0, n_wgs * sizeof(uint32_t)).wait();

    // warmup
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
            uint32_t sum = 0;
            for (size_t i = it.get_global_id(0); i < nbytes; i += n_threads)
                sum += buf[i];
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
            aref.fetch_add(sum);
        });
    }).wait();

    timer t;
    for (int i = 0; i < iters; i++) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
                uint32_t sum = 0;
                for (size_t i = it.get_global_id(0); i < nbytes; i += n_threads)
                    sum += buf[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
                aref.fetch_add(sum);
            });
        });
    }
    q.wait();
    double elapsed = t.ms() / iters;
    sycl::free(d_out, q);
    return elapsed;
}

// CPU: multi-threaded read of expert weight
static double bench_cpu_compute(const uint8_t * buf, size_t nbytes, int n_threads, int iters) {
    std::vector<uint64_t> partial(n_threads, 0);

    auto worker = [&](int tid) {
        size_t chunk = (nbytes + n_threads - 1) / n_threads;
        size_t start = tid * chunk;
        size_t end   = std::min(start + chunk, nbytes);
        uint64_t sum = 0;
        for (size_t i = start; i < end; i++)
            sum += buf[i];
        partial[tid] = sum;
    };

    // warmup
    { std::vector<std::thread> ts;
      for (int i = 0; i < n_threads; i++) ts.emplace_back(worker, i);
      for (auto & th : ts) th.join(); }

    timer t;
    for (int it = 0; it < iters; it++) {
        std::vector<std::thread> ts;
        for (int i = 0; i < n_threads; i++) ts.emplace_back(worker, i);
        for (auto & th : ts) th.join();
    }
    double elapsed = t.ms() / iters;

    volatile uint64_t sink = 0;
    for (auto v : partial) sink += v;
    return elapsed;
}

// DMA: H2D transfer time
static double bench_h2d_dma(sycl::queue & q, const uint8_t * host_buf, uint8_t * dev_buf,
                             size_t nbytes, int iters) {
    // warmup
    q.memcpy(dev_buf, host_buf, nbytes).wait();

    timer t;
    for (int i = 0; i < iters; i++) {
        q.memcpy(dev_buf, host_buf, nbytes).wait();
    }
    return t.ms() / iters;
}

// DMA + GPU compute (sequential: transfer then compute)
static double bench_dma_then_gpu(sycl::queue & q, const uint8_t * host_buf, uint8_t * dev_buf,
                                  size_t nbytes, int iters) {
    const int wg_size   = 256;
    const int n_wgs     = std::min(512, (int)((nbytes + wg_size - 1) / wg_size));
    const int n_threads = n_wgs * wg_size;
    auto * d_out = sycl::malloc_device<uint32_t>(n_wgs, q);

    // warmup
    q.memcpy(dev_buf, host_buf, nbytes).wait();
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
            uint32_t sum = 0;
            for (size_t i = it.get_global_id(0); i < nbytes; i += n_threads)
                sum += dev_buf[i];
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
            aref.fetch_add(sum);
        });
    }).wait();

    timer t;
    for (int i = 0; i < iters; i++) {
        q.memcpy(dev_buf, host_buf, nbytes);
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
                uint32_t sum = 0;
                for (size_t i = it.get_global_id(0); i < nbytes; i += n_threads)
                    sum += dev_buf[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
                aref.fetch_add(sum);
            });
        });
        q.wait();
    }
    double elapsed = t.ms() / iters;
    sycl::free(d_out, q);
    return elapsed;
}

// Pipelined: DMA expert[i+1] overlapped with GPU compute on expert[i]
// Uses two device buffers and an OOQ for DMA overlap
static double bench_pipelined(sycl::queue & compute_q, sycl::queue & dma_q,
                               const uint8_t * host_buf, uint8_t * dev_buf_a, uint8_t * dev_buf_b,
                               size_t nbytes, int n_experts) {
    const int wg_size   = 256;
    const int n_wgs     = std::min(512, (int)((nbytes + wg_size - 1) / wg_size));
    const int n_threads = n_wgs * wg_size;
    auto * d_out = sycl::malloc_device<uint32_t>(n_wgs, compute_q);

    // Prime: load first expert into buf_a
    compute_q.memcpy(dev_buf_a, host_buf, nbytes).wait();

    timer t;
    for (int i = 0; i < n_experts; i++) {
        uint8_t * compute_buf = (i % 2 == 0) ? dev_buf_a : dev_buf_b;
        uint8_t * dma_buf     = (i % 2 == 0) ? dev_buf_b : dev_buf_a;

        // Start DMA for next expert (overlapped)
        sycl::event dma_evt;
        if (i + 1 < n_experts) {
            dma_evt = dma_q.memcpy(dma_buf, host_buf, nbytes);
        }

        // Compute on current expert
        compute_q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
                uint32_t sum = 0;
                for (size_t i = it.get_global_id(0); i < nbytes; i += n_threads)
                    sum += compute_buf[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
                aref.fetch_add(sum);
            });
        });

        // Wait for both
        compute_q.wait();
        if (i + 1 < n_experts) dma_evt.wait();
    }
    double elapsed = t.ms() / n_experts;

    sycl::free(d_out, compute_q);
    return elapsed;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  MoE Expert Placement Strategy Micro-Benchmark          ║\n");
    printf("║  Static CPU vs Dynamic GPU shuttle for routed experts   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    sycl::queue compute_q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    // Separate OOQ for DMA overlap
    sycl::queue dma_q(compute_q.get_context(), compute_q.get_device(),
                       sycl::property::queue::in_order());

    auto dev = compute_q.get_device();
    printf("GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    int n_cpu_cores = std::thread::hardware_concurrency();
    int n_cpu_threads = std::max(1, n_cpu_cores - 2);
    printf("CPU cores: %d, worker threads: %d\n\n", n_cpu_cores, n_cpu_threads);

    // Expert sizes for different MoE models
    // Each expert = gate_proj + up_proj + down_proj
    // Mixtral 8x7B: 3 * [4096, 14336] Q4_0 = 3 * 33 MB = ~99 MB per expert
    // DeepSeek-V3:  3 * [7168, 2048] Q4_0 = 3 * 6.3 MB = ~19 MB per expert (but 256 experts)
    // GPT-OSS-120B: varies, ~40-130 MB per expert depending on architecture
    struct expert_config {
        const char * name;
        size_t bytes;
    };

    expert_config configs[] = {
        { "DeepSeek-V3-like (19 MB)",    19 * 1024 * 1024 },
        { "GPT-OSS-120B-like (40 MB)",   40 * 1024 * 1024 },
        { "Mixtral-like (99 MB)",         99 * 1024 * 1024 },
        { "Large expert (128 MB)",       128 * 1024 * 1024 },
    };

    const int ITERS = 10;

    for (auto & cfg : configs) {
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Expert: %s\n", cfg.name);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Allocate
        auto * host_buf  = sycl::malloc_host<uint8_t>(cfg.bytes, compute_q);
        auto * dev_buf_a = sycl::malloc_device<uint8_t>(cfg.bytes, compute_q);
        auto * dev_buf_b = sycl::malloc_device<uint8_t>(cfg.bytes, compute_q);

        if (!host_buf || !dev_buf_a || !dev_buf_b) {
            printf("  SKIP: allocation failed\n\n");
            if (host_buf)  sycl::free(host_buf, compute_q);
            if (dev_buf_a) sycl::free(dev_buf_a, compute_q);
            if (dev_buf_b) sycl::free(dev_buf_b, compute_q);
            continue;
        }

        // Fill with data
        memset(host_buf, 0x42, cfg.bytes);
        compute_q.memcpy(dev_buf_a, host_buf, cfg.bytes).wait();

        // ─── Test 1: GPU compute on VRAM expert ──────────
        double gpu_vram_ms = bench_gpu_compute(compute_q, dev_buf_a, cfg.bytes, ITERS);
        printf("  1. GPU compute (VRAM):       %8.3f ms  (%6.2f GB/s)\n",
            gpu_vram_ms, cfg.bytes / (gpu_vram_ms * 1e6));

        // ─── Test 2: CPU compute on host expert ──────────
        double cpu_host_ms = bench_cpu_compute(host_buf, cfg.bytes, n_cpu_threads, ITERS);
        printf("  2. CPU compute (host, %2dT):  %8.3f ms  (%6.2f GB/s)\n",
            n_cpu_threads, cpu_host_ms, cfg.bytes / (cpu_host_ms * 1e6));

        // ─── Test 3: H2D DMA only ───────────────────────
        double dma_ms = bench_h2d_dma(compute_q, host_buf, dev_buf_a, cfg.bytes, ITERS);
        printf("  3. H2D DMA only:             %8.3f ms  (%6.2f GB/s)\n",
            dma_ms, cfg.bytes / (dma_ms * 1e6));

        // ─── Test 4: DMA + GPU compute (sequential) ─────
        double dma_gpu_ms = bench_dma_then_gpu(compute_q, host_buf, dev_buf_a, cfg.bytes, ITERS);
        printf("  4. DMA + GPU (sequential):   %8.3f ms  (%6.2f GB/s)\n",
            dma_gpu_ms, cfg.bytes / (dma_gpu_ms * 1e6));

        // ─── Test 5: Pipelined DMA + GPU (double buffer) ─
        double pipe_ms = bench_pipelined(compute_q, dma_q, host_buf, dev_buf_a, dev_buf_b,
                                          cfg.bytes, 8);
        printf("  5. Pipelined (DMA||GPU):     %8.3f ms  (%6.2f GB/s)\n",
            pipe_ms, cfg.bytes / (pipe_ms * 1e6));

        // ─── Test 6: GPU zero-copy (host-pinned) ────────
        double gpu_zc_ms = bench_gpu_compute(compute_q, host_buf, cfg.bytes, ITERS);
        printf("  6. GPU zero-copy (host):     %8.3f ms  (%6.2f GB/s)\n",
            gpu_zc_ms, cfg.bytes / (gpu_zc_ms * 1e6));

        // ─── Analysis ────────────────────────────────────
        printf("\n  ┌─ ANALYSIS ─────────────────────────────────────────┐\n");

        // Break-even: how many consecutive uses justify DMA?
        // DMA cost = dma_ms
        // Per-use savings = cpu_host_ms - gpu_vram_ms
        double per_use_savings = cpu_host_ms - gpu_vram_ms;
        int break_even = (per_use_savings > 0) ?
            (int)std::ceil(dma_ms / per_use_savings) : -1;

        printf("  │ GPU VRAM vs CPU host:   GPU is %.1fx %s\n",
            gpu_vram_ms < cpu_host_ms ? cpu_host_ms / gpu_vram_ms : gpu_vram_ms / cpu_host_ms,
            gpu_vram_ms < cpu_host_ms ? "faster" : "slower");
        printf("  │ DMA cost:              %.3f ms (%.1f%% of CPU compute)\n",
            dma_ms, 100.0 * dma_ms / cpu_host_ms);
        printf("  │ DMA + GPU vs CPU-only: DMA+GPU is %.1fx %s\n",
            dma_gpu_ms < cpu_host_ms ? cpu_host_ms / dma_gpu_ms : dma_gpu_ms / cpu_host_ms,
            dma_gpu_ms < cpu_host_ms ? "faster" : "SLOWER");
        printf("  │ Pipelined vs CPU-only: Pipeline is %.1fx %s\n",
            pipe_ms < cpu_host_ms ? cpu_host_ms / pipe_ms : pipe_ms / cpu_host_ms,
            pipe_ms < cpu_host_ms ? "faster" : "SLOWER");

        if (break_even > 0) {
            printf("  │ Break-even reuses:    %d consecutive uses to justify DMA\n", break_even);
        } else {
            printf("  │ Break-even: N/A (CPU is faster than GPU VRAM)\n");
        }

        // Token-level analysis for MoE
        // Typical: 2-8 experts activated per token, each expert used for 1 mat-vec
        printf("  │\n");
        printf("  │ Per-token MoE cost (8 experts, typical DeepSeek):\n");
        printf("  │   All on CPU:              %8.3f ms (8 × %.3f ms)\n",
            8.0 * cpu_host_ms, cpu_host_ms);
        printf("  │   All on GPU (VRAM):       %8.3f ms (8 × %.3f ms)\n",
            8.0 * gpu_vram_ms, gpu_vram_ms);
        printf("  │   DMA+GPU (cold miss):     %8.3f ms (8 × %.3f ms)\n",
            8.0 * dma_gpu_ms, dma_gpu_ms);
        printf("  │   GPU+CPU parallel (4+4):  %8.3f ms\n",
            std::max(4.0 * gpu_vram_ms, 4.0 * cpu_host_ms));
        printf("  └────────────────────────────────────────────────────┘\n\n");

        sycl::free(host_buf, compute_q);
        sycl::free(dev_buf_a, compute_q);
        sycl::free(dev_buf_b, compute_q);
    }

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  PLACEMENT STRATEGY GUIDE                               ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  If DMA+GPU > CPU for all sizes:                        ║\n");
    printf("║    → Static CPU placement wins. Never shuttle experts.  ║\n");
    printf("║    → KTransformers approach: all experts on CPU.        ║\n");
    printf("║                                                          ║\n");
    printf("║  If GPU VRAM >> CPU and break-even < 3:                 ║\n");
    printf("║    → Dynamic caching with LRU. Shuttle hot experts.     ║\n");
    printf("║    → Prefetch next layer's experts during current layer. ║\n");
    printf("║                                                          ║\n");
    printf("║  If parallel GPU+CPU < either alone:                    ║\n");
    printf("║    → Split experts: hot on GPU, cold on CPU, parallel.  ║\n");
    printf("║    → Static placement with research-guided hot set.     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
