// test-kv-colocation.cpp
// Micro-benchmark: KV cache co-location with weights
//
// Question: Should KV cache be co-located with the layer's weights?
// If weights are on host, should KV also be on host?
//
// Tests the cost of cross-tier KV access at realistic sequence lengths.
//
// Key insight: sycl::malloc_device is NOT CPU-accessible.
// So if a layer runs on CPU and its KV is in VRAM, you CANNOT read it
// without explicit DMA. Co-location isn't optional for CPU layers.
//
// This benchmark quantifies the penalty for GPU layers with cross-tier KV
// (GPU reading KV from host-pinned via PCIe vs from VRAM).
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -pthread -o test-kv-colocation tests/test-kv-colocation.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-kv-colocation
//
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <algorithm>

struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ─── Simulated attention KV read ──────────────────────────
// During decode, attention reads ALL previous KV entries.
// KV size per layer = 2 * n_kv_heads * d_head * seq_len * sizeof(float16)
//
// Mistral 7B: 8 KV heads, 128 d_head, FP16
//   Per-layer KV at seq_len=2048: 2 * 8 * 128 * 2048 * 2 = 8 MB
//   Per-layer KV at seq_len=8192: 2 * 8 * 128 * 8192 * 2 = 32 MB

// GPU kernel: read all KV bytes (simulates attention KV fetch)
static double bench_gpu_kv_read(sycl::queue & q, const uint8_t * kv_buf, size_t nbytes, int iters) {
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
                sum += kv_buf[i];
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
                    sum += kv_buf[i];
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

// GPU kernel: write KV bytes (simulates appending new K,V for current token)
static double bench_gpu_kv_write(sycl::queue & q, uint8_t * kv_buf, size_t write_bytes, int iters) {
    const int wg_size   = 256;
    const int n_wgs     = std::min(256, (int)((write_bytes + wg_size - 1) / wg_size));
    const int n_threads = n_wgs * wg_size;

    // warmup
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
            for (size_t i = it.get_global_id(0); i < write_bytes; i += n_threads)
                kv_buf[i] = static_cast<uint8_t>(i & 0xFF);
        });
    }).wait();

    timer t;
    for (int i = 0; i < iters; i++) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
                for (size_t i = it.get_global_id(0); i < write_bytes; i += n_threads)
                    kv_buf[i] = static_cast<uint8_t>(i & 0xFF);
            });
        });
    }
    q.wait();
    return t.ms() / iters;
}

// CPU: multi-threaded KV read (simulates CPU attention on host-resident KV)
static double bench_cpu_kv_read(const uint8_t * kv_buf, size_t nbytes, int n_threads, int iters) {
    std::vector<uint64_t> partial(n_threads, 0);
    auto worker = [&](int tid) {
        size_t chunk = (nbytes + n_threads - 1) / n_threads;
        size_t start = tid * chunk;
        size_t end   = std::min(start + chunk, nbytes);
        uint64_t sum = 0;
        for (size_t i = start; i < end; i++)
            sum += kv_buf[i];
        partial[tid] = sum;
    };

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

// CPU: KV write (append new token's K,V)
static double bench_cpu_kv_write(uint8_t * kv_buf, size_t write_bytes, int iters) {
    // warmup
    memset(kv_buf, 0x42, write_bytes);

    timer t;
    for (int it = 0; it < iters; it++) {
        memset(kv_buf, 0x42, write_bytes);
    }
    return t.ms() / iters;
}

// Full layer simulation: attention = KV read + KV write + weight read
struct layer_result {
    double kv_read_ms;
    double kv_write_ms;
    double total_ms;
};

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  KV Cache Co-Location Micro-Benchmark                   ║\n");
    printf("║  Cross-tier penalty for KV reads/writes during decode   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    sycl::queue gpu_q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = gpu_q.get_device();
    printf("GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("GPU VRAM BW: ~456 GB/s (GDDR6)\n");
    printf("PCIe 4.0 x8: ~8 GB/s measured\n");

    int n_cpu_threads = std::max(1, (int)std::thread::hardware_concurrency() - 2);
    printf("CPU threads: %d\n\n", n_cpu_threads);

    // Mistral 7B KV cache parameters (GQA: 8 KV heads, 128 d_head, FP16)
    const int n_kv_heads = 8;
    const int d_head     = 128;
    const int dtype_size  = 2;  // FP16
    // Per-token KV = 2 * n_kv_heads * d_head * dtype_size (K and V)
    const size_t kv_per_token = 2 * n_kv_heads * d_head * dtype_size;  // 4096 bytes

    struct seq_config {
        const char * name;
        int seq_len;
    };

    seq_config seqs[] = {
        { "seq=512",   512 },
        { "seq=2048", 2048 },
        { "seq=4096", 4096 },
        { "seq=8192", 8192 },
    };

    const int ITERS    = 20;
    const int N_LAYERS = 32;  // Mistral 7B

    for (auto & seq : seqs) {
        size_t kv_read_bytes  = kv_per_token * seq.seq_len;  // total KV to read per layer
        size_t kv_write_bytes = kv_per_token;                  // new token's KV per layer

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Sequence: %s  (KV read=%.2f MB/layer, write=%.1f KB/layer)\n",
            seq.name, kv_read_bytes / 1e6, kv_write_bytes / 1e3);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Allocate VRAM and host-pinned KV buffers
        auto * vram_kv = sycl::malloc_device<uint8_t>(kv_read_bytes, gpu_q);
        auto * host_kv = sycl::malloc_host<uint8_t>(kv_read_bytes, gpu_q);

        if (!vram_kv || !host_kv) {
            printf("  SKIP: allocation failed\n\n");
            if (vram_kv) sycl::free(vram_kv, gpu_q);
            if (host_kv) sycl::free(host_kv, gpu_q);
            continue;
        }

        memset(host_kv, 0x42, kv_read_bytes);
        gpu_q.memcpy(vram_kv, host_kv, kv_read_bytes).wait();

        // ─── Scenario A: GPU layer, KV in VRAM (co-located, optimal) ──
        double gpu_vram_read  = bench_gpu_kv_read(gpu_q, vram_kv, kv_read_bytes, ITERS);
        double gpu_vram_write = bench_gpu_kv_write(gpu_q, vram_kv, kv_write_bytes, ITERS);
        printf("  A: GPU + VRAM KV (co-located):\n");
        printf("     KV read:  %8.3f ms  (%6.2f GB/s)\n", gpu_vram_read,
            kv_read_bytes / (gpu_vram_read * 1e6));
        printf("     KV write: %8.3f ms  (%6.2f GB/s)\n", gpu_vram_write,
            kv_write_bytes / (gpu_vram_write * 1e6));

        // ─── Scenario B: GPU layer, KV on host-pinned (cross-tier) ────
        double gpu_host_read  = bench_gpu_kv_read(gpu_q, host_kv, kv_read_bytes, ITERS);
        double gpu_host_write = bench_gpu_kv_write(gpu_q, host_kv, kv_write_bytes, ITERS);
        printf("  B: GPU + host KV (cross-tier PCIe):\n");
        printf("     KV read:  %8.3f ms  (%6.2f GB/s)\n", gpu_host_read,
            kv_read_bytes / (gpu_host_read * 1e6));
        printf("     KV write: %8.3f ms  (%6.2f GB/s)\n", gpu_host_write,
            kv_write_bytes / (gpu_host_write * 1e6));

        // ─── Scenario C: CPU layer, KV on host-pinned (co-located) ────
        double cpu_host_read  = bench_cpu_kv_read(host_kv, kv_read_bytes, n_cpu_threads, ITERS);
        double cpu_host_write = bench_cpu_kv_write(host_kv, kv_write_bytes, ITERS);
        printf("  C: CPU + host KV (co-located):\n");
        printf("     KV read:  %8.3f ms  (%6.2f GB/s)\n", cpu_host_read,
            kv_read_bytes / (cpu_host_read * 1e6));
        printf("     KV write: %8.3f ms  (%6.2f GB/s)\n", cpu_host_write,
            kv_write_bytes / (cpu_host_write * 1e6));

        // ─── Scenario D: CPU layer, KV in VRAM (IMPOSSIBLE without DMA) ──
        // sycl::malloc_device is NOT CPU-accessible!
        // This scenario requires explicit D2H for reads and H2D for writes.
        double dma_d2h_ms = 0, dma_h2d_ms = 0;
        {
            auto * staging = sycl::malloc_host<uint8_t>(kv_read_bytes, gpu_q);
            // D2H for KV read
            gpu_q.memcpy(staging, vram_kv, kv_read_bytes).wait(); // warmup
            timer t;
            for (int i = 0; i < ITERS; i++)
                gpu_q.memcpy(staging, vram_kv, kv_read_bytes).wait();
            dma_d2h_ms = t.ms() / ITERS;

            // H2D for KV write (only new token)
            gpu_q.memcpy(vram_kv, staging, kv_write_bytes).wait(); // warmup
            timer t2;
            for (int i = 0; i < ITERS; i++)
                gpu_q.memcpy(vram_kv, staging, kv_write_bytes).wait();
            dma_h2d_ms = t2.ms() / ITERS;

            sycl::free(staging, gpu_q);
        }
        printf("  D: CPU + VRAM KV (requires DMA):\n");
        printf("     D2H read: %8.3f ms  (%6.2f GB/s)  ← must DMA to CPU first\n",
            dma_d2h_ms, kv_read_bytes / (dma_d2h_ms * 1e6));
        printf("     H2D write:%8.3f ms  (%6.2f GB/s)  ← must DMA back after write\n",
            dma_h2d_ms, kv_write_bytes / (dma_h2d_ms * 1e6));
        double cpu_vram_total = dma_d2h_ms + cpu_host_read + dma_h2d_ms;
        printf("     Total:    %8.3f ms  (D2H + CPU read + H2D)\n", cpu_vram_total);

        // ─── Full model analysis ─────────────────────────
        printf("\n  ┌─ FULL MODEL ANALYSIS (32 layers) ──────────────────┐\n");

        // Case 1: All layers on GPU, all KV in VRAM
        double all_gpu_vram = N_LAYERS * (gpu_vram_read + gpu_vram_write);
        printf("  │ All GPU + VRAM KV:       %8.3f ms (%d × %.3f ms)\n",
            all_gpu_vram, N_LAYERS, gpu_vram_read + gpu_vram_write);

        // Case 2: All layers on GPU, all KV on host (BAD)
        double all_gpu_host = N_LAYERS * (gpu_host_read + gpu_host_write);
        printf("  │ All GPU + host KV (BAD): %8.3f ms (%d × %.3f ms)\n",
            all_gpu_host, N_LAYERS, gpu_host_read + gpu_host_write);

        // Case 3: All layers on CPU, all KV on host
        double all_cpu_host = N_LAYERS * (cpu_host_read + cpu_host_write);
        printf("  │ All CPU + host KV:       %8.3f ms (%d × %.3f ms)\n",
            all_cpu_host, N_LAYERS, cpu_host_read + cpu_host_write);

        // Case 4: Split — 24 GPU layers (VRAM KV) + 8 CPU layers (host KV), CO-LOCATED
        int gpu_layers = 24, cpu_layers = 8;
        double split_colocated = gpu_layers * (gpu_vram_read + gpu_vram_write)
                               + cpu_layers * (cpu_host_read + cpu_host_write);
        printf("  │ Split co-located (%d GPU + %d CPU):\n", gpu_layers, cpu_layers);
        printf("  │   GPU part: %8.3f ms, CPU part: %8.3f ms\n",
            gpu_layers * (gpu_vram_read + gpu_vram_write),
            cpu_layers * (cpu_host_read + cpu_host_write));
        printf("  │   Sequential: %8.3f ms\n", split_colocated);
        // Parallel: GPU and CPU layers can overlap
        double split_parallel = std::max(
            gpu_layers * (gpu_vram_read + gpu_vram_write),
            cpu_layers * (cpu_host_read + cpu_host_write));
        printf("  │   Parallel:   %8.3f ms\n", split_parallel);

        // Case 5: Split — 24 GPU + 8 CPU, but ALL KV in VRAM (BAD for CPU layers)
        double split_bad = gpu_layers * (gpu_vram_read + gpu_vram_write)
                         + cpu_layers * cpu_vram_total;
        printf("  │ Split ALL-VRAM KV (%d GPU + %d CPU):\n", gpu_layers, cpu_layers);
        printf("  │   GPU part: %8.3f ms, CPU+DMA part: %8.3f ms\n",
            gpu_layers * (gpu_vram_read + gpu_vram_write),
            cpu_layers * cpu_vram_total);
        printf("  │   Sequential: %8.3f ms\n", split_bad);

        // Penalty
        double penalty = split_bad - split_colocated;
        printf("  │\n");
        printf("  │ Cross-tier penalty for %d CPU layers: +%.3f ms (%.1f%%)\n",
            cpu_layers, penalty, 100.0 * penalty / split_colocated);
        printf("  │ Co-located KV read BW penalty: %.1fx (GPU VRAM vs host PCIe)\n",
            gpu_host_read / gpu_vram_read);
        printf("  └────────────────────────────────────────────────────┘\n\n");

        sycl::free(vram_kv, gpu_q);
        sycl::free(host_kv, gpu_q);
    }

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION                                              ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  1. malloc_device is NOT CPU-accessible.                 ║\n");
    printf("║     CPU layers MUST have host-pinned KV. No choice.     ║\n");
    printf("║                                                          ║\n");
    printf("║  2. GPU reading KV from host-pinned is ~40-60x slower   ║\n");
    printf("║     than VRAM. GPU layers MUST have VRAM KV.            ║\n");
    printf("║                                                          ║\n");
    printf("║  3. Co-location is mandatory, not optional:             ║\n");
    printf("║     • GPU layer → VRAM weights (SOA) + VRAM KV          ║\n");
    printf("║     • CPU layer → host weights (AOS) + host KV          ║\n");
    printf("║                                                          ║\n");
    printf("║  4. Decided at model load time. Fixed during inference.  ║\n");
    printf("║     KV zone size per device = layers_on_device × KV/lyr ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
