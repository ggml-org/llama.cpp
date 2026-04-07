// test-host-weight-strategy.cpp
// Micro-benchmark: GPU zero-copy vs CPU direct for host-resident Q4_0 weights
//
// Question: When weights are evicted from VRAM to host-pinned memory,
// should we keep SOA layout (for GPU zero-copy) or convert to AOS (for CPU compute)?
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -pthread -o test-host-weight-strategy tests/test-host-weight-strategy.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-host-weight-strategy
//
// Tests 6 scenarios at 3 tensor sizes (9MB, 33MB, 128MB):
//   1. GPU reading from malloc_device (VRAM baseline)
//   2. GPU reading from malloc_host, SOA layout (zero-copy)
//   3. GPU reading from malloc_host, AOS layout (zero-copy, non-coalesced)
//   4. CPU reading from malloc_host, AOS layout (single-thread)
//   5. CPU reading from malloc_host, AOS layout (multi-thread)
//   6. CPU reading from malloc_host, SOA layout (worst case)
//   7. SOA→AOS and AOS→SOA conversion cost
//   8. Parallel: GPU on VRAM + CPU on host simultaneously
//
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <functional>

// ─── Q4_0 block layout ────────────────────────────────────
struct block_q4_0 {
    uint16_t d;       // delta (f16)
    uint8_t  qs[16];  // 32 quants packed into 16 bytes
};
static_assert(sizeof(block_q4_0) == 18);

static constexpr int QK4_0 = 32;

// ─── f16 ↔ f32 (bit manipulation, no sycl::half on CPU) ──
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

// ─── SOA layout helpers ───────────────────────────────────
// SOA for Q4_0: all scales packed first, then all quants
// scales: n_blocks * 2 bytes, quants: n_blocks * 16 bytes

static void convert_aos_to_soa(const block_q4_0 * aos, uint8_t * soa, int n_blocks) {
    auto * scales = reinterpret_cast<uint16_t *>(soa);
    auto * quants = soa + n_blocks * sizeof(uint16_t);
    for (int i = 0; i < n_blocks; i++) {
        scales[i] = aos[i].d;
        memcpy(quants + i * 16, aos[i].qs, 16);
    }
}

static void convert_soa_to_aos(const uint8_t * soa, block_q4_0 * aos, int n_blocks) {
    const auto * scales = reinterpret_cast<const uint16_t *>(soa);
    const auto * quants = soa + n_blocks * sizeof(uint16_t);
    for (int i = 0; i < n_blocks; i++) {
        aos[i].d = scales[i];
        memcpy(aos[i].qs, quants + i * 16, 16);
    }
}

// ─── Timer ────────────────────────────────────────────────
struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ─── GPU kernel: read all bytes via reduction ─────────────
// Each work-item reads a strided portion of the buffer, reducing to a checksum.
// This measures effective read bandwidth from the GPU's perspective.

static double bench_gpu_read(sycl::queue & q, const uint8_t * buf, size_t nbytes, int iters) {
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

    // timed
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

// ─── CPU: Q4_0 vec_dot (single row), AOS layout ──────────
// Simulates the actual compute pattern for host-resident weights
static float cpu_vec_dot_q4_0_row_aos(const block_q4_0 * row, int row_blocks) {
    float sum = 0.0f;
    for (int b = 0; b < row_blocks; b++) {
        float d = f16_to_f32(row[b].d);
        for (int j = 0; j < 16; j++) {
            int8_t v0 = (row[b].qs[j] & 0xF) - 8;
            int8_t v1 = (row[b].qs[j] >> 4)   - 8;
            sum += d * (float)(v0 + v1);
        }
    }
    return sum;
}

// ─── CPU: Q4_0 vec_dot (single row), SOA layout ──────────
static float cpu_vec_dot_q4_0_row_soa(const uint16_t * scales, const uint8_t * quants,
                                       int row_start, int row_blocks) {
    float sum = 0.0f;
    for (int b = 0; b < row_blocks; b++) {
        float d = f16_to_f32(scales[row_start + b]);
        const uint8_t * qs = quants + (row_start + b) * 16;
        for (int j = 0; j < 16; j++) {
            int8_t v0 = (qs[j] & 0xF) - 8;
            int8_t v1 = (qs[j] >> 4)   - 8;
            sum += d * (float)(v0 + v1);
        }
    }
    return sum;
}

// ─── CPU bench: single-threaded, AOS ──────────────────────
static double bench_cpu_aos_1t(const block_q4_0 * data, int n_rows, int row_blocks, int iters) {
    volatile float sink = 0;

    // warmup
    for (int r = 0; r < n_rows; r++)
        sink += cpu_vec_dot_q4_0_row_aos(data + (size_t)r * row_blocks, row_blocks);

    timer t;
    for (int it = 0; it < iters; it++) {
        float s = 0;
        for (int r = 0; r < n_rows; r++)
            s += cpu_vec_dot_q4_0_row_aos(data + (size_t)r * row_blocks, row_blocks);
        sink += s;
    }
    return t.ms() / iters;
}

// ─── CPU bench: multi-threaded, AOS ──────────────────────
static double bench_cpu_aos_mt(const block_q4_0 * data, int n_rows, int row_blocks,
                                int n_threads, int iters) {
    std::vector<float> partial(n_threads, 0.0f);

    auto worker = [&](int tid) {
        int rows_per = (n_rows + n_threads - 1) / n_threads;
        int r_start  = tid * rows_per;
        int r_end    = std::min(r_start + rows_per, n_rows);
        float s = 0;
        for (int r = r_start; r < r_end; r++)
            s += cpu_vec_dot_q4_0_row_aos(data + (size_t)r * row_blocks, row_blocks);
        partial[tid] = s;
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

    volatile float sink = 0;
    for (auto v : partial) sink += v;
    return elapsed;
}

// ─── CPU bench: single-threaded, SOA ──────────────────────
static double bench_cpu_soa_1t(const uint8_t * soa, int n_rows, int row_blocks,
                                int n_blocks_total, int iters) {
    const auto * scales = reinterpret_cast<const uint16_t *>(soa);
    const auto * quants = soa + n_blocks_total * sizeof(uint16_t);
    volatile float sink = 0;

    // warmup
    for (int r = 0; r < n_rows; r++)
        sink += cpu_vec_dot_q4_0_row_soa(scales, quants, r * row_blocks, row_blocks);

    timer t;
    for (int it = 0; it < iters; it++) {
        float s = 0;
        for (int r = 0; r < n_rows; r++)
            s += cpu_vec_dot_q4_0_row_soa(scales, quants, r * row_blocks, row_blocks);
        sink += s;
    }
    return t.ms() / iters;
}

// ─── CPU bench: multi-threaded, SOA ──────────────────────
static double bench_cpu_soa_mt(const uint8_t * soa, int n_rows, int row_blocks,
                                int n_blocks_total, int n_threads, int iters) {
    const auto * scales = reinterpret_cast<const uint16_t *>(soa);
    const auto * quants = soa + n_blocks_total * sizeof(uint16_t);
    std::vector<float> partial(n_threads, 0.0f);

    auto worker = [&](int tid) {
        int rows_per = (n_rows + n_threads - 1) / n_threads;
        int r_start  = tid * rows_per;
        int r_end    = std::min(r_start + rows_per, n_rows);
        float s = 0;
        for (int r = r_start; r < r_end; r++)
            s += cpu_vec_dot_q4_0_row_soa(scales, quants, r * row_blocks, row_blocks);
        partial[tid] = s;
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

    volatile float sink = 0;
    for (auto v : partial) sink += v;
    return elapsed;
}

// ─── Parallel bench: GPU(VRAM) + CPU(host) simultaneously ─
struct parallel_result {
    double wall_ms;
    double gpu_ms;
    double cpu_ms;
};

static parallel_result bench_parallel(
    sycl::queue & gpu_q,
    const uint8_t * vram_buf,   size_t vram_bytes,   // GPU reads VRAM
    const block_q4_0 * host_buf, int n_rows, int row_blocks, int n_cpu_threads,
    int iters
) {
    size_t host_bytes = (size_t)n_rows * row_blocks * sizeof(block_q4_0);
    std::vector<float> cpu_partial(n_cpu_threads, 0.0f);

    // GPU setup
    const int wg_size   = 256;
    const int n_wgs     = std::min(512, (int)((vram_bytes + wg_size - 1) / wg_size));
    const int n_threads = n_wgs * wg_size;
    auto * d_out = sycl::malloc_device<uint32_t>(n_wgs, gpu_q);
    gpu_q.memset(d_out, 0, n_wgs * sizeof(uint32_t)).wait();

    // warmup
    gpu_q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
            uint32_t sum = 0;
            for (size_t i = it.get_global_id(0); i < vram_bytes; i += n_threads)
                sum += vram_buf[i];
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
            aref.fetch_add(sum);
        });
    }).wait();

    double cpu_total_ms = 0;
    double gpu_total_ms = 0;

    timer wall;
    for (int it = 0; it < iters; it++) {
        // Launch GPU (async)
        timer gt;
        auto gpu_evt = gpu_q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(n_threads, wg_size), [=](sycl::nd_item<1> it) {
                uint32_t sum = 0;
                for (size_t i = it.get_global_id(0); i < vram_bytes; i += n_threads)
                    sum += vram_buf[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space> aref(d_out[it.get_group_linear_id()]);
                aref.fetch_add(sum);
            });
        });

        // Launch CPU workers in parallel
        timer ct;
        {
            std::vector<std::thread> ts;
            for (int tid = 0; tid < n_cpu_threads; tid++) {
                ts.emplace_back([&, tid]() {
                    int rows_per = (n_rows + n_cpu_threads - 1) / n_cpu_threads;
                    int r_start  = tid * rows_per;
                    int r_end    = std::min(r_start + rows_per, n_rows);
                    float s = 0;
                    for (int r = r_start; r < r_end; r++)
                        s += cpu_vec_dot_q4_0_row_aos(host_buf + (size_t)r * row_blocks, row_blocks);
                    cpu_partial[tid] = s;
                });
            }
            for (auto & th : ts) th.join();
        }
        cpu_total_ms += ct.ms();

        gpu_q.wait();
        gpu_total_ms += gt.ms();
    }
    double wall_ms = wall.ms() / iters;

    sycl::free(d_out, gpu_q);

    volatile float sink = 0;
    for (auto v : cpu_partial) sink += v;

    return { wall_ms, gpu_total_ms / iters, cpu_total_ms / iters };
}

// ─── Conversion bench ─────────────────────────────────────
static double bench_conversion(std::function<void()> fn, int iters) {
    fn(); // warmup
    timer t;
    for (int i = 0; i < iters; i++) fn();
    return t.ms() / iters;
}

// ─── Main ─────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Host-Resident Weight Strategy Micro-Benchmark          ║\n");
    printf("║  GPU zero-copy vs CPU direct for evicted Q4_0 weights   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ─── SYCL setup ───────────────────────────────────────
    sycl::queue gpu_q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = gpu_q.get_device();
    printf("GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("GPU VRAM: %.1f GB,  Max Alloc: %.1f GB\n",
        dev.get_info<sycl::info::device::global_mem_size>() / 1e9,
        dev.get_info<sycl::info::device::max_mem_alloc_size>() / 1e9);

    int n_cpu_cores = std::thread::hardware_concurrency();
    int n_cpu_threads = std::max(1, n_cpu_cores - 2);  // reserve 2 for OS/GPU driver
    printf("CPU cores: %d,  worker threads: %d\n\n", n_cpu_cores, n_cpu_threads);

    // ─── Test configurations ──────────────────────────────
    // Realistic tensor sizes from Mistral 7B Q4_0:
    //   attention weight [4096,4096]: 4096 rows × 128 blocks/row × 18 B = 9.4 MB
    //   FFN weight [4096,14336]:    14336 rows × 128 blocks/row × 18 B = 33 MB
    //   large MoE expert:          ~128 MB (for 120B model experts)

    struct test_config {
        const char * name;
        int N;      // rows
        int K;      // cols (must be multiple of 32)
    };

    test_config configs[] = {
        { "attn [4096x4096]",   4096,  4096 },
        { "FFN  [14336x4096]", 14336,  4096 },
        { "MoE  [14336x14336]",14336, 14336 },
    };

    const int ITERS = 10;

    for (auto & cfg : configs) {
        int row_blocks   = cfg.K / QK4_0;
        int n_rows       = cfg.N;
        int n_blocks     = n_rows * row_blocks;
        size_t aos_bytes = (size_t)n_blocks * sizeof(block_q4_0);
        size_t soa_bytes = aos_bytes;  // same total size, different layout

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Tensor: %s  (%zu bytes = %.1f MB)\n", cfg.name, aos_bytes, aos_bytes / 1e6);
        printf("Blocks: %d  (rows=%d, blocks/row=%d)\n", n_blocks, n_rows, row_blocks);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // ─── Allocate ─────────────────────────────────────
        auto * host_aos     = sycl::malloc_host<block_q4_0>(n_blocks, gpu_q);
        auto * host_soa     = sycl::malloc_host<uint8_t>(soa_bytes, gpu_q);
        auto * device_buf   = sycl::malloc_device<uint8_t>(aos_bytes, gpu_q);
        auto * scratch_aos  = sycl::malloc_host<block_q4_0>(n_blocks, gpu_q);

        if (!host_aos || !host_soa || !device_buf || !scratch_aos) {
            printf("  SKIP: allocation failed (tensor too large for available memory)\n\n");
            if (host_aos)    sycl::free(host_aos, gpu_q);
            if (host_soa)    sycl::free(host_soa, gpu_q);
            if (device_buf)  sycl::free(device_buf, gpu_q);
            if (scratch_aos) sycl::free(scratch_aos, gpu_q);
            continue;
        }

        // Fill with pseudo-random data
        srand(42);
        for (int i = 0; i < n_blocks; i++) {
            host_aos[i].d = static_cast<uint16_t>(0x3C00 | (rand() & 0x3FF));  // ~1.0 in f16
            for (int j = 0; j < 16; j++)
                host_aos[i].qs[j] = static_cast<uint8_t>(rand());
        }

        // Prepare SOA and device copies
        convert_aos_to_soa(host_aos, host_soa, n_blocks);
        gpu_q.memcpy(device_buf, reinterpret_cast<uint8_t *>(host_aos), aos_bytes).wait();

        // ─── Bench 1: GPU reading VRAM (baseline) ─────────
        double gpu_vram_ms = bench_gpu_read(gpu_q, device_buf, aos_bytes, ITERS);
        printf("  GPU VRAM read (baseline): %8.3f ms  (%6.2f GB/s)\n",
            gpu_vram_ms, aos_bytes / (gpu_vram_ms * 1e6));

        // ─── Bench 2: GPU zero-copy, SOA layout ──────────
        double gpu_zc_soa_ms = bench_gpu_read(gpu_q, host_soa, soa_bytes, ITERS);
        printf("  GPU zero-copy SOA:        %8.3f ms  (%6.2f GB/s)\n",
            gpu_zc_soa_ms, soa_bytes / (gpu_zc_soa_ms * 1e6));

        // ─── Bench 3: GPU zero-copy, AOS layout ──────────
        double gpu_zc_aos_ms = bench_gpu_read(gpu_q,
            reinterpret_cast<uint8_t *>(host_aos), aos_bytes, ITERS);
        printf("  GPU zero-copy AOS:        %8.3f ms  (%6.2f GB/s)\n",
            gpu_zc_aos_ms, aos_bytes / (gpu_zc_aos_ms * 1e6));

        // ─── Bench 4: CPU single-thread, AOS ─────────────
        double cpu_aos_1t_ms = bench_cpu_aos_1t(host_aos, n_rows, row_blocks, ITERS);
        printf("  CPU AOS (1 thread):       %8.3f ms  (%6.2f GB/s)\n",
            cpu_aos_1t_ms, aos_bytes / (cpu_aos_1t_ms * 1e6));

        // ─── Bench 5: CPU multi-thread, AOS ──────────────
        double cpu_aos_mt_ms = bench_cpu_aos_mt(host_aos, n_rows, row_blocks, n_cpu_threads, ITERS);
        printf("  CPU AOS (%2d threads):     %8.3f ms  (%6.2f GB/s)\n",
            n_cpu_threads, cpu_aos_mt_ms, aos_bytes / (cpu_aos_mt_ms * 1e6));

        // ─── Bench 6: CPU single-thread, SOA ─────────────
        double cpu_soa_1t_ms = bench_cpu_soa_1t(host_soa, n_rows, row_blocks, n_blocks, ITERS);
        printf("  CPU SOA (1 thread):       %8.3f ms  (%6.2f GB/s)\n",
            cpu_soa_1t_ms, aos_bytes / (cpu_soa_1t_ms * 1e6));

        // ─── Bench 6b: CPU multi-thread, SOA ─────────────
        double cpu_soa_mt_ms = bench_cpu_soa_mt(host_soa, n_rows, row_blocks, n_blocks,
                                                 n_cpu_threads, ITERS);
        printf("  CPU SOA (%2d threads):     %8.3f ms  (%6.2f GB/s)\n",
            n_cpu_threads, cpu_soa_mt_ms, aos_bytes / (cpu_soa_mt_ms * 1e6));

        // ─── Bench 7: Layout conversion cost ─────────────
        double soa2aos_ms = bench_conversion([&]() {
            convert_soa_to_aos(host_soa, scratch_aos, n_blocks);
        }, ITERS);
        double aos2soa_ms = bench_conversion([&]() {
            convert_aos_to_soa(host_aos, host_soa, n_blocks);
        }, ITERS);
        printf("  SOA→AOS conversion:       %8.3f ms  (%6.2f GB/s)\n",
            soa2aos_ms, aos_bytes / (soa2aos_ms * 1e6));
        printf("  AOS→SOA conversion:       %8.3f ms  (%6.2f GB/s)\n",
            aos2soa_ms, aos_bytes / (aos2soa_ms * 1e6));

        // ─── Bench 8: Parallel GPU(VRAM) + CPU(host AOS) ─
        auto par = bench_parallel(gpu_q, device_buf, aos_bytes,
                                   host_aos, n_rows, row_blocks, n_cpu_threads, ITERS);
        printf("  Parallel GPU+CPU:         %8.3f ms wall  (GPU=%.3f, CPU=%.3f)\n",
            par.wall_ms, par.gpu_ms, par.cpu_ms);

        // ─── Analysis ─────────────────────────────────────
        printf("\n  ┌─ ANALYSIS ──────────────────────────────────────────┐\n");

        double best_gpu_zc = std::min(gpu_zc_soa_ms, gpu_zc_aos_ms);
        double best_cpu    = cpu_aos_mt_ms;
        const char * winner = best_cpu < best_gpu_zc ? "CPU AOS (multi-thread)" : "GPU zero-copy";
        double ratio = best_cpu < best_gpu_zc ? best_gpu_zc / best_cpu : best_cpu / best_gpu_zc;

        printf("  │ Best GPU zero-copy: %.3f ms (%s)\n",
            best_gpu_zc, gpu_zc_soa_ms < gpu_zc_aos_ms ? "SOA" : "AOS");
        printf("  │ Best CPU direct:    %.3f ms (AOS, %d threads)\n", best_cpu, n_cpu_threads);
        printf("  │ Winner: %s (%.1fx faster)\n", winner, ratio);
        printf("  │\n");
        printf("  │ SOA penalty on CPU: %.1fx slower than AOS\n",
            cpu_soa_mt_ms / cpu_aos_mt_ms);
        printf("  │ Parallel speedup:   %.1fx vs GPU-only zero-copy\n",
            best_gpu_zc / par.wall_ms);
        printf("  │ Conversion cost:    %.3f ms (%.1f%% of one GPU ZC pass)\n",
            soa2aos_ms, 100.0 * soa2aos_ms / best_gpu_zc);
        printf("  └─────────────────────────────────────────────────────┘\n\n");

        // ─── Cleanup ──────────────────────────────────────
        sycl::free(host_aos, gpu_q);
        sycl::free(host_soa, gpu_q);
        sycl::free(device_buf, gpu_q);
        sycl::free(scratch_aos, gpu_q);
    }

    // ─── Final verdict ────────────────────────────────────
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  STRATEGY DECISION GUIDE                                ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  If CPU AOS > GPU ZC for all sizes:                     ║\n");
    printf("║    → Store AOS in host. CPU compute for host weights.   ║\n");
    printf("║    → GPU stays on VRAM-resident work (no PCIe waste).   ║\n");
    printf("║    → On eviction: async D2H, then background SOA→AOS.  ║\n");
    printf("║    → On promotion: AOS→SOA on device (GPU reorder).     ║\n");
    printf("║                                                          ║\n");
    printf("║  If GPU ZC > CPU AOS for all sizes:                     ║\n");
    printf("║    → Store SOA in host. GPU zero-copy for host weights. ║\n");
    printf("║    → Simpler: no layout conversion, same kernel path.   ║\n");
    printf("║                                                          ║\n");
    printf("║  If parallel GPU+CPU < either alone:                    ║\n");
    printf("║    → Best of both worlds. GPU on VRAM, CPU on host.     ║\n");
    printf("║    → Store AOS in host (CPU-optimal).                   ║\n");
    printf("║    → Pipeline layers: GPU attn + CPU FFN in parallel.   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
