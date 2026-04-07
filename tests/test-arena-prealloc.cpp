// test-arena-prealloc.cpp
// Validates P1 assumptions:
// 1. Can we allocate 90% of B580 VRAM in a single malloc_device?
// 2. Atomic bump allocator performance (target < 100ns)
// 3. Multi-chunk fallback if single alloc fails
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -pthread -o test-arena-prealloc tests/test-arena-prealloc.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-arena-prealloc
//
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

struct timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    timer() : t0(clock::now()) {}
    double ns() const {
        return std::chrono::duration<double, std::nano>(clock::now() - t0).count();
    }
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  VRAM Arena Pre-Allocation Test (P1 Gate)               ║\n");
    printf("║  Single malloc_device + atomic bump allocator perf      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = q.get_device();
    size_t total_mem = dev.get_info<sycl::info::device::global_mem_size>();
    size_t max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();

    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Total VRAM: %.2f GB\n", total_mem / 1e9);
    printf("Max alloc:  %.2f GB\n\n", max_alloc / 1e9);

    // ─── Test 1: Single large allocation at various % ─
    printf("--- Test 1: Single malloc_device at various VRAM budgets ---\n");
    int pcts[] = {95, 90, 85, 80, 75, 50};
    for (int pct : pcts) {
        size_t target = (size_t)((double)total_mem * pct / 100.0);
        // Clamp to max_alloc
        if (target > max_alloc) target = max_alloc;

        timer t;
        void * ptr = sycl::malloc_device(target, q);
        double alloc_ms = t.ms();

        if (ptr) {
            printf("  %2d%% (%8.2f GB): PASS — alloc took %.1f ms\n",
                pct, target / 1e9, alloc_ms);
            sycl::free(ptr, q);
        } else {
            printf("  %2d%% (%8.2f GB): FAIL — malloc_device returned nullptr\n",
                pct, target / 1e9);
        }
    }

    // ─── Test 2: Find max single allocation ───────────
    printf("\n--- Test 2: Binary search for max single allocation ---\n");
    {
        size_t lo = 0, hi = std::min(total_mem, max_alloc);
        size_t best = 0;
        while (hi - lo > 1024 * 1024) {  // 1 MB precision
            size_t mid = lo + (hi - lo) / 2;
            void * ptr = sycl::malloc_device(mid, q);
            if (ptr) {
                best = mid;
                sycl::free(ptr, q);
                lo = mid;
            } else {
                hi = mid;
            }
        }
        printf("  Max single allocation: %.2f GB (%.1f%% of total VRAM)\n",
            best / 1e9, 100.0 * best / total_mem);
    }

    // ─── Test 3: Multi-chunk allocation ───────────────
    printf("\n--- Test 3: Multi-chunk allocation (fallback strategy) ---\n");
    {
        size_t target = (size_t)(total_mem * 0.90);
        size_t chunk_sizes[] = {
            target,                    // Try single
            target / 2, target / 2,    // Try 2 chunks
        };

        // Try single chunk first
        void * single = sycl::malloc_device(target, q);
        if (single) {
            printf("  Single chunk %.2f GB: PASS\n", target / 1e9);
            sycl::free(single, q);
        } else {
            printf("  Single chunk %.2f GB: FAIL — trying 2 chunks\n", target / 1e9);
            size_t half = target / 2;
            void * c1 = sycl::malloc_device(half, q);
            void * c2 = sycl::malloc_device(half, q);
            printf("  Chunk 1 (%.2f GB): %s\n", half / 1e9, c1 ? "PASS" : "FAIL");
            printf("  Chunk 2 (%.2f GB): %s\n", half / 1e9, c2 ? "PASS" : "FAIL");
            if (c1) sycl::free(c1, q);
            if (c2) sycl::free(c2, q);
        }
    }

    // ─── Test 4: Atomic bump allocator performance ────
    printf("\n--- Test 4: Atomic bump allocator latency ---\n");
    {
        // Simulate: one thread bumps an atomic, simulating arena_alloc()
        std::atomic<size_t> bump{0};
        const int N = 1000000;
        const size_t alloc_size = 256;  // Typical small allocation

        // Single-threaded
        timer t1;
        for (int i = 0; i < N; i++) {
            size_t off = bump.fetch_add(alloc_size, std::memory_order_relaxed);
            (void)off;
        }
        double ns_per_alloc_1t = t1.ns() / N;
        printf("  1 thread:  %.1f ns/alloc (%s)\n",
            ns_per_alloc_1t, ns_per_alloc_1t < 100 ? "PASS < 100ns" : "FAIL >= 100ns");

        // Multi-threaded (simulate contention from multiple backends)
        bump.store(0);
        int n_threads = 4;
        const int per_thread = N / n_threads;

        timer t2;
        {
            std::vector<std::thread> threads;
            for (int t = 0; t < n_threads; t++) {
                threads.emplace_back([&]() {
                    for (int i = 0; i < per_thread; i++) {
                        size_t off = bump.fetch_add(alloc_size, std::memory_order_relaxed);
                        (void)off;
                    }
                });
            }
            for (auto & th : threads) th.join();
        }
        double ns_per_alloc_mt = t2.ns() / N;
        printf("  %d threads: %.1f ns/alloc (%s)\n",
            n_threads, ns_per_alloc_mt, ns_per_alloc_mt < 100 ? "PASS < 100ns" : "WARN >= 100ns");

        // With CAS (compare-and-swap) for aligned allocation
        bump.store(0);
        const size_t alignment = 256;
        timer t3;
        for (int i = 0; i < N; i++) {
            size_t old_val = bump.load(std::memory_order_relaxed);
            size_t aligned, new_val;
            do {
                aligned = (old_val + alignment - 1) & ~(alignment - 1);
                new_val = aligned + alloc_size;
            } while (!bump.compare_exchange_weak(old_val, new_val,
                        std::memory_order_relaxed, std::memory_order_relaxed));
        }
        double ns_per_alloc_cas = t3.ns() / N;
        printf("  CAS align: %.1f ns/alloc (%s)\n",
            ns_per_alloc_cas, ns_per_alloc_cas < 100 ? "PASS < 100ns" : "WARN >= 100ns");
    }

    // ─── Test 5: malloc_device allocation time ────────
    printf("\n--- Test 5: malloc_device latency (what arena avoids) ---\n");
    {
        size_t sizes[] = {256, 4096, 1024*1024, 16*1024*1024, 256*1024*1024};
        const char * names[] = {"256 B", "4 KB", "1 MB", "16 MB", "256 MB"};

        for (int i = 0; i < 5; i++) {
            const int iters = 10;
            double total_ms = 0;
            for (int j = 0; j < iters; j++) {
                timer t;
                void * p = sycl::malloc_device(sizes[i], q);
                total_ms += t.ms();
                if (p) sycl::free(p, q);
            }
            double avg_ms = total_ms / iters;
            printf("  %8s: %.3f ms avg (arena saves this per allocation)\n",
                names[i], avg_ms);
        }
    }

    // ─── Test 6: Arena write + kernel verify ──────────
    printf("\n--- Test 6: Arena allocation functional test ---\n");
    {
        size_t arena_size = 256 * 1024 * 1024;  // 256 MB
        void * arena = sycl::malloc_device(arena_size, q);
        if (!arena) {
            printf("  SKIP: couldn't allocate 256 MB arena\n");
        } else {
            // Simulate zone-based sub-allocation
            uint8_t * base = static_cast<uint8_t *>(arena);
            size_t compute_off = 0;
            size_t compute_size = 64 * 1024 * 1024;   // 64 MB compute zone
            size_t weight_off = arena_size;             // Weights from end
            size_t weight_size = 128 * 1024 * 1024;    // 128 MB weight zone
            weight_off -= weight_size;

            // Write to compute zone from GPU
            q.memset(base + compute_off, 0xAA, compute_size).wait();
            // Write to weight zone from GPU
            q.memset(base + weight_off, 0xBB, weight_size).wait();

            // Verify from GPU
            auto * errors = sycl::malloc_host<int>(1, q);
            errors[0] = 0;
            q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> id) {
                    // Check compute zone
                    if (base[compute_off + id[0]] != 0xAA) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> a(errors[0]);
                        a.fetch_add(1);
                    }
                    // Check weight zone
                    if (base[weight_off + id[0]] != 0xBB) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space> a(errors[0]);
                        a.fetch_add(1);
                    }
                });
            }).wait();

            printf("  Zone isolation test: %s (%d errors)\n",
                errors[0] == 0 ? "PASS" : "FAIL", errors[0]);

            sycl::free(errors, q);
            sycl::free(arena, q);
        }
    }

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY                                                 ║\n");
    printf("║  • If 90%% single alloc passes → P1 can use 1 chunk     ║\n");
    printf("║  • If bump alloc < 100ns → P3 pool_leg can use arena    ║\n");
    printf("║  • malloc_device latency shows what arena SAVES          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}
