// Quick precision throughput benchmark for Intel Arc B580
// Tests FP16, FP32, FP64 compute throughput via FMA chains
// Usage: ./precision-bench

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

// FMA chain length — long enough to be compute-bound
static constexpr int FMA_ITERS = 4096;
// Number of work-items (saturate the GPU)
static constexpr int N_ITEMS   = 1024 * 1024;
// Repeat for stable timing
static constexpr int N_REPEATS = 5;

template <typename T>
double bench_fma(sycl::queue & q, int n_items, int fma_iters) {
    T * buf = sycl::malloc_device<T>(n_items, q);
    q.memset(buf, 0, n_items * sizeof(T)).wait();

    // Warmup
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            T a = static_cast<T>(1.0001);
            T b = static_cast<T>(0.9999);
            T c = static_cast<T>(i.get(0) & 0xFF) * static_cast<T>(1e-6);
            for (int j = 0; j < fma_iters; j++) {
                c = a * c + b;  // FMA
            }
            buf[i] = c;
        });
    }).wait();

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            T a = static_cast<T>(1.0001);
            T b = static_cast<T>(0.9999);
            T c = static_cast<T>(i.get(0) & 0xFF) * static_cast<T>(1e-6);
            for (int j = 0; j < fma_iters; j++) {
                c = a * c + b;
            }
            buf[i] = c;
        });
    }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Each FMA = 2 FLOP (multiply + add)
    double flops = 2.0 * (double)n_items * (double)fma_iters;
    double gflops = flops / (ms * 1e6);

    sycl::free(buf, q);
    return gflops;
}

// FP16 version using sycl::half
double bench_fma_fp16(sycl::queue & q, int n_items, int fma_iters) {
    sycl::half * buf = sycl::malloc_device<sycl::half>(n_items, q);
    q.memset(buf, 0, n_items * sizeof(sycl::half)).wait();

    // Warmup
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            sycl::half a = sycl::half(1.0f);
            sycl::half b = sycl::half(1.0f);
            sycl::half c = sycl::half(0.001f);
            for (int j = 0; j < fma_iters; j++) {
                c = a * c + b;
            }
            buf[i] = c;
        });
    }).wait();

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            sycl::half a = sycl::half(1.0f);
            sycl::half b = sycl::half(1.0f);
            sycl::half c = sycl::half(0.001f);
            for (int j = 0; j < fma_iters; j++) {
                c = a * c + b;
            }
            buf[i] = c;
        });
    }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double flops = 2.0 * (double)n_items * (double)fma_iters;
    double gflops = flops / (ms * 1e6);

    sycl::free(buf, q);
    return gflops;
}

// INT8 version
double bench_int8(sycl::queue & q, int n_items, int fma_iters) {
    int32_t * buf = sycl::malloc_device<int32_t>(n_items, q);
    q.memset(buf, 0, n_items * sizeof(int32_t)).wait();

    // Warmup
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            int8_t a = 3;
            int8_t b = 7;
            int32_t c = (int32_t)(i.get(0) & 0xFF);
            for (int j = 0; j < fma_iters; j++) {
                c = (int32_t)a * (int32_t)((int8_t)(c & 0x7F)) + (int32_t)b;
            }
            buf[i] = c;
        });
    }).wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
            int8_t a = 3;
            int8_t b = 7;
            int32_t c = (int32_t)(i.get(0) & 0xFF);
            for (int j = 0; j < fma_iters; j++) {
                c = (int32_t)a * (int32_t)((int8_t)(c & 0x7F)) + (int32_t)b;
            }
            buf[i] = c;
        });
    }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double flops = 2.0 * (double)n_items * (double)fma_iters;
    double gops = flops / (ms * 1e6);

    sycl::free(buf, q);
    return gops;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    auto dev = q.get_device();

    printf("=== Precision Throughput Benchmark ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Max compute units: %d\n", dev.get_info<sycl::info::device::max_compute_units>());
    printf("Max clock: %d MHz\n", dev.get_info<sycl::info::device::max_clock_frequency>());
    printf("Work-items: %d  FMA iters: %d\n\n", N_ITEMS, FMA_ITERS);

    // Check FP64 support
    bool has_fp64 = dev.has(sycl::aspect::fp64);
    bool has_fp16 = dev.has(sycl::aspect::fp16);
    printf("FP16 support: %s\n", has_fp16 ? "YES" : "NO");
    printf("FP64 support: %s\n\n", has_fp64 ? "YES" : "NO");

    // Run benchmarks with multiple repeats, take best
    printf("%-8s  %10s  %10s  %s\n", "Type", "GFLOPS", "Time (ms)", "Ratio vs FP32");
    printf("%-8s  %10s  %10s  %s\n", "--------", "----------", "----------", "-------------");

    // FP32 first (baseline)
    double best_fp32 = 0;
    for (int r = 0; r < N_REPEATS; r++) {
        double g = bench_fma<float>(q, N_ITEMS, FMA_ITERS);
        if (g > best_fp32) best_fp32 = g;
    }
    printf("%-8s  %10.1f  %10.2f  %.2fx\n", "FP32", best_fp32,
           2.0 * N_ITEMS * FMA_ITERS / (best_fp32 * 1e6), 1.0);

    // FP16
    if (has_fp16) {
        double best_fp16 = 0;
        for (int r = 0; r < N_REPEATS; r++) {
            double g = bench_fma_fp16(q, N_ITEMS, FMA_ITERS);
            if (g > best_fp16) best_fp16 = g;
        }
        printf("%-8s  %10.1f  %10.2f  %.2fx\n", "FP16", best_fp16,
               2.0 * N_ITEMS * FMA_ITERS / (best_fp16 * 1e6), best_fp16 / best_fp32);
    }

    // FP64
    if (has_fp64) {
        double best_fp64 = 0;
        for (int r = 0; r < N_REPEATS; r++) {
            double g = bench_fma<double>(q, N_ITEMS, FMA_ITERS);
            if (g > best_fp64) best_fp64 = g;
        }
        printf("%-8s  %10.1f  %10.2f  %.2fx\n", "FP64", best_fp64,
               2.0 * N_ITEMS * FMA_ITERS / (best_fp64 * 1e6), best_fp64 / best_fp32);
    }

    // INT8 (multiply-add with int8 operands, int32 accum)
    {
        double best_i8 = 0;
        for (int r = 0; r < N_REPEATS; r++) {
            double g = bench_int8(q, N_ITEMS, FMA_ITERS);
            if (g > best_i8) best_i8 = g;
        }
        printf("%-8s  %10.1f  %10.2f  %.2fx\n", "INT8", best_i8,
               2.0 * N_ITEMS * FMA_ITERS / (best_i8 * 1e6), best_i8 / best_fp32);
    }

    printf("\n");

    // Also test with larger work-groups to see if that changes things
    printf("=== Varying work-item count (FP32 vs FP64) ===\n");
    printf("%-12s  %10s  %10s  %s\n", "Work-items", "FP32 GF", "FP64 GF", "FP64/FP32");
    printf("%-12s  %10s  %10s  %s\n", "------------", "----------", "----------", "---------");

    int sizes[] = {64*1024, 256*1024, 1024*1024, 4*1024*1024};
    for (int sz : sizes) {
        double g32 = 0, g64 = 0;
        for (int r = 0; r < 3; r++) {
            double a = bench_fma<float>(q, sz, FMA_ITERS);
            if (a > g32) g32 = a;
        }
        if (has_fp64) {
            for (int r = 0; r < 3; r++) {
                double a = bench_fma<double>(q, sz, FMA_ITERS);
                if (a > g64) g64 = a;
            }
        }
        printf("%-12d  %10.1f  %10.1f  %.2fx\n", sz, g32, g64,
               g64 > 0 ? g64 / g32 : 0.0);
    }

    printf("\nDone.\n");
    return 0;
}
