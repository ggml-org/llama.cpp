// Expert Cache Bandwidth Microbenchmark
//
// Measures the key bandwidth/latency numbers that determine whether B50-local
// expert compute is faster than streaming expert weights to B580 over PCIe.
//
// Tests:
//   1. B50 VRAM read bandwidth (kernel reads from device-local memory)
//   2. Host→B50 H2D bandwidth (upload experts to B50 VRAM)
//   3. B50-local MMVQ latency (activation round-trip: H2D act + kernel + D2H result)
//   4. B580-stream latency (H2D expert weight + kernel on B580)
//
// Usage:
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR="level_zero:0,1" ./build/bin/test-sycl-expert-cache-bandwidth

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <sycl/sycl.hpp>

// Simulated expert weight: 4MB (typical MoE expert FFN weight block)
static constexpr size_t EXPERT_BYTES    = 4ULL * 1024 * 1024;
// Simulated activation: 16KB (batch=1 hidden state, e.g. 4096 floats)
static constexpr size_t ACTIVATION_BYTES = 16 * 1024;
// Simulated output: ~57KB (batch=1, intermediate_size=14336, sizeof(float))
// For simplicity we use same size as activation in the dot-product simulation
static constexpr size_t OUTPUT_FLOATS   = ACTIVATION_BYTES / sizeof(float);
static constexpr int    WARMUP_ITERS    = 3;
static constexpr int    BENCH_ITERS     = 10;

static double now_us() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t.time_since_epoch()).count();
}

// Simple simulated MMVQ: dot product of weight rows with activation vector.
// Each work-item computes one output element by reading ACTIVATION_BYTES/4 weight
// elements and ACTIVATION_BYTES/4 activation elements.
static void run_dot_kernel(sycl::queue & q, const float * weights, const float * act,
                           float * out, int n_cols, int n_rows) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> row) {
            float sum = 0.0f;
            const float * w = weights + row * n_cols;
            for (int c = 0; c < n_cols; c++) {
                sum += w[c] * act[c];
            }
            out[row] = sum;
        });
    });
    q.wait();
}

// Measure device-local read bandwidth: kernel reads from malloc_device buffer
static double measure_vram_bandwidth(sycl::queue & q) {
    float * buf = sycl::malloc_device<float>(EXPERT_BYTES / sizeof(float), q);
    float * out = sycl::malloc_device<float>(1, q);

    // Fill buffer
    q.memset(buf, 0x42, EXPERT_BYTES).wait();
    q.memset(out, 0, sizeof(float)).wait();

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> wg) {
                float sum = 0.0f;
                size_t n = EXPERT_BYTES / sizeof(float);
                for (size_t j = wg; j < n; j += 256) {
                    sum += buf[j];
                }
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device> aref(*out);
                aref.fetch_add(sum);
            });
        });
        q.wait();
    }

    // Benchmark
    double best_gbps = 0.0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        auto t0 = now_us();
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> wg) {
                float sum = 0.0f;
                size_t n = EXPERT_BYTES / sizeof(float);
                for (size_t j = wg; j < n; j += 256) {
                    sum += buf[j];
                }
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device> aref(*out);
                aref.fetch_add(sum);
            });
        });
        q.wait();
        double elapsed_us = now_us() - t0;
        double gbps = (double)EXPERT_BYTES / (elapsed_us * 1e-6) / 1e9;
        best_gbps = std::max(best_gbps, gbps);
    }

    sycl::free(buf, q);
    sycl::free(out, q);
    return best_gbps;
}

// Measure H2D memcpy bandwidth: sycl::memcpy from host-pinned to device
static double measure_h2d_bandwidth(sycl::queue & q) {
    float * host_buf   = sycl::malloc_host<float>(EXPERT_BYTES / sizeof(float), q);
    float * device_buf = sycl::malloc_device<float>(EXPERT_BYTES / sizeof(float), q);

    memset(host_buf, 0x42, EXPERT_BYTES);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        q.memcpy(device_buf, host_buf, EXPERT_BYTES).wait();
    }

    // Benchmark
    double best_gbps = 0.0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        auto t0 = now_us();
        q.memcpy(device_buf, host_buf, EXPERT_BYTES).wait();
        double elapsed_us = now_us() - t0;
        double gbps = (double)EXPERT_BYTES / (elapsed_us * 1e-6) / 1e9;
        best_gbps = std::max(best_gbps, gbps);
    }

    sycl::free(host_buf, q);
    sycl::free(device_buf, q);
    return best_gbps;
}

// Measure B50-local compute latency:
//   H2D activation (16KB) → dot kernel on B50 → D2H result
static double measure_local_compute_latency(sycl::queue & q) {
    int n_cols = ACTIVATION_BYTES / sizeof(float);  // 4096
    int n_rows = OUTPUT_FLOATS;                     // 4096

    // Expert weight resident on B50 VRAM
    float * weights = sycl::malloc_device<float>((size_t)n_rows * n_cols, q);
    // Activation: host-pinned → H2D to device
    float * host_act   = sycl::malloc_host<float>(n_cols, q);
    float * device_act = sycl::malloc_device<float>(n_cols, q);
    // Output: device → D2H to host
    float * device_out = sycl::malloc_device<float>(n_rows, q);
    float * host_out   = sycl::malloc_host<float>(n_rows, q);

    // Initialize
    q.memset(weights, 0x01, (size_t)n_rows * n_cols * sizeof(float)).wait();
    memset(host_act, 0x01, n_cols * sizeof(float));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        q.memcpy(device_act, host_act, n_cols * sizeof(float)).wait();
        run_dot_kernel(q, weights, device_act, device_out, n_cols, n_rows);
        q.memcpy(host_out, device_out, n_rows * sizeof(float)).wait();
    }

    // Benchmark full round-trip
    double best_us = 1e12;
    for (int i = 0; i < BENCH_ITERS; i++) {
        auto t0 = now_us();
        q.memcpy(device_act, host_act, n_cols * sizeof(float)).wait();
        run_dot_kernel(q, weights, device_act, device_out, n_cols, n_rows);
        q.memcpy(host_out, device_out, n_rows * sizeof(float)).wait();
        double elapsed_us = now_us() - t0;
        best_us = std::min(best_us, elapsed_us);
    }

    sycl::free(weights, q);
    sycl::free(host_act, q);
    sycl::free(device_act, q);
    sycl::free(device_out, q);
    sycl::free(host_out, q);

    return best_us;
}

// Measure B580-stream latency: H2D expert weight (4MB) + kernel on B580
static double measure_stream_compute_latency(sycl::queue & q) {
    int n_cols = ACTIVATION_BYTES / sizeof(float);  // 4096
    int n_rows = OUTPUT_FLOATS;                     // 4096

    // Expert weight: host-pinned, must be uploaded each time (simulating cache miss)
    float * host_weights   = sycl::malloc_host<float>((size_t)n_rows * n_cols, q);
    float * device_weights = sycl::malloc_device<float>((size_t)n_rows * n_cols, q);
    // Activation already on B580
    float * device_act = sycl::malloc_device<float>(n_cols, q);
    float * device_out = sycl::malloc_device<float>(n_rows, q);

    memset(host_weights, 0x01, (size_t)n_rows * n_cols * sizeof(float));
    q.memset(device_act, 0x01, n_cols * sizeof(float)).wait();

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        q.memcpy(device_weights, host_weights, (size_t)n_rows * n_cols * sizeof(float)).wait();
        run_dot_kernel(q, device_weights, device_act, device_out, n_cols, n_rows);
    }

    // Benchmark: upload weight + compute
    double best_us = 1e12;
    for (int i = 0; i < BENCH_ITERS; i++) {
        auto t0 = now_us();
        q.memcpy(device_weights, host_weights, (size_t)n_rows * n_cols * sizeof(float)).wait();
        run_dot_kernel(q, device_weights, device_act, device_out, n_cols, n_rows);
        double elapsed_us = now_us() - t0;
        best_us = std::min(best_us, elapsed_us);
    }

    sycl::free(host_weights, q);
    sycl::free(device_weights, q);
    sycl::free(device_act, q);
    sycl::free(device_out, q);

    return best_us;
}

int main() {
    printf("=== Expert Cache Bandwidth Microbenchmark ===\n");
    printf("Expert weight: %zu KB, Activation: %zu KB\n",
           EXPERT_BYTES / 1024, ACTIVATION_BYTES / 1024);
    printf("Warmup: %d iters, Bench: %d iters (best of N)\n\n",
           WARMUP_ITERS, BENCH_ITERS);

    // Discover Level Zero GPUs
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto & p : platforms) {
        for (auto & d : p.get_devices(sycl::info::device_type::gpu)) {
            auto backend = d.get_backend();
            if (backend == sycl::backend::ext_oneapi_level_zero) {
                gpus.push_back(d);
            }
        }
    }

    if (gpus.size() < 2) {
        printf("ERROR: Need at least 2 Level Zero GPUs (found %zu)\n", gpus.size());
        printf("Run with: ONEAPI_DEVICE_SELECTOR=\"level_zero:0,1\"\n");
        return 1;
    }

    // Identify devices (device 0 = B580, device 1 = B50 by convention)
    for (size_t i = 0; i < gpus.size(); i++) {
        printf("GPU %zu: %s\n", i, gpus[i].get_info<sycl::info::device::name>().c_str());
        printf("  Global mem: %zu MB\n",
               gpus[i].get_info<sycl::info::device::global_mem_size>() / (1024 * 1024));
        printf("  Max compute units: %u\n",
               gpus[i].get_info<sycl::info::device::max_compute_units>());
    }
    printf("\n");

    sycl::queue q_b580(gpus[0], sycl::property::queue::in_order{});
    sycl::queue q_b50(gpus[1], sycl::property::queue::in_order{});

    // Test 1: VRAM read bandwidth on each GPU
    printf("--- Test 1: VRAM Read Bandwidth (%zu KB buffer) ---\n", EXPERT_BYTES / 1024);
    double bw_b580 = measure_vram_bandwidth(q_b580);
    double bw_b50  = measure_vram_bandwidth(q_b50);
    printf("  B580 VRAM read: %.1f GB/s\n", bw_b580);
    printf("  B50  VRAM read: %.1f GB/s\n", bw_b50);
    printf("\n");

    // Test 2: Host→GPU H2D bandwidth
    printf("--- Test 2: Host→GPU H2D Bandwidth (%zu KB) ---\n", EXPERT_BYTES / 1024);
    double h2d_b580 = measure_h2d_bandwidth(q_b580);
    double h2d_b50  = measure_h2d_bandwidth(q_b50);
    printf("  Host→B580 H2D: %.1f GB/s\n", h2d_b580);
    printf("  Host→B50  H2D: %.1f GB/s\n", h2d_b50);
    printf("\n");

    // Test 3: B50-local compute (expert cached in B50 VRAM)
    printf("--- Test 3: B50-Local Compute Latency ---\n");
    printf("  H2D activation (%zu KB) + dot kernel + D2H result\n", ACTIVATION_BYTES / 1024);
    double local_us = measure_local_compute_latency(q_b50);
    printf("  B50-local round-trip: %.0f us\n", local_us);
    printf("\n");

    // Test 4: B580-stream compute (expert streamed from host)
    printf("--- Test 4: B580-Stream Compute Latency ---\n");
    printf("  H2D expert weight (%zu KB) + dot kernel on B580\n", EXPERT_BYTES / 1024);
    double stream_us = measure_stream_compute_latency(q_b580);
    printf("  B580-stream latency: %.0f us\n", stream_us);
    printf("\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("  B50-local:   %.0f us (expert pre-cached in B50 VRAM)\n", local_us);
    printf("  B580-stream: %.0f us (4MB expert upload + compute on B580)\n", stream_us);
    if (stream_us > local_us) {
        printf("  B50-local is %.1fx FASTER than B580-stream\n", stream_us / local_us);
        printf("  --> B50 expert caching WINS\n");
    } else {
        printf("  B580-stream is %.1fx FASTER than B50-local\n", local_us / stream_us);
        printf("  --> B580 streaming WINS (B50 too slow for compute)\n");
    }

    return 0;
}
