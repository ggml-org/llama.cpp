/*
 * CXL Type 2 Device Dual-GPU Benchmark
 *
 * Benchmarks the performance of one or two CXL Type 2 devices
 * connected to host GPUs, measuring:
 *   - Memory allocation/deallocation latency
 *   - Host-to-Device / Device-to-Host transfer bandwidth
 *   - Concurrent dual-device transfer throughput
 *   - Expert tensor placement simulation for MoE models
 */

#include "cxl-device.h"
#include "cxl_gpu_cmd.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>

// Suppress volatile cast warnings - intentional for MMIO
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

using Clock = std::chrono::high_resolution_clock;

static double to_ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

static double to_gbps(size_t bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return (double)bytes / (1024.0 * 1024.0 * 1024.0) / (ms / 1000.0);
}

// ----------------------------------------------------------------------------
// Benchmark: memory allocation / deallocation
// ----------------------------------------------------------------------------

static void bench_alloc(struct cxl_device * dev, const char * name) {
    printf("\n--- Memory Allocation (%s) ---\n", name);
    printf("  %-20s  %12s  %12s\n", "Size", "Alloc (us)", "Free (us)");

    static const size_t sizes[] = {
        4 * 1024,               // 4 KB
        64 * 1024,              // 64 KB
        1 * 1024 * 1024,        // 1 MB
        16 * 1024 * 1024,       // 16 MB
        256 * 1024 * 1024,      // 256 MB
    };

    for (size_t sz : sizes) {
        const int iters = 20;
        double alloc_total = 0, free_total = 0;
        bool ok = true;

        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            uint64_t ptr = cxl_device_alloc(dev, sz);
            auto t1 = Clock::now();

            if (ptr == 0) {
                ok = false;
                break;
            }

            auto t2 = Clock::now();
            cxl_device_free(dev, ptr);
            auto t3 = Clock::now();

            alloc_total += to_ms(t1 - t0);
            free_total  += to_ms(t3 - t2);
        }

        char sz_str[32];
        if (sz >= 1024 * 1024) {
            snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024 * 1024));
        } else {
            snprintf(sz_str, sizeof(sz_str), "%zu KB", sz / 1024);
        }

        if (ok) {
            printf("  %-20s  %9.1f     %9.1f\n",
                   sz_str,
                   (alloc_total / iters) * 1000.0,
                   (free_total / iters) * 1000.0);
        } else {
            printf("  %-20s  FAILED (out of memory)\n", sz_str);
        }
    }
}

// ----------------------------------------------------------------------------
// Benchmark: data transfer bandwidth
// ----------------------------------------------------------------------------

static void bench_transfer(struct cxl_device * dev, const char * name) {
    printf("\n--- Data Transfer Bandwidth (%s) ---\n", name);
    printf("  %-12s  %12s  %12s  %12s  %12s\n",
           "Size", "HtoD (GB/s)", "HtoD (ms)", "DtoH (GB/s)", "DtoH (ms)");

    static const size_t sizes[] = {
        4 * 1024,               // 4 KB
        64 * 1024,              // 64 KB
        1 * 1024 * 1024,        // 1 MB
        16 * 1024 * 1024,       // 16 MB
        64 * 1024 * 1024,       // 64 MB
    };

    for (size_t sz : sizes) {
        // Allocate device memory
        uint64_t dev_ptr = cxl_device_alloc(dev, sz);
        if (dev_ptr == 0) {
            printf("  %-12zu  FAILED (alloc)\n", sz);
            continue;
        }

        // Allocate and fill host buffer
        std::vector<uint8_t> host_buf(sz);
        for (size_t i = 0; i < sz; i++) {
            host_buf[i] = (uint8_t)(i & 0xFF);
        }

        const int iters = (sz >= 16 * 1024 * 1024) ? 5 : 10;

        // Warmup
        cxl_device_htod(dev, dev_ptr, host_buf.data(), sz);
        cxl_device_dtoh(dev, host_buf.data(), dev_ptr, sz);

        // Benchmark HtoD
        double htod_ms = 0;
        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            cxl_device_htod(dev, dev_ptr, host_buf.data(), sz);
            auto t1 = Clock::now();
            htod_ms += to_ms(t1 - t0);
        }
        htod_ms /= iters;

        // Benchmark DtoH
        double dtoh_ms = 0;
        for (int i = 0; i < iters; i++) {
            auto t0 = Clock::now();
            cxl_device_dtoh(dev, host_buf.data(), dev_ptr, sz);
            auto t1 = Clock::now();
            dtoh_ms += to_ms(t1 - t0);
        }
        dtoh_ms /= iters;

        char sz_str[32];
        if (sz >= 1024 * 1024) {
            snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024 * 1024));
        } else {
            snprintf(sz_str, sizeof(sz_str), "%zu KB", sz / 1024);
        }

        printf("  %-12s  %9.2f     %9.3f     %9.2f     %9.3f\n",
               sz_str,
               to_gbps(sz, htod_ms), htod_ms,
               to_gbps(sz, dtoh_ms), dtoh_ms);

        cxl_device_free(dev, dev_ptr);
    }
}

// ----------------------------------------------------------------------------
// Benchmark: command round-trip latency
// ----------------------------------------------------------------------------

static void bench_latency(struct cxl_device * dev, const char * name) {
    printf("\n--- Command Round-Trip Latency (%s) ---\n", name);

    // Allocate a small buffer for latency tests
    uint64_t dev_ptr = cxl_device_alloc(dev, 4096);
    if (dev_ptr == 0) {
        printf("  FAILED (alloc)\n");
        return;
    }

    uint8_t byte = 0;
    const int iters = 1000;

    // Measure 1-byte HtoD latency (pure command overhead)
    double htod_1b_total = 0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        cxl_device_htod(dev, dev_ptr, &byte, 1);
        auto t1 = Clock::now();
        htod_1b_total += to_ms(t1 - t0);
    }

    // Measure 1-byte DtoH latency
    double dtoh_1b_total = 0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        cxl_device_dtoh(dev, &byte, dev_ptr, 1);
        auto t1 = Clock::now();
        dtoh_1b_total += to_ms(t1 - t0);
    }

    // Measure memset latency
    double memset_total = 0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        cxl_device_memset(dev, dev_ptr, 0, 4096);
        auto t1 = Clock::now();
        memset_total += to_ms(t1 - t0);
    }

    printf("  HtoD 1-byte:    %8.1f us  (avg over %d)\n", (htod_1b_total / iters) * 1000.0, iters);
    printf("  DtoH 1-byte:    %8.1f us  (avg over %d)\n", (dtoh_1b_total / iters) * 1000.0, iters);
    printf("  Memset 4KB:     %8.1f us  (avg over %d)\n", (memset_total / iters) * 1000.0, iters);

    cxl_device_free(dev, dev_ptr);
}

// ----------------------------------------------------------------------------
// Benchmark: concurrent dual-device transfers
// ----------------------------------------------------------------------------

struct xfer_result {
    double ms;
    size_t bytes;
};

static xfer_result do_htod(struct cxl_device * dev, size_t sz, int iters) {
    uint64_t dev_ptr = cxl_device_alloc(dev, sz);
    if (dev_ptr == 0) return {-1, 0};

    std::vector<uint8_t> buf(sz, 0xAB);

    // Warmup
    cxl_device_htod(dev, dev_ptr, buf.data(), sz);

    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        cxl_device_htod(dev, dev_ptr, buf.data(), sz);
    }
    auto t1 = Clock::now();

    cxl_device_free(dev, dev_ptr);
    return {to_ms(t1 - t0), sz * (size_t)iters};
}

static void bench_concurrent(struct cxl_device * dev0, struct cxl_device * dev1) {
    printf("\n--- Concurrent Dual-Device Transfers ---\n");

    static const size_t sizes[] = {
        1  * 1024 * 1024,       // 1 MB
        16 * 1024 * 1024,       // 16 MB
        64 * 1024 * 1024,       // 64 MB
    };

    printf("  %-12s  %15s  %15s  %10s\n",
           "Size", "Sequential", "Concurrent", "Speedup");

    for (size_t sz : sizes) {
        const int iters = (sz >= 64 * 1024 * 1024) ? 3 : 5;

        // Sequential: dev0 then dev1
        auto seq_r0 = do_htod(dev0, sz, iters);
        auto seq_r1 = do_htod(dev1, sz, iters);

        if (seq_r0.ms < 0 || seq_r1.ms < 0) {
            printf("  %-12zu  FAILED (alloc)\n", sz);
            continue;
        }

        double seq_ms = seq_r0.ms + seq_r1.ms;
        double seq_gbps = to_gbps(seq_r0.bytes + seq_r1.bytes, seq_ms);

        // Concurrent: dev0 and dev1 in parallel
        xfer_result conc_r0, conc_r1;
        auto t0 = Clock::now();

        std::thread t_dev0([&]() { conc_r0 = do_htod(dev0, sz, iters); });
        std::thread t_dev1([&]() { conc_r1 = do_htod(dev1, sz, iters); });
        t_dev0.join();
        t_dev1.join();

        auto t1 = Clock::now();
        double conc_ms = to_ms(t1 - t0);
        double conc_gbps = to_gbps(conc_r0.bytes + conc_r1.bytes, conc_ms);

        char sz_str[32];
        snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024 * 1024));

        printf("  %-12s  %8.2f GB/s    %8.2f GB/s    %7.2fx\n",
               sz_str, seq_gbps, conc_gbps, conc_gbps / seq_gbps);
    }
}

// ----------------------------------------------------------------------------
// Benchmark: expert tensor placement simulation
// ----------------------------------------------------------------------------

static void bench_expert_placement(struct cxl_device * dev0, struct cxl_device * dev1) {
    printf("\n--- Expert Tensor Load Simulation ---\n");
    printf("  Simulating MoE model with 8 experts, 2 devices\n");

    // Simulate loading 8 expert weight tensors
    // Typical expert tensor: ~128 MiB (e.g., 7168 x 2048 x float16 x 3 matrices)
    const int n_experts = 8;
    const size_t expert_size = 32 * 1024 * 1024;  // 32 MiB per expert (scaled down)
    const int experts_per_device = n_experts / 2;

    std::vector<uint8_t> tensor_data(expert_size, 0x42);

    // --- Single device: all experts on dev0 ---
    printf("\n  Single device (all %d experts on %s):\n", n_experts, dev0->name);
    {
        std::vector<uint64_t> ptrs(n_experts);
        auto t0 = Clock::now();
        for (int i = 0; i < n_experts; i++) {
            ptrs[i] = cxl_device_alloc(dev0, expert_size);
            if (ptrs[i] == 0) {
                printf("    FAILED: could not allocate expert %d\n", i);
                for (int j = 0; j < i; j++) cxl_device_free(dev0, ptrs[j]);
                return;
            }
            cxl_device_htod(dev0, ptrs[i], tensor_data.data(), expert_size);
        }
        auto t1 = Clock::now();
        double single_ms = to_ms(t1 - t0);

        printf("    Total load time: %8.2f ms\n", single_ms);
        printf("    Throughput:      %8.2f GB/s\n",
               to_gbps((size_t)n_experts * expert_size, single_ms));

        for (int i = 0; i < n_experts; i++) cxl_device_free(dev0, ptrs[i]);
    }

    // --- Two devices: split experts across dev0 and dev1 ---
    printf("\n  Dual device (%d experts each on %s, %s):\n",
           experts_per_device, dev0->name, dev1->name);
    {
        std::vector<uint64_t> ptrs0(experts_per_device);
        std::vector<uint64_t> ptrs1(experts_per_device);

        auto t0 = Clock::now();

        // Load in parallel to both devices
        std::thread load0([&]() {
            for (int i = 0; i < experts_per_device; i++) {
                ptrs0[i] = cxl_device_alloc(dev0, expert_size);
                if (ptrs0[i] != 0) {
                    cxl_device_htod(dev0, ptrs0[i], tensor_data.data(), expert_size);
                }
            }
        });
        std::thread load1([&]() {
            for (int i = 0; i < experts_per_device; i++) {
                ptrs1[i] = cxl_device_alloc(dev1, expert_size);
                if (ptrs1[i] != 0) {
                    cxl_device_htod(dev1, ptrs1[i], tensor_data.data(), expert_size);
                }
            }
        });
        load0.join();
        load1.join();

        auto t1 = Clock::now();
        double dual_ms = to_ms(t1 - t0);

        printf("    Total load time: %8.2f ms\n", dual_ms);
        printf("    Throughput:      %8.2f GB/s\n",
               to_gbps((size_t)n_experts * expert_size, dual_ms));

        for (int i = 0; i < experts_per_device; i++) {
            if (ptrs0[i]) cxl_device_free(dev0, ptrs0[i]);
            if (ptrs1[i]) cxl_device_free(dev1, ptrs1[i]);
        }
    }

    // --- Two devices: round-robin by layer (current impl) ---
    printf("\n  Round-robin layers (alternating experts on 2 devices):\n");
    {
        std::vector<uint64_t> ptrs(n_experts);
        auto t0 = Clock::now();

        for (int i = 0; i < n_experts; i++) {
            struct cxl_device * dev = (i % 2 == 0) ? dev0 : dev1;
            ptrs[i] = cxl_device_alloc(dev, expert_size);
            if (ptrs[i] != 0) {
                cxl_device_htod(dev, ptrs[i], tensor_data.data(), expert_size);
            }
        }

        auto t1 = Clock::now();
        double rr_ms = to_ms(t1 - t0);

        printf("    Total load time: %8.2f ms  (sequential, alternating devices)\n", rr_ms);
        printf("    Throughput:      %8.2f GB/s\n",
               to_gbps((size_t)n_experts * expert_size, rr_ms));

        for (int i = 0; i < n_experts; i++) {
            struct cxl_device * dev = (i % 2 == 0) ? dev0 : dev1;
            if (ptrs[i]) cxl_device_free(dev, ptrs[i]);
        }
    }
}

// ----------------------------------------------------------------------------
// Benchmark: graph compute (if supported by host)
// ----------------------------------------------------------------------------

static void bench_graph_compute(struct cxl_device * dev, const char * name) {
    printf("\n--- Graph Compute Probe (%s) ---\n", name);

    // Create a minimal serialized graph to test if graph compute is supported
    // Header: magic + version + n_nodes + n_tensors
    uint32_t header[4] = {
        0x47475347,  // "GGSG" magic
        1,           // version
        0,           // n_nodes (empty graph)
        0,           // n_tensors
    };

    int status = cxl_device_graph_compute(dev, header, sizeof(header));
    if (status == 0) {
        printf("  Graph compute: SUPPORTED (empty graph executed successfully)\n");
    } else if (status == -1) {
        printf("  Graph compute: NOT SUPPORTED or TIMEOUT (host handler may not be implemented)\n");
    } else {
        printf("  Graph compute: returned status %d\n", status);
    }
}

// ----------------------------------------------------------------------------
// Benchmark: raw BAR2/BAR4 MMIO bandwidth (works without backend)
// ----------------------------------------------------------------------------

// MMIO-safe 64-bit copy (matches cxl-device.cpp)
static void bar_write64(volatile void * dst, const void * src, size_t len) {
    volatile uint64_t * d = (volatile uint64_t *)dst;
    const uint64_t * s = (const uint64_t *)src;
    size_t n = (len + 7) / 8;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    __sync_synchronize();
}

static void bar_read64(void * dst, const volatile void * src, size_t len) {
    uint64_t * d = (uint64_t *)dst;
    volatile uint64_t * s = (volatile uint64_t *)src;
    size_t n = (len + 7) / 8;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    __sync_synchronize();
}

static void bench_bar_bandwidth(struct cxl_device * dev, const char * name) {
    printf("\n--- Raw BAR MMIO Bandwidth (%s) ---\n", name);
    printf("  (Direct MMIO read/write, no GPU commands - measures CXL interconnect)\n");

    // Test BAR2 data region (1 MB)
    size_t bar2_data_size = CXL_GPU_DATA_SIZE;  // 1 MB
    volatile uint8_t * bar2_data = dev->data;

    if (!bar2_data) {
        printf("  BAR2 data region not available\n");
        return;
    }

    printf("  %-20s  %12s  %12s  %12s  %12s\n",
           "Region (size)", "Write (GB/s)", "Write (ms)", "Read (GB/s)", "Read (ms)");

    // Allocate host buffer
    static const size_t test_sizes[] = { 4096, 64*1024, 256*1024, 1024*1024 };

    for (size_t sz : test_sizes) {
        if (sz > bar2_data_size) break;

        std::vector<uint8_t> host_buf(sz);
        for (size_t i = 0; i < sz; i++) host_buf[i] = (uint8_t)(i & 0xFF);
        std::vector<uint8_t> read_buf(sz, 0);

        const int iters = (sz >= 256*1024) ? 10 : 50;

        // Write benchmark
        auto t0 = Clock::now();
        for (int i = 0; i < iters; i++) {
            bar_write64(bar2_data, host_buf.data(), sz);
        }
        auto t1 = Clock::now();
        double write_ms = to_ms(t1 - t0) / iters;

        // Read benchmark
        t0 = Clock::now();
        for (int i = 0; i < iters; i++) {
            bar_read64(read_buf.data(), bar2_data, sz);
        }
        t1 = Clock::now();
        double read_ms = to_ms(t1 - t0) / iters;

        char sz_str[32];
        if (sz >= 1024 * 1024) {
            snprintf(sz_str, sizeof(sz_str), "BAR2 (%zu MB)", sz / (1024 * 1024));
        } else {
            snprintf(sz_str, sizeof(sz_str), "BAR2 (%zu KB)", sz / 1024);
        }

        printf("  %-20s  %9.3f     %9.3f     %9.3f     %9.3f\n",
               sz_str, to_gbps(sz, write_ms), write_ms, to_gbps(sz, read_ms), read_ms);
    }

    // Test BAR4 if available
    if (dev->bar4 && dev->bar4_size > 0) {
        static const size_t bar4_test_sizes[] = { 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024 };

        for (size_t sz : bar4_test_sizes) {
            if (sz > dev->bar4_size) break;

            std::vector<uint8_t> host_buf(sz);
            for (size_t i = 0; i < sz; i++) host_buf[i] = (uint8_t)(i & 0xFF);
            std::vector<uint8_t> read_buf(sz, 0);

            const int iters = (sz >= 16*1024*1024) ? 3 : 10;

            // Write benchmark
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                bar_write64(dev->bar4, host_buf.data(), sz);
            }
            auto t1 = Clock::now();
            double write_ms = to_ms(t1 - t0) / iters;

            // Read benchmark
            t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                bar_read64(read_buf.data(), dev->bar4, sz);
            }
            t1 = Clock::now();
            double read_ms = to_ms(t1 - t0) / iters;

            char sz_str[32];
            if (sz >= 1024 * 1024) {
                snprintf(sz_str, sizeof(sz_str), "BAR4 (%zu MB)", sz / (1024 * 1024));
            } else {
                snprintf(sz_str, sizeof(sz_str), "BAR4 (%zu KB)", sz / 1024);
            }

            printf("  %-20s  %9.3f     %9.3f     %9.3f     %9.3f\n",
                   sz_str, to_gbps(sz, write_ms), write_ms, to_gbps(sz, read_ms), read_ms);
        }
    } else {
        printf("  BAR4 not available\n");
    }

    // Data verification on BAR2 data region
    {
        printf("\n  BAR2 data region verification: ");
        size_t verify_sz = 4096;
        std::vector<uint8_t> pattern(verify_sz);
        for (size_t i = 0; i < verify_sz; i++) pattern[i] = (uint8_t)((i * 7 + 13) & 0xFF);

        bar_write64(bar2_data, pattern.data(), verify_sz);

        std::vector<uint8_t> readback(verify_sz, 0);
        bar_read64(readback.data(), bar2_data, verify_sz);

        int errors = 0;
        for (size_t i = 0; i < verify_sz; i++) {
            if (pattern[i] != readback[i]) errors++;
        }
        if (errors == 0) {
            printf("PASSED (%zu bytes)\n", verify_sz);
        } else {
            printf("FAILED (%d mismatches in %zu bytes)\n", errors, verify_sz);
        }
    }
}

// ----------------------------------------------------------------------------
// Benchmark: concurrent BAR bandwidth (dual-device, works without backend)
// ----------------------------------------------------------------------------

static void bench_concurrent_bar(struct cxl_device * dev0, struct cxl_device * dev1) {
    printf("\n--- Concurrent Dual-Device BAR Bandwidth ---\n");
    printf("  (Measures if two CXL devices can sustain parallel bandwidth)\n");

    size_t sz = 1024 * 1024;  // 1 MB
    const int iters = 20;

    std::vector<uint8_t> buf0(sz, 0xAA);
    std::vector<uint8_t> buf1(sz, 0xBB);

    // Sequential: write to dev0, then dev1
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        bar_write64(dev0->data, buf0.data(), sz);
    }
    for (int i = 0; i < iters; i++) {
        bar_write64(dev1->data, buf1.data(), sz);
    }
    auto t1 = Clock::now();
    double seq_ms = to_ms(t1 - t0);
    double seq_gbps = to_gbps(sz * iters * 2, seq_ms);

    // Concurrent: write to both simultaneously
    t0 = Clock::now();
    std::thread tw0([&]() {
        for (int i = 0; i < iters; i++) bar_write64(dev0->data, buf0.data(), sz);
    });
    std::thread tw1([&]() {
        for (int i = 0; i < iters; i++) bar_write64(dev1->data, buf1.data(), sz);
    });
    tw0.join();
    tw1.join();
    t1 = Clock::now();
    double conc_ms = to_ms(t1 - t0);
    double conc_gbps = to_gbps(sz * iters * 2, conc_ms);

    printf("  Write 1MB x %d (BAR2 data region):\n", iters);
    printf("    Sequential:  %8.3f GB/s  (%8.2f ms)\n", seq_gbps, seq_ms);
    printf("    Concurrent:  %8.3f GB/s  (%8.2f ms)\n", conc_gbps, conc_ms);
    printf("    Speedup:     %8.2fx\n", conc_gbps / seq_gbps);
}

// ----------------------------------------------------------------------------
// Data verification
// ----------------------------------------------------------------------------

static void bench_verify(struct cxl_device * dev, const char * name) {
    printf("\n--- Data Integrity Verification (%s) ---\n", name);

    const size_t sz = 1 * 1024 * 1024;  // 1 MB
    uint64_t dev_ptr = cxl_device_alloc(dev, sz);
    if (dev_ptr == 0) {
        printf("  FAILED (alloc)\n");
        return;
    }

    // Write pattern
    std::vector<uint8_t> send_buf(sz);
    for (size_t i = 0; i < sz; i++) {
        send_buf[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    }

    cxl_device_htod(dev, dev_ptr, send_buf.data(), sz);

    // Read back
    std::vector<uint8_t> recv_buf(sz, 0);
    cxl_device_dtoh(dev, recv_buf.data(), dev_ptr, sz);

    // Compare
    int errors = 0;
    for (size_t i = 0; i < sz; i++) {
        if (send_buf[i] != recv_buf[i]) {
            if (errors < 5) {
                printf("  Mismatch at offset %zu: sent 0x%02x, got 0x%02x\n",
                       i, send_buf[i], recv_buf[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASSED: %zu bytes verified\n", sz);
    } else {
        printf("  FAILED: %d mismatches in %zu bytes\n", errors, sz);
    }

    cxl_device_free(dev, dev_ptr);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h, --help           Show this help\n");
    printf("  -d, --device <n>     Benchmark only device n (default: all)\n");
    printf("  --no-concurrent      Skip concurrent dual-device tests\n");
    printf("  --no-expert          Skip expert tensor simulation\n");
    printf("  --no-verify          Skip data integrity verification\n");
    printf("  --no-graph           Skip graph compute probe\n");
}

int main(int argc, char ** argv) {
    int target_device = -1;     // -1 = all
    bool do_concurrent = true;
    bool do_expert     = true;
    bool do_verify     = true;
    bool do_graph      = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            target_device = atoi(argv[++i]);
        } else if (arg == "--no-concurrent") {
            do_concurrent = false;
        } else if (arg == "--no-expert") {
            do_expert = false;
        } else if (arg == "--no-verify") {
            do_verify = false;
        } else if (arg == "--no-graph") {
            do_graph = false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=== CXL Type 2 Device Dual-GPU Benchmark ===\n\n");

    // Discover devices
    struct cxl_device devices[CXL_MAX_DEVICES];
    int n_devices = cxl_device_discover_all(devices, CXL_MAX_DEVICES);

    if (n_devices == 0) {
        printf("No CXL Type 2 devices found.\n");
        printf("  Expected PCI vendor=0x%04x device=0x%04x\n",
               CXL_GPU_PCI_VENDOR_ID, CXL_GPU_PCI_DEVICE_ID);
        return 1;
    }

    printf("Discovered %d CXL Type 2 device(s)\n", n_devices);

    // Map devices
    int mapped = 0;
    for (int i = 0; i < n_devices; i++) {
        if (cxl_device_map(&devices[i]) != 0) {
            fprintf(stderr, "  Failed to map device %d (%s)\n", i, devices[i].pci_addr);
            continue;
        }
        if (mapped != i) {
            devices[mapped] = devices[i];
        }
        devices[mapped].index = mapped;
        mapped++;
    }
    n_devices = mapped;

    if (n_devices == 0) {
        printf("No devices could be mapped.\n");
        return 1;
    }

    printf("Mapped %d device(s):\n", n_devices);
    bool any_ctx_active = false;
    for (int i = 0; i < n_devices; i++) {
        printf("  Device %d: %s at %s", i, devices[i].name, devices[i].pci_addr);
        if (devices[i].total_memory > 0) {
            printf(" (%.0f MiB)", (double)devices[i].total_memory / (1024.0 * 1024.0));
        }
        printf(" [%s]", devices[i].ctx_active ? "GPU ready" : "BAR only");
        printf("\n");
        if (devices[i].ctx_active) any_ctx_active = true;
    }

    if (!any_ctx_active) {
        printf("\n  NOTE: No GPU contexts active (host backend may not be running).\n");
        printf("  Running BAR MMIO bandwidth tests only.\n");
    }

    // Filter to target device if specified
    int start_dev = 0, end_dev = n_devices;
    if (target_device >= 0) {
        if (target_device >= n_devices) {
            fprintf(stderr, "Device %d not found (have %d devices)\n", target_device, n_devices);
            return 1;
        }
        start_dev = target_device;
        end_dev = target_device + 1;
    }

    // Per-device benchmarks
    for (int i = start_dev; i < end_dev; i++) {
        printf("\n============================================================\n");
        printf("Device %d: %s\n", i, devices[i].name);
        printf("============================================================\n");

        // Raw BAR bandwidth test (always works - no GPU commands needed)
        bench_bar_bandwidth(&devices[i], devices[i].name);

        // GPU command benchmarks (require active context)
        if (devices[i].ctx_active) {
            if (do_verify) {
                bench_verify(&devices[i], devices[i].name);
            }

            bench_latency(&devices[i], devices[i].name);
            bench_alloc(&devices[i], devices[i].name);
            bench_transfer(&devices[i], devices[i].name);

            if (do_graph) {
                bench_graph_compute(&devices[i], devices[i].name);
            }
        } else {
            printf("\n  (Skipping GPU command benchmarks: no active context)\n");
        }
    }

    // Dual-device benchmarks (require 2+ devices)
    if (n_devices >= 2 && target_device < 0) {
        printf("\n============================================================\n");
        printf("Dual-Device Benchmarks: %s + %s\n", devices[0].name, devices[1].name);
        printf("============================================================\n");

        // BAR concurrent test always works
        bench_concurrent_bar(&devices[0], &devices[1]);

        // GPU command dual-device tests require active contexts
        if (devices[0].ctx_active && devices[1].ctx_active) {
            if (do_concurrent) {
                bench_concurrent(&devices[0], &devices[1]);
            }

            if (do_expert) {
                bench_expert_placement(&devices[0], &devices[1]);
            }
        } else {
            printf("\n  (Skipping GPU dual-device benchmarks: contexts not active)\n");
        }
    } else if (n_devices < 2 && do_concurrent) {
        printf("\n  (Skipping dual-device benchmarks: need 2+ devices, found %d)\n", n_devices);
    }

    printf("\n=== Benchmark Complete ===\n");

    // Cleanup
    for (int i = 0; i < n_devices; i++) {
        cxl_device_unmap(&devices[i]);
    }

    return 0;
}
